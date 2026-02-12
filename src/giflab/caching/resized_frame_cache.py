"""
Resized Frame Cache for GifLab.

This module provides caching for resized frames to avoid redundant resizing operations
across different metrics. It includes memory pooling for efficient buffer reuse.
"""

import hashlib
import logging
import threading
import time
import weakref
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class InterpolationMethod(Enum):
    """Supported interpolation methods for resizing."""

    AREA = cv2.INTER_AREA
    LANCZOS4 = cv2.INTER_LANCZOS4
    CUBIC = cv2.INTER_CUBIC
    LINEAR = cv2.INTER_LINEAR
    NEAREST = cv2.INTER_NEAREST

    @classmethod
    def from_cv2(cls, cv2_flag: int) -> "InterpolationMethod":
        """Convert OpenCV interpolation flag to enum."""
        for method in cls:
            if method.value == cv2_flag:
                return method
        # Default to AREA if unknown
        return cls.AREA


@dataclass
class ResizedFrameInfo:
    """Metadata for a cached resized frame."""

    frame: np.ndarray
    original_hash: str
    target_size: tuple[int, int]  # (width, height)
    interpolation: InterpolationMethod
    timestamp: float
    memory_size: int
    hit_count: int = 0


class FrameBufferPool:
    """
    Memory pool for reusing frame buffers to reduce allocation overhead.
    """

    def __init__(self, max_buffers_per_size: int = 10):
        """
        Initialize the frame buffer pool.

        Args:
            max_buffers_per_size: Maximum number of buffers to keep per size
        """
        self.pools: dict[tuple[int, ...], list[np.ndarray]] = {}
        self.max_buffers_per_size = max_buffers_per_size
        self._lock = threading.RLock()
        self._stats = {
            "allocations": 0,
            "reuses": 0,
            "releases": 0,
        }

    def get_buffer(
        self, shape: tuple[int, ...], dtype: np.dtype = np.dtype(np.uint8)
    ) -> np.ndarray:
        """
        Get a buffer from the pool or allocate a new one.

        Args:
            shape: Shape of the required buffer
            dtype: Data type of the buffer (accepts np.dtype or numpy type like np.uint8)

        Returns:
            numpy array buffer
        """
        dtype = np.dtype(dtype)
        with self._lock:
            pool_key = (*shape, dtype.num)

            if pool_key in self.pools and self.pools[pool_key]:
                self._stats["reuses"] += 1
                return self.pools[pool_key].pop()

            # Allocate new buffer
            self._stats["allocations"] += 1
            return np.empty(shape, dtype=dtype)

    def release_buffer(self, buffer: np.ndarray) -> None:
        """
        Return a buffer to the pool for reuse.

        Args:
            buffer: Buffer to release back to the pool
        """
        with self._lock:
            pool_key = (*buffer.shape, buffer.dtype.num)

            if pool_key not in self.pools:
                self.pools[pool_key] = []

            if len(self.pools[pool_key]) < self.max_buffers_per_size:
                self.pools[pool_key].append(buffer)
                self._stats["releases"] += 1

    def clear(self) -> None:
        """Clear all buffers from the pool."""
        with self._lock:
            self.pools.clear()

    def get_stats(self) -> dict[str, Any]:
        """Get pool statistics."""
        with self._lock:
            total_buffers = sum(len(pool) for pool in self.pools.values())
            total_memory = sum(
                sum(buf.nbytes for buf in pool) for pool in self.pools.values()
            )
            return {
                **self._stats,
                "total_buffers": total_buffers,
                "total_memory_mb": total_memory / (1024 * 1024),
                "reuse_rate": (
                    self._stats["reuses"]
                    / max(1, self._stats["allocations"] + self._stats["reuses"])
                ),
            }


class ResizedFrameCache:
    """
    LRU cache for resized frames with memory management and buffer pooling.
    """

    def __init__(
        self,
        memory_limit_mb: float = 200,
        enable_pooling: bool = True,
        ttl_seconds: float = 3600,
    ):
        """
        Initialize the resized frame cache.

        Args:
            memory_limit_mb: Maximum memory to use for cache in MB
            enable_pooling: Whether to use buffer pooling
            ttl_seconds: Time to live for cache entries in seconds
        """
        self.memory_limit = int(memory_limit_mb * 1024 * 1024)
        self.ttl = ttl_seconds
        self.enable_pooling = enable_pooling

        self._cache: OrderedDict[str, ResizedFrameInfo] = OrderedDict()
        self._current_memory = 0
        self._lock = threading.RLock()

        # Buffer pool for memory reuse
        self._buffer_pool = FrameBufferPool() if enable_pooling else None

        # Statistics
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "ttl_evictions": 0,
        }

        logger.info(
            f"Initialized ResizedFrameCache: memory_limit={memory_limit_mb}MB, "
            f"pooling={'enabled' if enable_pooling else 'disabled'}, ttl={ttl_seconds}s"
        )

    def _generate_cache_key(
        self,
        frame_hash: str,
        target_size: tuple[int, int],
        interpolation: InterpolationMethod,
    ) -> str:
        """
        Generate a unique cache key for a resized frame.

        Args:
            frame_hash: Hash of the original frame
            target_size: Target size (width, height)
            interpolation: Interpolation method

        Returns:
            Cache key string
        """
        key_str = f"{frame_hash}:{target_size[0]}x{target_size[1]}:{interpolation.name}"
        return hashlib.md5(key_str.encode()).hexdigest()[:16]

    def _evict_lru(self) -> None:
        """Evict least recently used entries to stay within memory limit."""
        while self._current_memory > self.memory_limit and self._cache:
            _, evicted = self._cache.popitem(last=False)
            self._current_memory -= evicted.memory_size
            self._stats["evictions"] += 1

            # Return buffer to pool if pooling is enabled
            if self._buffer_pool and evicted.frame is not None:
                self._buffer_pool.release_buffer(evicted.frame)

    def _evict_expired(self) -> None:
        """Remove expired entries based on TTL."""
        current_time = time.time()
        expired_keys = []

        for key, info in self._cache.items():
            if current_time - info.timestamp > self.ttl:
                expired_keys.append(key)

        for key in expired_keys:
            info = self._cache.pop(key)
            self._current_memory -= info.memory_size
            self._stats["ttl_evictions"] += 1

            # Return buffer to pool
            if self._buffer_pool and info.frame is not None:
                self._buffer_pool.release_buffer(info.frame)

    def get(
        self,
        frame: np.ndarray,
        target_size: tuple[int, int],
        interpolation: int = cv2.INTER_AREA,
    ) -> np.ndarray | None:
        """
        Get a resized frame from cache or resize and cache it.

        Args:
            frame: Original frame
            target_size: Target size (width, height)
            interpolation: OpenCV interpolation method

        Returns:
            Resized frame or None if not in cache
        """
        # Convert interpolation to enum
        interp_method = InterpolationMethod.from_cv2(interpolation)

        # Generate frame hash
        frame_hash = hashlib.md5(frame.tobytes()).hexdigest()[:16]

        # Generate cache key
        cache_key = self._generate_cache_key(frame_hash, target_size, interp_method)

        with self._lock:
            # Check for expired entries periodically
            if len(self._cache) > 0 and np.random.random() < 0.01:  # 1% chance
                self._evict_expired()

            # Check if in cache
            if cache_key in self._cache:
                # Move to end (most recently used)
                info = self._cache.pop(cache_key)
                info.hit_count += 1
                info.timestamp = time.time()  # Update access time
                self._cache[cache_key] = info
                self._stats["hits"] += 1

                return info.frame.copy()

            self._stats["misses"] += 1

            # Resize the frame
            if self._buffer_pool:
                # Try to get buffer from pool
                resized_shape = (
                    (target_size[1], target_size[0], frame.shape[2])
                    if len(frame.shape) == 3
                    else (target_size[1], target_size[0])
                )
                resized = self._buffer_pool.get_buffer(resized_shape, frame.dtype)
                cv2.resize(frame, target_size, dst=resized, interpolation=interpolation)
            else:
                resized = cv2.resize(frame, target_size, interpolation=interpolation)

            # Create cache entry
            info = ResizedFrameInfo(
                frame=resized.copy(),  # Store a copy
                original_hash=frame_hash,
                target_size=target_size,
                interpolation=interp_method,
                timestamp=time.time(),
                memory_size=resized.nbytes,
            )

            # Add to cache
            self._cache[cache_key] = info
            self._current_memory += info.memory_size

            # Evict if necessary
            self._evict_lru()

            return resized

    def clear(self) -> None:
        """Clear all entries from the cache."""
        with self._lock:
            # Return all buffers to pool
            if self._buffer_pool:
                for info in self._cache.values():
                    if info.frame is not None:
                        self._buffer_pool.release_buffer(info.frame)

            self._cache.clear()
            self._current_memory = 0

            if self._buffer_pool:
                self._buffer_pool.clear()

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_hits = self._stats["hits"]
            total_misses = self._stats["misses"]
            total_requests = total_hits + total_misses

            stats = {
                **self._stats,
                "entries": len(self._cache),
                "memory_mb": self._current_memory / (1024 * 1024),
                "memory_limit_mb": self.memory_limit / (1024 * 1024),
                "hit_rate": total_hits / max(1, total_requests),
                "avg_hit_count": (
                    sum(info.hit_count for info in self._cache.values())
                    / max(1, len(self._cache))
                ),
            }

            if self._buffer_pool:
                stats["buffer_pool"] = self._buffer_pool.get_stats()

            return stats

    def get_most_used(self, top_n: int = 10) -> list[dict[str, Any]]:
        """
        Get the most frequently used cache entries.

        Args:
            top_n: Number of top entries to return

        Returns:
            list of entry information dictionaries
        """
        with self._lock:
            sorted_entries = sorted(
                self._cache.items(),
                key=lambda x: x[1].hit_count,
                reverse=True,
            )[:top_n]

            return [
                {
                    "key": key,
                    "size": info.target_size,
                    "interpolation": info.interpolation.name,
                    "hit_count": info.hit_count,
                    "memory_kb": info.memory_size / 1024,
                }
                for key, info in sorted_entries
            ]


# Global singleton instance
_resize_cache_instance: ResizedFrameCache | None = None
_resize_cache_lock = threading.Lock()


def get_resize_cache(
    memory_limit_mb: float | None = None,
    enable_pooling: bool | None = None,
    ttl_seconds: float | None = None,
) -> ResizedFrameCache:
    """
    Get or create the global resize cache instance.

    Args:
        memory_limit_mb: Maximum memory to use for cache in MB (uses config default if None)
        enable_pooling: Whether to use buffer pooling (uses config default if None)
        ttl_seconds: Time to live for cache entries (uses config default if None)

    Returns:
        Global ResizedFrameCache instance
    """
    global _resize_cache_instance

    # Import config here to avoid circular imports
    from giflab.config import FRAME_CACHE

    # Use config defaults if not specified
    if memory_limit_mb is None:
        memory_limit_mb = FRAME_CACHE.get("resize_cache_memory_mb", 200)
    if enable_pooling is None:
        enable_pooling = FRAME_CACHE.get("enable_buffer_pooling", True)
    if ttl_seconds is None:
        ttl_seconds = FRAME_CACHE.get("resize_cache_ttl_seconds", 3600)

    with _resize_cache_lock:
        if _resize_cache_instance is None:
            _resize_cache_instance = ResizedFrameCache(
                memory_limit_mb=memory_limit_mb,
                enable_pooling=enable_pooling,
                ttl_seconds=ttl_seconds,
            )
        return _resize_cache_instance


def resize_frame_cached(
    frame: np.ndarray,
    target_size: tuple[int, int],
    interpolation: int = cv2.INTER_AREA,
    use_cache: bool = True,
) -> np.ndarray:
    """
    Resize a frame with caching support.

    This is a convenience function that uses the global cache instance.

    Args:
        frame: Frame to resize
        target_size: Target size (width, height)
        interpolation: OpenCV interpolation method
        use_cache: Whether to use the cache

    Returns:
        Resized frame
    """
    # Check if we need to resize at all
    if frame.shape[1] == target_size[0] and frame.shape[0] == target_size[1]:
        return frame

    # Import config to check if caching is enabled
    from giflab.config import FRAME_CACHE

    # Check if caching is globally enabled and locally requested
    cache_enabled = FRAME_CACHE.get("resize_cache_enabled", True) and use_cache

    if not cache_enabled:
        return cv2.resize(frame, target_size, interpolation=interpolation)

    cache = get_resize_cache()
    return cache.get(frame, target_size, interpolation)
