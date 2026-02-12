"""Frame caching system for GIF validation performance optimization."""

import hashlib
import logging
import pickle
import sqlite3
import threading
import time
import zlib
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CacheStats:
    """Statistics for frame cache performance."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    memory_bytes: int = 0
    disk_entries: int = 0
    total_accesses: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        if self.total_accesses == 0:
            return 0.0
        return self.hits / self.total_accesses


@dataclass
class FrameCacheEntry:
    """Single entry in the frame cache."""

    cache_key: str
    frames: list[np.ndarray]
    dimensions: tuple[int, int]
    duration_ms: int
    frame_count: int
    file_path: str
    file_size: int
    file_mtime: float
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0

    def update_access(self) -> None:
        """Update access statistics."""
        self.last_accessed = time.time()
        self.access_count += 1

    def to_bytes(self) -> bytes:
        """Serialize entry for disk storage."""
        # Convert frames to compressed format
        frames_data = []
        for frame in self.frames:
            # Compress each frame individually
            frame_bytes = frame.tobytes()
            compressed = zlib.compress(frame_bytes, level=6)
            frames_data.append(
                {"data": compressed, "shape": frame.shape, "dtype": str(frame.dtype)}
            )

        data = {
            "cache_key": self.cache_key,
            "frames_data": frames_data,
            "dimensions": self.dimensions,
            "duration_ms": self.duration_ms,
            "frame_count": self.frame_count,
            "file_path": self.file_path,
            "file_size": self.file_size,
            "file_mtime": self.file_mtime,
            "created_at": self.created_at,
            "last_accessed": self.last_accessed,
            "access_count": self.access_count,
        }
        return pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def from_bytes(cls, data: bytes) -> "FrameCacheEntry":
        """Deserialize entry from disk storage."""
        obj = pickle.loads(data)

        # Decompress frames
        frames = []
        for frame_data in obj["frames_data"]:
            decompressed = zlib.decompress(frame_data["data"])
            frame = np.frombuffer(
                decompressed, dtype=np.dtype(frame_data["dtype"])
            ).reshape(frame_data["shape"])
            frames.append(frame)

        return cls(
            cache_key=obj["cache_key"],
            frames=frames,
            dimensions=obj["dimensions"],
            duration_ms=obj["duration_ms"],
            frame_count=obj["frame_count"],
            file_path=obj["file_path"],
            file_size=obj["file_size"],
            file_mtime=obj["file_mtime"],
            created_at=obj["created_at"],
            last_accessed=obj["last_accessed"],
            access_count=obj["access_count"],
        )

    def memory_size(self) -> int:
        """Estimate memory size in bytes."""
        # Rough estimate: frames + metadata
        frame_bytes = sum(frame.nbytes for frame in self.frames)
        metadata_bytes = 256  # Approximate overhead
        return frame_bytes + metadata_bytes


class FrameCache:
    """Two-tier frame cache with in-memory LRU and disk persistence."""

    def __init__(
        self,
        memory_limit_mb: int = 500,
        disk_path: Path | None = None,
        disk_limit_mb: int = 2000,
        ttl_seconds: int = 86400,  # 24 hours default
        enabled: bool = True,
    ):
        """Initialize frame cache.

        Args:
            memory_limit_mb: Maximum memory usage in MB
            disk_path: Path to disk cache database (None for temp)
            disk_limit_mb: Maximum disk cache size in MB
            ttl_seconds: Time-to-live for cache entries in seconds
            enabled: Whether caching is enabled
        """
        self.enabled = enabled
        self.memory_limit_bytes = memory_limit_mb * 1024 * 1024
        self.disk_limit_bytes = disk_limit_mb * 1024 * 1024
        self.ttl_seconds = ttl_seconds

        # In-memory LRU cache
        self._memory_cache: OrderedDict[str, FrameCacheEntry] = OrderedDict()
        self._memory_bytes = 0
        self._lock = threading.RLock()

        # Statistics
        self._stats = CacheStats()

        # Disk cache setup
        if disk_path is None:
            disk_path = Path.home() / ".giflab_cache" / "frame_cache.db"

        self.disk_path = disk_path
        self.disk_path.parent.mkdir(parents=True, exist_ok=True)

        if self.enabled:
            self._init_database()
            self._cleanup_expired()

    def _init_database(self) -> None:
        """Initialize SQLite database for disk cache."""
        with sqlite3.connect(str(self.disk_path)) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS frame_cache (
                    cache_key TEXT PRIMARY KEY,
                    data BLOB NOT NULL,
                    file_path TEXT NOT NULL,
                    file_size INTEGER NOT NULL,
                    file_mtime REAL NOT NULL,
                    created_at REAL NOT NULL,
                    last_accessed REAL NOT NULL,
                    access_count INTEGER NOT NULL,
                    data_size INTEGER NOT NULL
                )
            """
            )

            # Create indices for efficient queries
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_last_accessed
                ON frame_cache(last_accessed)
            """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_file_path
                ON frame_cache(file_path)
            """
            )
            conn.commit()

    def generate_cache_key(self, file_path: Path) -> str:
        """Generate stable cache key from file metadata.

        Args:
            file_path: Path to the GIF file

        Returns:
            Hexadecimal cache key string
        """
        try:
            stat = file_path.stat()
            key_data = f"{file_path.absolute()}:{stat.st_mtime}:{stat.st_size}"
            return hashlib.sha256(key_data.encode()).hexdigest()[:32]
        except OSError as e:
            logger.warning(f"Failed to generate cache key for {file_path}: {e}")
            return ""

    def get(
        self, file_path: Path, max_frames: int | None = None
    ) -> tuple[list[np.ndarray], int, tuple[int, int], int] | None:
        """Get frames from cache if available.

        Args:
            file_path: Path to the GIF file
            max_frames: Maximum frames requested (must match cached value)

        Returns:
            Tuple of (frames, frame_count, dimensions, duration_ms) or None
        """
        if not self.enabled:
            return None

        cache_key = self.generate_cache_key(file_path)
        if not cache_key:
            return None

        with self._lock:
            self._stats.total_accesses += 1

            # Check memory cache first
            if cache_key in self._memory_cache:
                entry = self._memory_cache.pop(cache_key)

                # Check TTL
                if time.time() - entry.created_at > self.ttl_seconds:
                    # Entry expired, remove from memory cache
                    self._memory_bytes -= entry.memory_size()
                    self._stats.memory_bytes = self._memory_bytes
                    self._stats.evictions += 1
                    self._stats.misses += 1
                    return None

                entry.update_access()
                self._memory_cache[cache_key] = entry  # Move to end (MRU)

                # Check if frames match requested max_frames
                if max_frames is not None and len(entry.frames) != min(
                    entry.frame_count, max_frames
                ):
                    self._stats.misses += 1
                    return None

                self._stats.hits += 1
                logger.debug(f"Cache hit (memory) for {file_path.name}")
                return (
                    entry.frames,
                    entry.frame_count,
                    entry.dimensions,
                    entry.duration_ms,
                )

            # Check disk cache
            entry = self._load_from_disk(cache_key)
            if entry:
                # Validate file hasn't changed
                try:
                    stat = file_path.stat()
                    if (
                        entry.file_size != stat.st_size
                        or abs(entry.file_mtime - stat.st_mtime) > 0.001
                    ):
                        # File has changed, invalidate cache
                        self._invalidate_disk_entry(cache_key)
                        self._stats.misses += 1
                        return None
                except OSError:
                    self._stats.misses += 1
                    return None

                # Check TTL
                if time.time() - entry.created_at > self.ttl_seconds:
                    self._invalidate_disk_entry(cache_key)
                    self._stats.misses += 1
                    return None

                # Check frames match requested max_frames
                if max_frames is not None and len(entry.frames) != min(
                    entry.frame_count, max_frames
                ):
                    self._stats.misses += 1
                    return None

                # Promote to memory cache
                self._add_to_memory(entry)

                self._stats.hits += 1
                logger.debug(f"Cache hit (disk) for {file_path.name}")
                return (
                    entry.frames,
                    entry.frame_count,
                    entry.dimensions,
                    entry.duration_ms,
                )

            self._stats.misses += 1
            return None

    def put(
        self,
        file_path: Path,
        frames: list[np.ndarray],
        frame_count: int,
        dimensions: tuple[int, int],
        duration_ms: int,
    ) -> None:
        """Store frames in cache.

        Args:
            file_path: Path to the GIF file
            frames: Extracted frames
            frame_count: Total frame count in original GIF
            dimensions: Frame dimensions (width, height)
            duration_ms: Total animation duration
        """
        if not self.enabled:
            return

        cache_key = self.generate_cache_key(file_path)
        if not cache_key:
            return

        try:
            stat = file_path.stat()
        except OSError as e:
            logger.warning(f"Failed to stat file {file_path}: {e}")
            return

        entry = FrameCacheEntry(
            cache_key=cache_key,
            frames=frames,
            dimensions=dimensions,
            duration_ms=duration_ms,
            frame_count=frame_count,
            file_path=str(file_path.absolute()),
            file_size=stat.st_size,
            file_mtime=stat.st_mtime,
        )

        with self._lock:
            # Add to memory cache
            self._add_to_memory(entry)

            # Also store to disk for persistence
            self._store_to_disk(entry)

            logger.debug(f"Cached frames for {file_path.name}")

    def _add_to_memory(self, entry: FrameCacheEntry) -> None:
        """Add entry to memory cache with LRU eviction."""
        entry_size = entry.memory_size()

        # Evict entries if needed
        while (
            self._memory_bytes + entry_size > self.memory_limit_bytes
            and self._memory_cache
        ):
            # Remove least recently used
            evicted_key, evicted_entry = self._memory_cache.popitem(last=False)
            self._memory_bytes -= evicted_entry.memory_size()
            self._stats.evictions += 1
            logger.debug(f"Evicted {evicted_key} from memory cache")

        # Add new entry
        if entry.cache_key in self._memory_cache:
            # Remove old entry first
            old_entry = self._memory_cache.pop(entry.cache_key)
            self._memory_bytes -= old_entry.memory_size()

        self._memory_cache[entry.cache_key] = entry
        self._memory_bytes += entry_size
        self._stats.memory_bytes = self._memory_bytes

    def _store_to_disk(self, entry: FrameCacheEntry) -> None:
        """Store entry to disk cache."""
        try:
            data = entry.to_bytes()
            data_size = len(data)

            # Check disk space limit
            current_size = self._get_disk_cache_size()
            if current_size + data_size > self.disk_limit_bytes:
                self._evict_disk_entries(data_size)

            with sqlite3.connect(str(self.disk_path)) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO frame_cache
                    (cache_key, data, file_path, file_size, file_mtime,
                     created_at, last_accessed, access_count, data_size)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        entry.cache_key,
                        data,
                        entry.file_path,
                        entry.file_size,
                        entry.file_mtime,
                        entry.created_at,
                        entry.last_accessed,
                        entry.access_count,
                        data_size,
                    ),
                )
                conn.commit()

        except Exception as e:
            logger.error(f"Failed to store to disk cache: {e}")

    def _load_from_disk(self, cache_key: str) -> FrameCacheEntry | None:
        """Load entry from disk cache."""
        try:
            with sqlite3.connect(str(self.disk_path)) as conn:
                cursor = conn.execute(
                    """
                    SELECT data FROM frame_cache WHERE cache_key = ?
                """,
                    (cache_key,),
                )
                row = cursor.fetchone()

                if row:
                    # Update access time
                    conn.execute(
                        """
                        UPDATE frame_cache
                        SET last_accessed = ?, access_count = access_count + 1
                        WHERE cache_key = ?
                    """,
                        (time.time(), cache_key),
                    )
                    conn.commit()

                    return FrameCacheEntry.from_bytes(row[0])

        except Exception as e:
            logger.error(f"Failed to load from disk cache: {e}")

        return None

    def _invalidate_disk_entry(self, cache_key: str) -> None:
        """Remove entry from disk cache."""
        try:
            with sqlite3.connect(str(self.disk_path)) as conn:
                conn.execute(
                    "DELETE FROM frame_cache WHERE cache_key = ?", (cache_key,)
                )
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to invalidate disk entry: {e}")

    def _get_disk_cache_size(self) -> int:
        """Get total size of disk cache in bytes."""
        try:
            with sqlite3.connect(str(self.disk_path)) as conn:
                cursor = conn.execute("SELECT SUM(data_size) FROM frame_cache")
                result = cursor.fetchone()[0]
                return result if result else 0
        except Exception as e:
            logger.error(f"Failed to get disk cache size: {e}")
            return 0

    def _evict_disk_entries(self, needed_bytes: int) -> None:
        """Evict oldest entries from disk to make space."""
        try:
            with sqlite3.connect(str(self.disk_path)) as conn:
                # Get entries sorted by last access time
                cursor = conn.execute(
                    """
                    SELECT cache_key, data_size
                    FROM frame_cache
                    ORDER BY last_accessed ASC
                """
                )

                evicted_bytes = 0
                evict_keys = []

                for cache_key, data_size in cursor:
                    evict_keys.append(cache_key)
                    evicted_bytes += data_size
                    if evicted_bytes >= needed_bytes:
                        break

                # Delete evicted entries
                if evict_keys:
                    placeholders = ",".join("?" * len(evict_keys))
                    conn.execute(
                        f"DELETE FROM frame_cache WHERE cache_key IN ({placeholders})",
                        evict_keys,
                    )
                    conn.commit()
                    self._stats.evictions += len(evict_keys)

        except Exception as e:
            logger.error(f"Failed to evict disk entries: {e}")

    def _cleanup_expired(self) -> None:
        """Remove expired entries from disk cache."""
        try:
            cutoff_time = time.time() - self.ttl_seconds
            with sqlite3.connect(str(self.disk_path)) as conn:
                conn.execute(
                    """
                    DELETE FROM frame_cache WHERE created_at < ?
                """,
                    (cutoff_time,),
                )
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to cleanup expired entries: {e}")

    def invalidate(self, file_path: Path) -> None:
        """Invalidate cache entry for a specific file.

        Args:
            file_path: Path to the file to invalidate
        """
        if not self.enabled:
            return

        cache_key = self.generate_cache_key(file_path)
        if not cache_key:
            return

        with self._lock:
            # Remove from memory cache
            if cache_key in self._memory_cache:
                entry = self._memory_cache.pop(cache_key)
                self._memory_bytes -= entry.memory_size()
                self._stats.memory_bytes = self._memory_bytes

            # Remove from disk cache
            self._invalidate_disk_entry(cache_key)

            logger.debug(f"Invalidated cache for {file_path.name}")

    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._memory_cache.clear()
            self._memory_bytes = 0
            self._stats = CacheStats()

            if self.enabled:
                try:
                    with sqlite3.connect(str(self.disk_path)) as conn:
                        conn.execute("DELETE FROM frame_cache")
                        conn.commit()
                except Exception as e:
                    logger.error(f"Failed to clear disk cache: {e}")

            logger.info("Cleared frame cache")

    def get_stats(self) -> CacheStats:
        """Get cache statistics.

        Returns:
            Cache statistics object
        """
        with self._lock:
            stats = CacheStats(
                hits=self._stats.hits,
                misses=self._stats.misses,
                evictions=self._stats.evictions,
                memory_bytes=self._memory_bytes,
                total_accesses=self._stats.total_accesses,
            )

            # Add disk cache stats
            if self.enabled:
                try:
                    with sqlite3.connect(str(self.disk_path)) as conn:
                        cursor = conn.execute("SELECT COUNT(*) FROM frame_cache")
                        stats.disk_entries = cursor.fetchone()[0]
                except Exception:
                    pass

            return stats

    def warm_cache(self, file_paths: list[Path], max_frames: int | None = None) -> None:
        """Pre-load files into cache for better performance.

        Args:
            file_paths: List of GIF files to pre-cache
            max_frames: Maximum frames to extract for each file
        """
        if not self.enabled:
            return

        from ..metrics import extract_gif_frames

        for file_path in file_paths:
            if not file_path.exists():
                continue

            cache_key = self.generate_cache_key(file_path)
            if cache_key in self._memory_cache:
                continue  # Already cached

            try:
                # Extract frames (this will automatically cache them)
                extract_gif_frames(file_path, max_frames=max_frames)
                logger.debug(f"Warmed cache for {file_path.name}")
            except Exception as e:
                logger.warning(f"Failed to warm cache for {file_path}: {e}")


# Global frame cache instance
_frame_cache_instance: FrameCache | None = None
_frame_cache_lock = threading.Lock()


def get_frame_cache() -> FrameCache:
    """Get the global frame cache instance.

    Returns:
        The global FrameCache instance
    """
    global _frame_cache_instance

    if _frame_cache_instance is None:
        with _frame_cache_lock:
            if _frame_cache_instance is None:
                from ..config import FRAME_CACHE

                _frame_cache_instance = FrameCache(
                    memory_limit_mb=FRAME_CACHE.get("memory_limit_mb", 500),
                    disk_path=FRAME_CACHE.get("disk_path"),
                    disk_limit_mb=FRAME_CACHE.get("disk_limit_mb", 2000),
                    ttl_seconds=FRAME_CACHE.get("ttl_seconds", 86400),
                    enabled=FRAME_CACHE.get("enabled", True),
                )

    return _frame_cache_instance


def reset_frame_cache() -> None:
    """Reset the global frame cache instance (mainly for testing)."""
    global _frame_cache_instance

    with _frame_cache_lock:
        if _frame_cache_instance:
            _frame_cache_instance.clear()
        _frame_cache_instance = None
