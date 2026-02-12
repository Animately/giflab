"""
ValidationCache: Caching system for frame-level metric validation results.

This module provides a two-tier caching system (in-memory LRU + SQLite disk)
for storing and retrieving validation metric results. It caches individual
metric calculations (SSIM, MS-SSIM, LPIPS, etc.) for frame pairs to avoid
redundant processing.
"""

import hashlib
import json
import logging
import pickle
import sqlite3
import threading
import time
import zlib
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class ValidationCacheStats:
    """Statistics for validation cache performance."""
    
    hits: int = 0
    misses: int = 0
    memory_entries: int = 0
    disk_entries: int = 0
    memory_bytes: int = 0
    disk_bytes: int = 0
    evictions: int = 0
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

@dataclass
class ValidationResult:
    """Container for cached validation metric results."""
    
    metric_type: str  # e.g., "ssim", "ms_ssim", "lpips", "gradient_color"
    value: float | dict[str, Any]  # Metric value or dict for complex metrics
    frame_indices: tuple[int, int] | None = None  # For frame-level caching
    config_hash: str | None = None  # Hash of relevant configuration
    timestamp: float = 0.0
    metadata: dict[str, Any] | None = None  # Additional metadata

class ValidationCache:
    """
    Two-tier cache for validation metric results.
    
    Features:
    - In-memory LRU cache for fast access
    - SQLite disk cache for persistence
    - Automatic invalidation on configuration changes
    - Thread-safe concurrent access
    - TTL-based expiration
    """
    
    def __init__(
        self,
        memory_limit_mb: int = 100,
        disk_path: Path | None = None,
        disk_limit_mb: int = 1000,
        ttl_seconds: int = 172800,  # 48 hours default
        enabled: bool = True,
    ):
        """
        Initialize the ValidationCache.
        
        Args:
            memory_limit_mb: Memory cache size limit in MB
            disk_path: Path to SQLite database (None for default)
            disk_limit_mb: Disk cache size limit in MB
            ttl_seconds: Time-to-live for cache entries in seconds
            enabled: Whether caching is enabled
        """
        self.enabled = enabled
        self.memory_limit_bytes = memory_limit_mb * 1024 * 1024
        self.disk_limit_bytes = disk_limit_mb * 1024 * 1024
        self.ttl_seconds = ttl_seconds
        
        # In-memory cache
        self._memory_cache: OrderedDict[str, ValidationResult] = OrderedDict()
        self._memory_bytes = 0
        self._lock = threading.RLock()
        
        # Statistics
        self._stats = ValidationCacheStats()
        
        # Initialize disk cache
        if disk_path is None:
            disk_path = Path.home() / ".giflab_cache" / "validation_cache.db"
        self.disk_path = disk_path
        self.disk_path.parent.mkdir(parents=True, exist_ok=True)
        
        if self.enabled:
            self._init_database()
    
    def _init_database(self) -> None:
        """Initialize SQLite database for disk cache."""
        with sqlite3.connect(str(self.disk_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS validation_cache (
                    cache_key TEXT PRIMARY KEY,
                    metric_type TEXT NOT NULL,
                    value_data BLOB NOT NULL,
                    frame1_hash TEXT,
                    frame2_hash TEXT,
                    frame_indices TEXT,
                    config_hash TEXT,
                    metadata TEXT,
                    timestamp REAL NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    access_count INTEGER DEFAULT 1,
                    last_accessed REAL NOT NULL
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_metric_type
                ON validation_cache(metric_type)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp
                ON validation_cache(timestamp)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_frame_hashes
                ON validation_cache(frame1_hash, frame2_hash)
            """)
            
            # Cache statistics table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache_stats (
                    stat_name TEXT PRIMARY KEY,
                    stat_value INTEGER NOT NULL
                )
            """)
            
            conn.commit()
    
    def generate_cache_key(
        self,
        frame1_hash: str,
        frame2_hash: str,
        metric_type: str,
        config: dict[str, Any] | None = None,
        frame_indices: tuple[int, int] | None = None,
    ) -> str:
        """
        Generate a unique cache key for a validation result.
        
        Args:
            frame1_hash: Hash of first frame
            frame2_hash: Hash of second frame
            metric_type: type of metric (e.g., "ssim", "lpips")
            config: Relevant configuration parameters
            frame_indices: Optional frame indices for identification
            
        Returns:
            Unique cache key string
        """
        # Create a stable configuration hash
        config_str = ""
        if config:
            # Sort keys for stability
            sorted_config = sorted(config.items())
            config_str = json.dumps(sorted_config, sort_keys=True)
        
        # Combine all components
        key_components = [
            frame1_hash,
            frame2_hash,
            metric_type,
            config_str,
        ]
        
        if frame_indices:
            key_components.append(f"{frame_indices[0]}_{frame_indices[1]}")
        
        key_str = "|".join(key_components)
        return hashlib.sha256(key_str.encode()).hexdigest()[:32]
    
    def get_frame_hash(self, frame: np.ndarray) -> str:
        """
        Generate a hash for a frame array.
        
        Args:
            frame: Frame array
            
        Returns:
            Hash string
        """
        # Use a subset of frame data for performance
        if frame.size > 10000:
            # Sample frame for faster hashing
            sample_indices = np.linspace(0, frame.size - 1, 10000, dtype=int)
            frame_sample = frame.flat[sample_indices]
        else:
            frame_sample = frame.flatten()
        
        return hashlib.md5(frame_sample.tobytes()).hexdigest()[:16]
    
    def get(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
        metric_type: str,
        config: dict[str, Any] | None = None,
        frame_indices: tuple[int, int] | None = None,
    ) -> float | dict[str, Any] | None:
        """
        Retrieve a cached validation result.
        
        Args:
            frame1: First frame array
            frame2: Second frame array
            metric_type: type of metric
            config: Configuration parameters
            frame_indices: Optional frame indices
            
        Returns:
            Cached metric value or None if not found
        """
        if not self.enabled:
            return None
        
        # Generate cache key
        frame1_hash = self.get_frame_hash(frame1)
        frame2_hash = self.get_frame_hash(frame2)
        cache_key = self.generate_cache_key(
            frame1_hash, frame2_hash, metric_type, config, frame_indices
        )
        
        with self._lock:
            # Check memory cache first
            if cache_key in self._memory_cache:
                entry = self._memory_cache[cache_key]
                
                # Check TTL
                if time.time() - entry.timestamp < self.ttl_seconds:
                    # Move to end (LRU)
                    self._memory_cache.move_to_end(cache_key)
                    self._stats.hits += 1
                    logger.debug(f"Cache hit (memory): {metric_type} for frames {frame_indices}")
                    return entry.value
                else:
                    # Expired, remove from cache
                    del self._memory_cache[cache_key]
                    self._memory_bytes -= len(pickle.dumps(entry.value))
            
            # Check disk cache
            cached_result = self._load_from_disk(cache_key)
            if cached_result:
                # Check TTL
                if time.time() - cached_result.timestamp < self.ttl_seconds:
                    # Add to memory cache
                    self._add_to_memory(cache_key, cached_result)
                    self._stats.hits += 1
                    logger.debug(f"Cache hit (disk): {metric_type} for frames {frame_indices}")
                    return cached_result.value
                else:
                    # Expired, remove from disk
                    self._invalidate_disk_entry(cache_key)
            
            self._stats.misses += 1
            logger.debug(f"Cache miss: {metric_type} for frames {frame_indices}")
            return None
    
    def put(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
        metric_type: str,
        value: float | dict[str, Any],
        config: dict[str, Any] | None = None,
        frame_indices: tuple[int, int] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Store a validation result in the cache.
        
        Args:
            frame1: First frame array
            frame2: Second frame array
            metric_type: type of metric
            value: Metric value to cache
            config: Configuration parameters
            frame_indices: Optional frame indices
            metadata: Optional metadata to store
        """
        if not self.enabled:
            return
        
        # Generate cache key
        frame1_hash = self.get_frame_hash(frame1)
        frame2_hash = self.get_frame_hash(frame2)
        cache_key = self.generate_cache_key(
            frame1_hash, frame2_hash, metric_type, config, frame_indices
        )
        
        # Create cache entry
        config_hash = hashlib.md5(
            json.dumps(config, sort_keys=True).encode() if config else b""
        ).hexdigest()[:16]
        
        entry = ValidationResult(
            metric_type=metric_type,
            value=value,
            frame_indices=frame_indices,
            config_hash=config_hash,
            timestamp=time.time(),
            metadata=metadata,
        )
        
        with self._lock:
            # Add to memory cache
            self._add_to_memory(cache_key, entry)
            
            # Store to disk
            self._store_to_disk(
                cache_key, entry, frame1_hash, frame2_hash
            )
            
            logger.debug(f"Cached: {metric_type} for frames {frame_indices}")
    
    def _add_to_memory(self, cache_key: str, entry: ValidationResult) -> None:
        """Add entry to memory cache with LRU eviction."""
        entry_size = len(pickle.dumps(entry.value))
        
        # Evict if necessary
        while (
            self._memory_bytes + entry_size > self.memory_limit_bytes
            and self._memory_cache
        ):
            # Remove oldest entry (LRU)
            oldest_key = next(iter(self._memory_cache))
            oldest_entry = self._memory_cache[oldest_key]
            del self._memory_cache[oldest_key]
            self._memory_bytes -= len(pickle.dumps(oldest_entry.value))
            self._stats.evictions += 1
        
        # Add new entry
        self._memory_cache[cache_key] = entry
        self._memory_bytes += entry_size
        self._stats.memory_entries = len(self._memory_cache)
        self._stats.memory_bytes = self._memory_bytes
    
    def _store_to_disk(
        self,
        cache_key: str,
        entry: ValidationResult,
        frame1_hash: str,
        frame2_hash: str,
    ) -> None:
        """Store entry to disk cache."""
        try:
            # Compress value data
            value_data = zlib.compress(pickle.dumps(entry.value))
            size_bytes = len(value_data)
            
            # Check disk cache size and evict if necessary
            current_size = self._get_disk_cache_size()
            if current_size + size_bytes > self.disk_limit_bytes:
                self._evict_disk_entries(size_bytes)
            
            with sqlite3.connect(str(self.disk_path)) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO validation_cache
                    (cache_key, metric_type, value_data, frame1_hash, frame2_hash,
                     frame_indices, config_hash, metadata, timestamp, size_bytes,
                     access_count, last_accessed)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?)
                """, (
                    cache_key,
                    entry.metric_type,
                    value_data,
                    frame1_hash,
                    frame2_hash,
                    json.dumps(entry.frame_indices) if entry.frame_indices else None,
                    entry.config_hash,
                    json.dumps(entry.metadata) if entry.metadata else None,
                    entry.timestamp,
                    size_bytes,
                    time.time(),
                ))
                conn.commit()
                
                # Update stats
                cursor = conn.execute("SELECT COUNT(*) FROM validation_cache")
                self._stats.disk_entries = cursor.fetchone()[0]
                
        except Exception as e:
            logger.warning(f"Failed to store to disk cache: {e}")
    
    def _load_from_disk(self, cache_key: str) -> ValidationResult | None:
        """Load entry from disk cache."""
        try:
            with sqlite3.connect(str(self.disk_path)) as conn:
                cursor = conn.execute("""
                    SELECT metric_type, value_data, frame_indices, config_hash,
                           metadata, timestamp
                    FROM validation_cache
                    WHERE cache_key = ?
                """, (cache_key,))
                
                row = cursor.fetchone()
                if row:
                    # Update access count and time
                    conn.execute("""
                        UPDATE validation_cache
                        SET access_count = access_count + 1,
                            last_accessed = ?
                        WHERE cache_key = ?
                    """, (time.time(), cache_key))
                    conn.commit()
                    
                    # Decompress and return
                    value = pickle.loads(zlib.decompress(row[1]))
                    
                    return ValidationResult(
                        metric_type=row[0],
                        value=value,
                        frame_indices=json.loads(row[2]) if row[2] else None,
                        config_hash=row[3],
                        metadata=json.loads(row[4]) if row[4] else None,
                        timestamp=row[5],
                    )
                    
        except Exception as e:
            logger.warning(f"Failed to load from disk cache: {e}")
        
        return None
    
    def _invalidate_disk_entry(self, cache_key: str) -> None:
        """Remove an entry from disk cache."""
        try:
            with sqlite3.connect(str(self.disk_path)) as conn:
                conn.execute("DELETE FROM validation_cache WHERE cache_key = ?", (cache_key,))
                conn.commit()
        except Exception as e:
            logger.warning(f"Failed to invalidate disk entry: {e}")
    
    def _get_disk_cache_size(self) -> int:
        """Get current disk cache size in bytes."""
        try:
            with sqlite3.connect(str(self.disk_path)) as conn:
                cursor = conn.execute("SELECT SUM(size_bytes) FROM validation_cache")
                result = cursor.fetchone()[0]
                return result if result else 0
        except Exception:
            return 0
    
    def _evict_disk_entries(self, needed_bytes: int) -> None:
        """Evict oldest disk entries to make room."""
        try:
            with sqlite3.connect(str(self.disk_path)) as conn:
                # Get entries sorted by last access time
                cursor = conn.execute("""
                    SELECT cache_key, size_bytes
                    FROM validation_cache
                    ORDER BY last_accessed ASC
                """)
                
                evicted_bytes = 0
                keys_to_evict = []
                
                for row in cursor:
                    keys_to_evict.append(row[0])
                    evicted_bytes += row[1]
                    if evicted_bytes >= needed_bytes:
                        break
                
                # Delete evicted entries
                if keys_to_evict:
                    placeholders = ",".join("?" * len(keys_to_evict))
                    conn.execute(
                        f"DELETE FROM validation_cache WHERE cache_key IN ({placeholders})",
                        keys_to_evict
                    )
                    conn.commit()
                    self._stats.evictions += len(keys_to_evict)
                    
        except Exception as e:
            logger.warning(f"Failed to evict disk entries: {e}")
    
    def _cleanup_expired(self) -> None:
        """Remove expired entries from both caches."""
        current_time = time.time()
        cutoff_time = current_time - self.ttl_seconds
        
        with self._lock:
            # Clean memory cache
            expired_keys = [
                key for key, entry in self._memory_cache.items()
                if entry.timestamp < cutoff_time
            ]
            for key in expired_keys:
                entry = self._memory_cache[key]
                del self._memory_cache[key]
                self._memory_bytes -= len(pickle.dumps(entry.value))
            
            # Clean disk cache
            try:
                with sqlite3.connect(str(self.disk_path)) as conn:
                    conn.execute(
                        "DELETE FROM validation_cache WHERE timestamp < ?",
                        (cutoff_time,)
                    )
                    conn.commit()
            except Exception as e:
                logger.warning(f"Failed to cleanup expired disk entries: {e}")
    
    def invalidate_by_metric(self, metric_type: str) -> None:
        """
        Invalidate all cache entries for a specific metric type.
        
        Args:
            metric_type: type of metric to invalidate
        """
        with self._lock:
            # Clear from memory cache
            keys_to_remove = [
                key for key, entry in self._memory_cache.items()
                if entry.metric_type == metric_type
            ]
            for key in keys_to_remove:
                entry = self._memory_cache[key]
                del self._memory_cache[key]
                self._memory_bytes -= len(pickle.dumps(entry.value))
            
            # Clear from disk cache
            try:
                with sqlite3.connect(str(self.disk_path)) as conn:
                    conn.execute(
                        "DELETE FROM validation_cache WHERE metric_type = ?",
                        (metric_type,)
                    )
                    conn.commit()
            except Exception as e:
                logger.warning(f"Failed to invalidate metric type from disk: {e}")
            
            logger.info(f"Invalidated all cache entries for metric: {metric_type}")
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._memory_cache.clear()
            self._memory_bytes = 0
            
            try:
                with sqlite3.connect(str(self.disk_path)) as conn:
                    conn.execute("DELETE FROM validation_cache")
                    conn.commit()
            except Exception as e:
                logger.warning(f"Failed to clear disk cache: {e}")
            
            # Reset stats
            self._stats = ValidationCacheStats()
            
            logger.info("Validation cache cleared")
    
    def get_stats(self) -> ValidationCacheStats:
        """Get cache statistics."""
        with self._lock:
            # Update disk stats
            try:
                with sqlite3.connect(str(self.disk_path)) as conn:
                    cursor = conn.execute("SELECT COUNT(*), SUM(size_bytes) FROM validation_cache")
                    row = cursor.fetchone()
                    self._stats.disk_entries = row[0] if row[0] else 0
                    self._stats.disk_bytes = row[1] if row[1] else 0
            except Exception:
                pass
            
            self._stats.memory_entries = len(self._memory_cache)
            self._stats.memory_bytes = self._memory_bytes
            
            return self._stats
    
    def get_metric_stats(self) -> dict[str, int]:
        """Get statistics grouped by metric type."""
        stats = {}
        
        try:
            with sqlite3.connect(str(self.disk_path)) as conn:
                cursor = conn.execute("""
                    SELECT metric_type, COUNT(*) as count
                    FROM validation_cache
                    GROUP BY metric_type
                """)
                
                for row in cursor:
                    stats[row[0]] = row[1]
                    
        except Exception as e:
            logger.warning(f"Failed to get metric stats: {e}")
        
        return stats
    
    def warm_cache(
        self,
        frames1: list[np.ndarray],
        frames2: list[np.ndarray],
        metric_types: list[str],
        config: dict[str, Any] | None = None,
    ) -> int:
        """
        Pre-warm cache with frame pairs and metrics.
        
        Args:
            frames1: list of first frames
            frames2: list of second frames
            metric_types: list of metric types to pre-compute
            config: Configuration parameters
            
        Returns:
            Number of entries added to cache
        """
        if not self.enabled or not frames1 or not frames2:
            return 0
        
        entries_added = 0
        
        # Note: Actual metric calculation would need to be done here
        # This is a placeholder for the warming logic
        logger.info(f"Cache warming requested for {len(frames1)} frame pairs")
        
        return entries_added

# Singleton instance management
_validation_cache_instance: ValidationCache | None = None
_validation_cache_lock = threading.Lock()

def get_validation_cache() -> ValidationCache:
    """Get the singleton ValidationCache instance."""
    global _validation_cache_instance
    
    if _validation_cache_instance is None:
        with _validation_cache_lock:
            if _validation_cache_instance is None:
                from ..config import VALIDATION_CACHE
                
                _validation_cache_instance = ValidationCache(
                    memory_limit_mb=VALIDATION_CACHE.get("memory_limit_mb", 100),
                    disk_path=VALIDATION_CACHE.get("disk_path"),
                    disk_limit_mb=VALIDATION_CACHE.get("disk_limit_mb", 1000),
                    ttl_seconds=VALIDATION_CACHE.get("ttl_seconds", 172800),
                    enabled=VALIDATION_CACHE.get("enabled", True),
                )
    
    return _validation_cache_instance

def reset_validation_cache() -> None:
    """Reset the singleton ValidationCache instance."""
    global _validation_cache_instance
    
    with _validation_cache_lock:
        if _validation_cache_instance:
            _validation_cache_instance.clear()
        _validation_cache_instance = None
