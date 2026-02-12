"""Unit tests for ValidationCache."""

import hashlib
import json
import sqlite3
import tempfile
import threading
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from giflab.caching.validation_cache import (
    ValidationCache,
    ValidationCacheStats,
    ValidationResult,
    get_validation_cache,
    reset_validation_cache,
)


class TestValidationCache:
    """Test suite for ValidationCache functionality."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create a temporary directory for cache testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def cache(self, temp_cache_dir):
        """Create a ValidationCache instance for testing."""
        cache_path = temp_cache_dir / "test_cache.db"
        cache = ValidationCache(
            memory_limit_mb=10,
            disk_path=cache_path,
            disk_limit_mb=50,
            ttl_seconds=3600,
            enabled=True,
        )
        yield cache
        cache.clear()

    @pytest.fixture
    def sample_frames(self):
        """Create sample frame arrays for testing."""
        np.random.seed(42)
        frame1 = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        frame2 = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        return frame1, frame2

    def test_cache_initialization(self, temp_cache_dir):
        """Test cache initialization with various parameters."""
        cache_path = temp_cache_dir / "init_test.db"

        cache = ValidationCache(
            memory_limit_mb=5,
            disk_path=cache_path,
            disk_limit_mb=20,
            ttl_seconds=1800,
            enabled=True,
        )

        assert cache.enabled is True
        assert cache.memory_limit_bytes == 5 * 1024 * 1024
        assert cache.disk_limit_bytes == 20 * 1024 * 1024
        assert cache.ttl_seconds == 1800
        assert cache_path.exists()

        # Check database tables exist
        with sqlite3.connect(str(cache_path)) as conn:
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = {row[0] for row in cursor}
            assert "validation_cache" in tables
            assert "cache_stats" in tables

    def test_cache_disabled(self, temp_cache_dir):
        """Test cache behavior when disabled."""
        cache = ValidationCache(
            disk_path=temp_cache_dir / "disabled.db",
            enabled=False,
        )

        frame1 = np.ones((10, 10, 3), dtype=np.uint8)
        frame2 = np.zeros((10, 10, 3), dtype=np.uint8)

        # Put should do nothing
        cache.put(frame1, frame2, "ssim", 0.5)

        # Get should return None
        result = cache.get(frame1, frame2, "ssim")
        assert result is None

        # Stats should be empty
        stats = cache.get_stats()
        assert stats.hits == 0
        assert stats.misses == 0

    def test_frame_hash_generation(self, cache, sample_frames):
        """Test frame hash generation."""
        frame1, frame2 = sample_frames

        # Same frame should produce same hash
        hash1 = cache.get_frame_hash(frame1)
        hash2 = cache.get_frame_hash(frame1)
        assert hash1 == hash2

        # Different frames should produce different hashes
        hash3 = cache.get_frame_hash(frame2)
        assert hash1 != hash3

        # Hash should be 16 characters (MD5 truncated)
        assert len(hash1) == 16

        # Large frame should use sampling
        large_frame = np.random.randint(0, 255, (1000, 1000, 3), dtype=np.uint8)
        large_hash = cache.get_frame_hash(large_frame)
        assert len(large_hash) == 16

    def test_cache_key_generation(self, cache):
        """Test cache key generation."""
        frame1_hash = "abc123"
        frame2_hash = "def456"
        metric_type = "ssim"
        config = {"param1": "value1", "param2": 42}
        frame_indices = (0, 1)

        key1 = cache.generate_cache_key(
            frame1_hash, frame2_hash, metric_type, config, frame_indices
        )

        # Same inputs should produce same key
        key2 = cache.generate_cache_key(
            frame1_hash, frame2_hash, metric_type, config, frame_indices
        )
        assert key1 == key2

        # Key should be 32 characters (SHA256 truncated)
        assert len(key1) == 32

        # Different inputs should produce different keys
        key3 = cache.generate_cache_key(
            frame2_hash, frame1_hash, metric_type, config, frame_indices
        )
        assert key1 != key3

        # Config order shouldn't matter
        config_reordered = {"param2": 42, "param1": "value1"}
        key4 = cache.generate_cache_key(
            frame1_hash, frame2_hash, metric_type, config_reordered, frame_indices
        )
        assert key1 == key4

    def test_cache_put_and_get(self, cache, sample_frames):
        """Test basic cache put and get operations."""
        frame1, frame2 = sample_frames
        metric_type = "ms_ssim"
        value = 0.95
        config = {"scales": 5}

        # Put value in cache
        cache.put(frame1, frame2, metric_type, value, config)

        # Get value from cache
        retrieved = cache.get(frame1, frame2, metric_type, config)
        assert retrieved == value

        # Check stats
        stats = cache.get_stats()
        assert stats.hits == 1
        assert stats.misses == 0
        assert stats.memory_entries > 0

    def test_cache_miss(self, cache, sample_frames):
        """Test cache miss behavior."""
        frame1, frame2 = sample_frames

        # Try to get non-existent value
        result = cache.get(frame1, frame2, "lpips", {"net": "alex"})
        assert result is None

        # Check stats
        stats = cache.get_stats()
        assert stats.hits == 0
        assert stats.misses == 1

    def test_complex_metric_caching(self, cache, sample_frames):
        """Test caching of complex metric results (dicts)."""
        frame1, frame2 = sample_frames
        metric_type = "gradient_color"
        value = {
            "gradient_score": 0.85,
            "color_artifacts": 0.12,
            "details": {"param1": 1.5, "param2": [1, 2, 3]},
        }

        cache.put(frame1, frame2, metric_type, value)
        retrieved = cache.get(frame1, frame2, metric_type)

        assert retrieved == value
        assert retrieved["gradient_score"] == 0.85
        assert retrieved["details"]["param2"] == [1, 2, 3]

    def test_memory_eviction(self, temp_cache_dir):
        """Test LRU eviction when memory limit is reached."""
        # Create cache with extremely small memory limit (~10 bytes)
        # Each pickled float is ~14 bytes, so only 1 entry fits at a time
        cache = ValidationCache(
            memory_limit_mb=0.00001,  # ~10 bytes
            disk_path=temp_cache_dir / "eviction_test.db",
            enabled=True,
        )

        frames = []
        for _ in range(10):
            frame = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
            frames.append(frame)

        # Add multiple entries - each new entry should evict the previous one
        for i in range(5):
            cache.put(frames[i], frames[i + 1], "ssim", float(i))

        # Check that eviction occurred
        stats = cache.get_stats()
        assert stats.evictions > 0
        assert stats.memory_entries <= 1  # Only the last entry should remain

    def test_disk_persistence(self, temp_cache_dir, sample_frames):
        """Test that cache persists to disk."""
        frame1, frame2 = sample_frames
        cache_path = temp_cache_dir / "persist_test.db"

        # Create cache and add entry
        cache1 = ValidationCache(disk_path=cache_path)
        cache1.put(frame1, frame2, "ssim", 0.88)

        # Clear memory cache but keep disk
        cache1._memory_cache.clear()
        cache1._memory_bytes = 0

        # Get should still work (loaded from disk)
        result = cache1.get(frame1, frame2, "ssim")
        assert result == 0.88

        # Create new cache instance with same disk path
        cache2 = ValidationCache(disk_path=cache_path)
        result2 = cache2.get(frame1, frame2, "ssim")
        assert result2 == 0.88

    def test_ttl_expiration(self, temp_cache_dir, sample_frames):
        """Test TTL-based cache expiration."""
        frame1, frame2 = sample_frames

        # Create cache with very short TTL
        cache = ValidationCache(
            disk_path=temp_cache_dir / "ttl_test.db",
            ttl_seconds=0.1,  # 100ms TTL
        )

        cache.put(frame1, frame2, "ssim", 0.75)

        # Should get value immediately
        result = cache.get(frame1, frame2, "ssim")
        assert result == 0.75

        # Wait for expiration
        time.sleep(0.2)

        # Should return None after expiration
        result = cache.get(frame1, frame2, "ssim")
        assert result is None

    def test_thread_safety(self, cache, sample_frames):
        """Test thread-safe concurrent access."""
        frame1, frame2 = sample_frames
        results = []
        errors = []

        def worker(thread_id):
            try:
                for i in range(10):
                    # Write
                    cache.put(
                        frame1,
                        frame2,
                        f"metric_{thread_id}_{i}",
                        float(thread_id * 100 + i),
                    )

                    # Read
                    value = cache.get(frame1, frame2, f"metric_{thread_id}_{i}")
                    results.append(value)

            except Exception as e:
                errors.append(e)

        # Run multiple threads
        threads = []
        for i in range(5):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Check no errors occurred
        assert len(errors) == 0

        # Check all results are valid
        assert all(r is not None for r in results)
        assert len(results) == 50  # 5 threads * 10 operations

    def test_invalidate_by_metric(self, cache, sample_frames):
        """Test invalidation by metric type."""
        frame1, frame2 = sample_frames

        # Add multiple metric types
        cache.put(frame1, frame2, "ssim", 0.9)
        cache.put(frame1, frame2, "ms_ssim", 0.85)
        cache.put(frame1, frame2, "lpips", 0.1)

        # Invalidate one metric type
        cache.invalidate_by_metric("ms_ssim")

        # Other metrics should still exist
        assert cache.get(frame1, frame2, "ssim") == 0.9
        assert cache.get(frame1, frame2, "lpips") == 0.1

        # Invalidated metric should be gone
        assert cache.get(frame1, frame2, "ms_ssim") is None

    def test_clear_cache(self, cache, sample_frames):
        """Test clearing entire cache."""
        frame1, frame2 = sample_frames

        # Add entries
        cache.put(frame1, frame2, "ssim", 0.9)
        cache.put(frame1, frame2, "ms_ssim", 0.85)

        # Clear cache
        cache.clear()

        # All entries should be gone
        assert cache.get(frame1, frame2, "ssim") is None
        assert cache.get(frame1, frame2, "ms_ssim") is None

        # Stats should be reset
        stats = cache.get_stats()
        assert stats.hits == 0
        assert stats.misses == 2  # From the get calls above
        assert stats.memory_entries == 0
        assert stats.disk_entries == 0

    def test_get_metric_stats(self, cache, sample_frames):
        """Test getting statistics by metric type."""
        frame1, frame2 = sample_frames

        # Add various metrics
        cache.put(frame1, frame2, "ssim", 0.9)
        cache.put(frame1, frame2, "ms_ssim", 0.85)
        cache.put(frame2, frame1, "ssim", 0.88)  # Different frame order

        metric_stats = cache.get_metric_stats()

        assert "ssim" in metric_stats
        assert "ms_ssim" in metric_stats
        assert metric_stats["ssim"] == 2
        assert metric_stats["ms_ssim"] == 1

    def test_cache_with_metadata(self, cache, sample_frames):
        """Test caching with metadata."""
        frame1, frame2 = sample_frames
        metadata = {
            "processing_time": 0.5,
            "algorithm_version": "1.2.3",
        }

        cache.put(frame1, frame2, "lpips", 0.15, metadata=metadata)

        # Retrieve and verify (metadata is stored but not returned by get)
        value = cache.get(frame1, frame2, "lpips")
        assert value == 0.15

    def test_disk_size_limit(self, temp_cache_dir):
        """Test disk cache size limiting."""
        # Create cache with small disk limit
        cache = ValidationCache(
            disk_path=temp_cache_dir / "size_test.db",
            disk_limit_mb=0.01,  # ~10KB
        )

        # Add many large entries
        for i in range(20):
            frame1 = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            frame2 = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

            # Large dictionary to increase storage size
            value = {
                "data": list(range(1000)),
                "index": i,
            }

            cache.put(frame1, frame2, f"metric_{i}", value)

        # Check that disk size is within limits
        disk_size = cache._get_disk_cache_size()
        assert disk_size <= cache.disk_limit_bytes

        # Check that eviction occurred
        stats = cache.get_stats()
        assert stats.evictions > 0

    def test_hit_rate_calculation(self):
        """Test cache hit rate calculation."""
        stats = ValidationCacheStats(hits=75, misses=25)
        assert stats.hit_rate == 0.75

        stats_no_traffic = ValidationCacheStats()
        assert stats_no_traffic.hit_rate == 0.0

    def test_singleton_instance(self):
        """Test singleton pattern for get_validation_cache."""
        with patch(
            "giflab.config.VALIDATION_CACHE",
            {
                "enabled": True,
                "memory_limit_mb": 50,
            },
        ):
            cache1 = get_validation_cache()
            cache2 = get_validation_cache()

            assert cache1 is cache2

            # Reset should clear the singleton
            reset_validation_cache()

            cache3 = get_validation_cache()
            assert cache3 is not cache1

    @pytest.mark.parametrize(
        "metric_type,value",
        [
            ("ssim", 0.95),
            ("ms_ssim", 0.88),
            ("lpips", 0.12),
            ("psnr", 35.5),
            ("custom", {"score": 0.75, "details": [1, 2, 3]}),
        ],
    )
    def test_various_metric_types(self, cache, sample_frames, metric_type, value):
        """Test caching various metric types."""
        frame1, frame2 = sample_frames

        cache.put(frame1, frame2, metric_type, value)
        retrieved = cache.get(frame1, frame2, metric_type)

        assert retrieved == value

    def test_frame_indices_in_key(self, cache, sample_frames):
        """Test that frame indices are included in cache key."""
        frame1, frame2 = sample_frames

        # Same frames but different indices should have different cache entries
        cache.put(frame1, frame2, "ssim", 0.9, frame_indices=(0, 1))
        cache.put(frame1, frame2, "ssim", 0.8, frame_indices=(5, 6))

        result1 = cache.get(frame1, frame2, "ssim", frame_indices=(0, 1))
        result2 = cache.get(frame1, frame2, "ssim", frame_indices=(5, 6))

        assert result1 == 0.9
        assert result2 == 0.8

    def test_config_affects_cache_key(self, cache, sample_frames):
        """Test that configuration affects cache key."""
        frame1, frame2 = sample_frames

        # Same frames but different config should have different cache entries
        cache.put(frame1, frame2, "lpips", 0.1, config={"net": "alex"})
        cache.put(frame1, frame2, "lpips", 0.15, config={"net": "vgg"})

        result1 = cache.get(frame1, frame2, "lpips", config={"net": "alex"})
        result2 = cache.get(frame1, frame2, "lpips", config={"net": "vgg"})

        assert result1 == 0.1
        assert result2 == 0.15


class TestValidationResult:
    """Test ValidationResult dataclass."""

    def test_validation_result_creation(self):
        """Test creating ValidationResult instances."""
        result = ValidationResult(
            metric_type="ssim",
            value=0.95,
            frame_indices=(0, 1),
            config_hash="abc123",
            timestamp=time.time(),
            metadata={"key": "value"},
        )

        assert result.metric_type == "ssim"
        assert result.value == 0.95
        assert result.frame_indices == (0, 1)
        assert result.config_hash == "abc123"
        assert result.metadata == {"key": "value"}

    def test_validation_result_optional_fields(self):
        """Test ValidationResult with optional fields."""
        result = ValidationResult(
            metric_type="lpips",
            value=0.1,
        )

        assert result.metric_type == "lpips"
        assert result.value == 0.1
        assert result.frame_indices is None
        assert result.config_hash is None
        assert result.timestamp == 0.0
        assert result.metadata is None
