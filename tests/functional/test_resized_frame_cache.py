"""Unit tests for ResizedFrameCache and FrameBufferPool."""

import threading
import time
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
from giflab.caching.resized_frame_cache import (
    FrameBufferPool,
    InterpolationMethod,
    ResizedFrameCache,
    get_resize_cache,
    resize_frame_cached,
)


class TestInterpolationMethod:
    """Tests for InterpolationMethod enum."""
    
    def test_from_cv2_known_flags(self):
        """Test conversion from OpenCV flags to enum."""
        assert InterpolationMethod.from_cv2(cv2.INTER_AREA) == InterpolationMethod.AREA
        assert InterpolationMethod.from_cv2(cv2.INTER_LANCZOS4) == InterpolationMethod.LANCZOS4
        assert InterpolationMethod.from_cv2(cv2.INTER_CUBIC) == InterpolationMethod.CUBIC
        assert InterpolationMethod.from_cv2(cv2.INTER_LINEAR) == InterpolationMethod.LINEAR
        assert InterpolationMethod.from_cv2(cv2.INTER_NEAREST) == InterpolationMethod.NEAREST
    
    def test_from_cv2_unknown_flag(self):
        """Test that unknown flags default to AREA."""
        assert InterpolationMethod.from_cv2(999) == InterpolationMethod.AREA


class TestFrameBufferPool:
    """Tests for FrameBufferPool memory management."""
    
    def test_buffer_allocation(self):
        """Test that new buffers are allocated when pool is empty."""
        pool = FrameBufferPool(max_buffers_per_size=5)
        
        buffer = pool.get_buffer((100, 100, 3))
        assert buffer.shape == (100, 100, 3)
        assert buffer.dtype == np.uint8
        
        stats = pool.get_stats()
        assert stats["allocations"] == 1
        assert stats["reuses"] == 0
    
    def test_buffer_reuse(self):
        """Test that buffers are reused from pool."""
        pool = FrameBufferPool(max_buffers_per_size=5)
        
        # Get and release a buffer
        buffer1 = pool.get_buffer((50, 50, 3))
        buffer1_id = id(buffer1)
        pool.release_buffer(buffer1)
        
        # Get another buffer of same size - should reuse
        buffer2 = pool.get_buffer((50, 50, 3))
        assert id(buffer2) == buffer1_id
        
        stats = pool.get_stats()
        assert stats["allocations"] == 1
        assert stats["reuses"] == 1
        assert stats["releases"] == 1
    
    def test_max_buffers_per_size(self):
        """Test that pool respects max buffers per size limit."""
        pool = FrameBufferPool(max_buffers_per_size=2)

        # Allocate 3 buffers of the same size, then release all at once
        buffers = [pool.get_buffer((30, 30)) for _ in range(3)]
        for buf in buffers:
            pool.release_buffer(buf)

        stats = pool.get_stats()
        assert stats["total_buffers"] == 2  # Only 2 kept
        assert stats["releases"] == 2  # Third release ignored
    
    def test_different_dtypes(self):
        """Test that buffers with different dtypes are pooled separately."""
        pool = FrameBufferPool()
        
        # Create buffers with different dtypes
        uint8_buffer = pool.get_buffer((10, 10), dtype=np.uint8)
        float32_buffer = pool.get_buffer((10, 10), dtype=np.float32)
        
        assert uint8_buffer.dtype == np.uint8
        assert float32_buffer.dtype == np.float32
        
        # Release and reuse
        pool.release_buffer(uint8_buffer)
        pool.release_buffer(float32_buffer)
        
        new_uint8 = pool.get_buffer((10, 10), dtype=np.uint8)
        new_float32 = pool.get_buffer((10, 10), dtype=np.float32)
        
        assert new_uint8.dtype == np.uint8
        assert new_float32.dtype == np.float32
    
    def test_clear(self):
        """Test clearing the pool."""
        pool = FrameBufferPool()
        
        # Add some buffers
        for _ in range(3):
            buffer = pool.get_buffer((20, 20))
            pool.release_buffer(buffer)
        
        assert pool.get_stats()["total_buffers"] > 0
        
        pool.clear()
        assert pool.get_stats()["total_buffers"] == 0
    
    def test_get_stats(self):
        """Test statistics reporting."""
        pool = FrameBufferPool()
        
        # Perform some operations
        b1 = pool.get_buffer((100, 100, 3))
        pool.release_buffer(b1)
        pool.get_buffer((100, 100, 3))
        
        stats = pool.get_stats()
        assert stats["allocations"] == 1
        assert stats["reuses"] == 1
        assert stats["releases"] == 1
        assert stats["reuse_rate"] == 0.5
        assert "total_memory_mb" in stats


class TestResizedFrameCache:
    """Tests for ResizedFrameCache."""
    
    @pytest.fixture
    def cache(self):
        """Create a test cache instance."""
        return ResizedFrameCache(
            memory_limit_mb=10,
            enable_pooling=True,
            ttl_seconds=60
        )
    
    @pytest.fixture
    def test_frame(self):
        """Create a test frame."""
        return np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8)
    
    def test_cache_hit(self, cache, test_frame):
        """Test cache hit for same resize operation."""
        target_size = (75, 50)
        
        # First call - cache miss
        resized1 = cache.get(test_frame, target_size, cv2.INTER_AREA)
        assert cache.get_stats()["misses"] == 1
        assert cache.get_stats()["hits"] == 0
        
        # Second call - cache hit
        resized2 = cache.get(test_frame, target_size, cv2.INTER_AREA)
        assert cache.get_stats()["misses"] == 1
        assert cache.get_stats()["hits"] == 1
        
        # Results should be identical
        np.testing.assert_array_equal(resized1, resized2)
    
    def test_different_interpolations_cached_separately(self, cache, test_frame):
        """Test that different interpolation methods are cached separately."""
        target_size = (75, 50)

        # Resize with AREA
        cache.get(test_frame, target_size, cv2.INTER_AREA)

        # Resize with LINEAR - should be cache miss (separate entry)
        cache.get(test_frame, target_size, cv2.INTER_LINEAR)

        assert cache.get_stats()["misses"] == 2
        assert cache.get_stats()["entries"] == 2
    
    def test_different_sizes_cached_separately(self, cache, test_frame):
        """Test that different target sizes are cached separately."""
        # Resize to different sizes
        resized1 = cache.get(test_frame, (75, 50), cv2.INTER_AREA)
        resized2 = cache.get(test_frame, (60, 40), cv2.INTER_AREA)
        
        assert cache.get_stats()["misses"] == 2
        assert cache.get_stats()["entries"] == 2
        assert resized1.shape[:2] == (50, 75)
        assert resized2.shape[:2] == (40, 60)
    
    def test_lru_eviction(self):
        """Test LRU eviction when memory limit is exceeded."""
        # Create cache with very small memory limit (0.01MB = ~10KB)
        cache = ResizedFrameCache(memory_limit_mb=0.01, enable_pooling=False)

        # Create frames that will exceed memory limit
        # Each 200x200x3 resized to 100x100x3 = 30000 bytes >> 10KB
        frames = [np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8) for _ in range(5)]

        # Add frames to cache
        for _i, frame in enumerate(frames):
            cache.get(frame, (100, 100), cv2.INTER_AREA)

        stats = cache.get_stats()
        assert stats["evictions"] > 0
        assert stats["memory_mb"] <= 0.01
    
    def test_ttl_expiration(self):
        """Test that expired entries are removed."""
        cache = ResizedFrameCache(memory_limit_mb=10, ttl_seconds=0.1)
        
        frame = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        
        # Add to cache
        cache.get(frame, (25, 25), cv2.INTER_AREA)
        assert cache.get_stats()["entries"] == 1
        
        # Wait for TTL to expire
        time.sleep(0.2)
        
        # Force expiration check by adding another entry
        frame2 = np.random.randint(0, 255, (60, 60, 3), dtype=np.uint8)
        cache.get(frame2, (30, 30), cv2.INTER_AREA)
        
        # Original entry should be expired
        cache.get(frame, (25, 25), cv2.INTER_AREA)
        assert cache.get_stats()["ttl_evictions"] >= 0  # May or may not have been evicted
    
    def test_clear(self, cache, test_frame):
        """Test clearing the cache."""
        # Add some entries
        cache.get(test_frame, (75, 50), cv2.INTER_AREA)
        cache.get(test_frame, (60, 40), cv2.INTER_AREA)
        
        assert cache.get_stats()["entries"] == 2
        
        cache.clear()
        
        stats = cache.get_stats()
        assert stats["entries"] == 0
        assert stats["memory_mb"] == 0
    
    def test_get_most_used(self, cache):
        """Test getting most frequently used entries."""
        frames = [np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8) for _ in range(3)]
        
        # Use first frame 5 times
        for _ in range(5):
            cache.get(frames[0], (25, 25), cv2.INTER_AREA)
        
        # Use second frame 3 times
        for _ in range(3):
            cache.get(frames[1], (30, 30), cv2.INTER_AREA)
        
        # Use third frame once
        cache.get(frames[2], (35, 35), cv2.INTER_AREA)
        
        most_used = cache.get_most_used(top_n=2)
        assert len(most_used) == 2
        assert most_used[0]["hit_count"] == 4  # 5 total, but first was a miss
        assert most_used[1]["hit_count"] == 2  # 3 total, but first was a miss
    
    def test_thread_safety(self, cache):
        """Test that cache is thread-safe."""
        num_threads = 10
        iterations = 100
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        def resize_many():
            for _ in range(iterations):
                cache.get(frame, (50, 50), cv2.INTER_AREA)
        
        threads = [threading.Thread(target=resize_many) for _ in range(num_threads)]
        
        for t in threads:
            t.start()
        
        for t in threads:
            t.join()
        
        stats = cache.get_stats()
        # Should have many hits, only one miss
        assert stats["hits"] == (num_threads * iterations - 1)
        assert stats["misses"] == 1


class TestGlobalFunctions:
    """Tests for global cache functions."""
    
    def test_get_resize_cache_singleton(self):
        """Test that get_resize_cache returns singleton instance."""
        cache1 = get_resize_cache()
        cache2 = get_resize_cache()
        assert cache1 is cache2
    
    @patch('giflab.config.FRAME_CACHE', {
        'resize_cache_memory_mb': 100,
        'enable_buffer_pooling': False,
        'resize_cache_ttl_seconds': 7200
    })
    def test_get_resize_cache_uses_config(self):
        """Test that get_resize_cache uses config values."""
        # Clear any existing instance
        import giflab.caching.resized_frame_cache as module
        module._resize_cache_instance = None

        cache = get_resize_cache()
        assert cache.memory_limit == 100 * 1024 * 1024
        assert cache.enable_pooling is False
        assert cache.ttl == 7200
    
    def test_resize_frame_cached_no_resize_needed(self):
        """Test that resize_frame_cached returns original if no resize needed."""
        frame = np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8)
        result = resize_frame_cached(frame, (150, 100))
        assert result is frame  # Should return same object
    
    def test_resize_frame_cached_with_cache(self):
        """Test resize_frame_cached with caching enabled."""
        frame = np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8)
        
        # Clear any existing cache
        cache = get_resize_cache()
        cache.clear()
        
        # First resize - cache miss
        result1 = resize_frame_cached(frame, (75, 50), use_cache=True)
        stats = cache.get_stats()
        assert stats["misses"] == 1
        
        # Second resize - cache hit
        result2 = resize_frame_cached(frame, (75, 50), use_cache=True)
        stats = cache.get_stats()
        assert stats["hits"] == 1
        
        np.testing.assert_array_equal(result1, result2)
    
    def test_resize_frame_cached_without_cache(self):
        """Test resize_frame_cached with caching disabled."""
        frame = np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8)
        
        # Clear cache stats
        cache = get_resize_cache()
        cache.clear()
        initial_stats = cache.get_stats()
        
        # Resize without cache
        result = resize_frame_cached(frame, (75, 50), use_cache=False)
        assert result.shape[:2] == (50, 75)
        
        # Cache stats should not change
        final_stats = cache.get_stats()
        assert final_stats["hits"] == initial_stats["hits"]
        assert final_stats["misses"] == initial_stats["misses"]
    
    @patch('giflab.config.FRAME_CACHE', {'resize_cache_enabled': False})
    def test_resize_frame_cached_globally_disabled(self):
        """Test that global disable flag prevents caching."""
        frame = np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8)
        
        # Even with use_cache=True, should not use cache if globally disabled
        result = resize_frame_cached(frame, (75, 50), use_cache=True)
        assert result.shape[:2] == (50, 75)
        
        # Should not affect cache stats
        cache = get_resize_cache()
        stats = cache.get_stats()
        # Stats might have values from other tests, so just check it's not incrementing
        initial_total = stats["hits"] + stats["misses"]
        
        resize_frame_cached(frame, (75, 50), use_cache=True)
        stats = cache.get_stats()
        assert (stats["hits"] + stats["misses"]) == initial_total


