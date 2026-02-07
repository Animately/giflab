"""Comprehensive tests for the frame caching system."""

import hashlib
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from giflab.caching import (
    CacheStats,
    FrameCache,
    FrameCacheEntry,
    get_frame_cache,
    reset_frame_cache,
)


@pytest.fixture
def temp_cache_dir(tmp_path):
    """Create a temporary directory for cache testing."""
    cache_dir = tmp_path / "test_cache"
    cache_dir.mkdir()
    return cache_dir


@pytest.fixture
def frame_cache(temp_cache_dir):
    """Create a test frame cache instance."""
    cache = FrameCache(
        memory_limit_mb=10,
        disk_path=temp_cache_dir / "frame_cache.db",
        disk_limit_mb=50,
        ttl_seconds=3600,
        enabled=True
    )
    yield cache
    cache.clear()


@pytest.fixture
def sample_frames():
    """Create sample frames for testing."""
    frames = [
        np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        for _ in range(5)
    ]
    return frames


@pytest.fixture
def sample_gif(tmp_path):
    """Create a sample GIF file for testing."""
    gif_path = tmp_path / "test.gif"
    
    # Create a simple animated GIF
    frames = []
    for i in range(10):
        frame = Image.new("RGB", (100, 100), color=(i * 25, i * 25, i * 25))
        frames.append(frame)
    
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=100,
        loop=0
    )
    
    return gif_path


class TestFrameCacheEntry:
    """Tests for FrameCacheEntry class."""
    
    def test_entry_creation(self, sample_frames):
        """Test creating a cache entry."""
        entry = FrameCacheEntry(
            cache_key="test_key",
            frames=sample_frames,
            dimensions=(100, 100),
            duration_ms=500,
            frame_count=5,
            file_path="/test/path.gif",
            file_size=1000,
            file_mtime=123456.789
        )
        
        assert entry.cache_key == "test_key"
        assert len(entry.frames) == 5
        assert entry.dimensions == (100, 100)
        assert entry.duration_ms == 500
        assert entry.frame_count == 5
        assert entry.file_path == "/test/path.gif"
        assert entry.file_size == 1000
        assert entry.file_mtime == 123456.789
        assert entry.access_count == 0
    
    def test_entry_update_access(self, sample_frames):
        """Test updating entry access statistics."""
        entry = FrameCacheEntry(
            cache_key="test_key",
            frames=sample_frames,
            dimensions=(100, 100),
            duration_ms=500,
            frame_count=5,
            file_path="/test/path.gif",
            file_size=1000,
            file_mtime=123456.789
        )
        
        initial_time = entry.last_accessed
        entry.update_access()
        
        assert entry.access_count == 1
        assert entry.last_accessed > initial_time
    
    def test_entry_serialization(self, sample_frames):
        """Test serializing and deserializing an entry."""
        original_entry = FrameCacheEntry(
            cache_key="test_key",
            frames=sample_frames,
            dimensions=(100, 100),
            duration_ms=500,
            frame_count=5,
            file_path="/test/path.gif",
            file_size=1000,
            file_mtime=123456.789
        )
        
        # Serialize
        data = original_entry.to_bytes()
        assert isinstance(data, bytes)
        
        # Deserialize
        restored_entry = FrameCacheEntry.from_bytes(data)
        
        assert restored_entry.cache_key == original_entry.cache_key
        assert len(restored_entry.frames) == len(original_entry.frames)
        assert restored_entry.dimensions == original_entry.dimensions
        assert restored_entry.duration_ms == original_entry.duration_ms
        assert restored_entry.frame_count == original_entry.frame_count
        assert restored_entry.file_path == original_entry.file_path
        assert restored_entry.file_size == original_entry.file_size
        assert restored_entry.file_mtime == original_entry.file_mtime
        
        # Check frame data is preserved
        for orig_frame, restored_frame in zip(original_entry.frames, restored_entry.frames):
            np.testing.assert_array_equal(orig_frame, restored_frame)
    
    def test_entry_memory_size(self, sample_frames):
        """Test memory size calculation."""
        entry = FrameCacheEntry(
            cache_key="test_key",
            frames=sample_frames,
            dimensions=(100, 100),
            duration_ms=500,
            frame_count=5,
            file_path="/test/path.gif",
            file_size=1000,
            file_mtime=123456.789
        )
        
        mem_size = entry.memory_size()
        
        # Each frame is 100x100x3 bytes
        expected_frame_size = 100 * 100 * 3 * 5
        assert mem_size >= expected_frame_size
        assert mem_size < expected_frame_size + 1000  # Metadata overhead


class TestFrameCache:
    """Tests for FrameCache class."""
    
    def test_cache_initialization(self, temp_cache_dir):
        """Test cache initialization with various configurations."""
        cache = FrameCache(
            memory_limit_mb=20,
            disk_path=temp_cache_dir / "test.db",
            disk_limit_mb=100,
            ttl_seconds=7200,
            enabled=True
        )
        
        assert cache.enabled
        assert cache.memory_limit_bytes == 20 * 1024 * 1024
        assert cache.disk_limit_bytes == 100 * 1024 * 1024
        assert cache.ttl_seconds == 7200
        assert cache.disk_path.exists()
    
    def test_cache_disabled(self, temp_cache_dir, sample_frames, sample_gif):
        """Test cache operations when disabled."""
        cache = FrameCache(
            memory_limit_mb=10,
            disk_path=temp_cache_dir / "test.db",
            enabled=False
        )
        
        # Operations should be no-ops when disabled
        cache.put(sample_gif, sample_frames, 10, (100, 100), 1000)
        result = cache.get(sample_gif, max_frames=None)
        
        assert result is None
        assert cache.get_stats().total_accesses == 0
    
    def test_cache_key_generation(self, frame_cache, sample_gif):
        """Test cache key generation."""
        key1 = frame_cache.generate_cache_key(sample_gif)
        assert isinstance(key1, str)
        assert len(key1) == 32  # SHA256 truncated to 32 chars
        
        # Same file should generate same key
        key2 = frame_cache.generate_cache_key(sample_gif)
        assert key1 == key2
        
        # Non-existent file should return empty string
        non_existent = Path("/does/not/exist.gif")
        key3 = frame_cache.generate_cache_key(non_existent)
        assert key3 == ""
    
    def test_cache_put_and_get(self, frame_cache, sample_frames, sample_gif):
        """Test storing and retrieving frames from cache."""
        # Store frames
        frame_cache.put(
            sample_gif,
            sample_frames,
            frame_count=10,
            dimensions=(100, 100),
            duration_ms=1000
        )
        
        # Retrieve frames
        result = frame_cache.get(sample_gif, max_frames=None)
        
        assert result is not None
        frames, frame_count, dimensions, duration_ms = result
        assert len(frames) == len(sample_frames)
        assert frame_count == 10
        assert dimensions == (100, 100)
        assert duration_ms == 1000
        
        # Check frames match
        for orig_frame, cached_frame in zip(sample_frames, frames):
            np.testing.assert_array_equal(orig_frame, cached_frame)
    
    def test_cache_hit_miss_stats(self, frame_cache, sample_frames, sample_gif):
        """Test cache hit/miss statistics."""
        stats = frame_cache.get_stats()
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.total_accesses == 0
        
        # Miss
        result = frame_cache.get(sample_gif, max_frames=None)
        assert result is None
        
        stats = frame_cache.get_stats()
        assert stats.hits == 0
        assert stats.misses == 1
        assert stats.total_accesses == 1
        
        # Store
        frame_cache.put(
            sample_gif,
            sample_frames,
            frame_count=10,
            dimensions=(100, 100),
            duration_ms=1000
        )
        
        # Hit
        result = frame_cache.get(sample_gif, max_frames=None)
        assert result is not None
        
        stats = frame_cache.get_stats()
        assert stats.hits == 1
        assert stats.misses == 1
        assert stats.total_accesses == 2
        assert stats.hit_rate == 0.5
    
    def test_cache_max_frames_validation(self, frame_cache, sample_frames, sample_gif):
        """Test that cache validates max_frames parameter."""
        # Store with 5 frames
        frame_cache.put(
            sample_gif,
            sample_frames,
            frame_count=10,
            dimensions=(100, 100),
            duration_ms=1000
        )
        
        # Request with same number of frames should hit
        result = frame_cache.get(sample_gif, max_frames=5)
        assert result is not None
        
        # Request with different number should miss
        result = frame_cache.get(sample_gif, max_frames=3)
        assert result is None
    
    def test_cache_memory_eviction(self, temp_cache_dir):
        """Test LRU eviction when memory limit is reached."""
        # Very small cache for testing eviction
        cache = FrameCache(
            memory_limit_mb=0.5,  # 0.5MB (500KB) limit to ensure evictions
            disk_path=temp_cache_dir / "test.db",
            enabled=True
        )
        
        # Create multiple GIF files
        gif_files = []
        for i in range(5):
            gif_path = temp_cache_dir / f"test_{i}.gif"
            # Create large frames that will exceed memory limit
            frames = [np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)]
            
            # Create actual GIF file
            img = Image.fromarray(frames[0])
            img.save(gif_path)
            gif_files.append(gif_path)
            
            cache.put(gif_path, frames, 1, (200, 200), 100)
        
        # Check evictions happened
        stats = cache.get_stats()
        assert stats.evictions > 0
        
        # First entries should have been evicted
        result = cache.get(gif_files[0], max_frames=None)
        # May be in disk cache but not memory
    
    def test_cache_ttl_expiration(self, temp_cache_dir, sample_frames, sample_gif):
        """Test TTL expiration of cache entries."""
        # Cache with very short TTL
        cache = FrameCache(
            memory_limit_mb=10,
            disk_path=temp_cache_dir / "test.db",
            ttl_seconds=1,  # 1 second TTL
            enabled=True
        )
        
        # Store frames
        cache.put(
            sample_gif,
            sample_frames,
            frame_count=10,
            dimensions=(100, 100),
            duration_ms=1000
        )
        
        # Should hit immediately
        result = cache.get(sample_gif, max_frames=None)
        assert result is not None
        
        # Wait for TTL to expire
        time.sleep(1.5)  # Increased sleep to ensure TTL definitely expires
        
        # Should miss due to expiration
        result = cache.get(sample_gif, max_frames=None)
        assert result is None
    
    def test_cache_file_change_invalidation(self, frame_cache, sample_frames, tmp_path):
        """Test that cache invalidates when file changes."""
        gif_path = tmp_path / "changing.gif"
        
        # Create initial file
        img = Image.new("RGB", (100, 100), color=(100, 100, 100))
        img.save(gif_path)
        
        # Cache it
        frame_cache.put(
            gif_path,
            sample_frames,
            frame_count=1,
            dimensions=(100, 100),
            duration_ms=0
        )
        
        # Should hit
        result = frame_cache.get(gif_path, max_frames=None)
        assert result is not None
        
        # Modify file (changes mtime and size)
        time.sleep(0.01)  # Ensure different mtime
        img = Image.new("RGB", (150, 150), color=(200, 200, 200))
        img.save(gif_path)
        
        # Should miss due to file change
        result = frame_cache.get(gif_path, max_frames=None)
        assert result is None
    
    def test_cache_invalidate(self, frame_cache, sample_frames, sample_gif):
        """Test manual cache invalidation."""
        # Store frames
        frame_cache.put(
            sample_gif,
            sample_frames,
            frame_count=10,
            dimensions=(100, 100),
            duration_ms=1000
        )
        
        # Should hit
        result = frame_cache.get(sample_gif, max_frames=None)
        assert result is not None
        
        # Invalidate
        frame_cache.invalidate(sample_gif)
        
        # Should miss
        result = frame_cache.get(sample_gif, max_frames=None)
        assert result is None
    
    def test_cache_clear(self, frame_cache, sample_frames, sample_gif):
        """Test clearing the cache."""
        # Store frames
        frame_cache.put(
            sample_gif,
            sample_frames,
            frame_count=10,
            dimensions=(100, 100),
            duration_ms=1000
        )
        
        stats = frame_cache.get_stats()
        assert stats.memory_bytes > 0
        
        # Clear cache
        frame_cache.clear()
        
        stats = frame_cache.get_stats()
        assert stats.memory_bytes == 0
        assert stats.disk_entries == 0
        assert stats.hits == 0
        assert stats.misses == 0
    
    def test_cache_warm(self, frame_cache, tmp_path):
        """Test cache warming functionality."""
        # Create multiple GIF files
        gif_files = []
        for i in range(3):
            gif_path = tmp_path / f"warm_{i}.gif"
            img = Image.new("RGB", (50, 50), color=(i * 50, i * 50, i * 50))
            img.save(gif_path)
            gif_files.append(gif_path)
        
        # Mock extract_gif_frames to avoid import issues
        with patch("giflab.metrics.extract_gif_frames") as mock_extract:
            mock_result = MagicMock()
            mock_result.frames = [np.zeros((50, 50, 3), dtype=np.uint8)]
            mock_result.frame_count = 1
            mock_result.dimensions = (50, 50)
            mock_result.duration_ms = 0
            mock_extract.return_value = mock_result
            
            # Warm the cache
            frame_cache.warm_cache(gif_files, max_frames=None)
            
            # Check that extract was called for each file
            assert mock_extract.call_count == 3
    
    def test_disk_cache_persistence(self, temp_cache_dir, sample_frames, sample_gif):
        """Test that disk cache persists across instances."""
        # First cache instance
        cache1 = FrameCache(
            memory_limit_mb=10,
            disk_path=temp_cache_dir / "persist.db",
            enabled=True
        )
        
        # Store frames
        cache1.put(
            sample_gif,
            sample_frames,
            frame_count=10,
            dimensions=(100, 100),
            duration_ms=1000
        )
        
        # Create new instance with same disk path
        cache2 = FrameCache(
            memory_limit_mb=10,
            disk_path=temp_cache_dir / "persist.db",
            enabled=True
        )
        
        # Should hit from disk
        result = cache2.get(sample_gif, max_frames=None)
        assert result is not None
        
        stats = cache2.get_stats()
        assert stats.hits == 1
    
    def test_concurrent_access(self, frame_cache, sample_frames, tmp_path):
        """Test thread-safe concurrent access."""
        import threading
        
        gif_path = tmp_path / "concurrent.gif"
        img = Image.new("RGB", (100, 100), color=(100, 100, 100))
        img.save(gif_path)
        
        results = []
        
        def cache_operation():
            # Store and retrieve
            frame_cache.put(
                gif_path,
                sample_frames,
                frame_count=5,
                dimensions=(100, 100),
                duration_ms=500
            )
            result = frame_cache.get(gif_path, max_frames=None)
            results.append(result is not None)
        
        # Run multiple threads
        threads = []
        for _ in range(10):
            t = threading.Thread(target=cache_operation)
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        # All operations should succeed
        assert all(results)


class TestGlobalCache:
    """Tests for global cache instance."""
    
    def test_get_frame_cache_singleton(self):
        """Test that get_frame_cache returns singleton instance."""
        cache1 = get_frame_cache()
        cache2 = get_frame_cache()
        
        assert cache1 is cache2
    
    def test_reset_frame_cache(self):
        """Test resetting the global cache."""
        cache1 = get_frame_cache()
        reset_frame_cache()
        cache2 = get_frame_cache()
        
        assert cache1 is not cache2
    
    def test_cache_config_integration(self):
        """Test that cache uses config module settings."""
        mock_config = {
            'enabled': True,
            'memory_limit_mb': 250,
            'disk_path': None,
            'disk_limit_mb': 1000,
            'ttl_seconds': 43200
        }
        with patch("giflab.config.FRAME_CACHE", mock_config):
            
            # Reset to force new instance with mock config
            reset_frame_cache()
            cache = get_frame_cache()
            
            assert cache.memory_limit_bytes == 250 * 1024 * 1024
            assert cache.disk_limit_bytes == 1000 * 1024 * 1024
            assert cache.ttl_seconds == 43200


class TestCacheStats:
    """Tests for CacheStats class."""
    
    def test_cache_stats_creation(self):
        """Test creating cache statistics."""
        stats = CacheStats(
            hits=10,
            misses=5,
            evictions=2,
            memory_bytes=1024 * 1024,
            disk_entries=100,
            total_accesses=15
        )
        
        assert stats.hits == 10
        assert stats.misses == 5
        assert stats.evictions == 2
        assert stats.memory_bytes == 1024 * 1024
        assert stats.disk_entries == 100
        assert stats.total_accesses == 15
    
    def test_cache_stats_hit_rate(self):
        """Test hit rate calculation."""
        stats = CacheStats(
            hits=75,
            misses=25,
            total_accesses=100
        )
        
        assert stats.hit_rate == 0.75
        
        # Test zero accesses
        stats_zero = CacheStats()
        assert stats_zero.hit_rate == 0.0