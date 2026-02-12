"""Integration tests for frame cache with metrics system."""

import time
from pathlib import Path

import numpy as np
import pytest
from giflab.caching import get_frame_cache, reset_frame_cache
from giflab.config import FRAME_CACHE
from giflab.metrics import extract_gif_frames
from PIL import Image


@pytest.fixture(autouse=True)
def reset_cache():
    """Reset global cache before and after each test."""
    reset_frame_cache()
    yield
    reset_frame_cache()


@pytest.fixture
def create_test_gif(tmp_path):
    """Factory fixture to create test GIF files."""
    def _create_gif(name: str, frames: int = 10, size: tuple[int, int] = (100, 100)):
        gif_path = tmp_path / f"{name}.gif"
        
        # Create animated GIF with specified number of frames
        images = []
        for i in range(frames):
            # Create frame with different colors for variety
            color = (
                (i * 255 // frames),
                ((i * 128) % 256),
                (255 - i * 255 // frames)
            )
            frame = Image.new("RGB", size, color=color)
            images.append(frame)
        
        images[0].save(
            gif_path,
            save_all=True,
            append_images=images[1:],
            duration=100,
            loop=0
        )
        
        return gif_path
    
    return _create_gif


@pytest.fixture
def enable_cache(monkeypatch):
    """Enable frame cache for testing."""
    monkeypatch.setitem(FRAME_CACHE, "enabled", True)
    monkeypatch.setitem(FRAME_CACHE, "memory_limit_mb", 100)
    monkeypatch.setitem(FRAME_CACHE, "ttl_seconds", 3600)


@pytest.fixture
def disable_cache(monkeypatch):
    """Disable frame cache for testing."""
    monkeypatch.setitem(FRAME_CACHE, "enabled", False)


class TestFrameCacheIntegration:
    """Integration tests for frame cache with extract_gif_frames."""
    
    def test_extract_gif_frames_uses_cache(self, create_test_gif, enable_cache):
        """Test that extract_gif_frames uses the cache."""
        gif_path = create_test_gif("test", frames=20)
        
        # First extraction should miss cache
        result1 = extract_gif_frames(gif_path, max_frames=10)
        
        cache = get_frame_cache()
        stats1 = cache.get_stats()
        assert stats1.misses == 1
        assert stats1.hits == 0
        
        # Second extraction should hit cache
        result2 = extract_gif_frames(gif_path, max_frames=10)
        
        stats2 = cache.get_stats()
        assert stats2.misses == 1
        assert stats2.hits == 1
        
        # Results should be identical
        assert result1.frame_count == result2.frame_count
        assert result1.dimensions == result2.dimensions
        assert result1.duration_ms == result2.duration_ms
        assert len(result1.frames) == len(result2.frames)
        
        # Frames should be identical
        for f1, f2 in zip(result1.frames, result2.frames, strict=True):
            np.testing.assert_array_equal(f1, f2)

    def test_extract_gif_frames_cache_disabled(self, create_test_gif, disable_cache):
        """Test that extract_gif_frames works without cache."""
        gif_path = create_test_gif("test", frames=10)
        
        # Both extractions should work but not cache
        result1 = extract_gif_frames(gif_path, max_frames=5)
        result2 = extract_gif_frames(gif_path, max_frames=5)
        
        cache = get_frame_cache()
        stats = cache.get_stats()
        
        # No cache activity when disabled
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.total_accesses == 0
        
        # Results should still be valid
        assert result1.frame_count == result2.frame_count
        assert len(result1.frames) == 5
        assert len(result2.frames) == 5
    
    def test_cache_different_max_frames(self, create_test_gif, enable_cache):
        """Test cache behavior with different max_frames values."""
        gif_path = create_test_gif("test", frames=30)
        
        # Extract with max_frames=10
        result1 = extract_gif_frames(gif_path, max_frames=10)
        assert len(result1.frames) == 10
        
        cache = get_frame_cache()
        stats1 = cache.get_stats()
        assert stats1.misses == 1
        
        # Extract with max_frames=10 again (should hit)
        result2 = extract_gif_frames(gif_path, max_frames=10)
        assert len(result2.frames) == 10
        
        stats2 = cache.get_stats()
        assert stats2.hits == 1
        
        # Extract with max_frames=20 (should miss)
        result3 = extract_gif_frames(gif_path, max_frames=20)
        assert len(result3.frames) == 20
        
        stats3 = cache.get_stats()
        assert stats3.misses == 2  # Second miss
    
    def test_cache_file_modification(self, tmp_path, enable_cache):
        """Test cache invalidation on file modification."""
        gif_path = tmp_path / "modifiable.gif"
        
        # Create initial GIF
        img1 = Image.new("RGB", (100, 100), color=(100, 0, 0))
        img1.save(gif_path)
        
        # Extract and cache
        result1 = extract_gif_frames(gif_path)
        
        cache = get_frame_cache()
        stats1 = cache.get_stats()
        assert stats1.misses == 1
        
        # Extract again (should hit)
        extract_gif_frames(gif_path)
        stats2 = cache.get_stats()
        assert stats2.hits == 1
        
        # Modify file
        time.sleep(0.01)  # Ensure different mtime
        img2 = Image.new("RGB", (100, 100), color=(0, 100, 0))
        img2.save(gif_path)
        
        # Extract again (should miss due to file change)
        result3 = extract_gif_frames(gif_path)
        stats3 = cache.get_stats()
        assert stats3.misses == 2
        
        # Verify the frame actually changed
        assert not np.array_equal(result1.frames[0], result3.frames[0])
    
    def test_cache_with_single_frame_image(self, tmp_path, enable_cache):
        """Test caching single-frame images (PNG, JPEG, single-frame GIF)."""
        # Test PNG
        png_path = tmp_path / "single.png"
        img = Image.new("RGB", (100, 100), color=(255, 0, 0))
        img.save(png_path)
        
        result1 = extract_gif_frames(png_path)
        assert result1.frame_count == 1
        assert len(result1.frames) == 1
        
        cache = get_frame_cache()
        stats1 = cache.get_stats()
        assert stats1.misses == 1
        
        # Second extraction should hit cache
        result2 = extract_gif_frames(png_path)
        stats2 = cache.get_stats()
        assert stats2.hits == 1
        
        # Results should be identical
        np.testing.assert_array_equal(result1.frames[0], result2.frames[0])
    
    def test_cache_memory_limits(self, create_test_gif, monkeypatch):
        """Test cache behavior when memory limits are reached."""
        # Set very low memory limit
        monkeypatch.setitem(FRAME_CACHE, "enabled", True)
        monkeypatch.setitem(FRAME_CACHE, "memory_limit_mb", 1)  # 1MB limit
        reset_frame_cache()
        
        # Create multiple large GIFs
        gifs = []
        for i in range(5):
            # Large frames that will exceed memory limit
            gif_path = create_test_gif(f"large_{i}", frames=10, size=(200, 200))
            gifs.append(gif_path)
        
        # Extract all GIFs
        for gif_path in gifs:
            extract_gif_frames(gif_path, max_frames=5)
        
        cache = get_frame_cache()
        stats = cache.get_stats()
        
        # Should have evictions due to memory limit
        assert stats.evictions > 0
        
        # Early GIFs should have been evicted from memory
        # (may still be in disk cache)
    
    def test_cache_performance_improvement(self, create_test_gif, enable_cache):
        """Test that cache actually improves performance."""
        gif_path = create_test_gif("perf_test", frames=100)
        
        # Time first extraction (cache miss)
        start1 = time.perf_counter()
        result1 = extract_gif_frames(gif_path, max_frames=50)
        time1 = time.perf_counter() - start1
        
        # Time second extraction (cache hit)
        start2 = time.perf_counter()
        result2 = extract_gif_frames(gif_path, max_frames=50)
        time2 = time.perf_counter() - start2
        
        # Cache hit should be significantly faster
        # (at least 10x faster, typically 100x or more)
        assert time2 < time1 / 10
        
        # Verify results are identical
        assert len(result1.frames) == len(result2.frames)
        for f1, f2 in zip(result1.frames, result2.frames, strict=True):
            np.testing.assert_array_equal(f1, f2)

    def test_cache_with_corrupted_gif(self, tmp_path, enable_cache):
        """Test cache behavior with corrupted GIF files."""
        # Create a corrupted GIF (invalid data)
        corrupted_path = tmp_path / "corrupted.gif"
        corrupted_path.write_bytes(b"GIF89a" + b"\x00" * 100)  # Invalid GIF data
        
        # Should raise error on first attempt
        with pytest.raises(OSError):
            extract_gif_frames(corrupted_path)
        
        # Should raise error on second attempt (not cached)
        with pytest.raises(OSError):
            extract_gif_frames(corrupted_path)
        
        # Cache should not have stored the failure
        cache = get_frame_cache()
        stats = cache.get_stats()
        # May have attempts but no successful caching
        assert stats.hits == 0
    
    def test_concurrent_extractions(self, create_test_gif, enable_cache):
        """Test concurrent frame extractions with caching."""
        import concurrent.futures
        
        gif_path = create_test_gif("concurrent", frames=50)
        
        def extract_frames():
            return extract_gif_frames(gif_path, max_frames=25)
        
        # Run multiple concurrent extractions
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(extract_frames) for _ in range(20)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        # All results should be valid and identical
        assert len(results) == 20
        reference = results[0]
        for result in results[1:]:
            assert result.frame_count == reference.frame_count
            assert result.dimensions == reference.dimensions
            assert len(result.frames) == len(reference.frames)
        
        # Cache should show activity
        cache = get_frame_cache()
        stats = cache.get_stats()
        assert stats.total_accesses == 20
        # At least some should be hits (exact number depends on timing)
        assert stats.hits > 0
    
    def test_cache_warm_and_use(self, create_test_gif, enable_cache):
        """Test warming cache and then using it."""
        # Create multiple test GIFs
        gif_paths = [
            create_test_gif(f"warm_{i}", frames=20)
            for i in range(5)
        ]
        
        cache = get_frame_cache()
        
        # Warm the cache
        cache.warm_cache(gif_paths, max_frames=10)
        
        # Now extract should hit cache for all files
        initial_stats = cache.get_stats()
        
        for gif_path in gif_paths:
            result = extract_gif_frames(gif_path, max_frames=10)
            assert len(result.frames) == 10
        
        final_stats = cache.get_stats()
        
        # All extractions should have been cache hits
        hits_added = final_stats.hits - initial_stats.hits
        assert hits_added == len(gif_paths)
