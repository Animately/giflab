"""Performance tests for ResizedFrameCache.

Moved from tests/functional/test_resized_frame_cache.py because these tests
contain timing comparisons and memory-pressure assertions that are flaky under
parallel execution with pytest-xdist (-n auto). They belong in the nightly
layer per the project's 4-layer test architecture.
"""

import time

import cv2
import numpy as np
import pytest

from giflab.caching.resized_frame_cache import ResizedFrameCache

pytestmark = [pytest.mark.performance, pytest.mark.nightly]


class TestCachePerformance:
    """Performance-related tests for cache."""

    @pytest.mark.serial
    def test_cache_speedup(self):
        """Test that cached resizes are significantly faster."""
        cache = ResizedFrameCache(memory_limit_mb=100)
        frame = np.random.randint(0, 255, (2000, 2000, 3), dtype=np.uint8)
        target_size = (1000, 1000)

        # Warm up - first call also hashes the frame
        cache.get(frame, target_size, cv2.INTER_AREA)
        cache.clear()

        # First resize - uncached (includes hashing + resize)
        start = time.perf_counter()
        cache.get(frame, target_size, cv2.INTER_AREA)
        uncached_time = time.perf_counter() - start

        # Second resize - cached (only hash + lookup + copy)
        start = time.perf_counter()
        cache.get(frame, target_size, cv2.INTER_AREA)
        cached_time = time.perf_counter() - start

        # Cached should generally be faster than uncached
        # Allow small margin for timing jitter on fast operations
        assert cached_time < uncached_time * 1.5

    def test_memory_limit_respected(self):
        """Test that cache respects memory limit."""
        memory_limit_mb = 1
        cache = ResizedFrameCache(memory_limit_mb=memory_limit_mb)

        # Create large frames that will exceed 1MB limit
        # Each 500x500x3 resized to 400x400x3 = 480000 bytes (~0.46MB)
        # 3 frames = ~1.4MB > 1MB limit
        for i in range(5):
            frame = np.random.randint(0, 255, (500, 500, 3), dtype=np.uint8)
            cache.get(frame, (400, 400), cv2.INTER_AREA)

        stats = cache.get_stats()
        assert stats["memory_mb"] <= memory_limit_mb
        assert stats["evictions"] > 0
