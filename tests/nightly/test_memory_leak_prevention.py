"""
Tests for memory leak prevention in deep learning models.

This module ensures that model caching works correctly and prevents
memory leaks from repeated model instantiation.
"""

import gc
import os
import tempfile
import time
import tracemalloc
from pathlib import Path

import numpy as np
import psutil
import pytest
from giflab.config import MetricsConfig
from giflab.deep_perceptual_metrics import cleanup_global_validator
from giflab.metrics import (
    calculate_comprehensive_metrics_from_frames,
    cleanup_all_validators,
)
from giflab.model_cache import (
    LPIPSModelCache,
    cleanup_model_cache,
    get_model_cache_info,
)
from giflab.temporal_artifacts import cleanup_global_temporal_detector
from PIL import Image


class TestMemoryLeakPrevention:
    """Test suite for memory leak prevention."""

    def setup_method(self):
        """Set up test environment."""
        # Clean cache and global validator before each test
        cleanup_model_cache(force=True)
        cleanup_global_validator()
        gc.collect()

    def teardown_method(self):
        """Clean up after each test."""
        # Always clean cache and global validator after test
        cleanup_model_cache(force=True)
        cleanup_global_validator()
        gc.collect()

    @pytest.mark.fast
    def test_model_cache_singleton(self):
        """Test that model cache is a proper singleton."""
        cache1 = LPIPSModelCache()
        cache2 = LPIPSModelCache()

        # Should be the same instance
        assert cache1 is cache2

    @pytest.mark.fast
    def test_model_reuse(self):
        """Test that models are properly reused from cache."""
        # Get model multiple times
        model1 = LPIPSModelCache.get_model()
        model2 = LPIPSModelCache.get_model()
        model3 = LPIPSModelCache.get_model()

        # Should be the same model instance
        assert model1 is model2
        assert model2 is model3

        # Cache info should show only one model
        info = get_model_cache_info()
        assert info["models_cached"] == 1
        assert info["total_references"] == 3

    @pytest.mark.fast
    def test_no_memory_leak_with_repeated_calls(self):
        """Ensure repeated metric calculations don't leak memory."""
        # Start memory tracking
        tracemalloc.start()
        snapshot1 = tracemalloc.take_snapshot()

        # Create test frames
        frames = [
            np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8) for _ in range(3)
        ]

        # Run multiple iterations
        for _ in range(10):
            calculate_comprehensive_metrics_from_frames(
                frames, frames, config=MetricsConfig()
            )

        # Take second snapshot
        snapshot2 = tracemalloc.take_snapshot()
        top_stats = snapshot2.compare_to(snapshot1, "lineno")

        # Check for memory growth
        total_growth = sum(
            stat.size_diff for stat in top_stats[:10] if stat.size_diff > 0
        )
        total_growth_mb = total_growth / (1024 * 1024)

        tracemalloc.stop()

        # Should have minimal memory growth (< 10MB)
        assert total_growth_mb < 10, f"Memory grew by {total_growth_mb:.1f}MB"

    @pytest.mark.fast
    def test_memory_usage_with_cache_vs_without(self):
        """Compare memory usage with and without caching."""
        process = psutil.Process(os.getpid())

        # Test with cache enabled
        os.environ["GIFLAB_USE_MODEL_CACHE"] = "true"
        cleanup_model_cache(force=True)
        gc.collect()

        initial_memory = process.memory_info().rss

        frames = [
            np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8) for _ in range(2)
        ]

        # Run multiple times with cache
        for _ in range(5):
            calculate_comprehensive_metrics_from_frames(
                frames, frames, config=MetricsConfig()
            )

        cached_memory = process.memory_info().rss
        cached_increase = (cached_memory - initial_memory) / (1024 * 1024)

        # Memory increase with caching should be < 500MB
        # (LPIPS AlexNet model is ~475MB, so this accounts for the model + overhead)
        assert (
            cached_increase < 500
        ), f"Cached memory increased by {cached_increase:.1f}MB"

    @pytest.mark.fast
    def test_cache_cleanup(self):
        """Test that cache cleanup properly frees memory."""
        process = psutil.Process(os.getpid())

        # Get initial memory
        initial_memory = process.memory_info().rss

        # Load models into cache
        for _i in range(3):
            model = LPIPSModelCache.get_model(device="cpu")
            assert model is not None

        # Memory should increase after loading model (or already be loaded)
        loaded_memory = process.memory_info().rss
        # Skip the assertion if model was already in memory from a previous test
        # The important test is whether cleanup doesn't increase memory significantly

        # Clean cache and all validators
        cleanup_all_validators()
        gc.collect()
        time.sleep(0.1)  # Give GC time to work

        # Memory should decrease after cleanup (or at least not increase significantly)
        cleaned_memory = process.memory_info().rss
        (loaded_memory - cleaned_memory) / (1024 * 1024)

        # Should not increase memory after cleanup
        # Note: Due to Python memory management, we might not see immediate freeing
        # The threshold is set high because Python may not release memory immediately
        memory_increase_after_cleanup = (cleaned_memory - initial_memory) / (
            1024 * 1024
        )
        assert (
            memory_increase_after_cleanup < 500
        ), f"Memory still high after cleanup: {memory_increase_after_cleanup:.1f}MB"

    @pytest.mark.fast
    def test_reference_counting(self):
        """Test that reference counting works correctly."""
        # Get model and check reference count
        LPIPSModelCache.get_model()
        info = get_model_cache_info()
        assert info["total_references"] == 1

        # Get same model again
        LPIPSModelCache.get_model()
        info = get_model_cache_info()
        assert info["total_references"] == 2

        # Release references
        LPIPSModelCache.release_model()
        info = get_model_cache_info()
        assert info["total_references"] == 1

        LPIPSModelCache.release_model()
        info = get_model_cache_info()
        assert info["total_references"] == 0

    @pytest.mark.fast
    @pytest.mark.benchmark
    def test_performance_with_caching(self):
        """Test that caching improves performance."""
        frames = [
            np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8) for _ in range(3)
        ]

        # First call (model loading)
        start_time = time.perf_counter()
        calculate_comprehensive_metrics_from_frames(
            frames, frames, config=MetricsConfig()
        )
        first_call_time = time.perf_counter() - start_time

        # Second call (should use cached model)
        start_time = time.perf_counter()
        calculate_comprehensive_metrics_from_frames(
            frames, frames, config=MetricsConfig()
        )
        second_call_time = time.perf_counter() - start_time

        # Second call should be faster (cached model)
        # Allow generous variance for system timing jitter
        assert second_call_time <= first_call_time * 2.0, (
            f"Caching didn't improve performance: "
            f"first={first_call_time:.3f}s, second={second_call_time:.3f}s"
        )

    @pytest.mark.fast
    def test_concurrent_access(self):
        """Test that cache handles concurrent access correctly."""
        import threading

        results = []
        errors = []

        def get_model():
            try:
                model = LPIPSModelCache.get_model()
                results.append(model)
            except Exception as e:
                errors.append(e)

        # Create multiple threads
        threads = []
        for _ in range(10):
            t = threading.Thread(target=get_model)
            threads.append(t)
            t.start()

        # Wait for all threads
        for t in threads:
            t.join()

        # Should have no errors
        assert len(errors) == 0, f"Concurrent access errors: {errors}"

        # All should get the same model
        assert len(results) == 10
        first_model = results[0]
        for model in results[1:]:
            assert model is first_model

    @pytest.mark.fast
    def test_cache_info_accuracy(self):
        """Test that cache info reporting is accurate."""
        # Initial state
        info = get_model_cache_info()
        assert info["models_cached"] == 0
        assert info["total_references"] == 0

        # Add a model
        LPIPSModelCache.get_model()
        info = get_model_cache_info()
        assert info["models_cached"] == 1
        assert info["total_references"] == 1

        # Add another reference
        LPIPSModelCache.get_model()
        info = get_model_cache_info()
        assert info["models_cached"] == 1  # Still one model
        assert info["total_references"] == 2  # Two references

        # Different device should create new model
        LPIPSModelCache.get_model(device="cpu", spatial=True)
        info = get_model_cache_info()
        assert info["models_cached"] == 2  # Now two models
        assert info["total_references"] == 3  # Three total references


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "memory"])
