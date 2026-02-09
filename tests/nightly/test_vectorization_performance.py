"""Performance tests for vectorized synthetic GIF generation.

Moved from tests/functional/test_synthetic_gifs_performance.py because these
tests contain tight timing assertions that are incompatible with parallel
execution under pytest-xdist (-n auto). They belong in the nightly layer
per the project's 4-layer test architecture.
"""

import os
import time

import numpy as np
import pytest
from giflab.synthetic_gifs import SyntheticFrameGenerator
from PIL import Image

pytestmark = [pytest.mark.performance, pytest.mark.nightly]


def _get_performance_threshold_multiplier():
    """Get performance threshold multiplier based on environment.

    Returns higher multipliers for CI environments to account for
    shared resources, variable load, and concurrent test execution.
    """
    ci_indicators = [
        "CI",
        "CONTINUOUS_INTEGRATION",
        "GITHUB_ACTIONS",
        "TRAVIS",
        "JENKINS_URL",
        "BUILDKITE",
        "CIRCLECI",
    ]
    if any(os.getenv(var) for var in ci_indicators):
        return 3.0
    return 1.0


class TestPerformanceCharacteristics:
    """Lightweight performance tests to verify vectorization benefits."""

    def setup_method(self):
        """Setup test fixtures."""
        self.generator = SyntheticFrameGenerator()

    def test_large_image_performance_reasonable(self):
        """Test that large images generate in reasonable time."""
        large_size = (500, 500)
        multiplier = _get_performance_threshold_multiplier()

        start_time = time.time()
        img = self.generator.create_frame("gradient", large_size, 0, 8)
        end_time = time.time()

        generation_time = end_time - start_time

        threshold = 0.1 * multiplier
        assert (
            generation_time < threshold
        ), f"Large image took {generation_time:.3f}s, expected < {threshold}s"
        assert isinstance(img, Image.Image)
        assert img.size == large_size

    @pytest.mark.serial
    def test_multiple_frames_performance(self):
        """Test that generating multiple frames is efficient."""
        size = (200, 200)
        num_frames = 10
        multiplier = _get_performance_threshold_multiplier()

        start_time = time.time()
        for frame_idx in range(num_frames):
            img = self.generator.create_frame("noise", size, frame_idx, num_frames)
            assert isinstance(img, Image.Image)
        end_time = time.time()

        total_time = end_time - start_time
        avg_time_per_frame = total_time / num_frames

        threshold = 0.01 * multiplier
        assert (
            avg_time_per_frame < threshold
        ), f"Average frame time {avg_time_per_frame:.3f}s too slow (threshold {threshold}s)"

    def test_different_content_types_all_fast(self):
        """Test that all vectorized content types perform well."""
        content_types = ["gradient", "complex_gradient", "noise", "texture", "solid"]
        size = (150, 150)
        multiplier = _get_performance_threshold_multiplier()

        for content_type in content_types:
            start_time = time.time()
            img = self.generator.create_frame(content_type, size, 0, 5)
            end_time = time.time()

            generation_time = end_time - start_time
            threshold = 0.05 * multiplier
            assert (
                generation_time < threshold
            ), f"{content_type} took {generation_time:.3f}s (threshold {threshold}s)"
            assert isinstance(img, Image.Image)


@pytest.mark.performance
class TestPerformanceRegression:
    """Performance regression tests to ensure optimizations are maintained."""

    @pytest.mark.serial
    def test_no_performance_regression_medium_size(self):
        """Test that medium-sized images generate quickly (regression test)."""
        generator = SyntheticFrameGenerator()
        size = (200, 200)
        multiplier = _get_performance_threshold_multiplier()

        times = []
        for i in range(5):
            start = time.time()
            generator.create_frame("gradient", size, i, 5)
            times.append(time.time() - start)

        avg_time = sum(times) / len(times)

        threshold = 0.01 * multiplier
        assert (
            avg_time < threshold
        ), f"Performance regression: avg {avg_time:.4f}s > {threshold}s"

    @pytest.mark.serial
    def test_vectorization_still_active(self):
        """Test that vectorized operations are still being used."""
        generator = SyntheticFrameGenerator()
        multiplier = _get_performance_threshold_multiplier()

        # Generate a complex gradient which uses heavy numpy operations
        start_time = time.time()
        img = generator.create_frame("complex_gradient", (300, 300), 0, 8)
        elapsed = time.time() - start_time

        threshold = 0.02 * multiplier
        assert (
            elapsed < threshold
        ), f"Vectorization may not be working: {elapsed:.4f}s too slow (threshold {threshold}s)"

        # Verify the image has the expected complex characteristics
        img_array = np.array(img)
        assert img_array.std() > 30
