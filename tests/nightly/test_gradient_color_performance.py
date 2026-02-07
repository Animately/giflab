"""Performance benchmark tests for gradient banding and color validation features.

This test suite benchmarks the performance characteristics of the gradient banding
detection and perceptual color validation systems to ensure acceptable performance
in production scenarios.
"""

import gc
import os
import statistics
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import psutil
import pytest
from giflab.gradient_color_artifacts import (
    GradientBandingDetector,
    PerceptualColorValidator,
    calculate_gradient_color_metrics,
)
from giflab.metrics import calculate_comprehensive_metrics
from PIL import Image


def _get_performance_threshold_multiplier():
    """Get performance threshold multiplier based on environment.

    Returns higher multipliers for CI environments to account for:
    - Shared resources and variable load
    - Different hardware characteristics
    - Concurrent test execution
    - Cold start effects

    Returns:
        float: Multiplier for performance thresholds (1.0 = local dev, >1.0 = CI)
    """
    # Common CI environment variables
    ci_indicators = [
        "CI",  # Generic CI flag
        "CONTINUOUS_INTEGRATION",  # Common CI flag
        "GITHUB_ACTIONS",  # GitHub Actions
        "TRAVIS",  # Travis CI
        "JENKINS_URL",  # Jenkins
        "BUILDKITE",  # Buildkite
        "CIRCLECI",  # Circle CI
    ]

    # Check if running in CI environment
    is_ci = any(os.getenv(var) for var in ci_indicators)

    if is_ci:
        # Use 2.0x multiplier for CI environments
        # This accounts for shared resources, variable load, and hardware differences
        return 2.0
    else:
        # Local development environment - use strict thresholds
        return 1.0


@pytest.mark.benchmark
class TestPerformanceBenchmarks:
    """Performance benchmarks for gradient and color validation."""

    def setup_method(self):
        """Set up performance testing environment."""
        self.process = psutil.Process(os.getpid())
        self.temp_dir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        """Clean up after performance tests."""
        import shutil

        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
        gc.collect()  # Force garbage collection

    @pytest.mark.fast
    def test_banding_detection_speed(self):
        """Benchmark banding detection for various image sizes."""
        detector = GradientBandingDetector()

        sizes = [
            (64, 64),  # Small
            (128, 128),  # Medium
            (256, 256),  # Large
            (512, 512),  # Very large
        ]

        results = {}

        for size in sizes:
            # Create gradient frames for testing
            frames = self._create_gradient_frames(size, num_frames=3)

            # Benchmark detection
            times = []
            for _ in range(5):  # Run multiple times for accuracy
                start_time = time.perf_counter()
                result = detector.detect_banding_artifacts(frames, frames)
                end_time = time.perf_counter()
                times.append(end_time - start_time)

            avg_time = statistics.mean(times)
            std_time = statistics.stdev(times) if len(times) > 1 else 0

            results[size] = {
                "avg_time": avg_time,
                "std_time": std_time,
                "result": result,
            }

            print(f"Banding detection {size}: {avg_time:.3f}s ± {std_time:.3f}s")

        # Performance targets
        assert results[(64, 64)]["avg_time"] < 0.050  # <50ms for small images
        assert results[(128, 128)]["avg_time"] < 0.200  # <200ms for medium images
        assert results[(256, 256)]["avg_time"] < 0.800  # <800ms for large images
        assert results[(512, 512)]["avg_time"] < 3.000  # <3s for very large images

        # Verify scaling is reasonable (not exponential)
        small_time = results[(64, 64)]["avg_time"]
        large_time = results[(256, 256)]["avg_time"]
        scaling_factor = large_time / max(
            small_time, 0.0001
        )  # Avoid division by tiny numbers

        # 256x256 is 16x the pixels of 64x64, but allow generous scaling for very fast operations
        assert scaling_factor < 1000, f"Excessive scaling: {scaling_factor:.1f}x"

    @pytest.mark.fast
    def test_color_validation_speed(self):
        """Benchmark ΔE00 calculation performance."""
        validator = PerceptualColorValidator()

        sizes = [
            (64, 64),  # Small
            (128, 128),  # Medium
            (256, 256),  # Large
        ]

        results = {}

        for size in sizes:
            # Create color frames for testing
            frames_orig = self._create_color_frames(size, num_frames=3)
            frames_comp = self._create_shifted_color_frames(size, num_frames=3)

            # Benchmark color validation
            times = []
            for _ in range(5):
                start_time = time.perf_counter()
                result = validator.calculate_color_difference_metrics(
                    frames_orig, frames_comp
                )
                end_time = time.perf_counter()
                times.append(end_time - start_time)

            avg_time = statistics.mean(times)
            std_time = statistics.stdev(times) if len(times) > 1 else 0

            results[size] = {
                "avg_time": avg_time,
                "std_time": std_time,
                "result": result,
            }

            print(f"Color validation {size}: {avg_time:.3f}s ± {std_time:.3f}s")

        # Performance targets
        assert results[(64, 64)]["avg_time"] < 0.080  # <80ms for small images
        assert results[(128, 128)]["avg_time"] < 0.300  # <300ms for medium images
        assert results[(256, 256)]["avg_time"] < 1.200  # <1.2s for large images

    @pytest.mark.serial
    def test_combined_metrics_speed(self):
        """Benchmark combined gradient and color metrics calculation."""
        sizes = [(128, 128), (256, 256)]  # Focus on realistic sizes
        frame_counts = [3, 10, 20]  # Different frame counts

        results = {}

        for size in sizes:
            for num_frames in frame_counts:
                # Create test frames
                orig_frames = self._create_gradient_frames(size, num_frames)
                comp_frames = self._create_shifted_gradient_frames(size, num_frames)

                # Benchmark combined calculation
                times = []
                for _ in range(3):  # Fewer runs for larger tests
                    start_time = time.perf_counter()
                    calculate_gradient_color_metrics(orig_frames, comp_frames)
                    end_time = time.perf_counter()
                    times.append(end_time - start_time)

                avg_time = statistics.mean(times)
                key = f"{size}_{num_frames}frames"
                results[key] = {
                    "avg_time": avg_time,
                    "frames": num_frames,
                    "size": size,
                }

                print(f"Combined metrics {size} x{num_frames}: {avg_time:.3f}s")

        # Performance should scale reasonably with frame count
        # Use environment-aware thresholds for CI reliability
        threshold_multiplier = _get_performance_threshold_multiplier()
        max_scaling = 7.0 * threshold_multiplier

        for size in sizes:
            key_3 = f"{size}_3frames"
            key_10 = f"{size}_10frames"

            if key_3 in results and key_10 in results:
                time_3 = results[key_3]["avg_time"]
                time_10 = results[key_10]["avg_time"]

                # 10 frames should be <7x slower than 3 frames (allow for system variance)
                # Adjust for CI environments: local=7.0x, CI=14.0x
                scaling = time_10 / time_3
                print(
                    f"Frame scaling {size}: {scaling:.1f}x (threshold: {max_scaling:.1f}x, multiplier: {threshold_multiplier:.1f}x)"
                )
                assert (
                    scaling < max_scaling
                ), f"Poor frame scaling for {size}: {scaling:.1f}x > {max_scaling:.1f}x"

    @pytest.mark.fast
    def test_memory_usage(self):
        """Profile memory consumption during processing."""
        initial_memory = self.process.memory_info().rss / (1024 * 1024)  # MB

        # Create large frames to test memory efficiency
        large_frames = self._create_gradient_frames((512, 512), num_frames=10)

        memory_before = self.process.memory_info().rss / (1024 * 1024)

        # Process the frames
        result = calculate_gradient_color_metrics(large_frames, large_frames)

        memory_after = self.process.memory_info().rss / (1024 * 1024)

        # Clean up
        del large_frames, result
        gc.collect()

        memory_final = self.process.memory_info().rss / (1024 * 1024)

        print(
            f"Memory usage: {initial_memory:.1f} -> {memory_before:.1f} -> {memory_after:.1f} -> {memory_final:.1f} MB"
        )

        # Memory increase during processing should be reasonable
        processing_increase = memory_after - memory_before
        assert (
            processing_increase < 500
        ), f"Excessive memory use: {processing_increase:.1f}MB"

        # Memory should be mostly freed after processing
        final_increase = memory_final - initial_memory
        assert final_increase < 200, f"Memory leak suspected: {final_increase:.1f}MB"

    @pytest.mark.fast
    def test_cpu_usage_efficiency(self):
        """Test CPU utilization patterns."""
        # This test is more observational than assertive
        size = (256, 256)
        frames = self._create_gradient_frames(size, num_frames=5)

        # Monitor CPU usage
        cpu_before = self.process.cpu_percent(interval=None)

        start_time = time.perf_counter()
        calculate_gradient_color_metrics(frames, frames)
        end_time = time.perf_counter()

        processing_time = end_time - start_time
        cpu_after = self.process.cpu_percent(interval=None)

        print(
            f"CPU usage: {cpu_before:.1f}% -> {cpu_after:.1f}%, time: {processing_time:.3f}s"
        )

        # Should complete in reasonable time
        assert (
            processing_time < 5.0
        ), f"Processing took too long: {processing_time:.3f}s"

    @pytest.mark.serial
    def test_thread_safety_performance(self):
        """Test performance with concurrent execution."""
        size = (128, 128)
        frames = self._create_gradient_frames(size, num_frames=3)

        # Test sequential execution
        start_time = time.perf_counter()
        for _ in range(5):
            calculate_gradient_color_metrics(frames, frames)
        sequential_time = time.perf_counter() - start_time

        # Test concurrent execution
        results = []
        exceptions = []

        def worker():
            try:
                result = calculate_gradient_color_metrics(frames, frames)
                results.append(result)
            except Exception as e:
                exceptions.append(e)

        start_time = time.perf_counter()
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        concurrent_time = time.perf_counter() - start_time

        # Use environment-aware thresholds for CI reliability
        threshold_multiplier = _get_performance_threshold_multiplier()
        max_concurrent_ratio = 2.0 * threshold_multiplier

        print(f"Sequential: {sequential_time:.3f}s, Concurrent: {concurrent_time:.3f}s")
        print(
            f"Thread performance ratio: {concurrent_time/sequential_time:.1f}x (threshold: {max_concurrent_ratio:.1f}x, multiplier: {threshold_multiplier:.1f}x)"
        )

        # Should not crash
        assert len(exceptions) == 0, f"Concurrent execution failed: {exceptions}"
        assert len(results) == 5, f"Not all threads completed: {len(results)}/5"

        # Concurrent might be faster (or at least not much slower)
        # Adjust for CI environments: local=2.0x, CI=4.0x
        assert (
            concurrent_time <= sequential_time * max_concurrent_ratio
        ), f"Concurrent execution too slow: {concurrent_time/sequential_time:.1f}x > {max_concurrent_ratio:.1f}x"

    @pytest.mark.fast
    def test_comprehensive_metrics_performance_impact(self):
        """Test performance impact on comprehensive metrics calculation."""
        # Create test GIFs
        original_gif = self._create_test_gif("original.gif", (128, 128), frames=5)
        compressed_gif = self._create_test_gif("compressed.gif", (128, 128), frames=5)

        # Measure time with new metrics
        times_with = []
        for _ in range(3):
            start_time = time.perf_counter()
            calculate_comprehensive_metrics(original_gif, compressed_gif)
            times_with.append(time.perf_counter() - start_time)

        avg_time_with = statistics.mean(times_with)

        # Mock out gradient/color metrics to measure without them
        from unittest.mock import patch

        with patch(
            "giflab.gradient_color_artifacts.calculate_gradient_color_metrics",
            return_value={},
        ):
            times_without = []
            for _ in range(3):
                start_time = time.perf_counter()
                calculate_comprehensive_metrics(original_gif, compressed_gif)
                times_without.append(time.perf_counter() - start_time)

        avg_time_without = statistics.mean(times_without)
        overhead = avg_time_with / max(avg_time_without, 0.001)

        print(
            f"Metrics performance: {avg_time_without:.3f}s -> {avg_time_with:.3f}s (overhead: {overhead:.2f}x)"
        )

        # Overhead should be reasonable
        assert overhead < 3.0, f"Too much overhead: {overhead:.2f}x"
        assert avg_time_with < 10.0, f"Total time too high: {avg_time_with:.2f}s"

    # Helper methods for creating test data
    def _create_gradient_frames(self, size, num_frames):
        """Create frames with gradients for testing."""
        frames = []
        for _i in range(num_frames):
            frame = np.zeros(
                (size[1], size[0], 3), dtype=np.uint8
            )  # Note: numpy is (height, width)

            # Create horizontal gradient
            for x in range(size[0]):
                intensity = int(x * 255 / (size[0] - 1))
                frame[:, x] = [intensity, intensity // 2, (255 - intensity) // 3]

            frames.append(frame)
        return frames

    def _create_color_frames(self, size, num_frames):
        """Create frames with solid colors for testing."""
        frames = []
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]

        for i in range(num_frames):
            color = colors[i % len(colors)]
            frame = np.full((size[1], size[0], 3), color, dtype=np.uint8)
            frames.append(frame)
        return frames

    def _create_shifted_color_frames(self, size, num_frames):
        """Create frames with slightly shifted colors."""
        frames = []
        colors = [(230, 20, 20), (20, 230, 20), (20, 20, 230), (230, 230, 20)]

        for i in range(num_frames):
            color = colors[i % len(colors)]
            frame = np.full((size[1], size[0], 3), color, dtype=np.uint8)
            frames.append(frame)
        return frames

    def _create_shifted_gradient_frames(self, size, num_frames):
        """Create frames with slightly modified gradients."""
        frames = []
        for i in range(num_frames):
            frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)

            # Create horizontal gradient with slight variations
            offset = i * 10  # Small color offset per frame
            for x in range(size[0]):
                intensity = int(x * 255 / (size[0] - 1))
                shifted_intensity = min(255, max(0, intensity + offset))
                frame[:, x] = [
                    shifted_intensity,
                    intensity // 2,
                    (255 - intensity) // 3,
                ]

            frames.append(frame)
        return frames

    def _create_test_gif(self, filename, size, frames):
        """Create a test GIF file."""
        gif_path = self.temp_dir / filename

        images = []
        for i in range(frames):
            # Create simple gradient
            img = Image.new("RGB", size)
            pixels = img.load()

            for x in range(size[0]):
                for y in range(size[1]):
                    intensity = int((x + i * 10) * 255 / (size[0] - 1)) % 256
                    pixels[x, y] = (intensity, intensity // 2, 255 - intensity)

            images.append(img)

        images[0].save(
            gif_path, save_all=True, append_images=images[1:], duration=200, loop=0
        )

        return gif_path


@pytest.mark.benchmark
class TestScalingBenchmarks:
    """Test scaling characteristics with different parameters."""

    def setup_method(self):
        """Set up scaling tests."""
        self.temp_dir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        """Clean up scaling tests."""
        import shutil

        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    @pytest.mark.fast
    def test_patch_size_scaling(self):
        """Test performance scaling with different patch sizes."""
        detector = GradientBandingDetector()
        validator = PerceptualColorValidator()

        patch_sizes = [16, 32, 64, 128]
        image_size = (256, 256)
        frames = self._create_gradient_frames(image_size, num_frames=3)

        results = {}

        for patch_size in patch_sizes:
            # Test banding detector
            detector.patch_size = patch_size

            start_time = time.perf_counter()
            banding_result = detector.detect_banding_artifacts(frames, frames)
            banding_time = time.perf_counter() - start_time

            # Test color validator
            validator.patch_size = patch_size

            start_time = time.perf_counter()
            color_result = validator.calculate_color_difference_metrics(frames, frames)
            color_time = time.perf_counter() - start_time

            results[patch_size] = {
                "banding_time": banding_time,
                "color_time": color_time,
                "total_patches": banding_result.get("banding_patch_count", 0)
                + color_result.get("color_patch_count", 0),
            }

            print(
                f"Patch size {patch_size}: banding={banding_time:.3f}s, color={color_time:.3f}s"
            )

        # Smaller patches should generally be faster (fewer calculations per patch)
        # but might have more patches total
        assert all(r["banding_time"] < 2.0 for r in results.values())
        assert all(r["color_time"] < 2.0 for r in results.values())

    @pytest.mark.fast
    def test_frame_count_scaling(self):
        """Test performance scaling with different frame counts."""
        frame_counts = [1, 5, 10, 20]
        image_size = (128, 128)

        results = {}

        for frame_count in frame_counts:
            frames = self._create_gradient_frames(image_size, frame_count)

            start_time = time.perf_counter()
            calculate_gradient_color_metrics(frames, frames)
            processing_time = time.perf_counter() - start_time

            results[frame_count] = {
                "time": processing_time,
                "time_per_frame": processing_time / frame_count,
            }

            print(
                f"Frames {frame_count}: {processing_time:.3f}s ({processing_time/frame_count:.3f}s/frame)"
            )

        # Time per frame should be relatively consistent
        times_per_frame = [
            r["time_per_frame"]
            for r in results.values()
            if frame_counts.index(list(results.keys())[list(results.values()).index(r)])
            > 0
        ]

        if len(times_per_frame) > 1:
            time_variation = max(times_per_frame) / min(times_per_frame)
            assert (
                time_variation < 3.0
            ), f"Poor frame scaling consistency: {time_variation:.2f}x variation"

    # Helper methods for creating test data

    def _create_gradient_frames(self, size, num_frames):
        """Create frames with gradients for testing."""
        frames = []
        for _i in range(num_frames):
            frame = np.zeros(
                (size[1], size[0], 3), dtype=np.uint8
            )  # Note: numpy is (height, width)

            # Create horizontal gradient
            for x in range(size[0]):
                intensity = int(x * 255 / (size[0] - 1))
                frame[:, x] = [intensity, intensity // 2, (255 - intensity) // 3]

            frames.append(frame)
        return frames

    def _create_color_frames(self, size, num_frames):
        """Create frames with solid colors for testing."""
        frames = []
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]

        for i in range(num_frames):
            color = colors[i % len(colors)]
            frame = np.full((size[1], size[0], 3), color, dtype=np.uint8)
            frames.append(frame)
        return frames

    def _create_shifted_color_frames(self, size, num_frames):
        """Create frames with slightly shifted colors."""
        frames = []
        colors = [(230, 20, 20), (20, 230, 20), (20, 20, 230), (230, 230, 20)]

        for i in range(num_frames):
            color = colors[i % len(colors)]
            frame = np.full((size[1], size[0], 3), color, dtype=np.uint8)
            frames.append(frame)
        return frames

    def _create_shifted_gradient_frames(self, size, num_frames):
        """Create frames with slightly modified gradients."""
        frames = []
        for i in range(num_frames):
            frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)

            # Create horizontal gradient with slight variations
            offset = i * 10  # Small color offset per frame
            for x in range(size[0]):
                intensity = int(x * 255 / (size[0] - 1))
                shifted_intensity = min(255, max(0, intensity + offset))
                frame[:, x] = [
                    shifted_intensity,
                    intensity // 2,
                    (255 - intensity) // 3,
                ]

            frames.append(frame)
        return frames

    def _create_test_gif(self, filename, size, frames):
        """Create a test GIF file."""
        gif_path = self.temp_dir / filename

        images = []
        for i in range(frames):
            # Create simple gradient
            img = Image.new("RGB", size)
            pixels = img.load()

            for x in range(size[0]):
                for y in range(size[1]):
                    intensity = int((x + i * 10) * 255 / (size[0] - 1)) % 256
                    pixels[x, y] = (intensity, intensity // 2, 255 - intensity)

            images.append(img)

        images[0].save(
            gif_path, save_all=True, append_images=images[1:], duration=200, loop=0
        )

        return gif_path


@pytest.mark.benchmark
class TestStressTesting:
    """Stress tests for extreme conditions."""

    @pytest.mark.slow  # Mark as slow since these are stress tests
    def test_large_image_stress(self):
        """Stress test with very large images.

        Note: This test is intentionally skipped during normal test runs.
        To run stress tests, set the environment variable:
        GIFLAB_STRESS_TESTS=1 poetry run pytest tests/test_gradient_color_performance.py::TestStressTesting
        """
        # Only run if explicitly requested
        if not os.getenv("GIFLAB_STRESS_TESTS"):
            pytest.skip("Stress tests require GIFLAB_STRESS_TESTS=1")

        size = (1024, 1024)  # Very large
        frames = [
            np.random.randint(0, 255, (size[1], size[0], 3), dtype=np.uint8)
            for _ in range(2)
        ]

        start_time = time.perf_counter()
        result = calculate_gradient_color_metrics(frames, frames)
        processing_time = time.perf_counter() - start_time

        print(f"Large image stress test: {processing_time:.3f}s")

        # Should complete within reasonable time (allow generous limits for stress test)
        assert processing_time < 30.0, f"Stress test too slow: {processing_time:.3f}s"
        assert isinstance(result, dict)

    @pytest.mark.slow
    def test_many_frames_stress(self):
        """Stress test with many frames.

        Note: This test is intentionally skipped during normal test runs.
        To run stress tests, set the environment variable:
        GIFLAB_STRESS_TESTS=1 poetry run pytest tests/test_gradient_color_performance.py::TestStressTesting
        """
        if not os.getenv("GIFLAB_STRESS_TESTS"):
            pytest.skip("Stress tests require GIFLAB_STRESS_TESTS=1")

        size = (256, 256)
        num_frames = 50  # Many frames
        frames = [
            np.random.randint(0, 255, (size[1], size[0], 3), dtype=np.uint8)
            for _ in range(num_frames)
        ]

        start_time = time.perf_counter()
        result = calculate_gradient_color_metrics(frames, frames)
        processing_time = time.perf_counter() - start_time

        print(f"Many frames stress test: {processing_time:.3f}s")

        # Should scale reasonably with frame count
        assert (
            processing_time < 60.0
        ), f"Many frames test too slow: {processing_time:.3f}s"
        assert isinstance(result, dict)


# Fixtures and utilities
@pytest.fixture
def performance_frames():
    """Fixture providing standard performance test frames."""
    size = (128, 128)
    frames = []

    for i in range(5):
        frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        # Create varied content for realistic testing
        for x in range(size[0]):
            intensity = int((x + i * 20) * 255 / (size[0] - 1)) % 256
            frame[:, x] = [intensity, (intensity + 50) % 256, (255 - intensity) % 256]
        frames.append(frame)

    return frames


# Integration with existing test markers
pytestmark = [pytest.mark.benchmark, pytest.mark.performance]
