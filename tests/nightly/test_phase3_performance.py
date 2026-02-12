"""Performance tests for Phase 3 conditional content-specific metrics.

This module tests the computational efficiency and performance characteristics
of Text/UI validation and SSIMULACRA2 components to ensure they meet performance
targets and don't introduce unacceptable overhead to the metrics pipeline.
"""

import gc
import os
import resource
import statistics
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np
import psutil
import pytest
from giflab.config import MetricsConfig
from giflab.metrics import calculate_comprehensive_metrics_from_frames
from giflab.ssimulacra2_metrics import (
    Ssimulacra2Validator,
    calculate_ssimulacra2_quality_metrics,
    should_use_ssimulacra2,
)
from giflab.text_ui_validation import (
    EdgeAcuityAnalyzer,
    OCRValidator,
    TextUIContentDetector,
    calculate_text_ui_metrics,
    should_validate_text_ui,
)

# Import fixture generator for consistent test data
try:
    from tests.fixtures.generate_phase3_fixtures import Phase3FixtureGenerator
except ImportError:
    Phase3FixtureGenerator = None


@pytest.fixture
def fixture_generator():
    """Create fixture generator for tests."""
    if Phase3FixtureGenerator is None:
        pytest.skip("Phase 3 fixture generator not available")

    with tempfile.TemporaryDirectory() as tmpdir:
        generator = Phase3FixtureGenerator(Path(tmpdir))
        yield generator


def measure_execution_time(func, *args, **kwargs):
    """Measure execution time of a function."""
    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    return result, execution_time


def measure_memory_usage(func, *args, **kwargs):
    """Measure peak memory usage during function execution."""
    process = psutil.Process()
    initial_memory = process.memory_info().rss

    result = func(*args, **kwargs)

    # Force garbage collection and measure again
    gc.collect()
    final_memory = process.memory_info().rss

    memory_increase = final_memory - initial_memory
    return result, memory_increase


class TestTextUIValidationPerformance:
    """Performance tests for Text/UI validation components."""

    def test_edge_detection_performance(self):
        """Test edge detection performance with different image sizes."""
        detector = TextUIContentDetector()

        # Test with different image sizes
        image_sizes = [(50, 50), (100, 100), (200, 200), (500, 500), (1000, 1000)]

        for width, height in image_sizes:
            frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)

            # Measure execution time
            _, execution_time = measure_execution_time(
                detector._detect_edge_density, frame
            )

            # Performance target: <10ms per 100x100 frame, scaling roughly O(n)
            pixels = width * height
            target_time = (pixels / 10000) * 0.01  # 10ms for 100x100

            assert (
                execution_time < target_time * 5
            ), f"Edge detection too slow: {execution_time:.4f}s > {target_time * 5:.4f}s for {width}x{height}"

    def test_component_analysis_performance(self):
        """Test connected component analysis performance."""
        detector = TextUIContentDetector()

        # Test with different numbers of components
        component_counts = [0, 5, 20, 50, 100]

        for count in component_counts:
            # Create binary image with specified number of components
            binary_image = np.zeros((200, 200), dtype=np.uint8)

            for i in range(count):
                x = (i * 15) % 180 + 10
                y = (i * 10) % 180 + 10
                binary_image[y : y + 8, x : x + 12] = 255

            _, execution_time = measure_execution_time(
                detector._find_text_like_components, binary_image
            )

            # Performance target: <50ms for 100 components
            target_time = 0.05 * (count / 100) if count > 0 else 0.01

            assert (
                execution_time < target_time * 2
            ), f"Component analysis too slow: {execution_time:.4f}s for {count} components"

    def test_ocr_validation_performance(self, fixture_generator):
        """Test OCR validation performance with different content types."""
        validator = OCRValidator()

        # Test different text content types
        content_types = ["clean_text", "blurry_text", "terminal_text"]

        for content_type in content_types:
            img_path = fixture_generator.create_text_ui_image(
                content_type, size=(150, 150)
            )
            frame = cv2.imread(str(img_path))

            # Define text regions
            regions = [(20, 30, 100, 20), (20, 60, 80, 20)]

            _, execution_time = measure_execution_time(
                validator.calculate_ocr_confidence_delta, frame, frame, regions
            )

            # Performance target: <1s per frame with text regions
            assert (
                execution_time < 1.0
            ), f"OCR validation too slow: {execution_time:.4f}s for {content_type}"

    def test_edge_acuity_performance(self):
        """Test edge acuity (MTF50) analysis performance."""
        analyzer = EdgeAcuityAnalyzer()

        # Test with different numbers of regions
        region_counts = [1, 3, 5, 10, 20]

        for count in region_counts:
            frame = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)

            # Create regions
            regions = []
            for i in range(count):
                x = (i * 30) % 150 + 20
                y = (i * 25) % 150 + 20
                regions.append((x, y, 25, 25))

            _, execution_time = measure_execution_time(
                analyzer.calculate_mtf50, frame, regions
            )

            # Performance target: <100ms for 10 regions
            target_time = 0.1 * (count / 10)

            assert (
                execution_time < target_time * 2
            ), f"Edge acuity analysis too slow: {execution_time:.4f}s for {count} regions"

    def test_text_ui_metrics_scalability(self):
        """Test scalability of text/UI metrics with frame count."""
        frame_counts = [1, 3, 5, 10, 20]

        for count in frame_counts:
            # Create frames with text-like content
            frames = []
            for _i in range(count):
                frame = np.zeros((100, 100, 3), dtype=np.uint8)
                # Add text-like rectangles
                frame[20:25, 10:50, :] = 255
                frame[30:35, 10:40, :] = 255
                frames.append(frame)

            _, execution_time = measure_execution_time(
                calculate_text_ui_metrics, frames, frames, max_frames=min(count, 5)
            )

            # Performance should be roughly linear with frame count (up to max_frames)
            effective_frames = min(count, 5)
            target_time = 0.1 * effective_frames  # 100ms per frame

            assert (
                execution_time < target_time * 2
            ), f"Text/UI metrics too slow: {execution_time:.4f}s for {count} frames"

    def test_conditional_execution_overhead(self):
        """Test overhead of conditional execution logic."""
        # Test content that should NOT trigger text/UI validation
        smooth_frames = [np.full((80, 80, 3), 128, dtype=np.uint8) for _ in range(5)]

        # Measure time for should_validate_text_ui check
        _, check_time = measure_execution_time(
            should_validate_text_ui, smooth_frames, quick_check=True
        )

        # Performance target: <10ms for conditional check
        assert check_time < 0.01, f"Conditional check too slow: {check_time:.4f}s"

        # Verify it correctly skips expensive processing
        should_validate, _ = should_validate_text_ui(smooth_frames, quick_check=True)
        if not should_validate:
            # If validation is skipped, full calculation should be very fast
            _, calc_time = measure_execution_time(
                calculate_text_ui_metrics, smooth_frames, smooth_frames
            )

            # Should be much faster when skipped
            assert (
                calc_time < 0.05
            ), f"Skipped calculation not fast enough: {calc_time:.4f}s"

    def test_memory_efficiency_text_ui(self):
        """Test memory efficiency of text/UI validation."""
        # Create large frames to test memory usage
        large_frames = [
            np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8) for _ in range(10)
        ]

        _, memory_increase = measure_memory_usage(
            calculate_text_ui_metrics, large_frames, large_frames, max_frames=5
        )

        # Memory increase should be reasonable (< 100MB for this test)
        max_memory_mb = 100 * 1024 * 1024  # 100MB in bytes
        assert (
            memory_increase < max_memory_mb
        ), f"Excessive memory usage: {memory_increase / 1024 / 1024:.1f}MB"


class TestSsimulacra2Performance:
    """Performance tests for SSIMULACRA2 integration."""

    def test_frame_sampling_efficiency(self):
        """Test efficiency of frame sampling strategies."""
        validator = Ssimulacra2Validator()

        # Test with very large frame counts
        large_counts = [100, 500, 1000, 5000, 10000]

        for count in large_counts:
            _, execution_time = measure_execution_time(
                validator._sample_frame_indices, count, 30
            )

            # Should be very fast regardless of frame count
            assert (
                execution_time < 0.001
            ), f"Frame sampling too slow: {execution_time:.6f}s for {count} frames"

    def test_score_normalization_performance(self):
        """Test performance of score normalization."""
        validator = Ssimulacra2Validator()

        # Test batch normalization
        batch_sizes = [10, 100, 1000, 10000]

        for size in batch_sizes:
            scores = np.random.uniform(-50, 150, size)

            _, execution_time = measure_execution_time(
                lambda scores=scores: [validator.normalize_score(s) for s in scores]
            )

            # Performance target: <1ms per 1000 scores
            target_time = (size / 1000) * 0.001

            assert (
                execution_time < target_time * 10
            ), f"Score normalization too slow: {execution_time:.6f}s for {size} scores"

    @patch.object(Ssimulacra2Validator, "is_available", return_value=True)
    @patch.object(Ssimulacra2Validator, "_export_frame_to_png")
    @patch.object(Ssimulacra2Validator, "_run_ssimulacra2_on_pair")
    def test_ssimulacra2_calculation_performance(
        self, mock_run, mock_export, mock_available
    ):
        """Test performance of SSIMULACRA2 calculation pipeline."""
        validator = Ssimulacra2Validator()
        config = MetricsConfig()

        # Mock subprocess to return quickly
        mock_run.return_value = 50.0

        # Test with different frame counts
        frame_counts = [1, 5, 10, 20, 30]

        for count in frame_counts:
            frames = [
                np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
                for _ in range(count)
            ]

            _, execution_time = measure_execution_time(
                validator.calculate_ssimulacra2_metrics, frames, frames, config
            )

            # Performance should be linear with frame count
            # (excluding actual subprocess time which is mocked)
            target_time = 0.05 * count  # 50ms per frame

            assert (
                execution_time < target_time
            ), f"SSIMULACRA2 pipeline too slow: {execution_time:.4f}s for {count} frames"

    def test_conditional_triggering_performance(self):
        """Test performance of conditional triggering logic."""
        validator = Ssimulacra2Validator()

        # Test many conditional checks
        quality_values = np.random.uniform(0.0, 1.0, 10000)

        _, execution_time = measure_execution_time(
            lambda: [validator.should_use_ssimulacra2(q) for q in quality_values]
        )

        # Should be very fast
        assert (
            execution_time < 0.01
        ), f"Conditional triggering too slow: {execution_time:.6f}s for 10000 checks"

    def test_temporary_file_management_performance(self):
        """Test performance impact of temporary file management."""
        validator = Ssimulacra2Validator()

        with patch.object(validator, "is_available", return_value=True), patch.object(
            validator, "_run_ssimulacra2_on_pair", return_value=50.0
        ):
            # Test frame export performance
            frame = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)

            times = []
            for _i in range(10):
                with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
                    _, exec_time = measure_execution_time(
                        validator._export_frame_to_png, frame, Path(tmp.name)
                    )
                    times.append(exec_time)

            # Should be consistently fast
            avg_time = statistics.mean(times)
            max_time = max(times)

            assert avg_time < 0.05, f"Average export time too slow: {avg_time:.4f}s"
            assert max_time < 0.1, f"Worst export time too slow: {max_time:.4f}s"

    def test_memory_efficiency_ssimulacra2(self):
        """Test memory efficiency of SSIMULACRA2 processing."""
        validator = Ssimulacra2Validator()
        config = MetricsConfig()
        config.SSIMULACRA2_MAX_FRAMES = 5

        with patch.object(validator, "is_available", return_value=True), patch.object(
            validator, "_export_frame_to_png"
        ), patch.object(validator, "_run_ssimulacra2_on_pair", return_value=50.0):
            # Create many large frames
            large_frames = [
                np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8)
                for _ in range(20)
            ]

            _, memory_increase = measure_memory_usage(
                validator.calculate_ssimulacra2_metrics,
                large_frames,
                large_frames,
                config,
            )

            # Should not use excessive memory despite large input
            # (due to frame sampling and temporary file cleanup)
            max_memory_mb = 150 * 1024 * 1024  # 150MB in bytes
            assert (
                memory_increase < max_memory_mb
            ), f"Excessive memory usage: {memory_increase / 1024 / 1024:.1f}MB"


class TestPhase3PipelinePerformance:
    """Performance tests for Phase 3 integration with main pipeline."""

    def test_comprehensive_metrics_performance_impact(self):
        """Test performance impact of Phase 3 on comprehensive metrics."""
        frames = [
            np.random.randint(0, 255, (150, 150, 3), dtype=np.uint8) for _ in range(5)
        ]

        config_without_phase3 = MetricsConfig()
        config_without_phase3.USE_COMPREHENSIVE_METRICS = True
        config_without_phase3.ENABLE_SSIMULACRA2 = False

        config_with_phase3 = MetricsConfig()
        config_with_phase3.USE_COMPREHENSIVE_METRICS = True
        config_with_phase3.ENABLE_SSIMULACRA2 = True

        # Mock Phase 3 components for consistent timing
        with patch(
            "giflab.text_ui_validation.calculate_text_ui_metrics"
        ) as mock_text_ui, patch(
            "giflab.ssimulacra2_metrics.calculate_ssimulacra2_quality_metrics"
        ) as mock_ssim2, patch(
            "giflab.text_ui_validation.should_validate_text_ui", return_value=(True, {})
        ), patch(
            "giflab.ssimulacra2_metrics.should_use_ssimulacra2", return_value=True
        ):
            # Quick mock responses
            mock_text_ui.return_value = {
                "has_text_ui_content": True,
                "text_ui_edge_density": 0.15,
            }
            mock_ssim2.return_value = {
                "ssimulacra2_mean": 0.70,
                "ssimulacra2_triggered": 1.0,
            }

            # Measure without Phase 3
            _, time_without = measure_execution_time(
                calculate_comprehensive_metrics_from_frames,
                frames,
                frames,
                config_without_phase3,
            )

            # Measure with Phase 3
            _, time_with = measure_execution_time(
                calculate_comprehensive_metrics_from_frames,
                frames,
                frames,
                config_with_phase3,
            )

        # Phase 3 overhead should be reasonable
        overhead = time_with - time_without
        overhead_percent = (overhead / time_without) * 100 if time_without > 0 else 0

        # Performance target: <50% overhead from Phase 3 additions
        assert (
            overhead_percent < 50
        ), f"Phase 3 adds too much overhead: {overhead_percent:.1f}%"

        # Absolute overhead should be reasonable
        assert overhead < 0.5, f"Phase 3 absolute overhead too high: {overhead:.4f}s"

    def test_conditional_execution_efficiency(self):
        """Test efficiency of conditional execution in pipeline."""
        # Create content that should skip both Phase 3 components
        smooth_frames = [np.full((100, 100, 3), 128, dtype=np.uint8) for _ in range(3)]

        config = MetricsConfig()
        config.USE_COMPREHENSIVE_METRICS = True
        config.ENABLE_SSIMULACRA2 = True

        # Measure time when both components should be skipped
        _, execution_time = measure_execution_time(
            calculate_comprehensive_metrics_from_frames,
            smooth_frames,
            smooth_frames,
            config,
        )

        # Should complete quickly when Phase 3 components are skipped
        # (Exact time depends on other metrics, but shouldn't be excessive)
        assert (
            execution_time < 2.0
        ), f"Pipeline too slow even with Phase 3 skipped: {execution_time:.4f}s"

    def test_parallel_processing_scalability(self):
        """Test scalability with parallel processing simulation."""
        # Simulate processing multiple frame pairs concurrently
        frame_pairs = [
            (
                [
                    np.random.randint(0, 255, (80, 80, 3), dtype=np.uint8)
                    for _ in range(2)
                ],
                [
                    np.random.randint(0, 255, (80, 80, 3), dtype=np.uint8)
                    for _ in range(2)
                ],
            )
            for _ in range(5)
        ]

        config = MetricsConfig()

        # Mock Phase 3 components to simulate processing time
        with patch(
            "giflab.text_ui_validation.calculate_text_ui_metrics"
        ) as mock_text_ui, patch(
            "giflab.ssimulacra2_metrics.calculate_ssimulacra2_quality_metrics"
        ) as mock_ssim2, patch(
            "giflab.text_ui_validation.should_validate_text_ui", return_value=(True, {})
        ), patch(
            "giflab.ssimulacra2_metrics.should_use_ssimulacra2", return_value=True
        ):

            def slow_text_ui(*args, **kwargs):
                time.sleep(0.01)  # Simulate 10ms processing
                return {"has_text_ui_content": False}

            def slow_ssim2(*args, **kwargs):
                time.sleep(0.02)  # Simulate 20ms processing
                return {"ssimulacra2_mean": 0.5, "ssimulacra2_triggered": 1.0}

            mock_text_ui.side_effect = slow_text_ui
            mock_ssim2.side_effect = slow_ssim2

            # Test sequential processing
            start_time = time.perf_counter()
            for orig_frames, comp_frames in frame_pairs:
                calculate_comprehensive_metrics_from_frames(
                    orig_frames, comp_frames, config
                )
            sequential_time = time.perf_counter() - start_time

            # Test parallel processing simulation
            def process_pair(pair):
                orig_frames, comp_frames = pair
                return calculate_comprehensive_metrics_from_frames(
                    orig_frames, comp_frames, config
                )

            start_time = time.perf_counter()
            with ThreadPoolExecutor(max_workers=3) as executor:
                list(executor.map(process_pair, frame_pairs))
            parallel_time = time.perf_counter() - start_time

            # Parallel should be faster (though limited by mock delays)
            speedup = sequential_time / parallel_time if parallel_time > 0 else 1
            assert (
                speedup > 1.2
            ), f"Insufficient parallelization speedup: {speedup:.2f}x"

    def test_large_batch_processing(self):
        """Test performance with large batches of frames."""
        # Test processing many frame sequences
        batch_sizes = [5, 10, 20, 50]

        config = MetricsConfig()
        config.USE_COMPREHENSIVE_METRICS = True

        for batch_size in batch_sizes:
            frame_sequences = []
            for _i in range(batch_size):
                frames = [
                    np.random.randint(0, 255, (60, 60, 3), dtype=np.uint8)
                    for _ in range(2)
                ]
                frame_sequences.append((frames, frames))

            with patch(
                "giflab.text_ui_validation.calculate_text_ui_metrics",
                return_value={"has_text_ui_content": False},
            ), patch(
                "giflab.ssimulacra2_metrics.calculate_ssimulacra2_quality_metrics",
                return_value={"ssimulacra2_triggered": 0.0},
            ):
                # Measure batch processing time
                start_time = time.perf_counter()
                for orig_frames, comp_frames in frame_sequences:
                    calculate_comprehensive_metrics_from_frames(
                        orig_frames, comp_frames, config
                    )
                batch_time = time.perf_counter() - start_time

                # Performance should scale reasonably
                time_per_sequence = batch_time / batch_size
                assert (
                    time_per_sequence < 0.2
                ), f"Batch processing too slow: {time_per_sequence:.4f}s per sequence"


class TestPerformanceRegression:
    """Tests to detect performance regressions."""

    def test_baseline_performance_benchmarks(self):
        """Establish baseline performance benchmarks."""
        # Standard test case
        frames = [
            np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8) for _ in range(3)
        ]
        config = MetricsConfig()

        # Run multiple iterations to get stable timing
        times = []
        for _i in range(5):
            _, exec_time = measure_execution_time(
                calculate_comprehensive_metrics_from_frames, frames, frames, config
            )
            times.append(exec_time)

        avg_time = statistics.mean(times)
        std_dev = statistics.stdev(times) if len(times) > 1 else 0

        # Document benchmark results
        print(f"Baseline performance: {avg_time:.4f}s ± {std_dev:.4f}s")

        # Performance regression threshold
        max_acceptable_time = 3.0  # 3 seconds for comprehensive metrics
        assert (
            avg_time < max_acceptable_time
        ), f"Performance regression: {avg_time:.4f}s > {max_acceptable_time}s"

    def test_memory_leak_detection(self):
        """Test for memory leaks in repeated operations."""
        frames = [
            np.random.randint(0, 255, (80, 80, 3), dtype=np.uint8) for _ in range(3)
        ]
        config = MetricsConfig()

        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss

        # Run many iterations
        for i in range(50):
            calculate_comprehensive_metrics_from_frames(frames, frames, config)

            # Force garbage collection periodically
            if i % 10 == 9:
                gc.collect()

        # Check final memory usage
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # Memory increase should be minimal
        max_increase_mb = 50 * 1024 * 1024  # 50MB
        assert (
            memory_increase < max_increase_mb
        ), f"Potential memory leak: {memory_increase / 1024 / 1024:.1f}MB increase"

    def test_cpu_usage_efficiency(self):
        """Test CPU usage efficiency during processing."""
        frames = [
            np.random.randint(0, 255, (120, 120, 3), dtype=np.uint8) for _ in range(5)
        ]
        config = MetricsConfig()

        # Monitor CPU usage during processing
        process = psutil.Process()

        def cpu_monitor():
            cpu_percentages = []
            for _ in range(20):  # Monitor for 2 seconds
                cpu_percentages.append(process.cpu_percent())
                time.sleep(0.1)
            return cpu_percentages

        # Start CPU monitoring
        cpu_thread = threading.Thread(target=cpu_monitor)
        cpu_thread.start()

        # Run processing
        start_time = time.time()
        calculate_comprehensive_metrics_from_frames(frames, frames, config)
        processing_time = time.time() - start_time

        cpu_thread.join()

        # CPU usage should be reasonable (not constantly at 100%)
        # This test is more informational than strict validation
        print(f"Processing completed in {processing_time:.2f}s")

    def test_performance_consistency(self):
        """Test that performance is consistent across multiple runs."""
        # Clean up any existing global instances before testing
        from giflab.metrics import cleanup_all_validators

        cleanup_all_validators()

        frames = [
            np.random.randint(0, 255, (90, 90, 3), dtype=np.uint8) for _ in range(3)
        ]
        config = MetricsConfig()

        # Warm-up run to initialize any lazy-loaded models (e.g., LPIPS)
        _, _ = measure_execution_time(
            calculate_comprehensive_metrics_from_frames, frames, frames, config
        )

        # Run multiple iterations
        times = []
        for _i in range(20):
            _, exec_time = measure_execution_time(
                calculate_comprehensive_metrics_from_frames, frames, frames, config
            )
            times.append(exec_time)

        # Calculate statistics
        mean_time = statistics.mean(times)
        std_dev = statistics.stdev(times)
        coefficient_of_variation = std_dev / mean_time if mean_time > 0 else 0

        # Performance should be consistent (low coefficient of variation)
        assert (
            coefficient_of_variation < 0.3
        ), f"Inconsistent performance: CV={coefficient_of_variation:.3f}"

        # No outliers should be more than 4 standard deviations from mean
        # (using 4 instead of 3 to account for system load variability)
        for time_val in times:
            z_score = abs(time_val - mean_time) / std_dev if std_dev > 0 else 0
            assert (
                z_score < 4
            ), f"Performance outlier detected: {time_val:.4f}s (z-score: {z_score:.2f})"


@pytest.mark.slow
class TestStressTestingPhase3:
    """Stress tests for Phase 3 components."""

    @pytest.mark.skipif(
        os.getenv("GIFLAB_STRESS_TESTS") != "1",
        reason="Stress tests require GIFLAB_STRESS_TESTS=1",
    )
    def test_large_image_processing(self):
        """Stress test with very large images."""
        # Test with large image sizes
        large_frames = [
            np.random.randint(0, 255, (2000, 2000, 3), dtype=np.uint8) for _ in range(3)
        ]

        config = MetricsConfig()

        # Should handle large images without crashing
        _, execution_time = measure_execution_time(
            calculate_comprehensive_metrics_from_frames,
            large_frames,
            large_frames,
            config,
        )

        # May be slow but should complete
        assert (
            execution_time < 60
        ), f"Large image processing too slow: {execution_time:.1f}s"

    @pytest.mark.skipif(
        os.getenv("GIFLAB_STRESS_TESTS") != "1",
        reason="Stress tests require GIFLAB_STRESS_TESTS=1",
    )
    def test_many_frame_processing(self):
        """Stress test with many frames."""
        # Test with many frames (but smaller size)
        many_frames = [
            np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8) for _ in range(100)
        ]

        config = MetricsConfig()

        # Should handle many frames efficiently due to sampling
        _, execution_time = measure_execution_time(
            calculate_comprehensive_metrics_from_frames,
            many_frames,
            many_frames,
            config,
        )

        # Should be efficient due to frame sampling
        assert (
            execution_time < 30
        ), f"Many frame processing too slow: {execution_time:.1f}s"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "not slow"])
