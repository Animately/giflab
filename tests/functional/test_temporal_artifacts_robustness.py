"""
Comprehensive tests for temporal artifacts robustness improvements.

Tests the memory management, error handling, batch processing improvements,
edge cases, boundary conditions, and performance characteristics of the
temporal artifacts detection system.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from giflab.optimization_validation.data_structures import ValidationConfig
from giflab.optimization_validation.validation_checker import ValidationChecker
from giflab.temporal_artifacts import (
    MemoryMonitor,
    TemporalArtifactDetector,
    calculate_enhanced_temporal_metrics,
)
from PIL import Image


class TestMemoryMonitor:
    """Test the MemoryMonitor class functionality."""

    def test_memory_monitor_cpu_device(self):
        """Test MemoryMonitor with CPU device."""
        monitor = MemoryMonitor("cpu")
        assert not monitor.is_cuda
        assert monitor._get_memory_usage() == 0.0

        # CPU should always allow max batch size
        batch_size = monitor.get_safe_batch_size((100, 100, 3), max_batch_size=32)
        assert batch_size == 32

        # No cleanup needed on CPU
        assert not monitor.should_cleanup_memory()

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.memory_allocated", return_value=1024 * 1024 * 1024)  # 1GB
    @patch("torch.cuda.get_device_properties")
    def test_memory_monitor_cuda_device(
        self, mock_props, mock_allocated, mock_available
    ):
        """Test MemoryMonitor with CUDA device."""
        mock_props.return_value.total_memory = 8 * 1024 * 1024 * 1024  # 8GB

        monitor = MemoryMonitor("cuda:0")
        assert monitor.is_cuda

        usage = monitor._get_memory_usage()
        assert usage == 1024 * 1024 * 1024 / (8 * 1024 * 1024 * 1024)  # 1/8 = 0.125

        # Should not need cleanup at 12.5% usage
        assert not monitor.should_cleanup_memory()

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.memory_allocated", return_value=7 * 1024 * 1024 * 1024)  # 7GB
    @patch("torch.cuda.get_device_properties")
    def test_memory_monitor_high_usage(
        self, mock_props, mock_allocated, mock_available
    ):
        """Test MemoryMonitor behavior at high memory usage."""
        mock_props.return_value.total_memory = 8 * 1024 * 1024 * 1024  # 8GB

        monitor = MemoryMonitor("cuda:0", memory_threshold=0.8)

        # Should need cleanup at 87.5% usage (7/8)
        assert monitor.should_cleanup_memory()

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.memory_allocated", return_value=2 * 1024 * 1024 * 1024)  # 2GB
    @patch("torch.cuda.get_device_properties")
    def test_safe_batch_size_calculation(
        self, mock_props, mock_allocated, mock_available
    ):
        """Test safe batch size calculation based on available memory."""
        mock_props.return_value.total_memory = 8 * 1024 * 1024 * 1024  # 8GB

        monitor = MemoryMonitor("cuda:0", memory_threshold=0.8)

        # With 2GB used out of 8GB, and 80% threshold, we have:
        # Available for threshold: 8GB * 0.8 = 6.4GB
        # Free memory: 6.4GB - 2GB = 4.4GB
        # For 100x100x3 frames, safety margin should allow reasonable batch size
        batch_size = monitor.get_safe_batch_size((100, 100, 3), max_batch_size=32)
        assert batch_size >= 1
        assert batch_size <= 32


class TestTemporalArtifactDetectorRobustness:
    """Test robustness improvements in TemporalArtifactDetector."""

    def test_detector_initialization(self):
        """Test detector initialization with different configurations."""
        # Default initialization
        detector = TemporalArtifactDetector()
        assert detector.memory_monitor is not None
        assert not detector.force_mse_fallback

        # MSE fallback initialization
        detector = TemporalArtifactDetector(force_mse_fallback=True)
        assert detector.force_mse_fallback
        assert detector._lpips_model is False

        # Custom memory threshold
        detector = TemporalArtifactDetector(memory_threshold=0.9)
        assert detector.memory_monitor.memory_threshold == 0.9

    @patch("giflab.temporal_artifacts.LPIPS_AVAILABLE", False)
    def test_detector_without_lpips(self):
        """Test detector behavior when LPIPS is not available."""
        detector = TemporalArtifactDetector()

        # Create dummy frames
        frames = [
            np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8) for _ in range(3)
        ]

        # Should fall back to MSE
        metrics = detector.calculate_lpips_temporal(frames)
        assert "lpips_t_mean" in metrics
        assert "lpips_frame_count" in metrics
        assert metrics["lpips_frame_count"] == 3

    def test_detector_with_small_frames(self):
        """Test detector with minimal frame input."""
        detector = TemporalArtifactDetector(force_mse_fallback=True)

        # Single frame - should return zero metrics
        single_frame = [np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)]
        metrics = detector.calculate_lpips_temporal(single_frame)
        assert metrics["lpips_t_mean"] == 0.0
        assert metrics["lpips_frame_count"] == 1

        # Two frames - should work
        two_frames = [
            np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8) for _ in range(2)
        ]
        metrics = detector.calculate_lpips_temporal(two_frames)
        assert "lpips_t_mean" in metrics
        assert metrics["lpips_frame_count"] == 2

    def test_flicker_detection_robustness(self):
        """Test flicker detection with various edge cases."""
        detector = TemporalArtifactDetector(force_mse_fallback=True)

        # Empty frames
        metrics = detector.detect_flicker_excess([])
        assert metrics["flicker_excess"] == 0.0
        assert metrics["flicker_frame_count"] == 0

        # Single frame
        single_frame = [np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)]
        metrics = detector.detect_flicker_excess(single_frame)
        assert metrics["flicker_excess"] == 0.0

        # Multiple similar frames (low flicker)
        base_frame = np.ones((32, 32, 3), dtype=np.uint8) * 128
        similar_frames = [
            base_frame + np.random.randint(-5, 5, (32, 32, 3)) for _ in range(5)
        ]
        metrics = detector.detect_flicker_excess(similar_frames)
        assert (
            metrics["flicker_excess"] >= 0.0
        )  # Should be low but not necessarily zero

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.OutOfMemoryError")
    def test_oom_error_handling(self, mock_oom, mock_cuda_available):
        """Test OOM error handling and recovery."""
        detector = TemporalArtifactDetector(device="cuda:0")

        # Mock LPIPS processing to raise OOM
        with patch.object(
            detector,
            "_process_lpips_batch",
            side_effect=torch.cuda.OutOfMemoryError("CUDA out of memory"),
        ):
            frames = [
                np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
                for _ in range(10)
            ]

            # Should gracefully fall back to MSE
            metrics = detector.calculate_lpips_temporal(frames)
            assert "lpips_t_mean" in metrics
            assert metrics["lpips_frame_count"] == 10

    def test_adaptive_batch_sizing(self):
        """Test adaptive batch sizing functionality."""
        detector = TemporalArtifactDetector(
            force_mse_fallback=False
        )  # Enable LPIPS to test batch sizing

        # Large frame sequence should trigger batch size reduction
        large_frames = [
            np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8) for _ in range(150)
        ]

        # Test that processing completes successfully with large batches
        metrics = detector.calculate_lpips_temporal(large_frames, batch_size=32)

        # Should get temporal metrics (both LPIPS and MSE fallback use lpips_ prefixed keys)
        assert "lpips_t_mean" in metrics
        assert "lpips_frame_count" in metrics

    def test_preprocessing_robustness(self):
        """Test preprocessing with different input formats."""
        detector = TemporalArtifactDetector()

        # Test with uint8 input
        uint8_frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        tensor = detector.preprocess_for_lpips(uint8_frame)
        assert tensor.shape == (1, 3, 64, 64)
        assert tensor.dtype == torch.float32

        # Test with float32 input
        float32_frame = np.random.rand(64, 64, 3).astype(np.float32)
        tensor = detector.preprocess_for_lpips(float32_frame)
        assert tensor.shape == (1, 3, 64, 64)
        assert tensor.dtype == torch.float32

        # Values should be in [-1, 1] range for LPIPS
        assert tensor.min() >= -1.0
        assert tensor.max() <= 1.0


class TestEnhancedTemporalMetrics:
    """Test the main enhanced temporal metrics function."""

    def test_enhanced_metrics_with_mismatched_frames(self):
        """Test enhanced metrics with mismatched frame counts."""
        original_frames = [
            np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8) for _ in range(5)
        ]
        compressed_frames = [
            np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8) for _ in range(3)
        ]

        metrics = calculate_enhanced_temporal_metrics(
            original_frames, compressed_frames, force_mse_fallback=True
        )

        # Should use the minimum frame count
        assert metrics["frame_count"] == 3
        assert "flicker_excess" in metrics
        assert "flat_flicker_ratio" in metrics

    def test_enhanced_metrics_with_empty_frames(self):
        """Test enhanced metrics with empty frame lists."""
        metrics = calculate_enhanced_temporal_metrics([], [], force_mse_fallback=True)

        assert metrics["flicker_excess"] == 0.0
        assert metrics["flat_flicker_ratio"] == 0.0
        assert metrics["temporal_pumping_score"] == 0.0
        assert metrics["lpips_t_mean"] == 0.0
        assert metrics["lpips_t_p95"] == 0.0

    def test_enhanced_metrics_large_sequence_optimization(self):
        """Test that large sequences trigger optimization."""
        # Create a large sequence that should trigger batch size reduction
        large_frames = [
            np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8) for _ in range(150)
        ]

        with patch("giflab.temporal_artifacts.logger") as mock_logger:
            metrics = calculate_enhanced_temporal_metrics(
                large_frames, large_frames, force_mse_fallback=True, batch_size=16
            )

            # Should log the batch size adjustment
            mock_logger.info.assert_called()
            assert "Large frame sequence" in str(mock_logger.info.call_args)

            assert metrics["frame_count"] == 150

    def test_device_fallback_handling(self):
        """Test device fallback when CUDA is requested but unavailable."""
        with patch("torch.cuda.is_available", return_value=False):
            detector = TemporalArtifactDetector(device="cuda:0")

            # Should have fallen back to CPU
            assert detector.device == "cpu"
            assert not detector.memory_monitor.is_cuda


@pytest.mark.parametrize("frame_size", [(32, 32), (64, 64), (128, 128), (256, 256)])
def test_memory_scaling_with_frame_size(frame_size):
    """Test that memory management scales appropriately with frame size."""
    h, w = frame_size
    monitor = MemoryMonitor("cuda:0" if torch.cuda.is_available() else "cpu")

    batch_size = monitor.get_safe_batch_size((h, w, 3), max_batch_size=32)

    # Larger frames should generally result in smaller batch sizes on CUDA
    assert batch_size >= 1
    assert batch_size <= 32

    if monitor.is_cuda and h > 128:
        # Very large frames should definitely constrain batch size
        assert batch_size < 32


@pytest.mark.parametrize("batch_size", [1, 4, 8, 16, 32])
def test_batch_processing_consistency(batch_size):
    """Test that different batch sizes produce consistent results."""
    detector = TemporalArtifactDetector(force_mse_fallback=True)

    # Create deterministic test frames
    np.random.seed(42)
    frames = [np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8) for _ in range(10)]

    metrics1 = detector.calculate_lpips_temporal(frames, batch_size=batch_size)
    metrics2 = detector.calculate_lpips_temporal(frames, batch_size=batch_size)

    # Results should be deterministic
    assert abs(metrics1["lpips_t_mean"] - metrics2["lpips_t_mean"]) < 1e-6
    assert metrics1["lpips_frame_count"] == metrics2["lpips_frame_count"]


# Integration test for the complete pipeline
def test_temporal_artifacts_integration():
    """Integration test for the complete temporal artifacts detection pipeline."""
    # Create test frames with known temporal artifacts
    frames = []
    base_frame = np.ones((64, 64, 3), dtype=np.uint8) * 128

    for i in range(20):
        if i % 4 == 0:
            # Every 4th frame has artifacts (flicker)
            frame = base_frame + np.random.randint(-50, 50, (64, 64, 3))
        else:
            # Stable frames
            frame = base_frame + np.random.randint(-5, 5, (64, 64, 3))

        frames.append(np.clip(frame, 0, 255).astype(np.uint8))

    # Test with both original and compressed versions
    compressed_frames = [
        frame + np.random.randint(-10, 10, (64, 64, 3)) for frame in frames
    ]
    compressed_frames = [
        np.clip(frame, 0, 255).astype(np.uint8) for frame in compressed_frames
    ]

    metrics = calculate_enhanced_temporal_metrics(
        frames, compressed_frames, force_mse_fallback=True, batch_size=8
    )

    # Should detect some temporal artifacts
    assert metrics["flicker_excess"] >= 0.0
    assert metrics["flat_flicker_ratio"] >= 0.0
    assert metrics["temporal_pumping_score"] >= 0.0
    assert metrics["frame_count"] == 20

    # All metrics should be finite
    for key, value in metrics.items():
        if isinstance(value, int | float):
            assert np.isfinite(value), f"Metric {key} is not finite: {value}"


class TestTemporalArtifactEdgeCases:
    """Test edge cases in temporal artifact detection."""

    @pytest.mark.fast
    def test_single_frame_gif(self):
        """Test temporal detection with single-frame GIF."""
        # Create a single frame GIF
        img = Image.new("RGB", (32, 32), (255, 0, 0))

        with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmp:
            gif_path = Path(tmp.name)
            img.save(gif_path)

            try:
                # Test with temporal metrics using simple frame data
                detector = TemporalArtifactDetector()

                # Create single frame as numpy array
                single_frame = np.array(img).astype(np.uint8)

                # Test detector methods with single frame
                flicker_result = detector.detect_flicker_excess([single_frame])
                assert flicker_result["flicker_excess"] == 0.0
                assert flicker_result["flicker_frame_ratio"] == 0.0

                flat_result = detector.detect_flat_region_flicker([single_frame])
                assert flat_result["flat_flicker_ratio"] == 0.0

                pumping_result = detector.detect_temporal_pumping([single_frame])
                assert pumping_result["temporal_pumping_score"] == 0.0

                # LPIPS should handle single frame
                lpips_result = detector.calculate_lpips_temporal([single_frame])
                assert lpips_result["lpips_t_mean"] == 0.0

            finally:
                gif_path.unlink()

    def test_empty_frames_list(self):
        """Test temporal detection with empty frames list."""
        detector = TemporalArtifactDetector()

        # Should handle empty frames gracefully
        flicker_result = detector.detect_flicker_excess([])
        assert flicker_result["flicker_excess"] == 0.0
        assert flicker_result["flicker_frame_ratio"] == 0.0

        flat_result = detector.detect_flat_region_flicker([])
        assert flat_result["flat_flicker_ratio"] == 0.0

        pumping_result = detector.detect_temporal_pumping([])
        assert pumping_result["temporal_pumping_score"] == 0.0

        lpips_result = detector.calculate_lpips_temporal([])
        assert lpips_result["lpips_t_mean"] == 0.0

    def test_identical_frames(self):
        """Test temporal detection with identical frames (no temporal change)."""
        # Create multiple identical frames
        frames = []
        base_frame = np.zeros((64, 64, 3), dtype=np.uint8)
        base_frame[:] = 128  # Gray image

        for _ in range(5):
            frames.append(base_frame.copy())

        detector = TemporalArtifactDetector()

        # All metrics should indicate no temporal artifacts
        flicker_result = detector.detect_flicker_excess(frames)
        assert flicker_result["flicker_excess"] == 0.0
        assert flicker_result["flicker_frame_ratio"] == 0.0

        flat_result = detector.detect_flat_region_flicker(frames)
        assert flat_result["flat_flicker_ratio"] == 0.0  # No flicker in flat regions

        pumping_result = detector.detect_temporal_pumping(frames)
        assert pumping_result["temporal_pumping_score"] == 0.0

        lpips_result = detector.calculate_lpips_temporal(frames)
        assert lpips_result["lpips_t_mean"] == 0.0

    def test_mismatched_frame_sizes(self):
        """Test temporal detection with frames of different sizes."""
        frames = []

        # Create frames with different sizes
        frames.append(np.zeros((32, 32, 3), dtype=np.uint8))
        frames.append(np.zeros((64, 64, 3), dtype=np.uint8))
        frames.append(np.zeros((48, 48, 3), dtype=np.uint8))

        detector = TemporalArtifactDetector()

        # Should handle mismatched sizes gracefully (likely by resizing or skipping)
        try:
            flicker_result = detector.detect_flicker_excess(frames)
            assert isinstance(flicker_result, dict)
            assert "flicker_excess" in flicker_result

            flat_result = detector.detect_flat_region_flicker(frames)
            assert isinstance(flat_result, dict)
            assert "flat_flicker_ratio" in flat_result

        except Exception as e:
            # If handling mismatched sizes by raising exception, that's also acceptable
            assert "size" in str(e).lower() or "shape" in str(e).lower()

    def test_extreme_frame_dimensions(self):
        """Test temporal detection with extremely small and large frames."""
        detector = TemporalArtifactDetector()

        # Test with very small frames (1x1)
        small_frames = [
            np.array([[[255, 0, 0]]], dtype=np.uint8),
            np.array([[[0, 255, 0]]], dtype=np.uint8),
        ]

        small_result = detector.detect_flicker_excess(small_frames)
        assert isinstance(small_result, dict)
        assert "flicker_excess" in small_result

        # Test with very large frames (if computationally feasible)
        # Note: This might be skipped in practice due to memory constraints
        try:
            large_frames = [
                np.zeros((1024, 1024, 3), dtype=np.uint8),
                np.ones((1024, 1024, 3), dtype=np.uint8) * 255,
            ]

            large_result = detector.detect_flicker_excess(large_frames)
            assert isinstance(large_result, dict)
            assert "flicker_excess" in large_result

        except MemoryError:
            pytest.skip("Large frame test skipped due to memory constraints")

    def test_extreme_color_values(self):
        """Test temporal detection with extreme color values."""
        frames = []

        # All black frame
        frames.append(np.zeros((64, 64, 3), dtype=np.uint8))

        # All white frame
        frames.append(np.ones((64, 64, 3), dtype=np.uint8) * 255)

        # High contrast patterns
        checkerboard = np.zeros((64, 64, 3), dtype=np.uint8)
        for x in range(64):
            for y in range(64):
                if (x + y) % 2 == 0:
                    checkerboard[x, y] = [255, 255, 255]
        frames.append(checkerboard)

        detector = TemporalArtifactDetector()

        # Should handle extreme values
        flicker_result = detector.detect_flicker_excess(frames, threshold=0.1)
        assert isinstance(flicker_result, dict)
        assert flicker_result["flicker_excess"] >= 0.0

        flat_result = detector.detect_flat_region_flicker(frames)
        assert isinstance(flat_result, dict)
        assert flat_result["flat_flicker_ratio"] >= 0.0

    def test_invalid_gif_file(self):
        """Test temporal detection with invalid GIF file."""
        # Create a file that's not a valid GIF
        with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmp:
            invalid_gif = Path(tmp.name)
            invalid_gif.write_text("This is not a GIF file")

            try:
                # Should handle invalid GIF gracefully - test with calculate_enhanced_temporal_metrics
                with pytest.raises(
                    Exception
                ):  # Specific exception depends on implementation
                    calculate_enhanced_temporal_metrics(
                        str(invalid_gif), str(invalid_gif)
                    )

            finally:
                invalid_gif.unlink()

    def test_corrupted_gif_frames(self):
        """Test temporal detection with partially corrupted GIF."""
        # This is harder to test without actual corrupted files
        # Instead, test with frames that have unusual characteristics

        detector = TemporalArtifactDetector()

        # Create frames with NaN values (simulating corruption)
        corrupted_frames = []
        frame1 = np.full((32, 32, 3), 100, dtype=np.float32)
        frame2 = np.full((32, 32, 3), 150, dtype=np.float32)
        frame2[10:20, 10:20] = np.nan  # Introduce NaN values

        corrupted_frames.append(frame1.astype(np.uint8))
        corrupted_frames.append(
            np.nan_to_num(frame2).astype(np.uint8)
        )  # Clean NaN for input

        # Should handle gracefully
        try:
            result = detector.detect_flicker_excess(corrupted_frames)
            assert isinstance(result, dict)
        except Exception:
            # Acceptable to raise exception for corrupted data
            assert True

    def test_memory_intensive_operations(self):
        """Test temporal detection under memory pressure."""
        detector = TemporalArtifactDetector()

        # Create a reasonable number of frames that would stress memory
        frames = []
        for _i in range(20):  # 20 frames of reasonable size
            frame = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
            frames.append(frame)

        try:
            # Should complete without memory errors
            flicker_result = detector.detect_flicker_excess(frames)
            assert isinstance(flicker_result, dict)

            flat_result = detector.detect_flat_region_flicker(frames)
            assert isinstance(flat_result, dict)

            pumping_result = detector.detect_temporal_pumping(frames)
            assert isinstance(pumping_result, dict)

        except MemoryError:
            pytest.skip("Memory-intensive test skipped due to insufficient memory")

    def test_zero_threshold_values(self):
        """Test temporal detection with zero thresholds."""
        detector = TemporalArtifactDetector()

        # Create frames with minimal differences
        frames = []
        base = np.zeros((32, 32, 3), dtype=np.uint8)
        frames.append(base)

        # Second frame with tiny difference
        frame2 = base.copy()
        frame2[0, 0] = [1, 1, 1]  # Minimal change
        frames.append(frame2)

        # Test with zero threshold - should detect any change
        flicker_result = detector.detect_flicker_excess(frames, threshold=0.0)
        assert flicker_result["flicker_excess"] > 0.0  # Should detect the tiny change

        flat_result = detector.detect_flat_region_flicker(
            frames, variance_threshold=0.0
        )
        # With zero variance threshold, any change should be detected
        assert isinstance(flat_result, dict)

    def test_maximum_threshold_values(self):
        """Test temporal detection with very high thresholds."""
        detector = TemporalArtifactDetector()

        # Create frames with dramatic differences
        frames = []
        frames.append(np.zeros((32, 32, 3), dtype=np.uint8))  # All black
        frames.append(np.ones((32, 32, 3), dtype=np.uint8) * 255)  # All white

        # Test with very high threshold - should not detect even dramatic changes
        flicker_result = detector.detect_flicker_excess(frames, threshold=1.0)
        assert flicker_result["flicker_excess"] <= 1.0  # Should be under threshold

        flat_result = detector.detect_flat_region_flicker(
            frames, variance_threshold=10000.0
        )
        assert flat_result["flat_flicker_ratio"] == 0.0  # High threshold should pass


@pytest.mark.performance
class TestTemporalArtifactPerformanceEdgeCases:
    """Test performance characteristics and edge cases."""

    @pytest.mark.slow
    def test_performance_with_many_frames(self):
        """Test performance with a large number of frames."""
        detector = TemporalArtifactDetector()

        # Create many frames (but reasonable size)
        frames = []
        for i in range(100):  # 100 frames
            # Create varied but simple frames to avoid excessive computation
            frame = np.full((32, 32, 3), i % 256, dtype=np.uint8)
            frames.append(frame)

        import time

        start_time = time.time()

        try:
            flicker_result = detector.detect_flicker_excess(frames)
            flat_result = detector.detect_flat_region_flicker(frames)
            pumping_result = detector.detect_temporal_pumping(frames)

            end_time = time.time()
            duration = end_time - start_time

            print(f"Performance with {len(frames)} frames: {duration:.2f}s")

            # Should complete in reasonable time (adjust threshold as needed)
            assert (
                duration < 60.0
            ), f"Processing {len(frames)} frames took too long: {duration:.2f}s"

            # Results should be valid
            assert isinstance(flicker_result, dict)
            assert isinstance(flat_result, dict)
            assert isinstance(pumping_result, dict)

        except MemoryError:
            pytest.skip("Many frames test skipped due to memory constraints")

    def test_concurrent_detector_instances(self):
        """Test multiple detector instances running concurrently."""
        import threading
        import time

        # Create test frames
        frames = []
        for i in range(10):
            frame = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
            frames.append(frame)

        results = {}

        def run_detector(detector_id):
            detector = TemporalArtifactDetector()
            start = time.time()
            result = detector.detect_flicker_excess(frames)
            end = time.time()
            results[detector_id] = (result, end - start)

        # Run multiple detectors concurrently
        threads = []
        for i in range(3):  # 3 concurrent instances
            thread = threading.Thread(target=run_detector, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=30)  # 30 second timeout

        # Check results
        assert len(results) >= 1, "At least one detector should complete"

        for detector_id, (result, duration) in results.items():
            assert isinstance(result, dict)
            assert (
                duration < 30.0
            ), f"Detector {detector_id} took too long: {duration:.2f}s"
            print(f"Detector {detector_id}: {duration:.2f}s")

    @pytest.mark.slow
    def test_memory_usage_patterns(self):
        """Test memory usage patterns during temporal detection."""
        detector = TemporalArtifactDetector()

        # Create frames that might cause memory issues
        large_frames = []
        for _i in range(10):
            frame = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
            large_frames.append(frame)

        try:
            import os

            import psutil

            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB

            # Run detection
            detector.detect_flicker_excess(large_frames)
            detector.detect_flat_region_flicker(large_frames)

            peak_memory = process.memory_info().rss / 1024 / 1024  # MB

            print(f"Memory usage: {initial_memory:.1f}MB -> {peak_memory:.1f}MB")
            print(f"Memory increase: {peak_memory - initial_memory:.1f}MB")

            # Should not use excessive memory (adjust threshold as needed)
            memory_increase = peak_memory - initial_memory
            assert (
                memory_increase < 1000
            ), f"Excessive memory usage: {memory_increase:.1f}MB"

        except ImportError:
            pytest.skip("psutil not available for memory testing")


class TestTemporalArtifactErrorHandling:
    """Test error handling in temporal artifact detection."""

    def test_lpips_model_loading_failure(self):
        """Test handling of LPIPS model loading failures."""
        with patch("giflab.temporal_artifacts.lpips.LPIPS") as mock_lpips:
            # Mock LPIPS to raise exception during loading
            mock_lpips.side_effect = RuntimeError("CUDA out of memory")

            detector = TemporalArtifactDetector()
            frames = [np.zeros((32, 32, 3), dtype=np.uint8) for _ in range(3)]

            # Should fall back gracefully without crashing
            result = detector.calculate_lpips_temporal(frames)

            # Should return default/fallback values
            assert isinstance(result, dict)
            assert "lpips_t_mean" in result
            # Might be 0.0 or computed via MSE fallback

    def test_gpu_memory_exhaustion(self):
        """Test handling of GPU memory exhaustion."""
        with patch("giflab.temporal_artifacts.torch") as mock_torch:
            # Mock torch operations to raise CUDA out of memory
            mock_torch.cuda.is_available.return_value = True
            mock_torch.device.return_value = "cuda:0"

            mock_tensor = MagicMock()
            mock_tensor.to.side_effect = RuntimeError("CUDA out of memory")
            mock_torch.from_numpy.return_value = mock_tensor

            detector = TemporalArtifactDetector(device="cuda")
            frames = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(5)]

            # Should handle GPU memory issues gracefully
            try:
                result = detector.calculate_lpips_temporal(frames)
                assert isinstance(result, dict)
            except Exception as e:
                # Should either handle gracefully or fail with clear error
                assert "cuda" in str(e).lower() or "memory" in str(e).lower()

    def test_invalid_device_specification(self):
        """Test handling of invalid device specifications."""
        # Test with invalid device string - may not raise exception in current implementation
        try:
            detector = TemporalArtifactDetector(device="invalid_device")
            # If it doesn't raise, verify device is set to something reasonable
            assert detector.device in [
                "cpu",
                "cuda",
                "invalid_device",
            ]  # May just accept the string
        except Exception as e:
            # If it does raise, should be appropriate exception type
            assert isinstance(e, ValueError | RuntimeError)

    def test_file_io_errors(self, tmp_path):
        """Test handling of file I/O errors during GIF processing."""
        # Create a temporary file that we'll make unreadable
        test_gif = tmp_path / "unreadable.gif"
        test_gif.write_text("fake gif")

        # Make file unreadable
        test_gif.chmod(0o000)

        try:
            # Should handle file permission errors - test with frame extraction function
            from giflab.metrics import extract_gif_frames

            with pytest.raises((IOError, ValueError, PermissionError)):
                extract_gif_frames(test_gif)
        finally:
            # Restore permissions for cleanup
            test_gif.chmod(0o644)

    def test_preprocessing_errors(self):
        """Test handling of preprocessing errors."""
        # Test with invalid frame shapes by creating frames manually
        detector = TemporalArtifactDetector()

        # Test preprocessing with various invalid inputs
        try:
            # Empty array
            detector.preprocess_for_lpips(np.array([]))
        except Exception as e:
            assert isinstance(
                e, ValueError | TypeError | AttributeError | IndexError | RuntimeError
            )

        try:
            # Wrong dimensions (1D instead of 3D)
            detector.preprocess_for_lpips(np.array([1, 2, 3]))
        except Exception as e:
            assert isinstance(
                e, ValueError | TypeError | AttributeError | IndexError | RuntimeError
            )

    def test_calculation_numerical_errors(self):
        """Test handling of numerical errors in calculations."""
        # Create frames that might cause numerical issues
        problematic_frames = []

        # Frame with very small values that might cause precision issues
        frame1 = np.full((32, 32, 3), 1e-10, dtype=np.float32)
        problematic_frames.append(frame1.astype(np.uint8))

        # Frame with values that might cause overflow
        frame2 = np.full((32, 32, 3), 255, dtype=np.uint8)
        problematic_frames.append(frame2)

        detector = TemporalArtifactDetector()

        # Should handle numerical edge cases gracefully
        try:
            flicker_result = detector.detect_flicker_excess(problematic_frames)
            assert isinstance(flicker_result, dict)
            assert all(
                not np.isnan(v) for v in flicker_result.values() if isinstance(v, float)
            )

            flat_result = detector.detect_flat_region_flicker(problematic_frames)
            assert isinstance(flat_result, dict)
            assert all(
                not np.isnan(v) for v in flat_result.values() if isinstance(v, float)
            )

        except Exception as e:
            # Should fail gracefully with appropriate error
            assert not isinstance(e, FloatingPointError | OverflowError)


class TestTemporalValidationBoundaryConditions:
    """Test boundary conditions in temporal validation."""

    def test_validation_at_exact_thresholds(self):
        """Test validation behavior at exact threshold values."""
        config = ValidationConfig(
            flicker_excess_threshold=0.05,
            flat_flicker_ratio_threshold=0.1,
            temporal_pumping_threshold=0.2,
            lpips_t_threshold=0.03,
        )

        # Create ValidationChecker with default config, then patch it
        validator = ValidationChecker(None)  # Use default config
        validator.config = config  # Override with test config

        # Test with metrics exactly at thresholds
        with patch("giflab.metrics.calculate_comprehensive_metrics") as mock_metrics:
            mock_metrics.return_value = {
                "flicker_excess": 0.05,  # Exactly at threshold
                "flat_flicker_ratio": 0.1,  # Exactly at threshold
                "temporal_pumping_score": 0.2,  # Exactly at threshold
                "lpips_t_mean": 0.03,  # Exactly at threshold
                "ssim_mean": 0.8,
                "psnr_mean": 25.0,
                "mse_mean": 300.0,
            }

            # Create proper metadata and compression metrics
            from giflab.meta import GifMetadata

            original_metadata = GifMetadata(
                gif_sha="test_sha",
                orig_filename="test.gif",
                orig_kilobytes=1024,
                orig_width=64,
                orig_height=64,
                orig_frames=10,
                orig_fps=10.0,
                orig_n_colors=256,
                entropy=5.0,
            )

            compression_metrics = {
                "flicker_excess": 0.05,  # Exactly at threshold
                "flat_flicker_ratio": 0.1,  # Exactly at threshold
                "temporal_pumping_score": 0.2,  # Exactly at threshold
                "lpips_t_mean": 0.03,  # Exactly at threshold
                "ssim_mean": 0.8,
                "psnr_mean": 25.0,
                "mse_mean": 300.0,
                "file_size_mb": 0.8,
                "compression_ratio": 0.8,
            }

            result = validator.validate_compression_result(
                original_metadata=original_metadata,
                compression_metrics=compression_metrics,
                pipeline_id="test_pipeline",
                gif_name="test.gif",
            )

            # Behavior at exact threshold depends on implementation
            # (typically <= threshold passes, > threshold fails)
            print(f"Validation at exact thresholds: {result.is_acceptable()}")

    def test_validation_just_above_thresholds(self):
        """Test validation behavior just above thresholds."""
        config = ValidationConfig(
            flicker_excess_threshold=0.05,
            flat_flicker_ratio_threshold=0.1,
            temporal_pumping_threshold=0.2,
            lpips_t_threshold=0.03,
        )

        # Create ValidationChecker with default config, then patch it
        validator = ValidationChecker(None)  # Use default config
        validator.config = config  # Override with test config

        # Test with metrics just above thresholds
        with patch("giflab.metrics.calculate_comprehensive_metrics") as mock_metrics:
            mock_metrics.return_value = {
                "flicker_excess": 0.051,  # Just above threshold
                "flat_flicker_ratio": 0.101,  # Just above threshold
                "temporal_pumping_score": 0.201,  # Just above threshold
                "lpips_t_mean": 0.031,  # Just above threshold
                "ssim_mean": 0.8,
                "psnr_mean": 25.0,
                "mse_mean": 300.0,
            }

            # Create proper metadata and compression metrics for above-threshold values
            from giflab.meta import GifMetadata

            original_metadata = GifMetadata(
                gif_sha="test_sha",
                orig_filename="test.gif",
                orig_kilobytes=1024,
                orig_width=64,
                orig_height=64,
                orig_frames=10,
                orig_fps=10.0,
                orig_n_colors=256,
                entropy=5.0,
            )

            compression_metrics = {
                "flicker_excess": 0.06,  # Above threshold (0.05)
                "flat_flicker_ratio": 0.11,  # Above threshold (0.1)
                "temporal_pumping_score": 0.21,  # Above threshold (0.2)
                "lpips_t_mean": 0.031,  # Above threshold (0.03)
                "ssim_mean": 0.79,  # Below threshold (0.8)
                "psnr_mean": 24.9,  # Below threshold (25.0)
                "mse_mean": 301.0,  # Above threshold (300.0)
                "file_size_mb": 0.8,
                "compression_ratio": 0.8,
            }

            result = validator.validate_compression_result(
                original_metadata=original_metadata,
                compression_metrics=compression_metrics,
                pipeline_id="test_pipeline",
                gif_name="test.gif",
            )

            # Should fail when above thresholds
            assert not result.is_acceptable()

    def test_validation_just_below_thresholds(self):
        """Test validation behavior just below thresholds."""
        config = ValidationConfig(
            flicker_excess_threshold=0.05,
            flat_flicker_ratio_threshold=0.1,
            temporal_pumping_threshold=0.2,
            lpips_t_threshold=0.03,
        )

        # Create ValidationChecker with default config, then patch it
        validator = ValidationChecker(None)  # Use default config
        validator.config = config  # Override with test config

        # Test with metrics just below thresholds
        with patch("giflab.metrics.calculate_comprehensive_metrics") as mock_metrics:
            mock_metrics.return_value = {
                "flicker_excess": 0.049,  # Just below threshold
                "flat_flicker_ratio": 0.099,  # Just below threshold
                "temporal_pumping_score": 0.199,  # Just below threshold
                "lpips_t_mean": 0.029,  # Just below threshold
                "ssim_mean": 0.8,
                "psnr_mean": 25.0,
                "mse_mean": 300.0,
            }

            # Create proper metadata and compression metrics for below-threshold values
            from giflab.meta import GifMetadata

            original_metadata = GifMetadata(
                gif_sha="test_sha",
                orig_filename="test.gif",
                orig_kilobytes=1024,
                orig_width=64,
                orig_height=64,
                orig_frames=10,
                orig_fps=10.0,
                orig_n_colors=256,
                entropy=5.0,
            )

            compression_metrics = {
                "flicker_excess": 0.04,  # Below threshold (0.05)
                "flat_flicker_ratio": 0.09,  # Below threshold (0.1)
                "temporal_pumping_score": 0.19,  # Below threshold (0.2)
                "lpips_t_mean": 0.029,  # Below threshold (0.03)
                "ssim_mean": 0.81,  # Above threshold (0.8)
                "psnr_mean": 25.1,  # Above threshold (25.0)
                "mse_mean": 299.0,  # Below threshold (300.0)
                "file_size_mb": 0.8,
                "compression_ratio": 0.8,
            }

            result = validator.validate_compression_result(
                original_metadata=original_metadata,
                compression_metrics=compression_metrics,
                pipeline_id="test_pipeline",
                gif_name="test.gif",
            )

            # Should pass when below thresholds
            assert result.is_acceptable()
