"""
Tests for Deep Perceptual Metrics functionality (Task 2.2)

This module tests the LPIPS-based spatial perceptual quality metrics that catch
perceptual issues traditional metrics miss.
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest
from giflab.deep_perceptual_metrics import (
    DeepPerceptualMetrics,
    DeepPerceptualValidator,
    PerceptualValidationResult,
    calculate_deep_perceptual_quality_metrics,
    should_use_deep_perceptual,
)


class TestShouldUseDeepPerceptual:
    """Test conditional triggering logic for deep perceptual metrics."""

    def test_should_use_for_borderline_quality(self):
        """Test that deep perceptual is used for borderline quality cases."""
        assert should_use_deep_perceptual(0.4)  # Borderline
        assert should_use_deep_perceptual(0.5)  # Borderline
        assert should_use_deep_perceptual(0.6)  # Borderline

    def test_should_use_for_poor_quality(self):
        """Test that deep perceptual is used for poor quality cases."""
        assert should_use_deep_perceptual(0.2)  # Poor quality
        assert should_use_deep_perceptual(0.1)  # Poor quality

    def test_should_not_use_for_high_quality(self):
        """Test that deep perceptual is skipped for high quality cases."""
        assert not should_use_deep_perceptual(0.8)  # Good quality
        assert not should_use_deep_perceptual(0.9)  # Excellent quality

    def test_should_use_when_quality_unknown(self):
        """Test that deep perceptual is used when composite quality is None."""
        assert should_use_deep_perceptual(None)


class TestDeepPerceptualValidator:
    """Test DeepPerceptualValidator class functionality."""

    @pytest.fixture
    def validator(self):
        """Create a DeepPerceptualValidator for testing."""
        return DeepPerceptualValidator(device="cpu", force_fallback=False)

    @pytest.fixture
    def fallback_validator(self):
        """Create a fallback DeepPerceptualValidator (no LPIPS)."""
        return DeepPerceptualValidator(device="cpu", force_fallback=True)

    @pytest.fixture
    def test_frames(self):
        """Create test frames for validation."""
        # Create simple test frames (RGB)
        original_frames = [
            np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8) for _ in range(5)
        ]
        # Create slightly different compressed frames
        compressed_frames = []
        for frame in original_frames:
            # Add small noise to simulate compression
            noise = np.random.randint(-10, 10, frame.shape, dtype=np.int16)
            compressed = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(
                np.uint8
            )
            compressed_frames.append(compressed)

        return original_frames, compressed_frames

    def test_device_determination(self):
        """Test device determination logic."""
        # Test auto device selection
        validator_auto = DeepPerceptualValidator(device="auto")
        assert validator_auto.device in ["cuda", "cpu"]

        # Test CPU fallback
        validator_cpu = DeepPerceptualValidator(device="cpu")
        assert validator_cpu.device == "cpu"

    def test_downscale_frame_if_needed(self, validator):
        """Test frame downscaling functionality."""
        # Small frame - no downscaling needed
        small_frame = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        result, was_downscaled = validator._downscale_frame_if_needed(small_frame)
        assert not was_downscaled
        assert result.shape == small_frame.shape

        # Large frame - should be downscaled
        large_frame = np.random.randint(0, 256, (1000, 1000, 3), dtype=np.uint8)
        result, was_downscaled = validator._downscale_frame_if_needed(large_frame)
        assert was_downscaled
        assert max(result.shape[:2]) <= validator.downscale_size

    def test_preprocess_for_lpips(self, validator):
        """Test LPIPS preprocessing functionality."""
        frame = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)

        # Mock torch to avoid requiring actual PyTorch installation in tests
        with patch("giflab.deep_perceptual_metrics.TORCH_AVAILABLE", True), patch(
            "giflab.deep_perceptual_metrics.torch"
        ) as mock_torch:
            # Mock tensor operations - need to handle arithmetic operations
            mock_tensor = Mock()

            # Handle arithmetic operations
            mock_tensor.__mul__ = Mock(return_value=mock_tensor)
            mock_tensor.__sub__ = Mock(return_value=mock_tensor)
            mock_tensor.unsqueeze.return_value = mock_tensor
            mock_tensor.to.return_value = mock_tensor
            mock_tensor.permute.return_value = mock_tensor
            mock_torch.from_numpy.return_value = mock_tensor

            validator._preprocess_for_lpips(frame)

            # Verify the preprocessing chain was called
            mock_torch.from_numpy.assert_called_once()
            mock_tensor.permute.assert_called_once_with(2, 0, 1)
            mock_tensor.unsqueeze.assert_called_once_with(0)
            mock_tensor.to.assert_called_once_with(validator.device)

    def test_calculate_deep_perceptual_metrics_fallback(
        self, fallback_validator, test_frames
    ):
        """Test fallback behavior when LPIPS is not available."""
        original_frames, compressed_frames = test_frames

        metrics = fallback_validator.calculate_deep_perceptual_metrics(
            original_frames, compressed_frames
        )

        # Should return fallback values
        assert isinstance(metrics, DeepPerceptualMetrics)
        assert metrics.lpips_quality_mean == 0.5
        assert metrics.lpips_quality_p95 == 0.5
        assert metrics.lpips_quality_max == 0.5
        assert metrics.frame_count == len(original_frames)
        assert metrics.device_used == "fallback"

    def test_validate_perceptual_quality_fallback(
        self, fallback_validator, test_frames
    ):
        """Test validation with fallback metrics."""
        original_frames, compressed_frames = test_frames

        result = fallback_validator.validate_perceptual_quality(
            original_frames, compressed_frames, quality_threshold=0.3
        )

        assert isinstance(result, PerceptualValidationResult)
        assert result.lpips_quality_mean == 0.5
        assert not result.quality_acceptable  # 0.5 > 0.3 threshold
        assert result.frames_processed == len(original_frames)

    def test_frame_sampling(self, validator, test_frames):
        """Test frame sampling for large frame counts."""
        original_frames, compressed_frames = test_frames

        # Create many frames to trigger sampling
        many_original = original_frames * 30  # 150 frames
        many_compressed = compressed_frames * 30

        metrics = validator.calculate_deep_perceptual_metrics(
            many_original, many_compressed, max_frames=20
        )

        # Should have sampled down to max_frames
        assert metrics.frame_count <= 20

    @patch("giflab.deep_perceptual_metrics.LPIPS_AVAILABLE", False)
    def test_lpips_unavailable_handling(self, validator, test_frames):
        """Test handling when LPIPS is not available."""
        original_frames, compressed_frames = test_frames

        metrics = validator.calculate_deep_perceptual_metrics(
            original_frames, compressed_frames
        )

        # Should return fallback metrics
        assert metrics.lpips_quality_mean == 0.5
        assert metrics.device_used == "fallback"


class TestCalculateDeepPerceptualQualityMetrics:
    """Test the main entry point function."""

    @pytest.fixture
    def test_frames(self):
        """Create test frames for validation."""
        original_frames = [
            np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8) for _ in range(3)
        ]
        compressed_frames = [
            np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8) for _ in range(3)
        ]
        return original_frames, compressed_frames

    def test_basic_functionality(self, test_frames):
        """Test basic functionality of the main entry point."""
        original_frames, compressed_frames = test_frames

        result = calculate_deep_perceptual_quality_metrics(
            original_frames, compressed_frames
        )

        assert isinstance(result, dict)
        assert "lpips_quality_mean" in result
        assert "lpips_quality_p95" in result
        assert "lpips_quality_max" in result
        assert "deep_perceptual_frame_count" in result
        assert "deep_perceptual_downscaled" in result
        assert "deep_perceptual_device" in result

    def test_with_custom_config(self, test_frames):
        """Test with custom configuration parameters."""
        original_frames, compressed_frames = test_frames

        config = {
            "device": "cpu",
            "lpips_downscale_size": 256,
            "lpips_max_frames": 50,
            "disable_deep_perceptual": False,
        }

        result = calculate_deep_perceptual_quality_metrics(
            original_frames, compressed_frames, config
        )

        assert isinstance(result, dict)
        assert result["deep_perceptual_device"] == "cpu"

    def test_with_disabled_deep_perceptual(self, test_frames):
        """Test when deep perceptual is disabled in config."""
        original_frames, compressed_frames = test_frames

        config = {"disable_deep_perceptual": True}

        result = calculate_deep_perceptual_quality_metrics(
            original_frames, compressed_frames, config
        )

        # Should return fallback values
        assert result["lpips_quality_mean"] == 0.5
        assert result["deep_perceptual_device"] == "fallback"

    def test_error_handling(self, test_frames):
        """Test error handling in main entry point."""
        original_frames, compressed_frames = test_frames

        # Simulate an error in the validator
        with patch.object(
            DeepPerceptualValidator,
            "calculate_deep_perceptual_metrics",
            side_effect=Exception("Test error"),
        ):
            result = calculate_deep_perceptual_quality_metrics(
                original_frames, compressed_frames
            )

            # Should return fallback values on error
            assert result["lpips_quality_mean"] == 0.5
            assert result["deep_perceptual_device"] == "fallback"


class TestIntegrationWithMetrics:
    """Test integration with the main metrics calculation system."""

    def test_metrics_integration_import(self):
        """Test that the module can be imported by the metrics system."""
        try:
            from giflab.deep_perceptual_metrics import (
                calculate_deep_perceptual_quality_metrics,
            )

            assert callable(calculate_deep_perceptual_quality_metrics)
        except ImportError as e:
            pytest.fail(f"Failed to import deep perceptual metrics: {e}")

    def test_conditional_triggering_in_metrics(self):
        """Test that conditional triggering works as expected."""
        # Test borderline quality case
        assert should_use_deep_perceptual(0.45)

        # Test high quality case
        assert not should_use_deep_perceptual(0.85)

        # Test None case (no composite quality available yet)
        assert should_use_deep_perceptual(None)


class TestDataStructures:
    """Test data structure integrity."""

    def test_deep_perceptual_metrics_dataclass(self):
        """Test DeepPerceptualMetrics dataclass."""
        metrics = DeepPerceptualMetrics(
            lpips_quality_mean=0.3,
            lpips_quality_p95=0.45,
            lpips_quality_max=0.6,
            frame_count=10,
            downscaled=True,
            device_used="cpu",
        )

        assert metrics.lpips_quality_mean == 0.3
        assert metrics.lpips_quality_p95 == 0.45
        assert metrics.lpips_quality_max == 0.6
        assert metrics.frame_count == 10
        assert metrics.downscaled
        assert metrics.device_used == "cpu"

    def test_perceptual_validation_result_dataclass(self):
        """Test PerceptualValidationResult dataclass."""
        result = PerceptualValidationResult(
            lpips_quality_mean=0.25,
            lpips_quality_p95=0.4,
            lpips_quality_max=0.55,
            quality_acceptable=True,
            frames_processed=15,
            downscaled=False,
        )

        assert result.lpips_quality_mean == 0.25
        assert result.quality_acceptable
        assert result.frames_processed == 15
        assert not result.downscaled


class TestPerformance:
    """Test performance characteristics of deep perceptual metrics."""

    def test_memory_efficiency(self):
        """Test that memory monitoring is integrated."""
        validator = DeepPerceptualValidator()

        # Check that memory monitor is initialized
        assert hasattr(validator, "memory_monitor")
        assert validator.memory_monitor is not None

    def test_downscaling_performance_benefit(self):
        """Test that downscaling provides expected performance benefits."""
        # This is more of a smoke test - actual performance would need real benchmarking
        validator = DeepPerceptualValidator(downscale_size=128)

        large_frame = np.random.randint(0, 256, (1024, 1024, 3), dtype=np.uint8)
        small_result, was_downscaled = validator._downscale_frame_if_needed(large_frame)

        assert was_downscaled
        assert small_result.shape[0] <= 128
        assert small_result.shape[1] <= 128


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_empty_frame_list(self):
        """Test handling of empty frame lists."""
        validator = DeepPerceptualValidator(force_fallback=True)

        metrics = validator.calculate_deep_perceptual_metrics([], [])

        assert metrics.frame_count == 0
        assert metrics.lpips_quality_mean == 0.5  # Fallback value

    def test_mismatched_frame_counts(self):
        """Test handling of mismatched frame counts."""
        validator = DeepPerceptualValidator(force_fallback=True)

        original_frames = [
            np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8) for _ in range(5)
        ]
        compressed_frames = [
            np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8) for _ in range(3)
        ]  # Different count

        # Should handle gracefully - in fallback mode, uses original frame count
        metrics = validator.calculate_deep_perceptual_metrics(
            original_frames, compressed_frames
        )
        assert metrics.frame_count == 5  # Uses original frames count in fallback

    def test_invalid_frame_shapes(self):
        """Test handling of frames with different shapes."""
        validator = DeepPerceptualValidator(force_fallback=True)

        # Different sized frames
        original_frames = [np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)]
        compressed_frames = [np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)]

        # Should handle gracefully (downscaling will make them compatible)
        metrics = validator.calculate_deep_perceptual_metrics(
            original_frames, compressed_frames
        )
        assert metrics.frame_count >= 0


if __name__ == "__main__":
    pytest.main([__file__])
