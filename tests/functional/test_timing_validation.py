"""Tests for timing validation functionality.

This module tests the TimingGridValidator class and related timing validation
functions to ensure frame timing integrity is properly validated.
"""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.giflab.config import ValidationConfig
from src.giflab.wrapper_validation.timing_validation import (
    TimingGridValidator,
    TimingMetrics,
    extract_timing_metrics_for_csv,
    validate_frame_timing_for_operation,
)
from src.giflab.wrapper_validation.types import ValidationResult


class TestTimingGridValidator:
    """Test cases for TimingGridValidator class."""

    def test_init_default_config(self):
        """Test initialization with default configuration."""
        validator = TimingGridValidator()

        # Should use default configuration values
        assert validator.grid_ms == 10  # Default TIMING_GRID_MS
        assert validator.timing_thresholds["max_drift_ms"] == 100
        assert validator.timing_thresholds["duration_diff_threshold"] == 50
        assert validator.timing_thresholds["alignment_accuracy_min"] == 0.9

    def test_init_custom_config(self):
        """Test initialization with custom configuration."""
        config = ValidationConfig(
            TIMING_GRID_MS=20,
            TIMING_MAX_DRIFT_MS=200,
            TIMING_DURATION_DIFF_THRESHOLD=100,
            TIMING_ALIGNMENT_ACCURACY_MIN=0.8,
        )
        validator = TimingGridValidator(config=config)

        assert validator.grid_ms == 20
        assert validator.timing_thresholds["max_drift_ms"] == 200
        assert validator.timing_thresholds["duration_diff_threshold"] == 100
        assert validator.timing_thresholds["alignment_accuracy_min"] == 0.8

    def test_align_to_grid(self):
        """Test frame duration alignment to timing grid."""
        validator = TimingGridValidator()

        # Test alignment to 10ms grid (Python uses banker's rounding)
        assert validator.align_to_grid(15) == 20  # 1.5 rounds to 2 (even)
        assert validator.align_to_grid(25) == 20  # 2.5 rounds to 2 (even)
        assert validator.align_to_grid(13) == 10  # 1.3 rounds down
        assert validator.align_to_grid(17) == 20  # 1.7 rounds up
        assert validator.align_to_grid(20) == 20  # Exact match
        assert validator.align_to_grid(0) == 0  # Zero case

        # Test with different grid size
        validator.grid_ms = 25
        assert validator.align_to_grid(30) == 25  # 1.2 rounds down
        assert validator.align_to_grid(40) == 50  # 1.6 rounds up

    def test_align_to_grid_zero_grid(self):
        """Test alignment with zero grid size (should return original duration)."""
        validator = TimingGridValidator()
        validator.grid_ms = 0

        assert validator.align_to_grid(15) == 15
        assert validator.align_to_grid(100) == 100

    def test_calculate_grid_length(self):
        """Test calculation of total animation length in grid units."""
        validator = TimingGridValidator()  # 10ms grid

        durations = [
            15,
            25,
            35,
        ]  # Align to [20, 20, 40] = 80ms total (banker's rounding)
        assert validator.calculate_grid_length(durations) == 8  # 80ms / 10ms

        durations = []
        assert validator.calculate_grid_length(durations) == 0

    def test_calculate_timing_drift_empty_lists(self):
        """Test timing drift calculation with empty duration lists."""
        validator = TimingGridValidator()

        result = validator.calculate_timing_drift([], [])
        expected = {
            "max_drift_ms": 0,
            "total_duration_diff_ms": 0,
            "drift_points": [],
            "cumulative_drift": [],
            "frames_analyzed": 0,
        }
        assert result == expected

    def test_calculate_timing_drift_perfect_match(self):
        """Test timing drift with perfectly matching durations."""
        validator = TimingGridValidator()

        original = [100, 200, 150]
        compressed = [100, 200, 150]

        result = validator.calculate_timing_drift(original, compressed)

        assert result["max_drift_ms"] == 0
        assert result["total_duration_diff_ms"] == 0
        assert result["frames_analyzed"] == 3
        assert result["cumulative_drift"] == [0, 0, 0]

    def test_calculate_timing_drift_with_drift(self):
        """Test timing drift with actual drift between durations."""
        validator = TimingGridValidator()

        original = [100, 200, 150]  # Total: 450ms
        compressed = [110, 190, 140]  # Total: 440ms, drift accumulates

        result = validator.calculate_timing_drift(original, compressed)

        assert result["total_duration_diff_ms"] == 10  # 450 - 440
        assert result["frames_analyzed"] == 3
        assert len(result["cumulative_drift"]) == 3

        # Check drift progression: 10ms, then 0ms, then 10ms
        assert result["cumulative_drift"][0] == 10  # |110 - 100| = 10
        assert (
            result["cumulative_drift"][1] == 0
        )  # |(110+190) - (100+200)| = |300-300| = 0
        assert (
            result["cumulative_drift"][2] == 10
        )  # |(110+190+140) - (100+200+150)| = |440-450| = 10
        assert result["max_drift_ms"] == 10

    def test_calculate_timing_drift_mismatched_lengths(self):
        """Test timing drift with different length duration lists."""
        validator = TimingGridValidator()

        original = [100, 200, 150, 300]  # 4 frames
        compressed = [100, 200]  # 2 frames

        result = validator.calculate_timing_drift(original, compressed)

        # Should only analyze the minimum number of frames
        assert result["frames_analyzed"] == 2
        assert len(result["cumulative_drift"]) == 2

    def test_calculate_alignment_accuracy_perfect(self):
        """Test alignment accuracy with perfect grid alignment."""
        validator = TimingGridValidator()  # 10ms grid

        original = [100, 200, 150]  # All align to themselves (multiples of 10)
        compressed = [100, 200, 150]  # Perfect match

        accuracy = validator.calculate_alignment_accuracy(original, compressed)
        assert accuracy == 1.0

    def test_calculate_alignment_accuracy_partial(self):
        """Test alignment accuracy with partial grid alignment."""
        validator = TimingGridValidator()  # 10ms grid

        original = [100, 200, 150]  # Align to [100, 200, 150]
        compressed = [
            105,
            195,
            150,
        ]  # Align to [100, 200, 150] - 2 of 3 match after alignment

        accuracy = validator.calculate_alignment_accuracy(original, compressed)
        assert accuracy == 1.0  # All align to the same values

    def test_calculate_alignment_accuracy_no_match(self):
        """Test alignment accuracy with no grid alignment matches."""
        validator = TimingGridValidator()  # 10ms grid

        original = [100, 200]  # Align to [100, 200]
        compressed = [105, 205]  # Align to [100, 200] - actually should match

        accuracy = validator.calculate_alignment_accuracy(original, compressed)
        assert accuracy == 1.0  # Both align to same grid values

    def test_calculate_alignment_accuracy_empty(self):
        """Test alignment accuracy with empty duration lists."""
        validator = TimingGridValidator()

        assert validator.calculate_alignment_accuracy([], []) == 0.0
        assert validator.calculate_alignment_accuracy([100], []) == 0.0
        assert validator.calculate_alignment_accuracy([], [100]) == 0.0

    @patch("src.giflab.wrapper_validation.timing_validation.Image.open")
    def test_extract_frame_durations_single_frame(self, mock_image_open):
        """Test frame duration extraction for single frame GIF."""
        validator = TimingGridValidator()

        # Mock single frame GIF
        mock_img = Mock()
        mock_img.__enter__ = Mock(return_value=mock_img)
        mock_img.__exit__ = Mock(return_value=None)
        mock_img.n_frames = 1
        mock_image_open.return_value = mock_img

        path = Path("/fake/single.gif")
        durations = validator.extract_frame_durations(path)

        assert durations == [100]  # Default duration for single frame

    @patch("src.giflab.wrapper_validation.timing_validation.Image.open")
    def test_extract_frame_durations_multi_frame(self, mock_image_open):
        """Test frame duration extraction for multi-frame GIF."""
        validator = TimingGridValidator()

        # Mock multi-frame GIF
        mock_img = Mock()
        mock_img.__enter__ = Mock(return_value=mock_img)
        mock_img.__exit__ = Mock(return_value=None)
        mock_img.n_frames = 3

        # Mock frame durations
        frame_durations = [100, 200, 150]

        def get_duration(key, default):
            return (
                frame_durations[mock_img.seek.call_count - 1]
                if key == "duration"
                else default
            )

        mock_info = Mock()
        mock_info.get = get_duration
        mock_img.info = mock_info
        mock_image_open.return_value = mock_img

        path = Path("/fake/multi.gif")
        durations = validator.extract_frame_durations(path)

        assert durations == frame_durations
        assert mock_img.seek.call_count == 3

    @patch("src.giflab.wrapper_validation.timing_validation.Image.open")
    def test_extract_frame_durations_duration_validation(self, mock_image_open):
        """Test frame duration validation and capping."""
        validator = TimingGridValidator()

        # Mock GIF with extreme durations
        mock_img = Mock()
        mock_img.__enter__ = Mock(return_value=mock_img)
        mock_img.__exit__ = Mock(return_value=None)
        mock_img.n_frames = 3

        # Mock extreme durations
        extreme_durations = [0, 15000, 50]  # Too small, too large, normal
        call_count = 0

        def get_duration(key, default):
            nonlocal call_count
            if key == "duration":
                result = extreme_durations[call_count]
                call_count += 1
                return result
            return default

        mock_info = Mock()
        mock_info.get = get_duration
        mock_img.info = mock_info
        mock_image_open.return_value = mock_img

        path = Path("/fake/extreme.gif")
        durations = validator.extract_frame_durations(path)

        # Should be: [100, 10000, 50] (capped and defaulted)
        assert durations == [100, 10000, 50]

    @patch("src.giflab.wrapper_validation.timing_validation.Image.open")
    def test_extract_frame_durations_file_error(self, mock_image_open):
        """Test frame duration extraction with file access error."""
        validator = TimingGridValidator()

        mock_image_open.side_effect = OSError("File not found")

        path = Path("/fake/missing.gif")
        with pytest.raises(ValueError, match="Cannot extract frame durations"):
            validator.extract_frame_durations(path)

    @patch.object(TimingGridValidator, "extract_frame_durations")
    def test_validate_timing_integrity_success(self, mock_extract):
        """Test successful timing validation."""
        validator = TimingGridValidator()

        # Mock frame durations that should pass validation
        original_durations = [100, 200, 150]
        compressed_durations = [100, 200, 150]

        mock_extract.side_effect = [original_durations, compressed_durations]

        original_path = Path("/fake/original.gif")
        compressed_path = Path("/fake/compressed.gif")

        result = validator.validate_timing_integrity(original_path, compressed_path)

        assert result.is_valid is True
        assert result.validation_type == "timing_grid_validation"
        assert result.error_message is None
        assert "timing_metrics" in result.details

    @patch.object(TimingGridValidator, "extract_frame_durations")
    def test_validate_timing_integrity_failure(self, mock_extract):
        """Test timing validation failure due to excessive drift."""
        # Create validator with strict thresholds
        config = ValidationConfig(
            TIMING_MAX_DRIFT_MS=10,  # Very strict
            TIMING_DURATION_DIFF_THRESHOLD=5,  # Very strict
        )
        validator = TimingGridValidator(config=config)

        # Mock frame durations with significant drift
        original_durations = [100, 200, 150]
        compressed_durations = [200, 100, 250]  # Large differences

        mock_extract.side_effect = [original_durations, compressed_durations]

        original_path = Path("/fake/original.gif")
        compressed_path = Path("/fake/compressed.gif")

        result = validator.validate_timing_integrity(original_path, compressed_path)

        assert result.is_valid is False
        assert result.validation_type == "timing_grid_validation"
        assert "timing drift" in result.error_message.lower()

    @patch.object(TimingGridValidator, "extract_frame_durations")
    def test_validate_timing_integrity_file_error(self, mock_extract):
        """Test timing validation with file access error."""
        validator = TimingGridValidator()

        mock_extract.side_effect = OSError("File not found")

        original_path = Path("/fake/original.gif")
        compressed_path = Path("/fake/compressed.gif")

        result = validator.validate_timing_integrity(original_path, compressed_path)

        assert result.is_valid is False
        assert result.validation_type == "timing_file_error"
        assert "cannot access gif files" in result.error_message.lower()


class TestTimingMetrics:
    """Test cases for TimingMetrics dataclass."""

    def test_timing_metrics_creation(self):
        """Test creation of TimingMetrics instance."""
        metrics = TimingMetrics(
            original_durations=[100, 200],
            compressed_durations=[100, 200],
            grid_ms=10,
            total_duration_diff_ms=0,
            max_timing_drift_ms=0,
            timing_drift_score=0.0,
            grid_length_original=30,
            grid_length_compressed=30,
            alignment_accuracy=1.0,
        )

        assert metrics.original_durations == [100, 200]
        assert metrics.compressed_durations == [100, 200]
        assert metrics.grid_ms == 10
        assert metrics.total_duration_diff_ms == 0
        assert metrics.max_timing_drift_ms == 0
        assert metrics.timing_drift_score == 0.0
        assert metrics.grid_length_original == 30
        assert metrics.grid_length_compressed == 30
        assert metrics.alignment_accuracy == 1.0


class TestHelperFunctions:
    """Test cases for helper functions."""

    @patch.object(TimingGridValidator, "validate_timing_integrity")
    def test_validate_frame_timing_for_operation(self, mock_validate):
        """Test convenience function for operation validation."""
        # Mock successful validation
        mock_result = ValidationResult(
            is_valid=True,
            validation_type="timing_grid_validation",
            expected={},
            actual={},
            details={"some": "data"},  # Initialize with some data
        )
        mock_validate.return_value = mock_result

        original_path = Path("/fake/original.gif")
        compressed_path = Path("/fake/compressed.gif")

        result = validate_frame_timing_for_operation(
            original_path, compressed_path, operation_type="test_operation"
        )

        assert result.is_valid is True
        assert result.details["operation_type"] == "test_operation"
        mock_validate.assert_called_once_with(original_path, compressed_path)

    def test_extract_timing_metrics_for_csv_success(self):
        """Test extraction of timing metrics for CSV output."""
        timing_metrics = TimingMetrics(
            original_durations=[100, 200],
            compressed_durations=[100, 200],
            grid_ms=10,
            total_duration_diff_ms=5,
            max_timing_drift_ms=2,
            timing_drift_score=0.1,
            grid_length_original=30,
            grid_length_compressed=30,
            alignment_accuracy=0.95,
        )

        validation_result = ValidationResult(
            is_valid=True,
            validation_type="timing_grid_validation",
            expected={},
            actual={},
            details={"timing_metrics": timing_metrics},
        )

        csv_metrics = extract_timing_metrics_for_csv(validation_result)

        expected = {
            "timing_grid_ms": 10,
            "grid_length": 30,
            "duration_diff_ms": 5,
            "timing_drift_score": 0.1,
            "max_timing_drift_ms": 2,
            "alignment_accuracy": 0.95,
        }

        assert csv_metrics == expected

    def test_extract_timing_metrics_for_csv_missing_data(self):
        """Test extraction with missing timing metrics data."""
        validation_result = ValidationResult(
            is_valid=False,
            validation_type="timing_file_error",
            expected={},
            actual={},
            details={},  # No timing_metrics
        )

        csv_metrics = extract_timing_metrics_for_csv(validation_result)

        # Should return default/empty metrics
        expected = {
            "timing_grid_ms": 0,
            "grid_length": 0,
            "duration_diff_ms": 0,
            "timing_drift_score": 1.0,
            "max_timing_drift_ms": 0,
            "alignment_accuracy": 0.0,
        }

        assert csv_metrics == expected
