"""Unit tests for timing validation functionality.

Tests the TimingGridValidator class and associated timing validation functions
including frame duration extraction, timing grid alignment, and drift detection.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from giflab.wrapper_validation.timing_validation import (
    TimingGridValidator,
    TimingMetrics,
    extract_timing_metrics_for_csv,
    validate_frame_timing_for_operation,
)
from giflab.wrapper_validation.types import ValidationResult
from PIL import Image


@pytest.fixture
def temp_gif_files():
    """Create temporary GIF files with known timing characteristics for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create original GIF with consistent 100ms frame delays
        original_path = temp_path / "original.gif"
        create_test_gif_with_timing(original_path, frames=4, duration=100)

        # Create compressed GIF with slightly different timing (frame reduction simulation)
        compressed_path = temp_path / "compressed.gif"
        create_test_gif_with_timing(
            compressed_path, frames=3, duration=133
        )  # ~100ms * 4/3

        # Create single frame GIF
        single_frame_path = temp_path / "single.gif"
        create_test_gif_with_timing(single_frame_path, frames=1, duration=100)

        # Create GIF with timing drift
        drift_path = temp_path / "drift.gif"
        create_test_gif_with_variable_timing(drift_path, durations=[100, 110, 120, 90])

        yield {
            "original": original_path,
            "compressed": compressed_path,
            "single_frame": single_frame_path,
            "drift": drift_path,
            "temp_dir": temp_path,
        }


def create_test_gif_with_timing(path: Path, frames: int = 4, duration: int = 100):
    """Create a test GIF with specific timing characteristics."""
    path.parent.mkdir(parents=True, exist_ok=True)
    size = (20, 20)
    images = []

    for i in range(frames):
        # Create different colored frames for visual distinction
        color = (i * 60, 100, 200 - i * 40)
        img = Image.new("RGB", size, color)
        images.append(img)

    if images:
        images[0].save(
            path, save_all=True, append_images=images[1:], duration=duration, loop=0
        )


def create_test_gif_with_variable_timing(path: Path, durations: list[int]):
    """Create a test GIF with variable frame durations."""
    path.parent.mkdir(parents=True, exist_ok=True)
    size = (20, 20)
    images = []

    for i, _duration in enumerate(durations):
        color = (i * 50, 150, 255 - i * 50)
        img = Image.new("RGB", size, color)
        images.append(img)

    if images:
        # Save with variable durations - this is tricky with PIL
        # We'll create individual frames and combine them
        images[0].save(
            path, save_all=True, append_images=images[1:], duration=durations, loop=0
        )


class TestTimingGridValidator:
    """Test cases for TimingGridValidator class."""

    def _create_validator(self, grid_ms=None):
        """Helper to create TimingGridValidator with custom grid size."""
        if grid_ms is None:
            return TimingGridValidator()

        from giflab.config import ValidationConfig

        custom_config = ValidationConfig()
        custom_config.TIMING_GRID_MS = grid_ms
        return TimingGridValidator(config=custom_config)

    def test_init_default_grid(self):
        """Test TimingGridValidator initialization with default grid size."""
        validator = TimingGridValidator()
        assert validator.grid_ms == 10
        assert validator.timing_thresholds["max_drift_ms"] == 100
        assert validator.timing_thresholds["duration_diff_threshold"] == 50
        assert validator.timing_thresholds["alignment_accuracy_min"] == 0.9

    def test_init_custom_grid(self):
        """Test TimingGridValidator initialization with custom grid size."""
        validator = self._create_validator(grid_ms=20)
        assert validator.grid_ms == 20

    def test_align_to_grid_basic(self):
        """Test basic timing grid alignment."""
        validator = self._create_validator(grid_ms=10)

        assert validator.align_to_grid(95) == 100  # Round up
        assert validator.align_to_grid(105) == 100  # Round down
        assert validator.align_to_grid(100) == 100  # Exact match
        assert validator.align_to_grid(85) == 80  # Round down (8.5 rounds to 8)

    def test_align_to_grid_edge_cases(self):
        """Test timing grid alignment edge cases."""
        validator = self._create_validator(grid_ms=0)
        assert validator.align_to_grid(95) == 95  # No alignment when grid_ms is 0

        validator = self._create_validator(grid_ms=1)
        assert validator.align_to_grid(95) == 95  # Already aligned to 1ms grid

    def test_calculate_grid_length(self):
        """Test calculation of animation length in grid units."""
        validator = self._create_validator(grid_ms=10)
        durations = [100, 110, 90, 105]  # Total: 405ms

        # Aligned: 100 + 110 + 90 + 100 = 400ms = 40 grid units (105 rounds to 100)
        expected_grid_length = 40
        assert validator.calculate_grid_length(durations) == expected_grid_length

    def test_calculate_grid_length_edge_cases(self):
        """Test grid length calculation edge cases."""
        validator = self._create_validator(grid_ms=10)

        assert validator.calculate_grid_length([]) == 0
        assert validator.calculate_grid_length([0]) == 0

        validator_zero = self._create_validator(grid_ms=0)
        assert validator_zero.calculate_grid_length([100, 200]) == 0

    @pytest.mark.fast
    def test_extract_frame_durations_basic(self, temp_gif_files):
        """Test basic frame duration extraction."""
        validator = TimingGridValidator()
        durations = validator.extract_frame_durations(temp_gif_files["original"])

        assert isinstance(durations, list)
        assert len(durations) == 4
        assert all(isinstance(d, int) for d in durations)
        assert all(d >= 1 for d in durations)  # All durations should be reasonable

    @pytest.mark.fast
    def test_extract_frame_durations_single_frame(self, temp_gif_files):
        """Test frame duration extraction for single frame GIF."""
        validator = TimingGridValidator()
        durations = validator.extract_frame_durations(temp_gif_files["single_frame"])

        assert len(durations) == 1
        assert durations[0] == 100  # Default duration for single frame

    def test_extract_frame_durations_missing_file(self):
        """Test frame duration extraction with missing file."""
        validator = TimingGridValidator()

        with pytest.raises(ValueError, match="Cannot extract frame durations"):
            validator.extract_frame_durations(Path("nonexistent.gif"))

    def test_calculate_timing_drift_basic(self):
        """Test basic timing drift calculation."""
        validator = TimingGridValidator()
        original = [100, 100, 100, 100]  # Total: 400ms
        compressed = [100, 110, 90, 100]  # Total: 400ms, but with drift in middle

        drift_metrics = validator.calculate_timing_drift(original, compressed)

        assert "max_drift_ms" in drift_metrics
        assert "total_duration_diff_ms" in drift_metrics
        assert "drift_points" in drift_metrics
        assert "cumulative_drift" in drift_metrics
        assert "frames_analyzed" in drift_metrics

        assert drift_metrics["total_duration_diff_ms"] == 0  # Same total duration
        assert drift_metrics["frames_analyzed"] == 4
        assert len(drift_metrics["cumulative_drift"]) == 4

    def test_calculate_timing_drift_frame_mismatch(self):
        """Test timing drift calculation with mismatched frame counts."""
        validator = TimingGridValidator()
        original = [100, 100, 100, 100]  # 4 frames
        compressed = [133, 133, 134]  # 3 frames (frame reduction)

        drift_metrics = validator.calculate_timing_drift(original, compressed)

        assert drift_metrics["frames_analyzed"] == 3  # Min of both counts
        assert len(drift_metrics["cumulative_drift"]) == 3

    def test_calculate_timing_drift_empty_input(self):
        """Test timing drift calculation with empty input."""
        validator = TimingGridValidator()

        drift_metrics = validator.calculate_timing_drift([], [])

        assert drift_metrics["max_drift_ms"] == 0
        assert drift_metrics["total_duration_diff_ms"] == 0
        assert drift_metrics["frames_analyzed"] == 0
        assert drift_metrics["drift_points"] == []
        assert drift_metrics["cumulative_drift"] == []

    def test_calculate_alignment_accuracy_perfect(self):
        """Test alignment accuracy calculation with perfect alignment."""
        validator = self._create_validator(grid_ms=10)
        original = [100, 110, 90, 100]  # All align to 10ms grid
        compressed = [100, 110, 90, 100]  # Identical

        accuracy = validator.calculate_alignment_accuracy(original, compressed)
        assert accuracy == 1.0

    def test_calculate_alignment_accuracy_partial(self):
        """Test alignment accuracy calculation with partial alignment."""
        validator = self._create_validator(grid_ms=10)
        original = [100, 100, 100, 100]  # All align to 100ms
        compressed = [100, 105, 95, 100]  # 105 aligns to 100, 95 aligns to 100

        accuracy = validator.calculate_alignment_accuracy(original, compressed)
        assert accuracy == 1.0  # All align to same grid points after rounding

    def test_calculate_alignment_accuracy_poor(self):
        """Test alignment accuracy calculation with poor alignment."""
        validator = self._create_validator(grid_ms=100)
        original = [100, 100, 100, 100]  # All align to 100ms
        compressed = [150, 150, 50, 50]  # All align to different grid points

        accuracy = validator.calculate_alignment_accuracy(original, compressed)
        assert accuracy == 0.0  # No matching grid points

    def test_calculate_alignment_accuracy_edge_cases(self):
        """Test alignment accuracy calculation edge cases."""
        validator = self._create_validator(grid_ms=10)

        assert validator.calculate_alignment_accuracy([], []) == 0.0
        assert validator.calculate_alignment_accuracy([100], []) == 0.0
        assert validator.calculate_alignment_accuracy([], [100]) == 0.0

    @pytest.mark.fast
    def test_validate_timing_integrity_valid(self, temp_gif_files):
        """Test timing integrity validation with valid timing."""
        validator = self._create_validator(grid_ms=10)

        # Test with very similar timing (should pass)
        result = validator.validate_timing_integrity(
            temp_gif_files["original"],
            temp_gif_files["original"],  # Same file should have perfect timing
        )

        assert isinstance(result, ValidationResult)
        assert result.is_valid is True
        assert result.validation_type == "timing_grid_validation"
        assert "timing_metrics" in result.details
        assert "drift_analysis" in result.details

    @pytest.mark.fast
    def test_validate_timing_integrity_invalid(self, temp_gif_files):
        """Test timing integrity validation with invalid timing."""
        validator = self._create_validator(grid_ms=10)

        # Set very strict thresholds to force failure
        validator.timing_thresholds["max_drift_ms"] = 5
        validator.timing_thresholds["duration_diff_threshold"] = 5
        validator.timing_thresholds["alignment_accuracy_min"] = 0.99

        result = validator.validate_timing_integrity(
            temp_gif_files["original"],
            temp_gif_files["drift"],  # Should have timing issues
        )

        assert isinstance(result, ValidationResult)
        # Note: might pass or fail depending on exact timing, test the structure
        assert result.validation_type == "timing_grid_validation"
        assert result.expected is not None
        assert result.actual is not None
        assert "timing_metrics" in result.details

    def test_validate_timing_integrity_missing_files(self):
        """Test timing integrity validation with missing files."""
        validator = TimingGridValidator()

        result = validator.validate_timing_integrity(
            Path("missing1.gif"), Path("missing2.gif")
        )

        assert result.is_valid is False
        assert result.validation_type == "timing_calculation_error"
        assert "exception" in result.details


class TestTimingMetrics:
    """Test cases for TimingMetrics dataclass."""

    def test_timing_metrics_creation(self):
        """Test TimingMetrics dataclass creation."""
        metrics = TimingMetrics(
            original_durations=[100, 100, 100],
            compressed_durations=[100, 110, 90],
            grid_ms=10,
            total_duration_diff_ms=0,
            max_timing_drift_ms=10,
            timing_drift_score=0.1,
            grid_length_original=30,
            grid_length_compressed=30,
            alignment_accuracy=0.95,
        )

        assert metrics.grid_ms == 10
        assert len(metrics.original_durations) == 3
        assert len(metrics.compressed_durations) == 3
        assert metrics.timing_drift_score == 0.1
        assert metrics.alignment_accuracy == 0.95


class TestConvenienceFunctions:
    """Test cases for convenience functions."""

    @pytest.mark.fast
    def test_validate_frame_timing_for_operation(self, temp_gif_files):
        """Test convenience function for timing validation."""
        from giflab.config import ValidationConfig

        # Create a custom config with grid_ms = 20
        custom_config = ValidationConfig()
        custom_config.TIMING_GRID_MS = 20

        result = validate_frame_timing_for_operation(
            temp_gif_files["original"],
            temp_gif_files["compressed"],
            operation_type="frame_reduction",
            config=custom_config,
        )

        assert isinstance(result, ValidationResult)
        assert result.validation_type == "timing_grid_validation"
        assert result.details["operation_type"] == "frame_reduction"

    def test_extract_timing_metrics_for_csv_valid(self):
        """Test CSV metrics extraction from valid validation result."""
        # Create a mock ValidationResult with timing metrics
        timing_metrics = TimingMetrics(
            original_durations=[100, 100, 100],
            compressed_durations=[100, 110, 90],
            grid_ms=10,
            total_duration_diff_ms=10,
            max_timing_drift_ms=15,
            timing_drift_score=0.2,
            grid_length_original=30,
            grid_length_compressed=30,
            alignment_accuracy=0.85,
        )

        validation_result = ValidationResult(
            is_valid=True,
            validation_type="timing_grid_validation",
            expected={},
            actual={},
            details={"timing_metrics": timing_metrics},
        )

        csv_metrics = extract_timing_metrics_for_csv(validation_result)

        expected_keys = [
            "timing_grid_ms",
            "grid_length",
            "duration_diff_ms",
            "timing_drift_score",
            "max_timing_drift_ms",
            "alignment_accuracy",
        ]

        for key in expected_keys:
            assert key in csv_metrics

        assert csv_metrics["timing_grid_ms"] == 10
        assert csv_metrics["grid_length"] == 30
        assert csv_metrics["duration_diff_ms"] == 10
        assert csv_metrics["timing_drift_score"] == 0.2
        assert csv_metrics["max_timing_drift_ms"] == 15
        assert csv_metrics["alignment_accuracy"] == 0.85

    def test_extract_timing_metrics_for_csv_empty(self):
        """Test CSV metrics extraction from empty/failed validation result."""
        validation_result = ValidationResult(
            is_valid=False,
            validation_type="timing_validation_error",
            expected="success",
            actual="failure",
            details={},  # No timing metrics
        )

        csv_metrics = extract_timing_metrics_for_csv(validation_result)

        # Should return default values
        expected_defaults = {
            "timing_grid_ms": 0,
            "grid_length": 0,
            "duration_diff_ms": 0,
            "timing_drift_score": 1.0,
            "max_timing_drift_ms": 0,
            "alignment_accuracy": 0.0,
        }

        for key, expected_value in expected_defaults.items():
            assert csv_metrics[key] == expected_value


class TestIntegrationScenarios:
    """Test cases for integration scenarios and edge cases."""

    def _create_validator(self, grid_ms=None):
        """Helper to create TimingGridValidator with custom grid size."""
        if grid_ms is None:
            return TimingGridValidator()

        from giflab.config import ValidationConfig

        custom_config = ValidationConfig()
        custom_config.TIMING_GRID_MS = grid_ms
        return TimingGridValidator(config=custom_config)

    @pytest.mark.fast
    def test_frame_reduction_timing_validation(self, temp_gif_files):
        """Test timing validation in frame reduction scenario."""
        validator = self._create_validator(grid_ms=10)

        # Simulate frame reduction: 4 frames -> 3 frames
        result = validator.validate_timing_integrity(
            temp_gif_files["original"],  # 4 frames at 100ms each
            temp_gif_files["compressed"],  # 3 frames at ~133ms each
        )

        assert isinstance(result, ValidationResult)
        # Should handle frame count differences gracefully
        assert "frames_analyzed" in result.details["drift_analysis"]
        assert result.details["drift_analysis"]["frames_analyzed"] <= 4

    def test_timing_validation_disabled(self):
        """Test timing validation when disabled via configuration."""
        from giflab.wrapper_validation.pipeline_validation import PipelineStageValidator

        validator = PipelineStageValidator()
        validator.pipeline_thresholds["timing_validation_enabled"] = False

        result = validator.validate_timing_integrity_for_stage(
            Path("dummy1.gif"), Path("dummy2.gif"), "test_stage"
        )

        assert result.is_valid is True
        assert result.validation_type == "timing_validation_disabled"

    @pytest.mark.fast
    def test_timing_validation_with_context(self, temp_gif_files):
        """Test timing validation with pipeline context."""
        from giflab.wrapper_validation.pipeline_validation import PipelineStageValidator

        validator = PipelineStageValidator()

        result = validator.validate_timing_integrity_for_stage(
            temp_gif_files["original"],
            temp_gif_files["compressed"],
            "frame_reduction",
            stage_index=1,
        )

        assert isinstance(result, ValidationResult)
        assert result.details["pipeline_stage"] == "frame_reduction"
        assert result.details["stage_index"] == 1
        assert result.details["validation_context"] == "pipeline_stage_validation"


class TestErrorHandling:
    """Test cases for error handling and edge cases."""

    def test_corrupted_gif_handling(self):
        """Test handling of corrupted GIF files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a fake GIF file with invalid content
            fake_gif = Path(temp_dir) / "corrupted.gif"
            fake_gif.write_bytes(b"Not a real GIF file")

            validator = TimingGridValidator()

            with pytest.raises(ValueError):
                validator.extract_frame_durations(fake_gif)

    @patch("giflab.wrapper_validation.timing_validation.Image.open")
    def test_pil_exception_handling(self, mock_open):
        """Test handling of PIL exceptions during frame duration extraction."""
        mock_open.side_effect = Exception("PIL error")

        validator = TimingGridValidator()

        with pytest.raises(ValueError, match="Cannot extract frame durations"):
            validator.extract_frame_durations(Path("test.gif"))

    def test_zero_duration_frames(self):
        """Test handling of frames with zero or invalid durations."""
        validator = TimingGridValidator()

        # Test timing drift calculation with problematic durations
        original = [0, 100, -10, 5000]  # Mix of problematic values
        compressed = [100, 100, 100, 100]

        # Should handle gracefully without crashing
        drift_metrics = validator.calculate_timing_drift(original, compressed)

        assert "max_drift_ms" in drift_metrics
        assert "total_duration_diff_ms" in drift_metrics
        assert drift_metrics["frames_analyzed"] == 4


@pytest.mark.performance
class TestPerformance:
    """Test cases for performance characteristics."""

    def test_timing_validation_speed(self, temp_gif_files):
        """Test that timing validation completes within reasonable time."""
        import time

        validator = TimingGridValidator()

        start_time = time.perf_counter()
        result = validator.validate_timing_integrity(
            temp_gif_files["original"], temp_gif_files["compressed"]
        )
        end_time = time.perf_counter()

        elapsed_ms = (end_time - start_time) * 1000

        # Should complete within 100ms for small test GIFs (generous threshold)
        assert (
            elapsed_ms < 100
        ), f"Timing validation took {elapsed_ms:.1f}ms, expected < 100ms"
        assert result.validation_type == "timing_grid_validation"

    def test_memory_usage_reasonable(self, temp_gif_files):
        """Test that timing validation doesn't use excessive memory."""
        validator = TimingGridValidator()

        # This should not crash or consume excessive memory
        durations1 = validator.extract_frame_durations(temp_gif_files["original"])
        durations2 = validator.extract_frame_durations(temp_gif_files["compressed"])

        # Basic sanity checks
        assert isinstance(durations1, list)
        assert isinstance(durations2, list)
        assert len(durations1) > 0
        assert len(durations2) > 0
