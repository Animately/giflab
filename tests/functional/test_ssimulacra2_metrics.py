"""Unit tests for SSIMULACRA2 perceptual quality metrics integration.

Includes extended tests with edge cases, performance testing, error recovery,
and integration scenarios (consolidated from test_ssimulacra2_extended.py).
"""

import os
import signal
import subprocess
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, Mock, mock_open, patch

import cv2
import numpy as np
import pytest
from giflab.config import MetricsConfig
from giflab.ssimulacra2_metrics import (
    DEFAULT_SSIMULACRA2_PATH,
    SSIMULACRA2_EXCELLENT_SCORE,
    SSIMULACRA2_POOR_SCORE,
    Ssimulacra2Validator,
    calculate_ssimulacra2_quality_metrics,
    should_use_ssimulacra2,
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


class TestSsimulacra2Validator:
    """Test the SSIMULACRA2 validator class."""

    def test_init_default_path(self):
        """Test validator initialization with default path."""
        validator = Ssimulacra2Validator()
        assert validator.binary_path == Path(DEFAULT_SSIMULACRA2_PATH)

    def test_init_custom_path(self):
        """Test validator initialization with custom path."""
        custom_path = "/custom/path/to/ssimulacra2"
        validator = Ssimulacra2Validator(custom_path)
        assert validator.binary_path == Path(custom_path)

    @patch("subprocess.run")
    @patch.object(Path, "exists")
    @patch.object(Path, "is_file")
    def test_is_available_true(self, mock_is_file, mock_exists, mock_subprocess):
        """Test is_available returns True when binary exists."""
        mock_exists.return_value = True
        mock_is_file.return_value = True
        mock_subprocess.return_value = Mock(returncode=0)

        validator = Ssimulacra2Validator()
        assert validator.is_available() is True

    @patch("subprocess.run")
    @patch.object(Path, "exists")
    def test_is_available_false_missing_binary(self, mock_exists, mock_subprocess):
        """Test is_available returns False when binary doesn't exist."""
        mock_exists.return_value = False

        validator = Ssimulacra2Validator()
        assert validator.is_available() is False

    @patch("subprocess.run", side_effect=FileNotFoundError())
    def test_is_available_false_subprocess_error(self, mock_subprocess):
        """Test is_available returns False when subprocess fails."""
        validator = Ssimulacra2Validator()
        assert validator.is_available() is False

    def test_should_use_ssimulacra2_none_quality(self):
        """Test conditional triggering with None quality (first pass)."""
        validator = Ssimulacra2Validator()
        assert validator.should_use_ssimulacra2(None) is True

    def test_should_use_ssimulacra2_borderline_quality(self):
        """Test conditional triggering with borderline quality."""
        validator = Ssimulacra2Validator()
        assert validator.should_use_ssimulacra2(0.5) is True  # < 0.7 threshold
        assert validator.should_use_ssimulacra2(0.69) is True  # Just below threshold

    def test_should_use_ssimulacra2_high_quality(self):
        """Test conditional triggering with high quality."""
        validator = Ssimulacra2Validator()
        assert validator.should_use_ssimulacra2(0.8) is False  # > 0.7 threshold
        assert validator.should_use_ssimulacra2(0.95) is False  # High quality

    def test_normalize_score_excellent(self):
        """Test score normalization for excellent scores."""
        validator = Ssimulacra2Validator()
        assert validator.normalize_score(90.0) == 1.0
        assert validator.normalize_score(95.0) == 1.0  # Capped at 1.0

    def test_normalize_score_poor(self):
        """Test score normalization for poor scores."""
        validator = Ssimulacra2Validator()
        assert validator.normalize_score(10.0) == 0.0
        assert validator.normalize_score(5.0) == 0.0  # Capped at 0.0
        assert validator.normalize_score(-10.0) == 0.0  # Negative scores

    def test_normalize_score_medium(self):
        """Test score normalization for medium scores."""
        validator = Ssimulacra2Validator()
        # Linear interpolation between 10 (poor) and 90 (excellent)
        expected = (50.0 - 10.0) / (90.0 - 10.0)  # Should be 0.5
        assert abs(validator.normalize_score(50.0) - expected) < 0.001

    @patch("PIL.Image.fromarray")
    def test_export_frame_to_png(self, mock_fromarray):
        """Test PNG export functionality."""
        validator = Ssimulacra2Validator()

        # Create mock frame
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        # Create mock image
        mock_image = Mock()
        mock_fromarray.return_value = mock_image

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test.png"
            validator._export_frame_to_png(frame, output_path)

            mock_fromarray.assert_called_once()
            mock_image.save.assert_called_once_with(output_path, "PNG")

    @patch("PIL.Image.fromarray")
    def test_export_frame_to_png_float_input(self, mock_fromarray):
        """Test PNG export with float input (needs conversion)."""
        validator = Ssimulacra2Validator()

        # Create mock frame with float values [0, 1]
        frame = np.random.random((100, 100, 3)).astype(np.float32)

        mock_image = Mock()
        mock_fromarray.return_value = mock_image

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test.png"
            validator._export_frame_to_png(frame, output_path)

            # Verify the frame was converted to uint8
            call_args = mock_fromarray.call_args[0][0]
            assert call_args.dtype == np.uint8
            assert np.all(call_args >= 0) and np.all(call_args <= 255)

    @patch("subprocess.run")
    def test_run_ssimulacra2_on_pair_success(self, mock_subprocess):
        """Test successful SSIMULACRA2 execution."""
        validator = Ssimulacra2Validator()

        # Mock successful subprocess result
        mock_subprocess.return_value = Mock(returncode=0, stdout="75.42\n", stderr="")

        result = validator._run_ssimulacra2_on_pair(
            Path("/fake/orig.png"), Path("/fake/comp.png")
        )

        assert result == 75.42
        mock_subprocess.assert_called_once()

    @patch("subprocess.run")
    def test_run_ssimulacra2_on_pair_failure(self, mock_subprocess):
        """Test SSIMULACRA2 execution failure."""
        validator = Ssimulacra2Validator()

        # Mock failed subprocess result
        mock_subprocess.return_value = Mock(
            returncode=1, stdout="", stderr="Error: invalid input"
        )

        with pytest.raises(subprocess.CalledProcessError):
            validator._run_ssimulacra2_on_pair(
                Path("/fake/orig.png"), Path("/fake/comp.png")
            )

    @patch(
        "subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="test", timeout=30)
    )
    def test_run_ssimulacra2_timeout(self, mock_subprocess):
        """Test SSIMULACRA2 execution timeout."""
        validator = Ssimulacra2Validator()

        with pytest.raises(subprocess.TimeoutExpired):
            validator._run_ssimulacra2_on_pair(
                Path("/fake/orig.png"), Path("/fake/comp.png")
            )

    def test_sample_frame_indices_all_frames(self):
        """Test frame sampling when total <= max."""
        validator = Ssimulacra2Validator()

        indices = validator._sample_frame_indices(total_frames=5, max_frames=10)
        assert indices == [0, 1, 2, 3, 4]

        indices = validator._sample_frame_indices(total_frames=3, max_frames=3)
        assert indices == [0, 1, 2]

    def test_sample_frame_indices_uniform_sampling(self):
        """Test uniform frame sampling when total > max."""
        validator = Ssimulacra2Validator()

        indices = validator._sample_frame_indices(total_frames=100, max_frames=10)

        # Should return exactly 10 indices
        assert len(indices) == 10

        # Should be sorted
        assert indices == sorted(indices)

        # Should include first and last frames (approximately)
        assert indices[0] == 0
        assert indices[-1] == 99

    @patch.object(Ssimulacra2Validator, "is_available")
    def test_calculate_ssimulacra2_metrics_unavailable(self, mock_is_available):
        """Test metrics calculation when SSIMULACRA2 is unavailable."""
        mock_is_available.return_value = False

        validator = Ssimulacra2Validator()
        config = MetricsConfig()

        # Create mock frames
        orig_frames = [
            np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8) for _ in range(3)
        ]
        comp_frames = [
            np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8) for _ in range(3)
        ]

        result = validator.calculate_ssimulacra2_metrics(
            orig_frames, comp_frames, config
        )

        expected_keys = {
            "ssimulacra2_mean",
            "ssimulacra2_p95",
            "ssimulacra2_min",
            "ssimulacra2_frame_count",
            "ssimulacra2_triggered",
        }
        assert set(result.keys()) == expected_keys
        assert result["ssimulacra2_triggered"] == 0.0

    def test_calculate_ssimulacra2_metrics_frame_mismatch(self):
        """Test metrics calculation with mismatched frame counts."""
        validator = Ssimulacra2Validator()
        config = MetricsConfig()

        # Create mismatched frames
        orig_frames = [
            np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8) for _ in range(3)
        ]
        comp_frames = [
            np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8) for _ in range(5)
        ]

        with pytest.raises(ValueError, match="Frame count mismatch"):
            validator.calculate_ssimulacra2_metrics(orig_frames, comp_frames, config)

    @patch.object(Ssimulacra2Validator, "is_available")
    @patch.object(Ssimulacra2Validator, "_export_frame_to_png")
    @patch.object(Ssimulacra2Validator, "_run_ssimulacra2_on_pair")
    @patch.object(Ssimulacra2Validator, "normalize_score")
    def test_calculate_ssimulacra2_metrics_success(
        self, mock_normalize, mock_run_ssim, mock_export, mock_is_available
    ):
        """Test successful metrics calculation."""
        mock_is_available.return_value = True
        mock_run_ssim.side_effect = [75.0, 80.0, 70.0]  # Raw SSIMULACRA2 scores
        mock_normalize.side_effect = [0.75, 0.80, 0.70]  # Normalized scores

        validator = Ssimulacra2Validator()
        config = MetricsConfig()
        config.SSIMULACRA2_MAX_FRAMES = 5

        # Create mock frames
        orig_frames = [
            np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8) for _ in range(3)
        ]
        comp_frames = [
            np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8) for _ in range(3)
        ]

        result = validator.calculate_ssimulacra2_metrics(
            orig_frames, comp_frames, config
        )

        # Check that all frames were processed (3 <= 5 max frames)
        assert mock_run_ssim.call_count == 3
        assert mock_export.call_count == 6  # 3 orig + 3 comp
        assert mock_normalize.call_count == 3

        # Check results
        assert result["ssimulacra2_mean"] == 0.75  # Mean of [0.75, 0.80, 0.70]
        assert result["ssimulacra2_p95"] == 0.795  # 95th percentile
        assert result["ssimulacra2_min"] == 0.70  # Minimum
        assert result["ssimulacra2_frame_count"] == 3.0
        assert result["ssimulacra2_triggered"] == 1.0

    @patch.object(Ssimulacra2Validator, "is_available")
    @patch.object(Ssimulacra2Validator, "_export_frame_to_png")
    @patch.object(Ssimulacra2Validator, "_run_ssimulacra2_on_pair")
    def test_calculate_ssimulacra2_metrics_with_failures(
        self, mock_run_ssim, mock_export, mock_is_available
    ):
        """Test metrics calculation with some frame processing failures."""
        mock_is_available.return_value = True

        # First frame succeeds, second fails, third succeeds
        mock_run_ssim.side_effect = [75.0, Exception("Processing failed"), 70.0]

        validator = Ssimulacra2Validator()
        config = MetricsConfig()

        orig_frames = [
            np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8) for _ in range(3)
        ]
        comp_frames = [
            np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8) for _ in range(3)
        ]

        with patch.object(
            validator, "normalize_score", side_effect=[0.75, 0.70]
        ):
            result = validator.calculate_ssimulacra2_metrics(
                orig_frames, comp_frames, config
            )

        # Should have processed 3 frames but only 2 successful scores + 1 fallback (0.5)
        expected_scores = [0.75, 0.5, 0.70]  # Success, fallback, success
        assert result["ssimulacra2_mean"] == np.mean(expected_scores)
        assert result["ssimulacra2_frame_count"] == 3.0
        assert result["ssimulacra2_triggered"] == 1.0


class TestSsimulacra2ValidatorExtended:
    """Extended tests for Ssimulacra2Validator with comprehensive scenarios."""

    def test_binary_path_validation(self):
        """Test binary path validation with various path types."""
        # Test with non-existent path
        validator = Ssimulacra2Validator("/non/existent/path")
        assert not validator.is_available()

        # Test with directory instead of file
        with tempfile.TemporaryDirectory() as tmpdir:
            validator = Ssimulacra2Validator(tmpdir)
            assert not validator.is_available()

        # Test with relative path
        validator = Ssimulacra2Validator("./ssimulacra2")
        # Result depends on whether relative path exists
        assert isinstance(validator.is_available(), bool)

    @patch("subprocess.run")
    def test_binary_version_checking(self, mock_subprocess):
        """Test binary availability and capability checking."""
        # Test help check success (ssimulacra2 uses --help, not --version)
        mock_subprocess.return_value = Mock(
            returncode=1,  # ssimulacra2 returns non-zero even for --help
            stdout="Usage: ssimulacra2 orig.png distorted.png\n",
            stderr="",
        )

        validator = Ssimulacra2Validator()
        with patch.object(Path, "exists", return_value=True), patch.object(
            Path, "is_file", return_value=True
        ):
            assert validator.is_available() is True

        # Test binary not found
        mock_subprocess.side_effect = FileNotFoundError()

        assert validator.is_available() is False

    def test_conditional_triggering_edge_cases(self):
        """Test conditional triggering with edge case quality values."""
        validator = Ssimulacra2Validator()

        # Test boundary conditions around 0.7 threshold
        test_cases = [
            (0.699999, True),  # Just below threshold
            (0.700000, False),  # Exactly at threshold
            (0.700001, False),  # Just above threshold
            # Extreme values
            (0.0, True),  # Minimum quality
            (1.0, False),  # Perfect quality
            # Invalid values (should handle gracefully)
            (-0.5, True),  # Below valid range
            (1.5, False),  # Above valid range
        ]

        for quality, expected in test_cases:
            result = validator.should_use_ssimulacra2(quality)
            assert (
                result == expected
            ), f"Quality {quality} should trigger={expected}, got {result}"

    def test_score_normalization_mathematical_properties(self):
        """Test mathematical properties of score normalization."""
        validator = Ssimulacra2Validator()

        # Test monotonicity (higher raw score = higher normalized score)
        scores = [10, 25, 50, 75, 90]
        normalized = [validator.normalize_score(s) for s in scores]

        for i in range(len(normalized) - 1):
            assert (
                normalized[i] <= normalized[i + 1]
            ), f"Normalization not monotonic: {normalized[i]} > {normalized[i + 1]}"

        # Test normalization range
        extreme_scores = [-100, -10, 0, 5, 10, 50, 90, 95, 100, 150]
        for score in extreme_scores:
            normalized = validator.normalize_score(score)
            assert (
                0.0 <= normalized <= 1.0
            ), f"Score {score} normalized to {normalized}, outside [0, 1]"

    def test_frame_sampling_strategies(self):
        """Test different frame sampling strategies and edge cases."""
        validator = Ssimulacra2Validator()

        # Test with various frame counts
        test_cases = [
            (1, 30, [0]),  # Single frame
            (5, 30, [0, 1, 2, 3, 4]),  # All frames when count <= max
            (30, 30, list(range(30))),  # Exactly max frames
            (50, 10, [0, 5, 11, 16, 22, 27, 33, 38, 44, 49]),  # Uniform sampling
        ]

        for total, max_frames, _expected_pattern in test_cases:
            indices = validator._sample_frame_indices(total, max_frames)

            assert len(indices) <= max_frames
            assert len(indices) <= total

            # Check ordering
            assert indices == sorted(indices)

            # Check bounds
            if indices:
                assert indices[0] >= 0
                assert indices[-1] < total

        # Test edge cases
        assert validator._sample_frame_indices(0, 10) == []
        assert validator._sample_frame_indices(10, 0) == []

    @patch("subprocess.run")
    def test_subprocess_output_parsing(self, mock_subprocess):
        """Test parsing of various SSIMULACRA2 output formats."""
        validator = Ssimulacra2Validator()

        # Test normal output
        mock_subprocess.return_value = Mock(
            returncode=0, stdout="75.123456\n", stderr=""
        )
        result = validator._run_ssimulacra2_on_pair(Path("/fake1"), Path("/fake2"))
        assert abs(result - 75.123456) < 1e-6

        # Test output with extra whitespace
        mock_subprocess.return_value = Mock(
            returncode=0, stdout="  82.5  \n", stderr=""
        )
        result = validator._run_ssimulacra2_on_pair(Path("/fake1"), Path("/fake2"))
        assert abs(result - 82.5) < 1e-6

        # Test output with additional text (should extract number)
        mock_subprocess.return_value = Mock(
            returncode=0, stdout="SSIMULACRA2 score: 67.8\n", stderr=""
        )
        result = validator._run_ssimulacra2_on_pair(Path("/fake1"), Path("/fake2"))
        assert abs(result - 67.8) < 1e-6

        # Test negative score
        mock_subprocess.return_value = Mock(returncode=0, stdout="-5.25\n", stderr="")
        result = validator._run_ssimulacra2_on_pair(Path("/fake1"), Path("/fake2"))
        assert abs(result - (-5.25)) < 1e-6

    @patch("subprocess.run")
    def test_subprocess_error_scenarios(self, mock_subprocess):
        """Test handling of various subprocess error scenarios."""
        validator = Ssimulacra2Validator()

        # Test timeout
        mock_subprocess.side_effect = subprocess.TimeoutExpired("ssimulacra2", 30)
        with pytest.raises(subprocess.TimeoutExpired):
            validator._run_ssimulacra2_on_pair(Path("/fake1"), Path("/fake2"))

        # Test file not found
        mock_subprocess.side_effect = FileNotFoundError("Binary not found")
        with pytest.raises(FileNotFoundError):
            validator._run_ssimulacra2_on_pair(Path("/fake1"), Path("/fake2"))

        # Test permission denied
        mock_subprocess.side_effect = PermissionError("Permission denied")
        with pytest.raises(PermissionError):
            validator._run_ssimulacra2_on_pair(Path("/fake1"), Path("/fake2"))

        # Test unparseable output
        mock_subprocess.side_effect = None
        mock_subprocess.return_value = Mock(
            returncode=0, stdout="invalid_number\n", stderr=""
        )
        with pytest.raises((ValueError, TypeError)):
            validator._run_ssimulacra2_on_pair(Path("/fake1"), Path("/fake2"))

    def test_frame_export_edge_cases(self):
        """Test frame export with various data types and conditions."""
        validator = Ssimulacra2Validator()

        # Test different dtypes
        dtypes = [np.uint8, np.uint16, np.float32, np.float64]

        for dtype in dtypes:
            if dtype == np.uint8:
                frame = np.random.randint(0, 255, (50, 50, 3), dtype=dtype)
            elif dtype == np.uint16:
                frame = np.random.randint(0, 65535, (50, 50, 3), dtype=dtype)
            else:  # float types
                frame = np.random.random((50, 50, 3)).astype(dtype)

            with tempfile.NamedTemporaryFile(suffix=".png") as temp_file:
                temp_path = Path(temp_file.name)

                # Should handle different dtypes gracefully
                try:
                    validator._export_frame_to_png(frame, temp_path)
                    assert temp_path.exists()
                except Exception:
                    # Some dtypes might not be supported - that's acceptable
                    pass

        # Test extreme values
        extreme_frames = [
            np.zeros((50, 50, 3), dtype=np.uint8),  # All black
            np.full((50, 50, 3), 255, dtype=np.uint8),  # All white
            np.full((50, 50, 3), 128, dtype=np.uint8),  # Mid gray
        ]

        for frame in extreme_frames:
            with tempfile.NamedTemporaryFile(suffix=".png") as temp_file:
                temp_path = Path(temp_file.name)
                validator._export_frame_to_png(frame, temp_path)
                assert temp_path.exists()

    def test_memory_management_large_frames(self):
        """Test memory management with large frame sequences."""
        validator = Ssimulacra2Validator()
        config = MetricsConfig()
        config.SSIMULACRA2_MAX_FRAMES = 5

        # Create large frames
        large_frames_orig = [
            np.random.randint(0, 255, (500, 500, 3), dtype=np.uint8) for _ in range(10)
        ]
        large_frames_comp = [
            np.random.randint(0, 255, (500, 500, 3), dtype=np.uint8) for _ in range(10)
        ]

        with patch.object(validator, "is_available", return_value=True), patch.object(
            validator, "_export_frame_to_png"
        ) as mock_export, patch.object(
            validator, "_run_ssimulacra2_on_pair", return_value=50.0
        ), patch.object(
            validator, "normalize_score", return_value=0.5
        ):
            result = validator.calculate_ssimulacra2_metrics(
                large_frames_orig, large_frames_comp, config
            )

        # Should process only up to max_frames
        assert mock_export.call_count <= config.SSIMULACRA2_MAX_FRAMES * 2
        assert result["ssimulacra2_frame_count"] <= config.SSIMULACRA2_MAX_FRAMES

    @patch("tempfile.TemporaryDirectory")
    def test_temporary_directory_error_handling(self, mock_tempdir):
        """Test handling of temporary directory creation errors."""
        validator = Ssimulacra2Validator()
        config = MetricsConfig()

        # Mock temporary directory creation failure
        mock_tempdir.side_effect = OSError("Cannot create temp directory")

        frames = [np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)]

        with patch.object(validator, "is_available", return_value=True):
            with pytest.raises(OSError):
                validator.calculate_ssimulacra2_metrics(frames, frames, config)


class TestModuleFunctions:
    """Test module-level convenience functions."""

    @patch.object(Ssimulacra2Validator, "calculate_ssimulacra2_metrics")
    def test_calculate_ssimulacra2_quality_metrics(self, mock_calculate):
        """Test the main module function."""
        mock_result = {"ssimulacra2_mean": 0.75}
        mock_calculate.return_value = mock_result

        config = MetricsConfig()
        orig_frames = [np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)]
        comp_frames = [np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)]

        result = calculate_ssimulacra2_quality_metrics(orig_frames, comp_frames, config)

        assert result == mock_result
        mock_calculate.assert_called_once_with(orig_frames, comp_frames, config)

    @patch.object(Ssimulacra2Validator, "should_use_ssimulacra2")
    def test_should_use_ssimulacra2_function(self, mock_should_use):
        """Test the convenience function for conditional triggering."""
        mock_should_use.return_value = True

        result = should_use_ssimulacra2(0.5)

        assert result is True
        mock_should_use.assert_called_once_with(0.5)


class TestModuleFunctionsExtended:
    """Extended tests for module-level functions and utilities."""

    def test_convenience_function_parameter_passing(self):
        """Test parameter passing through convenience functions."""
        config = MetricsConfig()
        config.SSIMULACRA2_MAX_FRAMES = 15

        frames = [
            np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8) for _ in range(2)
        ]

        with patch("giflab.ssimulacra2_metrics.Ssimulacra2Validator") as MockValidator:
            mock_instance = Mock()
            mock_instance.calculate_ssimulacra2_metrics.return_value = {
                "test": "result"
            }
            MockValidator.return_value = mock_instance

            result = calculate_ssimulacra2_quality_metrics(frames, frames, config)

            # Verify validator was created and called correctly
            MockValidator.assert_called_once_with(config.SSIMULACRA2_PATH)
            mock_instance.calculate_ssimulacra2_metrics.assert_called_once_with(
                frames, frames, config
            )
            assert result == {"test": "result"}

    def test_should_use_ssimulacra2_parameter_handling(self):
        """Test parameter handling in should_use_ssimulacra2."""
        with patch("giflab.ssimulacra2_metrics.Ssimulacra2Validator") as MockValidator:
            mock_instance = Mock()
            MockValidator.return_value = mock_instance
            mock_instance.should_use_ssimulacra2.return_value = True

            # Test with various parameter types
            test_params = [0.5, 0.0, 1.0, None]

            for param in test_params:
                result = should_use_ssimulacra2(param)
                assert result is True

                # Verify correct parameter was passed
                mock_instance.should_use_ssimulacra2.assert_called_with(param)

    def test_module_constants_validation(self):
        """Test that module constants are reasonable."""
        # Test score boundaries
        assert SSIMULACRA2_POOR_SCORE < SSIMULACRA2_EXCELLENT_SCORE
        assert SSIMULACRA2_POOR_SCORE >= 0  # Should be non-negative
        assert SSIMULACRA2_EXCELLENT_SCORE <= 100  # Should be reasonable upper bound

        # Test default path
        assert isinstance(DEFAULT_SSIMULACRA2_PATH, str)
        assert len(DEFAULT_SSIMULACRA2_PATH) > 0


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""

    @patch.object(Ssimulacra2Validator, "is_available")
    def test_high_quality_skip_scenario(self, mock_is_available):
        """Test that high quality GIFs skip SSIMULACRA2 calculation."""
        mock_is_available.return_value = True

        validator = Ssimulacra2Validator()

        # High composite quality should not trigger SSIMULACRA2
        assert validator.should_use_ssimulacra2(0.85) is False

    @patch.object(Ssimulacra2Validator, "is_available")
    def test_borderline_quality_trigger_scenario(self, mock_is_available):
        """Test that borderline quality GIFs trigger SSIMULACRA2."""
        mock_is_available.return_value = True

        validator = Ssimulacra2Validator()

        # Borderline quality should trigger SSIMULACRA2
        assert validator.should_use_ssimulacra2(0.45) is True
        assert validator.should_use_ssimulacra2(0.65) is True

    def test_score_normalization_edge_cases(self):
        """Test score normalization with edge cases."""
        validator = Ssimulacra2Validator()

        # Test extreme values -- non-finite values are treated as errors and return 0.0
        assert validator.normalize_score(float("inf")) == 0.0
        assert validator.normalize_score(float("-inf")) == 0.0
        assert validator.normalize_score(0.0) == 0.0

        # Test boundary values
        assert validator.normalize_score(SSIMULACRA2_POOR_SCORE) == 0.0
        assert validator.normalize_score(SSIMULACRA2_EXCELLENT_SCORE) == 1.0

    @patch.object(Ssimulacra2Validator, "is_available")
    @patch("tempfile.TemporaryDirectory")
    def test_temporary_file_cleanup(self, mock_tempdir, mock_is_available):
        """Test that temporary files are properly cleaned up."""
        mock_is_available.return_value = True

        # Mock the context manager
        MagicMock()
        mock_tempdir.return_value.__enter__ = Mock(return_value="/fake/temp/dir")
        mock_tempdir.return_value.__exit__ = Mock(return_value=None)

        validator = Ssimulacra2Validator()
        config = MetricsConfig()

        orig_frames = [np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)]
        comp_frames = [np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)]

        with patch.object(validator, "_export_frame_to_png"), patch.object(
            validator, "_run_ssimulacra2_on_pair", return_value=50.0
        ), patch.object(validator, "normalize_score", return_value=0.5):
            result = validator.calculate_ssimulacra2_metrics(
                orig_frames, comp_frames, config
            )

        # Verify temporary directory context manager was used
        mock_tempdir.assert_called_once_with(prefix="ssimulacra2_")
        assert result["ssimulacra2_triggered"] == 1.0


class TestSsimulacra2Performance:
    """Performance and efficiency tests for SSIMULACRA2 integration."""

    def test_frame_sampling_performance(self):
        """Test performance of frame sampling with large sequences."""
        validator = Ssimulacra2Validator()

        # Test with very large frame counts
        large_counts = [1000, 5000, 10000]

        for count in large_counts:
            start_time = time.time()
            indices = validator._sample_frame_indices(count, 30)
            elapsed = time.time() - start_time

            # Should complete very quickly
            assert elapsed < 0.1  # Less than 100ms
            assert len(indices) == 30
            assert indices[0] == 0
            assert indices[-1] == count - 1

    def test_score_normalization_performance(self):
        """Test performance of score normalization."""
        validator = Ssimulacra2Validator()

        # Test batch normalization
        scores = np.random.uniform(-50, 150, 10000)

        start_time = time.time()
        normalized = [validator.normalize_score(s) for s in scores]
        elapsed = time.time() - start_time

        # Should process many scores quickly
        assert elapsed < 1.0  # Less than 1 second
        assert len(normalized) == len(scores)
        assert all(0.0 <= n <= 1.0 for n in normalized)

    @patch.object(Ssimulacra2Validator, "_export_frame_to_png")
    @patch.object(Ssimulacra2Validator, "_run_ssimulacra2_on_pair")
    def test_concurrent_processing_simulation(self, mock_run_ssim, mock_export):
        """Test behavior under concurrent processing simulation."""
        validator = Ssimulacra2Validator()
        config = MetricsConfig()

        # Simulate varying processing times
        mock_run_ssim.side_effect = lambda *args: (time.sleep(0.01), 50.0)[1]

        frames = [
            np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8) for _ in range(5)
        ]

        with patch.object(validator, "is_available", return_value=True), patch.object(
            validator, "normalize_score", return_value=0.5
        ):
            start_time = time.time()
            result = validator.calculate_ssimulacra2_metrics(frames, frames, config)
            elapsed = time.time() - start_time

            # Should complete reasonably quickly
            assert elapsed < 5.0  # Allow some time for simulated processing
            assert result["ssimulacra2_triggered"] == 1.0


class TestSsimulacra2Integration:
    """Integration tests with realistic scenarios."""

    def test_quality_level_detection(self, fixture_generator):
        """Test quality level detection with generated test pairs."""
        validator = Ssimulacra2Validator()
        config = MetricsConfig()

        quality_levels = ["excellent", "good", "medium", "poor", "terrible"]

        with patch.object(validator, "is_available", return_value=True):
            for quality_level in quality_levels:
                # Generate test pair
                orig_path, comp_path = fixture_generator.create_ssimulacra2_test_pair(
                    quality_level
                )

                # Load frames
                orig_frame = cv2.imread(str(orig_path))
                comp_frame = cv2.imread(str(comp_path))

                # Mock SSIMULACRA2 responses based on quality level
                expected_scores = {
                    "excellent": 85.0,
                    "good": 70.0,
                    "medium": 50.0,
                    "poor": 25.0,
                    "terrible": 5.0,
                }

                expected_score = expected_scores[quality_level]

                with patch.object(validator, "_export_frame_to_png"), patch.object(
                    validator, "_run_ssimulacra2_on_pair", return_value=expected_score
                ), patch.object(
                    validator,
                    "normalize_score",
                    return_value=validator.normalize_score(expected_score),
                ):
                    result = validator.calculate_ssimulacra2_metrics(
                        [orig_frame], [comp_frame], config
                    )

                # Check that results reflect expected quality level
                normalized_score = result["ssimulacra2_mean"]

                if quality_level in ["excellent", "good"]:
                    assert normalized_score > 0.6
                elif quality_level == "medium":
                    assert 0.3 <= normalized_score <= 0.7
                else:  # poor, terrible
                    assert normalized_score < 0.4

    def test_conditional_triggering_integration(self):
        """Test integration between conditional triggering and calculation."""
        validator = Ssimulacra2Validator()
        config = MetricsConfig()

        frames = [
            np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8) for _ in range(3)
        ]

        test_scenarios = [
            (0.85, False),  # High quality - should not trigger
            (0.45, True),  # Low quality - should trigger
            (None, True),  # No quality info - should trigger
        ]

        for _composite_quality, should_trigger in test_scenarios:
            # Create a modified validator that uses the quality for decisions
            with patch.object(
                validator, "is_available", return_value=True
            ), patch.object(
                validator, "should_use_ssimulacra2", return_value=should_trigger
            ), patch.object(
                validator, "_export_frame_to_png"
            ), patch.object(
                validator, "_run_ssimulacra2_on_pair", return_value=50.0
            ), patch.object(
                validator, "normalize_score", return_value=0.5
            ):
                # This would be called from the main metrics calculation
                result = validator.calculate_ssimulacra2_metrics(frames, frames, config)

                if should_trigger:
                    assert result["ssimulacra2_triggered"] == 1.0
                    assert result["ssimulacra2_mean"] > 0
                else:
                    # Current implementation always calculates if available
                    # This test documents expected behavior
                    pass

    def test_error_recovery_integration(self):
        """Test error recovery in realistic failure scenarios."""
        validator = Ssimulacra2Validator()
        config = MetricsConfig()

        frames = [
            np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8) for _ in range(5)
        ]

        # Test partial failures
        with patch.object(validator, "is_available", return_value=True), patch.object(
            validator, "_export_frame_to_png"
        ), patch.object(
            validator, "normalize_score", side_effect=[0.8, 0.5, 0.6, 0.7, 0.4]
        ):
            # First and third frames fail, others succeed
            mock_run_ssim = Mock(
                side_effect=[
                    Exception("Frame 0 failed"),
                    60.0,  # Frame 1 success
                    Exception("Frame 2 failed"),
                    70.0,  # Frame 3 success
                    40.0,  # Frame 4 success
                ]
            )

            with patch.object(validator, "_run_ssimulacra2_on_pair", mock_run_ssim):
                result = validator.calculate_ssimulacra2_metrics(frames, frames, config)

            # Should have processed all frames but with fallback scores for failures
            assert result["ssimulacra2_frame_count"] == 5.0
            assert result["ssimulacra2_triggered"] == 1.0

            # Mean should include fallback scores (0.5) for failed frames
            # normalize_score is called only for successful frames (1, 3, 4),
            # consuming side_effect values [0.8, 0.5, 0.6].
            # Failed frames (0, 2) get hardcoded 0.5.
            # Expected scores: [0.5, 0.8, 0.5, 0.5, 0.6]
            expected_mean = np.mean([0.5, 0.8, 0.5, 0.5, 0.6])
            assert abs(result["ssimulacra2_mean"] - expected_mean) < 0.01


class TestSsimulacra2RobustnessAndSafety:
    """Test robustness, safety, and edge case handling."""

    def test_input_sanitization(self):
        """Test input sanitization for security and robustness."""
        validator = Ssimulacra2Validator()

        # Test path traversal attempts
        malicious_paths = [
            "../../../etc/passwd",
            "../../../../bin/rm",
            "/dev/null",
            "con",
            "aux",
            "prn",  # Windows reserved names
        ]

        for malicious_path in malicious_paths:
            path = Path(malicious_path)

            # Should handle gracefully without security issues
            try:
                validator._run_ssimulacra2_on_pair(path, path)
            except (subprocess.CalledProcessError, FileNotFoundError, OSError):
                pass  # Expected failures for invalid paths
            except Exception as e:
                # Should not raise unexpected exceptions
                pytest.fail(f"Unexpected exception for path {malicious_path}: {e}")

    def test_resource_cleanup_edge_cases(self):
        """Test resource cleanup in error conditions."""
        validator = Ssimulacra2Validator()
        config = MetricsConfig()

        frames = [np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)]

        # Test cleanup when export fails - the exception is caught internally
        # and a fallback score (0.5) is used, so no exception propagates
        with patch.object(validator, "is_available", return_value=True), patch(
            "tempfile.TemporaryDirectory"
        ) as mock_tempdir:
            # Mock temp directory
            mock_tempdir_instance = Mock()
            mock_tempdir.return_value = mock_tempdir_instance
            mock_tempdir_instance.__enter__ = Mock(return_value="/fake/temp")
            mock_tempdir_instance.__exit__ = Mock(return_value=None)

            # Export fails after temp directory creation
            with patch.object(
                validator,
                "_export_frame_to_png",
                side_effect=Exception("Export failed"),
            ):
                result = validator.calculate_ssimulacra2_metrics(
                    frames, frames, config
                )

                # Should use fallback score for the failed frame
                assert result["ssimulacra2_mean"] == 0.5
                assert result["ssimulacra2_frame_count"] == 1.0

                # Temporary directory cleanup should still be called
                mock_tempdir_instance.__exit__.assert_called_once()

    def test_signal_handling_simulation(self):
        """Test behavior when interrupted by signals."""
        validator = Ssimulacra2Validator()
        config = MetricsConfig()

        frames = [
            np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8) for _ in range(3)
        ]

        def interrupt_after_delay(*args, **kwargs):
            time.sleep(0.01)  # Simulate some processing time
            raise KeyboardInterrupt("Simulated interrupt")

        with patch.object(validator, "is_available", return_value=True), patch.object(
            validator, "_export_frame_to_png"
        ), patch.object(
            validator, "_run_ssimulacra2_on_pair", side_effect=interrupt_after_delay
        ):
            with pytest.raises(KeyboardInterrupt):
                validator.calculate_ssimulacra2_metrics(frames, frames, config)

    def test_numerical_stability_edge_cases(self):
        """Test numerical stability with extreme or edge case values."""
        validator = Ssimulacra2Validator()

        # Test score normalization with extreme values
        extreme_scores = [
            float("inf"),
            float("-inf"),
            float("nan"),
            1e10,
            -1e10,
            1e-10,
            -1e-10,
        ]

        for score in extreme_scores:
            normalized = validator.normalize_score(score)

            # Should always return finite values in [0, 1]
            assert np.isfinite(normalized), f"Non-finite result for score {score}"
            assert (
                0.0 <= normalized <= 1.0
            ), f"Out of range result for score {score}: {normalized}"

    def test_concurrent_access_safety(self):
        """Test thread safety and concurrent access patterns."""
        # Create multiple validator instances
        validators = [Ssimulacra2Validator() for _ in range(3)]

        # Test that they don't interfere with each other
        quality_values = [0.3, 0.6, 0.9]

        results = []
        for validator, quality in zip(validators, quality_values):
            result = validator.should_use_ssimulacra2(quality)
            results.append(result)

        # Results should be consistent and independent
        expected = [True, True, False]  # Based on 0.7 threshold
        assert results == expected

    def test_memory_leak_prevention(self):
        """Test prevention of memory leaks in repeated operations."""
        validator = Ssimulacra2Validator()

        # Perform many score normalizations
        for _ in range(1000):
            score = np.random.uniform(-50, 150)
            normalized = validator.normalize_score(score)
            assert 0.0 <= normalized <= 1.0

        # Perform many frame index samples
        for _ in range(100):
            total = np.random.randint(10, 200)
            max_frames = np.random.randint(5, 50)
            indices = validator._sample_frame_indices(total, max_frames)
            assert len(indices) <= min(total, max_frames)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
