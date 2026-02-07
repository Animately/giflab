"""Tests for wrapper output validation system.

This test suite validates that the wrapper validation framework correctly
identifies issues with frame count, color count, timing preservation, and
file integrity across different wrapper implementations.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from giflab.config import DEFAULT_VALIDATION_CONFIG
from giflab.wrapper_validation import (
    ValidationConfig,
    WrapperOutputValidator,
)
from giflab.wrapper_validation.integration import (
    add_validation_to_result,
    get_wrapper_type_from_class,
    validate_wrapper_apply_result,
)


class TestWrapperOutputValidator:
    """Test core validation functionality."""

    def test_validator_initialization_default_config(self):
        """Test validator initializes with default config."""
        validator = WrapperOutputValidator()
        assert validator.config == DEFAULT_VALIDATION_CONFIG

    def test_validator_initialization_custom_config(self):
        """Test validator initializes with custom config."""
        custom_config = ValidationConfig(FRAME_RATIO_TOLERANCE=0.05)
        validator = WrapperOutputValidator(custom_config)
        assert validator.config.FRAME_RATIO_TOLERANCE == 0.05

    @patch("giflab.wrapper_validation.core.extract_gif_metadata")
    def test_validate_frame_reduction_success(self, mock_extract):
        """Test successful frame reduction validation."""
        validator = WrapperOutputValidator()

        # Mock metadata for input (10 frames) and output (5 frames)
        input_metadata = Mock()
        input_metadata.orig_frames = 10
        output_metadata = Mock()
        output_metadata.orig_frames = 5

        mock_extract.side_effect = [input_metadata, output_metadata]

        result = validator.validate_frame_reduction(
            Path("input.gif"),
            Path("output.gif"),
            expected_ratio=0.5,
            wrapper_metadata={},
        )

        assert result.is_valid
        assert result.validation_type == "frame_count"
        assert result.expected["ratio"] == 0.5
        assert result.actual["ratio"] == 0.5
        assert result.error_message is None

    @patch("giflab.wrapper_validation.core.extract_gif_metadata")
    def test_validate_frame_reduction_failure(self, mock_extract):
        """Test frame reduction validation failure."""
        validator = WrapperOutputValidator()

        # Mock metadata for input (10 frames) and output (8 frames) - should be 5 for 0.5 ratio
        input_metadata = Mock()
        input_metadata.orig_frames = 10
        output_metadata = Mock()
        output_metadata.orig_frames = 8

        mock_extract.side_effect = [input_metadata, output_metadata]

        result = validator.validate_frame_reduction(
            Path("input.gif"),
            Path("output.gif"),
            expected_ratio=0.5,
            wrapper_metadata={},
        )

        assert not result.is_valid
        assert result.validation_type == "frame_count"
        assert "differs from expected" in result.error_message
        assert (
            result.details["ratio_difference"]
            > DEFAULT_VALIDATION_CONFIG.FRAME_RATIO_TOLERANCE
        )

    @patch("giflab.wrapper_validation.core.extract_gif_metadata")
    @patch.object(WrapperOutputValidator, "_count_unique_colors")
    def test_validate_color_reduction_success(self, mock_count_colors, mock_extract):
        """Test successful color reduction validation."""
        validator = WrapperOutputValidator()

        # Mock metadata for input (256 colors)
        input_metadata = Mock()
        input_metadata.orig_n_colors = 256
        mock_extract.return_value = input_metadata

        # Mock output with 32 colors (within tolerance of expected 32)
        mock_count_colors.return_value = 32

        result = validator.validate_color_reduction(
            Path("input.gif"),
            Path("output.gif"),
            expected_colors=32,
            wrapper_metadata={},
        )

        assert result.is_valid
        assert result.validation_type == "color_count"
        assert result.expected == 32
        assert result.actual == 32

    @patch("giflab.wrapper_validation.core.extract_gif_metadata")
    @patch.object(WrapperOutputValidator, "_count_unique_colors")
    def test_validate_color_reduction_failure_too_many_colors(
        self, mock_count_colors, mock_extract
    ):
        """Test color reduction validation failure - too many colors."""
        validator = WrapperOutputValidator()

        # Mock metadata for input (256 colors)
        input_metadata = Mock()
        input_metadata.orig_n_colors = 256
        mock_extract.return_value = input_metadata

        # Mock output with 50 colors (exceeds expected 32 + tolerance)
        mock_count_colors.return_value = 50

        result = validator.validate_color_reduction(
            Path("input.gif"),
            Path("output.gif"),
            expected_colors=32,
            wrapper_metadata={},
        )

        assert not result.is_valid
        assert "exceeds expected" in result.error_message

    @patch("giflab.wrapper_validation.core.extract_gif_metadata")
    def test_validate_timing_preservation_success(self, mock_extract):
        """Test successful timing preservation validation."""
        validator = WrapperOutputValidator()

        # Mock metadata with similar FPS
        input_metadata = Mock()
        input_metadata.orig_fps = 10.0
        output_metadata = Mock()
        output_metadata.orig_fps = 10.5  # Within 20% tolerance

        mock_extract.side_effect = [input_metadata, output_metadata]

        result = validator.validate_timing_preservation(
            Path("input.gif"), Path("output.gif"), wrapper_metadata={}
        )

        assert result.is_valid
        assert result.validation_type == "timing_preservation"

    @patch("giflab.wrapper_validation.core.extract_gif_metadata")
    def test_validate_timing_preservation_failure(self, mock_extract):
        """Test timing preservation validation failure."""
        validator = WrapperOutputValidator()

        # Mock metadata with significantly different FPS
        input_metadata = Mock()
        input_metadata.orig_fps = 10.0
        output_metadata = Mock()
        output_metadata.orig_fps = 30.0  # 200% increase, exceeds 20% tolerance

        mock_extract.side_effect = [input_metadata, output_metadata]

        result = validator.validate_timing_preservation(
            Path("input.gif"), Path("output.gif"), wrapper_metadata={}
        )

        assert not result.is_valid
        assert "FPS changed" in result.error_message

    def test_validate_file_integrity_missing_file(self):
        """Test file integrity validation for missing file."""
        validator = WrapperOutputValidator()

        result = validator.validate_file_integrity(
            Path("/nonexistent/file.gif"), wrapper_metadata={}
        )

        assert not result.is_valid
        assert result.validation_type == "file_integrity"
        assert "does not exist" in result.error_message

    def test_validate_file_integrity_small_file(self):
        """Test file integrity validation for file too small."""
        validator = WrapperOutputValidator()

        with tempfile.NamedTemporaryFile(suffix=".gif") as tmp_file:
            # Write minimal content (less than minimum required)
            tmp_file.write(b"GIF")
            tmp_file.flush()

            result = validator.validate_file_integrity(
                Path(tmp_file.name), wrapper_metadata={}
            )

            assert not result.is_valid
            assert "too small" in result.error_message


class TestValidationIntegration:
    """Test validation integration utilities."""

    def test_get_wrapper_type_from_variable_attribute(self):
        """Test wrapper type detection from VARIABLE attribute."""
        mock_wrapper = Mock()
        mock_wrapper.VARIABLE = "color_reduction"

        wrapper_type = get_wrapper_type_from_class(mock_wrapper)
        assert wrapper_type == "color_reduction"

    def test_get_wrapper_type_from_class_name(self):
        """Test wrapper type detection from class name."""
        mock_wrapper = Mock()
        mock_wrapper.__class__.__name__ = "GifsicleFrameReducer"
        del mock_wrapper.VARIABLE  # Ensure VARIABLE is not present

        wrapper_type = get_wrapper_type_from_class(mock_wrapper)
        assert wrapper_type == "frame_reduction"

    def test_get_wrapper_type_unknown(self):
        """Test wrapper type detection for unknown wrapper."""
        mock_wrapper = Mock()
        mock_wrapper.__class__.__name__ = "UnknownWrapper"
        del mock_wrapper.VARIABLE

        wrapper_type = get_wrapper_type_from_class(mock_wrapper)
        assert wrapper_type == "unknown"

    @patch("giflab.wrapper_validation.integration.WrapperOutputValidator")
    def test_add_validation_to_result_success(self, mock_validator_class):
        """Test adding validation results to wrapper result."""
        # Mock validator and validation results
        mock_validator = Mock()
        mock_validation = Mock()
        mock_validation.is_valid = True
        mock_validation.validation_type = "frame_count"
        mock_validation.expected = {"ratio": 0.5}
        mock_validation.actual = {"ratio": 0.5}
        mock_validation.error_message = None
        mock_validation.details = {}

        mock_validator.validate_wrapper_output.return_value = [mock_validation]
        mock_validator_class.return_value = mock_validator

        wrapper_result = {"render_ms": 1000, "engine": "test"}

        enhanced_result = add_validation_to_result(
            input_path=Path("input.gif"),
            output_path=Path("output.gif"),
            wrapper_params={"ratio": 0.5},
            wrapper_result=wrapper_result,
            wrapper_type="frame_reduction",
        )

        assert enhanced_result["validation_passed"] is True
        assert enhanced_result["validation_count"] == 1
        assert len(enhanced_result["validations"]) == 1
        assert enhanced_result["validations"][0]["is_valid"] is True

    @patch("giflab.wrapper_validation.integration.WrapperOutputValidator")
    def test_add_validation_to_result_failure(self, mock_validator_class):
        """Test adding failed validation results to wrapper result."""
        # Mock validator and failed validation results
        mock_validator = Mock()
        mock_validation = Mock()
        mock_validation.is_valid = False
        mock_validation.validation_type = "color_count"
        mock_validation.expected = 32
        mock_validation.actual = 50
        mock_validation.error_message = "Too many colors"
        mock_validation.details = {}

        mock_validator.validate_wrapper_output.return_value = [mock_validation]
        mock_validator_class.return_value = mock_validator

        wrapper_result = {"render_ms": 1000, "engine": "test"}

        enhanced_result = add_validation_to_result(
            input_path=Path("input.gif"),
            output_path=Path("output.gif"),
            wrapper_params={"colors": 32},
            wrapper_result=wrapper_result,
            wrapper_type="color_reduction",
        )

        assert enhanced_result["validation_passed"] is False
        assert enhanced_result["validation_count"] == 1
        assert enhanced_result["validations"][0]["is_valid"] is False
        assert enhanced_result["validations"][0]["error_message"] == "Too many colors"

    def test_add_validation_disabled(self):
        """Test validation is skipped when disabled."""
        config = ValidationConfig(ENABLE_WRAPPER_VALIDATION=False)
        wrapper_result = {"render_ms": 1000, "engine": "test"}

        enhanced_result = add_validation_to_result(
            input_path=Path("input.gif"),
            output_path=Path("output.gif"),
            wrapper_params={"ratio": 0.5},
            wrapper_result=wrapper_result,
            wrapper_type="frame_reduction",
            config=config,
        )

        # Should return original result unchanged
        assert enhanced_result == wrapper_result
        assert "validations" not in enhanced_result

    @patch("giflab.wrapper_validation.integration.add_validation_to_result")
    def test_validate_wrapper_apply_result(self, mock_add_validation):
        """Test convenience function for wrapper apply result validation."""
        mock_wrapper = Mock()
        mock_wrapper.VARIABLE = "frame_reduction"

        mock_add_validation.return_value = {"validation_passed": True}

        result = validate_wrapper_apply_result(
            wrapper_instance=mock_wrapper,
            input_path=Path("input.gif"),
            output_path=Path("output.gif"),
            params={"ratio": 0.5},
            result={"render_ms": 1000},
        )

        mock_add_validation.assert_called_once_with(
            input_path=Path("input.gif"),
            output_path=Path("output.gif"),
            wrapper_params={"ratio": 0.5},
            wrapper_result={"render_ms": 1000},
            wrapper_type="frame_reduction",
        )

        assert result["validation_passed"] is True


class TestValidationConfig:
    """Test validation configuration."""

    def test_default_config_values(self):
        """Test default configuration values are reasonable."""
        config = ValidationConfig()

        assert config.ENABLE_WRAPPER_VALIDATION is True
        assert config.FRAME_RATIO_TOLERANCE == 0.1
        assert config.MIN_FRAMES_REQUIRED >= 1
        assert config.FPS_TOLERANCE > 0
        assert config.FAIL_ON_VALIDATION_ERROR is False  # Don't break pipelines

    def test_config_validation_tolerances(self):
        """Test configuration validation for tolerance values."""
        # Valid configuration
        config = ValidationConfig(FRAME_RATIO_TOLERANCE=0.05)
        assert config.FRAME_RATIO_TOLERANCE == 0.05

        # Invalid tolerance - should raise ValueError
        with pytest.raises(
            ValueError, match="FRAME_RATIO_TOLERANCE must be between 0 and 1"
        ):
            ValidationConfig(FRAME_RATIO_TOLERANCE=1.5)

        with pytest.raises(
            ValueError, match="FRAME_RATIO_TOLERANCE must be between 0 and 1"
        ):
            ValidationConfig(FRAME_RATIO_TOLERANCE=-0.1)

    def test_config_validation_fps_bounds(self):
        """Test configuration validation for FPS bounds."""
        # Valid configuration
        config = ValidationConfig(MIN_FPS=0.5, MAX_FPS=30.0)
        assert config.MIN_FPS == 0.5
        assert config.MAX_FPS == 30.0

        # Invalid FPS bounds
        with pytest.raises(ValueError, match="MIN_FPS must be positive"):
            ValidationConfig(MIN_FPS=-1.0)

        with pytest.raises(ValueError, match="MAX_FPS must be greater than MIN_FPS"):
            ValidationConfig(MIN_FPS=30.0, MAX_FPS=15.0)

    def test_config_validation_file_size_limits(self):
        """Test configuration validation for file size limits."""
        # Valid configuration
        config = ValidationConfig(MIN_FILE_SIZE_BYTES=50, MAX_FILE_SIZE_MB=50.0)
        assert config.MIN_FILE_SIZE_BYTES == 50
        assert config.MAX_FILE_SIZE_MB == 50.0

        # Invalid file size limits
        with pytest.raises(ValueError, match="MIN_FILE_SIZE_BYTES must be positive"):
            ValidationConfig(MIN_FILE_SIZE_BYTES=-1)

        with pytest.raises(ValueError, match="MAX_FILE_SIZE_MB must be positive"):
            ValidationConfig(MAX_FILE_SIZE_MB=-1.0)


@pytest.mark.external_tools
class TestWrapperValidationIntegration:
    """Integration tests with actual wrapper classes (requires external tools)."""

    @pytest.fixture
    def test_gif(self):
        """Path to a simple test GIF fixture."""
        return Path(__file__).parent / "fixtures" / "simple_4frame.gif"

    def test_gifsicle_color_reducer_validation_integration(self, test_gif):
        """Test validation integration with GifsicleColorReducer."""
        from giflab.tool_wrappers import GifsicleColorReducer

        wrapper = GifsicleColorReducer()
        if not wrapper.available():
            pytest.skip("Gifsicle not available")

        with tempfile.NamedTemporaryFile(suffix=".gif") as tmp_file:
            output_path = Path(tmp_file.name)

            # Test with reasonable parameters
            result = wrapper.apply(test_gif, output_path, params={"colors": 16})

            # Check that validation results were added
            assert "validations" in result
            assert "validation_passed" in result
            assert "validation_count" in result
            assert isinstance(result["validations"], list)

            # Should have multiple validations (file integrity, timing, color count)
            assert result["validation_count"] >= 2

            # Check validation structure
            for validation in result["validations"]:
                assert "is_valid" in validation
                assert "validation_type" in validation
                assert "expected" in validation
                assert "actual" in validation
