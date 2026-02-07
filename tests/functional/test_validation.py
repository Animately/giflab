"""Tests for input validation functionality."""

from pathlib import Path

import pytest
from giflab.input_validation import (
    ValidationError,
    sanitize_filename,
    validate_config_paths,
    validate_file_extension,
    validate_output_path,
    validate_path_security,
    validate_raw_dir,
    validate_worker_count,
)


class TestValidateRawDir:
    """Tests for validate_raw_dir function."""

    def test_validate_raw_dir_success(self, tmp_path):
        """Test successful RAW_DIR validation."""
        # Create a test directory with a GIF file
        test_dir = tmp_path / "test_raw"
        test_dir.mkdir()
        (test_dir / "test.gif").write_text("fake gif content")

        result = validate_raw_dir(test_dir)
        assert result == test_dir
        assert result.exists()
        assert result.is_dir()

    def test_validate_raw_dir_empty_path(self):
        """Test validation with empty path."""
        with pytest.raises(ValidationError, match="RAW_DIR cannot be empty"):
            validate_raw_dir("")

    def test_validate_raw_dir_nonexistent(self, tmp_path):
        """Test validation with non-existent directory."""
        nonexistent = tmp_path / "nonexistent"

        with pytest.raises(ValidationError, match="RAW_DIR does not exist"):
            validate_raw_dir(nonexistent)


class TestValidatePathSecurity:
    """Tests for validate_path_security function."""

    def test_validate_path_security_success(self, tmp_path):
        """Test successful path security validation."""
        test_path = tmp_path / "safe_path"
        result = validate_path_security(test_path)
        assert result == test_path

    def test_validate_path_security_dangerous_chars(self):
        """Test validation with dangerous characters."""
        dangerous_paths = [
            "/path/with;semicolon",
            "/path/with&ampersand",
            "/path/with|pipe",
            "/path/with`backtick",
            "/path/with$dollar",
            "/path/with$(command)",
            "/path/with\nnewline",
            "/path/with\rcarriage",
        ]

        for dangerous_path in dangerous_paths:
            with pytest.raises(
                ValidationError, match="potentially dangerous characters"
            ):
                validate_path_security(dangerous_path)

    def test_validate_path_security_traversal(self):
        """Test validation with path traversal attempts."""
        traversal_paths = [
            "../../../etc/passwd",
            "safe/../../../etc/passwd",
            "/path/../../../etc/passwd",
        ]

        for traversal_path in traversal_paths:
            with pytest.raises(ValidationError, match="directory traversal"):
                validate_path_security(traversal_path)


class TestValidateOutputPath:
    """Tests for validate_output_path function."""

    def test_validate_output_path_success(self, tmp_path):
        """Test successful output path validation."""
        output_path = tmp_path / "output.csv"
        result = validate_output_path(output_path)
        assert result == output_path

    def test_validate_output_path_no_create_parent(self, tmp_path):
        """Test output path validation without parent creation."""
        output_path = tmp_path / "nonexistent" / "output.csv"
        with pytest.raises(ValidationError, match="Parent directory does not exist"):
            validate_output_path(output_path, create_parent=False)


class TestValidateWorkerCount:
    """Tests for validate_worker_count function."""

    def test_validate_worker_count_success(self):
        """Test successful worker count validation."""
        assert validate_worker_count(4) == 4
        assert validate_worker_count(0) == 0

    def test_validate_worker_count_negative(self):
        """Test validation with negative worker count."""
        with pytest.raises(ValidationError, match="cannot be negative"):
            validate_worker_count(-1)


class TestValidateFileExtension:
    """Tests for validate_file_extension function."""

    def test_validate_file_extension_success(self):
        """Test successful file extension validation."""
        path = Path("test.gif")
        result = validate_file_extension(path, [".gif", ".GIF"])
        assert result == path

    def test_validate_file_extension_invalid(self):
        """Test validation with invalid extension."""
        path = Path("test.txt")
        with pytest.raises(ValidationError, match="Invalid file extension"):
            validate_file_extension(path, [".gif", ".png"])


class TestValidateConfigPaths:
    """Tests for validate_config_paths function."""

    def test_validate_config_paths_success(self, tmp_path):
        """Test successful config paths validation."""
        config = {
            "RAW_DIR": tmp_path / "raw",
            "OUTPUT_DIR": tmp_path / "output",
            "SOME_PATH": tmp_path / "path",
            "OTHER_VALUE": "not_a_path",
        }

        result = validate_config_paths(config)
        assert "RAW_DIR" in result
        assert "OUTPUT_DIR" in result
        assert "SOME_PATH" in result
        assert "OTHER_VALUE" not in result


class TestSanitizeFilename:
    """Tests for sanitize_filename function."""

    def test_sanitize_filename_success(self):
        """Test successful filename sanitization."""
        result = sanitize_filename("normal_filename.txt")
        assert result == "normal_filename.txt"

    def test_sanitize_filename_invalid_chars(self):
        """Test sanitization with invalid characters."""
        result = sanitize_filename('file<>:"|?*name.txt')
        assert result == "file_______name.txt"


class TestValidationError:
    """Tests for ValidationError exception."""

    def test_validation_error_inheritance(self):
        """Test that ValidationError inherits from ValueError."""
        error = ValidationError("test message")
        assert isinstance(error, ValueError)
        assert str(error) == "test message"
