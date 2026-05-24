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
        """Control chars (newline / carriage return) must still be rejected.

        Shell metacharacters such as ``;``, ``&``, ``|``, ``$``, and backticks
        are *no longer* on the blacklist: giflab pipes paths through
        ``subprocess.run(cmd_list, shell=False)`` everywhere, so the args are
        passed straight to ``execve`` and never re-parsed by a shell. The only
        characters that remain unambiguously dangerous in a path string are
        control characters that can corrupt logs / break downstream tooling.
        """
        dangerous_paths = [
            "/path/with\nnewline",
            "/path/with\rcarriage",
        ]

        for dangerous_path in dangerous_paths:
            with pytest.raises(
                ValidationError, match="potentially dangerous characters"
            ):
                validate_path_security(dangerous_path)

    def test_validate_path_security_null_byte(self):
        """Null bytes are rejected separately (they truncate C strings)."""
        with pytest.raises(ValidationError, match="null bytes"):
            validate_path_security("/path/with\x00null")

    def test_validate_path_security_accepts_cosmetic_chars(self, tmp_path):
        """Legitimate filenames with cosmetic punctuation must pass.

        Real-world GIFs (e.g. downloaded from HubSpot / generic CDN URLs)
        regularly contain parens, spaces, %-encoded sequences, ampersands,
        equals signs and query-string punctuation in the *filename*. None of
        these characters can hurt anything once we stop running paths through
        a shell — see the module docstring on ``validate_path_security``.
        """
        cosmetic_filenames = [
            # Parens — Apple-style "file (1).gif" duplicates
            "file (1).gif",
            "Untitled-1-2 (copy).gif",
            # Spaces
            "my file with spaces.gif",
            # %-encoded sequences (re-encoded URLs)
            "name%20with%20spaces.gif",
            "image%2Ffoo.gif",
            # Shell-meta-chars that aren't actually dangerous off-shell
            "Untitled-1-2.gif_upscale=true&width=1600&name=foo.gif",
            "thing;other.gif",
            "thing|pipe.gif",
            "thing$dollar.gif",
            "thing`backtick.gif",
            "thing$(command).gif",
            # URL query-style punctuation
            "name=value&other=thing.gif",
            "file?query=1.gif",
            "file+plus.gif",
            "file@host.gif",
            "file#fragment.gif",
            "file~tilde.gif",
            # The exact path from audit/sweep.csv that triggered the failure
            "https___hs-8497520.f.hubspotemail.net_hub_8497520_hubfs_"
            "Untitled-1-2.gif_upscale=true&width=1600&upscale=true&"
            "name=Untitled-1-2.gif",
        ]

        for name in cosmetic_filenames:
            full = tmp_path / name
            result = validate_path_security(full)
            assert result == full, f"Should accept cosmetic filename: {name!r}"

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
