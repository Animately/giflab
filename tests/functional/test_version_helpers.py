"""Tests for version helper functions with stub binaries."""

import subprocess
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from giflab.lossy import (
    LossyEngine,
    get_animately_version,
    get_engine_version,
    get_gifsicle_version,
)
from giflab.system_tools import _extract_version, _run_version_cmd


class TestVersionHelpers:
    """Test version helper functions with stub binaries."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for stub binaries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def create_stub_binary(self, path: Path, script_content: str) -> Path:
        """Create a stub binary that responds to version commands.

        Args:
            path: Path where to create the stub binary
            script_content: Shell script content for the stub

        Returns:
            Path to the created stub binary
        """
        path.write_text(f"#!/bin/bash\n{script_content}\n")
        path.chmod(0o755)
        return path

    def test_get_gifsicle_version_success(self, temp_dir):
        """Test successful gifsicle version detection."""
        stub_binary = temp_dir / "gifsicle"
        self.create_stub_binary(
            stub_binary,
            'if [[ "$1" == "--version" ]]; then echo "LCDF Gifsicle 1.94"; exit 0; fi',
        )

        with patch("giflab.lossy.DEFAULT_ENGINE_CONFIG") as mock_config:
            mock_config.GIFSICLE_PATH = str(stub_binary)

            version = get_gifsicle_version()
            assert version == "1.94"

    def test_get_gifsicle_version_not_found(self):
        """Test gifsicle version when binary not found."""
        with patch("giflab.lossy.DEFAULT_ENGINE_CONFIG") as mock_config:
            mock_config.GIFSICLE_PATH = "/nonexistent/gifsicle"

            with pytest.raises(RuntimeError, match="Gifsicle not found"):
                get_gifsicle_version()

    def test_get_gifsicle_version_command_fails(self, temp_dir):
        """Test gifsicle version when command fails."""
        stub_binary = temp_dir / "gifsicle"
        self.create_stub_binary(
            stub_binary,
            'if [[ "$1" == "--version" ]]; then echo "Error" >&2; exit 1; fi',
        )

        with patch("giflab.lossy.DEFAULT_ENGINE_CONFIG") as mock_config:
            mock_config.GIFSICLE_PATH = str(stub_binary)

            with pytest.raises(RuntimeError, match="Gifsicle version check failed"):
                get_gifsicle_version()

    def test_get_gifsicle_version_timeout(self, temp_dir):
        """Test gifsicle version when command times out."""
        stub_binary = temp_dir / "gifsicle"
        self.create_stub_binary(
            stub_binary,
            'if [[ "$1" == "--version" ]]; then sleep 15; fi',  # Sleep longer than timeout
        )

        with patch("giflab.lossy.DEFAULT_ENGINE_CONFIG") as mock_config:
            mock_config.GIFSICLE_PATH = str(stub_binary)

            with pytest.raises(RuntimeError, match="Gifsicle version check timed out"):
                get_gifsicle_version()

    def test_get_gifsicle_version_fallback(self, temp_dir):
        """Test gifsicle version fallback when parsing fails."""
        stub_binary = temp_dir / "gifsicle"
        self.create_stub_binary(
            stub_binary,
            'if [[ "$1" == "--version" ]]; then echo "Some weird output without version"; exit 0; fi',
        )

        with patch("giflab.lossy.DEFAULT_ENGINE_CONFIG") as mock_config:
            mock_config.GIFSICLE_PATH = str(stub_binary)

            version = get_gifsicle_version()
            assert version == "Some weird output without version"

    def test_get_animately_version_success_with_version_flag(self, temp_dir):
        """Test successful animately version detection with --version flag."""
        stub_binary = temp_dir / "animately"
        self.create_stub_binary(
            stub_binary,
            'if [[ "$1" == "--version" ]]; then echo "animately 1.2.3"; exit 0; fi',
        )

        with patch("giflab.lossy.DEFAULT_ENGINE_CONFIG") as mock_config:
            mock_config.ANIMATELY_PATH = str(stub_binary)

            version = get_animately_version()
            assert version == "1.2.3"

    def test_get_animately_version_success_with_help_flag(self, temp_dir):
        """Test animately version detection fallback to --help."""
        stub_binary = temp_dir / "animately"
        self.create_stub_binary(
            stub_binary,
            """
            if [[ "$1" == "--version" ]]; then exit 1; fi
            if [[ "$1" == "--help" ]]; then echo "animately version 2.0.1 - GIF processing tool"; exit 0; fi
            """,
        )

        with patch("giflab.lossy.DEFAULT_ENGINE_CONFIG") as mock_config:
            mock_config.ANIMATELY_PATH = str(stub_binary)

            version = get_animately_version()
            assert version == "2.0.1"

    def test_get_animately_version_fallback_to_generic(self, temp_dir):
        """Test animately version fallback to generic identifier."""
        stub_binary = temp_dir / "animately"
        self.create_stub_binary(
            stub_binary,
            """
            if [[ "$1" == "--version" ]]; then exit 1; fi
            if [[ "$1" == "--help" ]]; then echo "No version info here"; exit 0; fi
            """,
        )

        with patch("giflab.lossy.DEFAULT_ENGINE_CONFIG") as mock_config:
            mock_config.ANIMATELY_PATH = str(stub_binary)

            version = get_animately_version()
            assert version == "animately-engine"

    def test_get_animately_version_not_found(self):
        """Test animately version when binary not found."""
        with patch("giflab.lossy.DEFAULT_ENGINE_CONFIG") as mock_config:
            mock_config.ANIMATELY_PATH = "/nonexistent/animately"

            with pytest.raises(RuntimeError, match="Animately not found"):
                get_animately_version()

    def test_get_animately_version_timeout(self, temp_dir):
        """Test animately version when command times out."""
        stub_binary = temp_dir / "animately"
        self.create_stub_binary(
            stub_binary,
            'if [[ "$1" == "--version" ]]; then sleep 15; fi',  # Sleep longer than timeout
        )

        with patch("giflab.lossy.DEFAULT_ENGINE_CONFIG") as mock_config:
            mock_config.ANIMATELY_PATH = str(stub_binary)

            with pytest.raises(RuntimeError, match="Animately version check timed out"):
                get_animately_version()

    def test_get_engine_version_gifsicle(self, temp_dir):
        """Test get_engine_version for gifsicle."""
        stub_binary = temp_dir / "gifsicle"
        self.create_stub_binary(
            stub_binary,
            'if [[ "$1" == "--version" ]]; then echo "LCDF Gifsicle 1.95"; exit 0; fi',
        )

        with patch("giflab.lossy.DEFAULT_ENGINE_CONFIG") as mock_config:
            mock_config.GIFSICLE_PATH = str(stub_binary)

            version = get_engine_version(LossyEngine.GIFSICLE)
            assert version == "1.95"

    def test_get_engine_version_animately(self, temp_dir):
        """Test get_engine_version for animately."""
        stub_binary = temp_dir / "animately"
        self.create_stub_binary(
            stub_binary,
            'if [[ "$1" == "--version" ]]; then echo "animately 3.0.0"; exit 0; fi',
        )

        with patch("giflab.lossy.DEFAULT_ENGINE_CONFIG") as mock_config:
            mock_config.ANIMATELY_PATH = str(stub_binary)

            version = get_engine_version(LossyEngine.ANIMATELY)
            assert version == "3.0.0"

    def test_get_engine_version_unsupported_engine(self):
        """Test get_engine_version with unsupported engine."""
        # Create a fake enum value
        fake_engine = "FAKE_ENGINE"

        with pytest.raises(ValueError, match="Unsupported engine"):
            get_engine_version(fake_engine)


class TestSystemToolsVersionHelpers:
    """Test system tools version helper functions."""

    def test_extract_version_success(self):
        """Test successful version extraction."""
        output = "LCDF Gifsicle 1.94\nCopyright notice..."
        pattern = r"Gifsicle (\d+\.\d+)"

        version = _extract_version(output, pattern)
        assert version == "1.94"

    def test_extract_version_no_match(self):
        """Test version extraction when pattern doesn't match."""
        output = "Some output without version"
        pattern = r"version (\d+\.\d+)"

        version = _extract_version(output, pattern)
        assert version is None

    def test_extract_version_complex_pattern(self):
        """Test version extraction with complex pattern."""
        output = "ImageMagick 7.1.0-57 Q16 x86_64 https://imagemagick.org"
        pattern = r"ImageMagick (\d+\.\d+\.\d+-\d+)"

        version = _extract_version(output, pattern)
        assert version == "7.1.0-57"

    def test_run_version_cmd_success(self):
        """Test successful version command execution."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout="Tool version 2.3.4\n", stderr=""
            )

            version = _run_version_cmd(
                ["tool", "--version"], r"version (\d+\.\d+\.\d+)"
            )
            assert version == "2.3.4"

    def test_run_version_cmd_not_found(self):
        """Test version command when binary not found."""
        with patch("subprocess.run", side_effect=FileNotFoundError()):
            version = _run_version_cmd(
                ["nonexistent", "--version"], r"version (\d+\.\d+)"
            )
            assert version is None

    def test_run_version_cmd_timeout(self):
        """Test version command when it times out."""
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("cmd", 5)):
            version = _run_version_cmd(
                ["slow_tool", "--version"], r"version (\d+\.\d+)"
            )
            assert version is None

    def test_run_version_cmd_stderr_version(self):
        """Test version command that outputs version to stderr."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=1,  # Non-zero exit code
                stdout="",
                stderr="animately version 1.5.0\n",
            )

            version = _run_version_cmd(
                ["animately", "--version"], r"version (\d+\.\d+\.\d+)"
            )
            assert version == "1.5.0"

    def test_run_version_cmd_nonzero_exit_no_version(self):
        """Test version command with non-zero exit and no version found."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=1, stdout="Error: unknown option", stderr=""
            )

            version = _run_version_cmd(["tool", "--version"], r"version (\d+\.\d+)")
            assert version is None

    def test_run_version_cmd_generic_exception(self):
        """Test version command with generic exception."""
        with patch("subprocess.run", side_effect=Exception("Generic error")):
            version = _run_version_cmd(["tool", "--version"], r"version (\d+\.\d+)")
            assert version is None


@pytest.mark.fast
class TestVersionHelpersFast:
    """Fast version helper tests without subprocess calls."""

    def test_extract_version_edge_cases(self):
        """Test version extraction edge cases."""
        # Empty output
        assert _extract_version("", r"version (\d+\.\d+)") is None

        # Multiple matches - should return first
        output = "version 1.0 and version 2.0"
        assert _extract_version(output, r"version (\d+\.\d+)") == "1.0"

        # Case insensitive
        output = "VERSION 3.2.1"
        assert _extract_version(output, r"(?i)version (\d+\.\d+\.\d+)") == "3.2.1"

        # Complex version format
        output = "Tool v1.2.3-beta.4+build.567"
        pattern = r"v(\d+\.\d+\.\d+-[a-z]+\.\d+\+[a-z]+\.\d+)"
        assert _extract_version(output, pattern) == "1.2.3-beta.4+build.567"
