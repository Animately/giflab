"""Integration tests for compression engines (gifsicle, animately, and external engines).

Combines slow integration tests, fast mocked tests, and external engine helper tests.
"""

import subprocess
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from giflab.config import DEFAULT_ENGINE_CONFIG
from giflab.external_engines import (
    gifski_lossy_compress,
    imagemagick_color_reduce,
)
from giflab.external_engines.common import run_command
from giflab.lossy import (
    LossyEngine,
    _is_executable,
    apply_lossy_compression,
    compress_with_animately,
    compress_with_gifsicle,
)
from PIL import Image, ImageDraw


def create_test_gif(path: Path, frames: int = 5, size: tuple = (50, 50)) -> None:
    """Create a simple test GIF for testing purposes.

    Args:
        path: Path where to save the GIF
        frames: Number of frames in the GIF
        size: Size of each frame (width, height)
    """
    images = []
    for i in range(frames):
        # Create a simple colored square that changes color
        img = Image.new("RGB", size, color=(i * 50 % 255, 100, 150))
        draw = ImageDraw.Draw(img)
        # Add a simple shape that moves
        draw.rectangle([i * 5, i * 5, i * 5 + 10, i * 5 + 10], fill=(255, 255, 255))
        images.append(img)

    # Save as GIF
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=200,  # 200ms per frame
        loop=0,
    )


# ===========================================================================
# Slow integration tests (require actual engine binaries)
# ===========================================================================


@pytest.mark.slow
class TestEngineAvailability:
    """Test that both engines are available and properly configured."""

    def test_gifsicle_available(self):
        """Test that gifsicle is available and executable."""
        gifsicle_path = DEFAULT_ENGINE_CONFIG.GIFSICLE_PATH

        if not _is_executable(gifsicle_path):
            pytest.skip(f"gifsicle not found at {gifsicle_path}")

        try:
            result = subprocess.run(
                [gifsicle_path, "--version"], capture_output=True, text=True, timeout=9
            )
            assert result.returncode == 0, f"gifsicle not working: {result.stderr}"
            assert (
                "gifsicle" in result.stdout.lower()
            ), f"Unexpected version output: {result.stdout}"
        except FileNotFoundError:
            pytest.skip(f"gifsicle not found at {gifsicle_path}")
        except subprocess.TimeoutExpired:
            pytest.fail("gifsicle --version timed out")

    def test_animately_available(self):
        """Test that animately is available and executable."""
        animately_path = DEFAULT_ENGINE_CONFIG.ANIMATELY_PATH

        if not _is_executable(animately_path):
            pytest.skip(f"Animately not found at {animately_path}")

        try:
            result = subprocess.run(
                [animately_path, "--help"], capture_output=True, text=True, timeout=9
            )
            # Animately might return non-zero for --help, so check output instead
            assert (
                "--input" in result.stdout or "--input" in result.stderr
            ), f"Unexpected help output: {result.stdout}"
        except FileNotFoundError:
            pytest.skip(f"Animately not found at {animately_path}")
        except subprocess.TimeoutExpired:
            pytest.fail("animately --help timed out")


@pytest.mark.slow
class TestGifsicleIntegration:
    """Integration tests for gifsicle engine."""

    @pytest.fixture
    def test_gif(self):
        """Create a temporary test GIF."""
        with tempfile.TemporaryDirectory() as temp_dir:
            gif_path = Path(temp_dir) / "test.gif"
            create_test_gif(gif_path)
            yield gif_path

    @pytest.fixture
    def output_path(self):
        """Create a temporary output path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir) / "output.gif"

    def test_gifsicle_lossless_compression(self, test_gif, output_path):
        """Test gifsicle lossless compression."""
        gifsicle_path = DEFAULT_ENGINE_CONFIG.GIFSICLE_PATH

        # Check if gifsicle is available
        try:
            subprocess.run(
                [gifsicle_path, "--version"], capture_output=True, check=True, timeout=5
            )
        except (
            FileNotFoundError,
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
        ):
            pytest.skip(f"gifsicle not available at {gifsicle_path}")

        # Test compression
        result = compress_with_gifsicle(test_gif, output_path, lossy_level=0)

        # Verify results
        assert output_path.exists(), "Output file was not created"
        assert result["engine"] == "gifsicle"
        assert result["lossy_level"] == 0
        assert result["render_ms"] > 0
        assert "command" in result

        # Verify output is a valid GIF
        assert output_path.stat().st_size > 0, "Output file is empty"


@pytest.mark.slow
class TestAnimatelyIntegration:
    """Integration tests for animately engine."""

    @pytest.fixture
    def test_gif(self):
        """Create a temporary test GIF."""
        with tempfile.TemporaryDirectory() as temp_dir:
            gif_path = Path(temp_dir) / "test.gif"
            create_test_gif(gif_path)
            yield gif_path

    @pytest.fixture
    def output_path(self):
        """Create a temporary output path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir) / "output.gif"

    def test_animately_lossless_compression(self, test_gif, output_path):
        """Test animately lossless compression."""
        animately_path = DEFAULT_ENGINE_CONFIG.ANIMATELY_PATH

        # Check if animately is available
        if not _is_executable(animately_path):
            pytest.skip(f"Animately not found at {animately_path}")

        # Test compression
        result = compress_with_animately(test_gif, output_path, lossy_level=0)

        # Verify results
        assert output_path.exists(), "Output file was not created"
        assert result["engine"] == "animately"
        assert result["lossy_level"] == 0
        assert result["render_ms"] > 0
        assert "command" in result

        # Verify output is a valid GIF
        assert output_path.stat().st_size > 0, "Output file is empty"


@pytest.mark.slow
class TestHighLevelAPI:
    """Test the high-level API with both engines."""

    @pytest.fixture
    def test_gif(self):
        """Create a temporary test GIF."""
        with tempfile.TemporaryDirectory() as temp_dir:
            gif_path = Path(temp_dir) / "test.gif"
            create_test_gif(gif_path)
            yield gif_path

    def test_apply_lossy_compression_gifsicle(self, test_gif):
        """Test high-level API with gifsicle."""
        gifsicle_path = DEFAULT_ENGINE_CONFIG.GIFSICLE_PATH

        try:
            subprocess.run(
                [gifsicle_path, "--version"], capture_output=True, check=True, timeout=5
            )
        except Exception:
            pytest.skip(f"gifsicle not available at {gifsicle_path}")

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "output.gif"

            result = apply_lossy_compression(
                test_gif,
                output_path,
                lossy_level=0,
                frame_keep_ratio=1.0,
                engine=LossyEngine.GIFSICLE,
            )

            assert output_path.exists()
            assert result["engine"] == "gifsicle"


# ===========================================================================
# Fast integration tests (mocked engines, no binary dependencies)
# ===========================================================================


class TestEngineIntegrationFast:
    """Fast integration tests for compression engines using fast_compress fixture."""

    @patch("giflab.lossy.compress_with_gifsicle")
    def test_gifsicle_compression_fast(self, mock_gifsicle):
        """Test gifsicle compression with fast mocked implementation."""

        # Configure mock to copy file and return expected results
        def mock_compress(input_path, output_path, **kwargs):
            import shutil

            shutil.copyfile(input_path, output_path)
            return {
                "render_ms": 1,
                "engine": "noop",
                "command": "noop-copy",
                "ssim": 1.0,
                "lossy_level": kwargs.get("lossy_level", 0),
            }

        mock_gifsicle.side_effect = mock_compress

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "test_input.gif"
            output_path = Path(tmpdir) / "test_output.gif"

            # Create test GIF
            create_test_gif(input_path)

            # Test compression (should use mocked implementation)
            from giflab.lossy import compress_with_gifsicle

            result = compress_with_gifsicle(
                input_path=input_path, output_path=output_path, lossy_level=50
            )

            # Verify mock was called
            mock_gifsicle.assert_called_once_with(
                input_path=input_path, output_path=output_path, lossy_level=50
            )

            # Verify the fast fixture behavior
            assert output_path.exists()
            assert isinstance(result, dict)
            assert result["engine"] == "noop"
            assert result["render_ms"] == 1
            assert result["ssim"] == 1.0  # Perfect similarity in fast mode
            assert result["lossy_level"] == 50

    @patch("giflab.lossy.compress_with_gifsicle")
    def test_apply_lossy_compression_gifsicle_fast(self, mock_gifsicle):
        """Test apply_lossy_compression with gifsicle engine using fast fixture."""

        # Configure mock to copy file and return expected results
        def mock_compress(
            input_path,
            output_path,
            lossy_level=0,
            frame_keep_ratio=1.0,
            color_keep_count=None,
        ):
            import shutil

            shutil.copyfile(input_path, output_path)
            return {
                "render_ms": 1,
                "engine": "noop",
                "command": "noop-copy",
                "ssim": 1.0,
                "lossy_level": lossy_level,
                "frame_keep_ratio": frame_keep_ratio,
            }

        mock_gifsicle.side_effect = mock_compress

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "test_input.gif"
            output_path = Path(tmpdir) / "test_output.gif"

            create_test_gif(input_path)

            result = apply_lossy_compression(
                input_path=input_path,
                output_path=output_path,
                lossy_level=40,
                frame_keep_ratio=1.0,
                engine=LossyEngine.GIFSICLE,
            )

            # Check result structure
            assert isinstance(result, dict)
            assert "render_ms" in result
            assert "engine" in result
            assert result["engine"] == "noop"
            assert result["frame_keep_ratio"] == 1.0
            assert result["lossy_level"] == 40

    @patch("giflab.lossy.compress_with_animately")
    def test_apply_lossy_compression_animately_fast(self, mock_animately):
        """Test apply_lossy_compression with animately engine using fast fixture."""

        # Configure mock to copy file and return expected results
        def mock_compress(
            input_path,
            output_path,
            lossy_level=0,
            frame_keep_ratio=1.0,
            color_keep_count=None,
        ):
            import shutil

            shutil.copyfile(input_path, output_path)
            return {
                "render_ms": 1,
                "engine": "noop",
                "command": "noop-copy",
                "ssim": 1.0,
                "lossy_level": lossy_level,
                "frame_keep_ratio": frame_keep_ratio,
            }

        mock_animately.side_effect = mock_compress

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "test_input.gif"
            output_path = Path(tmpdir) / "test_output.gif"

            create_test_gif(input_path)

            result = apply_lossy_compression(
                input_path=input_path,
                output_path=output_path,
                lossy_level=60,
                frame_keep_ratio=1.0,
                engine=LossyEngine.ANIMATELY,
            )

            # Check result structure
            assert isinstance(result, dict)
            assert "render_ms" in result
            assert "engine" in result
            assert result["engine"] == "noop"
            assert result["frame_keep_ratio"] == 1.0
            assert result["lossy_level"] == 60

    @patch("giflab.lossy.compress_with_gifsicle")
    def test_compression_error_handling_fast(self, mock_gifsicle):
        """Test error handling in compression functions using fast fixture."""

        # Configure mock to raise FileNotFoundError for nonexistent file
        def mock_compress(input_path, output_path, **kwargs):
            if not Path(input_path).exists():
                raise FileNotFoundError(f"No such file: {input_path}")
            import shutil

            shutil.copyfile(input_path, output_path)
            return {
                "render_ms": 1,
                "engine": "noop",
                "command": "noop-copy",
                "ssim": 1.0,
                "lossy_level": kwargs.get("lossy_level", 0),
            }

        mock_gifsicle.side_effect = mock_compress

        with tempfile.TemporaryDirectory() as tmpdir:
            # Test with non-existent input file
            input_path = Path(tmpdir) / "nonexistent.gif"
            output_path = Path(tmpdir) / "test_output.gif"

            # The fast fixture should still handle this gracefully
            # by trying to copy a non-existent file, which should raise an error
            with pytest.raises((FileNotFoundError, OSError, RuntimeError)):
                compress_with_gifsicle(
                    input_path=input_path, output_path=output_path, lossy_level=50
                )


class TestEngineUtilsFast:
    """Fast tests for engine utility functions."""

    def test_engine_enum_values(self):
        """Test LossyEngine enum values."""
        assert LossyEngine.GIFSICLE.value == "gifsicle"
        assert LossyEngine.ANIMATELY.value == "animately"

        # Test that enum can be converted to string
        assert str(LossyEngine.GIFSICLE) == "LossyEngine.GIFSICLE"
        assert str(LossyEngine.ANIMATELY) == "LossyEngine.ANIMATELY"


# ===========================================================================
# Common utility tests
# ===========================================================================


def test_run_command_success():
    """Test run_command with a successful command."""
    with tempfile.NamedTemporaryFile(suffix=".txt") as tmp:
        output_path = Path(tmp.name)

        # Simple command that should succeed
        result = run_command(["echo", "hello"], engine="test", output_path=output_path)

        assert result["engine"] == "test"
        assert result["render_ms"] >= 0
        assert "echo hello" in result["command"]
        assert "kilobytes" in result


def test_run_command_failure():
    """Test run_command with a failing command."""
    with tempfile.NamedTemporaryFile(suffix=".txt") as tmp:
        output_path = Path(tmp.name)

        with pytest.raises(RuntimeError, match="test command failed"):
            run_command(
                ["false"],  # Command that always fails
                engine="test",
                output_path=output_path,
            )


# ===========================================================================
# External engine fixtures
# ===========================================================================


@pytest.fixture
def external_test_gif():
    """Path to a simple test GIF fixture."""
    return Path(__file__).parent.parent / "fixtures" / "simple_4frame.gif"


# ===========================================================================
# Parameter validation tests (no external tools required)
# ===========================================================================


class TestExternalEngineParameterValidation:
    """Test parameter validation without requiring external tools."""

    def test_imagemagick_color_reduce_invalid_colors(self, external_test_gif):
        """Test color reduction with invalid color count."""
        with tempfile.NamedTemporaryFile(suffix=".gif") as tmp:
            output_path = Path(tmp.name)

            with pytest.raises(ValueError, match="colors must be between 1 and 256"):
                imagemagick_color_reduce(external_test_gif, output_path, colors=300)

    def test_gifski_lossy_compress_invalid_quality(self, external_test_gif):
        """Test gifski lossy compression with invalid quality."""
        with tempfile.NamedTemporaryFile(suffix=".gif") as tmp:
            output_path = Path(tmp.name)

            with pytest.raises(ValueError, match="quality must be in 0–100"):
                gifski_lossy_compress(external_test_gif, output_path, quality=150)
