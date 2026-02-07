"""Tests for AnimatelyAdvancedLossyCompressor and PNG sequence optimization.

Trimmed to ~20 tests covering distinct failure modes and public API surface.
Internal helper functions (_extract_frame_timing, _generate_frame_list, etc.)
are tested indirectly through integration/end-to-end tests.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
from PIL import Image, ImageDraw

from giflab.lossy import (
    _execute_animately_advanced,
    _extract_gif_metadata,
    _generate_json_config,
    _setup_png_sequence_directory,
    _validate_animately_availability,
    compress_with_animately_advanced_lossy,
)
from giflab.tool_wrappers import AnimatelyAdvancedLossyCompressor


def create_test_gif(path: Path, frames: int = 5, size: tuple = (50, 50)) -> None:
    """Create a simple test GIF for testing purposes."""
    images = []
    for i in range(frames):
        img = Image.new("RGB", size, color=(i * 50 % 255, 100, 150))
        draw = ImageDraw.Draw(img)
        draw.rectangle([i * 5, i * 5, i * 5 + 10, i * 5 + 10], fill=(255, 255, 255))
        images.append(img)

    images[0].save(path, save_all=True, append_images=images[1:], duration=200, loop=0)


class TestAnimatelyAdvancedLossyCompressor:
    """Test the AnimatelyAdvancedLossyCompressor public API."""

    def test_tool_registration(self):
        """Test that the tool is properly registered with correct attributes."""
        tool = AnimatelyAdvancedLossyCompressor()

        assert tool.NAME == "animately-advanced-lossy"
        assert tool.COMBINE_GROUP == "animately"
        assert tool.VARIABLE == "lossy_compression"
        assert hasattr(tool, "available")
        assert hasattr(tool, "version")
        assert hasattr(tool, "apply")

    def test_available_method(self):
        """Test the availability check."""
        with patch("giflab.tool_wrappers._is_executable") as mock_is_executable:
            mock_is_executable.return_value = True
            tool = AnimatelyAdvancedLossyCompressor()
            assert tool.available() is True

            mock_is_executable.return_value = False
            assert tool.available() is False

    def test_version_method(self):
        """Test version retrieval and fallback on failure."""
        with patch("giflab.tool_wrappers.get_animately_version") as mock_version:
            mock_version.return_value = "1.1.20.0"
            tool = AnimatelyAdvancedLossyCompressor()
            assert tool.version() == "1.1.20.0"

            mock_version.side_effect = Exception("Failed")
            assert tool.version() == "unknown"

    def test_apply_missing_lossy_level(self):
        """Test that apply raises error when lossy_level is missing."""
        tool = AnimatelyAdvancedLossyCompressor()

        with pytest.raises(ValueError, match="params must include 'lossy_level'"):
            tool.apply(Path("input.gif"), Path("output.gif"), params={})

        with pytest.raises(ValueError, match="params must include 'lossy_level'"):
            tool.apply(Path("input.gif"), Path("output.gif"), params=None)

    @patch("giflab.lossy.compress_with_animately_advanced_lossy")
    def test_apply_basic_params(self, mock_compress):
        """Test apply delegates to compress function with correct params."""
        mock_compress.return_value = {"render_ms": 100, "engine": "animately-advanced"}

        tool = AnimatelyAdvancedLossyCompressor()
        result = tool.apply(
            Path("input.gif"), Path("output.gif"), params={"lossy_level": 60}
        )

        mock_compress.assert_called_once_with(
            Path("input.gif"),
            Path("output.gif"),
            lossy_level=60,
            color_keep_count=None,
            png_sequence_dir=None,
        )
        assert result["engine"] == "animately-advanced"


class TestCompressWithAnimatelyAdvancedLossy:
    """Test the core compress_with_animately_advanced_lossy function."""

    @patch("giflab.lossy._is_executable")
    def test_animately_not_available(self, mock_is_executable):
        """Test error when Animately is not available."""
        mock_is_executable.return_value = False

        with pytest.raises(RuntimeError, match="Animately launcher not found"):
            compress_with_animately_advanced_lossy(
                Path("input.gif"), Path("output.gif"), lossy_level=60
            )

    @patch("giflab.lossy.subprocess.run")
    @patch("giflab.lossy.export_png_sequence")
    @patch("giflab.lossy.extract_gif_metadata")
    @patch("giflab.lossy._is_executable")
    @patch("giflab.lossy.validate_path_security")
    def test_successful_compression_without_provided_sequence(
        self, mock_validate, mock_is_executable, mock_extract, mock_export, mock_run
    ):
        """Test successful compression when creating new PNG sequence."""
        mock_is_executable.return_value = True
        mock_validate.side_effect = lambda x: x

        mock_metadata = Mock()
        mock_metadata.orig_frames = 3
        mock_metadata.orig_n_colors = 128
        mock_extract.return_value = mock_metadata

        mock_export.return_value = {
            "frame_count": 3,
            "render_ms": 10,
            "engine": "imagemagick",
        }

        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.gif"
            output_path.touch()

            png_dir = Path(tmpdir) / "animately_png_test"
            png_dir.mkdir()
            for i in range(3):
                (png_dir / f"frame_{i:04d}.png").touch()

            with patch("tempfile.mkdtemp", return_value=str(png_dir)):
                with patch("PIL.Image.open") as mock_pil:
                    mock_img = Mock()
                    mock_img.seek = Mock()
                    mock_img.info = {"duration": 100}
                    mock_pil.return_value.__enter__.return_value = mock_img

                    result = compress_with_animately_advanced_lossy(
                        Path("input.gif"),
                        output_path,
                        lossy_level=60,
                        color_keep_count=32,
                    )

        assert result["engine"] == "animately-advanced"
        assert result["lossy_level"] == 60
        assert result["color_keep_count"] == 32
        assert result["frames_processed"] == 3
        assert "png_sequence_metadata" in result
        assert "json_config_path" in result

    @patch("giflab.lossy.subprocess.run")
    @patch("giflab.lossy.extract_gif_metadata")
    @patch("giflab.lossy._is_executable")
    @patch("giflab.lossy.validate_path_security")
    def test_successful_compression_with_provided_sequence(
        self, mock_validate, mock_is_executable, mock_extract, mock_run
    ):
        """Test successful compression when PNG sequence is provided by previous step."""
        mock_is_executable.return_value = True
        mock_validate.side_effect = lambda x: x

        mock_metadata = Mock()
        mock_metadata.orig_frames = 3
        mock_metadata.orig_n_colors = 128
        mock_extract.return_value = mock_metadata

        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.gif"
            output_path.touch()

            png_dir = Path(tmpdir) / "provided_png_sequence"
            png_dir.mkdir()
            for i in range(3):
                (png_dir / f"frame_{i:04d}.png").touch()

            with patch("PIL.Image.open") as mock_pil:
                mock_img = Mock()
                mock_img.seek = Mock()
                mock_img.info = {"duration": 100}
                mock_pil.return_value.__enter__.return_value = mock_img

                result = compress_with_animately_advanced_lossy(
                    Path("input.gif"),
                    output_path,
                    lossy_level=60,
                    color_keep_count=32,
                    png_sequence_dir=png_dir,
                )

        assert result["engine"] == "animately-advanced"
        assert result["png_sequence_metadata"]["engine"] == "provided_by_previous_step"
        assert result["png_sequence_metadata"]["render_ms"] == 0


class TestPNGSequenceOptimization:
    """Test PNG sequence optimization in pipeline generation."""

    def test_pipeline_generation_includes_advanced_lossy(self):
        """Test that pipeline generation includes the new advanced lossy compressor."""
        from giflab.dynamic_pipeline import generate_all_pipelines

        all_pipelines = generate_all_pipelines()
        advanced_pipelines = [
            p
            for p in all_pipelines
            if any(
                step.tool_cls.__name__ == "AnimatelyAdvancedLossyCompressor"
                for step in p.steps
            )
        ]

        assert (
            len(advanced_pipelines) > 0
        ), "Should have pipelines with AnimatelyAdvancedLossyCompressor"
        assert (
            len(advanced_pipelines) >= 5
        ), f"Should have multiple advanced pipelines, found {len(advanced_pipelines)}"


class TestBoundaryAndEdgeCases:
    """Test boundary conditions and edge cases."""

    def test_lossy_level_clamping(self):
        """Test that lossy level is clamped to valid range in JSON config."""
        with tempfile.TemporaryDirectory() as temp_dir:
            png_dir = Path(temp_dir)
            frame_files = [{"png": "/tmp/frame_001.png", "delay": 100}]

            # Test clamping high value
            config_path = _generate_json_config(
                png_dir, lossy_level=150, color_keep_count=None, frame_files=frame_files
            )
            with open(config_path) as f:
                config = json.load(f)
            assert config["lossy"] == 100  # Clamped to max

            # Test clamping negative value
            config_path2 = _generate_json_config(
                png_dir, lossy_level=-10, color_keep_count=None, frame_files=frame_files
            )
            with open(config_path2) as f:
                config2 = json.load(f)
            assert config2["lossy"] == 0  # Clamped to min

    def test_provided_directory_no_png_files(self):
        """Test error when provided directory has no PNG files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            png_dir = Path(temp_dir) / "png_sequence"
            png_dir.mkdir()

            with pytest.raises(RuntimeError, match="contains no frames"):
                _setup_png_sequence_directory(png_dir, Path("input.gif"), 2)

    def test_invalid_gif_metadata(self):
        """Test handling of invalid GIF files for metadata extraction."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_gif = Path(tmpdir) / "fake.gif"
            fake_gif.write_text("not a gif")

            with pytest.raises(RuntimeError):
                _extract_gif_metadata(fake_gif)

    @patch("giflab.lossy.subprocess.run")
    @patch("giflab.lossy.validate_path_security")
    def test_execution_timeout(self, mock_validate, mock_run):
        """Test Animately execution timeout."""
        from subprocess import TimeoutExpired

        mock_validate.return_value = Path("/safe/output.gif")
        mock_run.side_effect = TimeoutExpired("animately", 30)

        with pytest.raises(RuntimeError, match="timed out after"):
            _execute_animately_advanced(
                "/usr/bin/animately", Path("/tmp/config.json"), Path("output.gif")
            )


class TestEndToEndIntegration:
    """Test the main compress_with_animately_advanced_lossy function end-to-end."""

    @patch("giflab.lossy._process_advanced_lossy")
    @patch("giflab.lossy._setup_png_sequence_directory")
    @patch("giflab.lossy._extract_gif_metadata")
    @patch("giflab.lossy._validate_animately_availability")
    def test_successful_compression_with_provided_png(
        self, mock_validate, mock_metadata, mock_setup, mock_process
    ):
        """Test successful compression using provided PNG sequence."""
        mock_validate.return_value = "/usr/bin/animately"
        mock_metadata.return_value = (5, 256)
        mock_setup.return_value = (
            Path("/provided/png_seq"),
            {"frame_count": 5, "engine": "provided_by_previous_step"},
            True,
        )
        mock_process.return_value = {
            "render_ms": 3000,
            "engine": "animately-advanced",
            "lossy_level": 80,
        }

        result = compress_with_animately_advanced_lossy(
            input_path=Path("input.gif"),
            output_path=Path("output.gif"),
            lossy_level=80,
            color_keep_count=32,
            png_sequence_dir=Path("/provided/png_seq"),
        )

        assert result["render_ms"] == 3000
        assert result["engine"] == "animately-advanced"
        mock_process.assert_called_once()

    @patch("giflab.lossy.rmtree")
    @patch("giflab.lossy._process_advanced_lossy")
    @patch("giflab.lossy._setup_png_sequence_directory")
    @patch("giflab.lossy._extract_gif_metadata")
    @patch("giflab.lossy._validate_animately_availability")
    def test_successful_compression_with_cleanup(
        self, mock_validate, mock_metadata, mock_setup, mock_process, mock_rmtree
    ):
        """Test successful compression with automatic temp dir cleanup."""
        mock_validate.return_value = "/usr/bin/animately"
        mock_metadata.return_value = (3, 128)

        temp_dir = Path("/tmp/animately_png_12345")
        mock_setup.return_value = (
            temp_dir,
            {"frame_count": 3, "engine": "imagemagick"},
            False,
        )
        mock_process.return_value = {"render_ms": 2000, "engine": "animately-advanced"}

        result = compress_with_animately_advanced_lossy(
            input_path=Path("input.gif"), output_path=Path("output.gif"), lossy_level=60
        )

        assert result["render_ms"] == 2000
        mock_rmtree.assert_called_once_with(temp_dir)

    @patch("giflab.lossy.rmtree")
    @patch("giflab.lossy._process_advanced_lossy")
    @patch("giflab.lossy._setup_png_sequence_directory")
    @patch("giflab.lossy._extract_gif_metadata")
    @patch("giflab.lossy._validate_animately_availability")
    def test_cleanup_on_processing_failure(
        self, mock_validate, mock_metadata, mock_setup, mock_process, mock_rmtree
    ):
        """Test that cleanup happens even when processing fails."""
        mock_validate.return_value = "/usr/bin/animately"
        mock_metadata.return_value = (3, 128)

        temp_dir = Path("/tmp/animately_png_67890")
        mock_setup.return_value = (
            temp_dir,
            {"frame_count": 3, "engine": "imagemagick"},
            False,
        )
        mock_process.side_effect = RuntimeError("Processing failed")

        with pytest.raises(RuntimeError, match="Processing failed"):
            compress_with_animately_advanced_lossy(
                input_path=Path("input.gif"),
                output_path=Path("output.gif"),
                lossy_level=60,
            )

        mock_rmtree.assert_called_once_with(temp_dir)


@pytest.mark.slow
class TestIntegrationWithRealFiles:
    """Integration tests with real GIF files (marked as slow)."""

    @patch("giflab.external_engines.imagemagick._magick_binary")
    @patch("giflab.lossy._is_executable")
    @patch("giflab.lossy.DEFAULT_ENGINE_CONFIG")
    def test_end_to_end_png_sequence_optimization(
        self, mock_config, mock_is_executable, mock_magick_binary
    ):
        """Test the complete PNG sequence optimization flow with real files."""
        import shutil
        import tempfile
        from pathlib import Path

        mock_magick_binary.return_value = "/usr/bin/convert"
        mock_config.ANIMATELY_PATH = "/usr/bin/animately"
        mock_is_executable.return_value = True

        sample_gif = Path("test-workspace/samples/simple_4frame.gif")
        if not sample_gif.exists():
            sample_gif = Path("tests/fixtures/simple_4frame.gif")

        if sample_gif.exists():
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                input_gif = temp_path / "input.gif"
                output_gif = temp_path / "output.gif"
                png_sequence_dir = temp_path / "png_frames"
                png_sequence_dir.mkdir(parents=True, exist_ok=True)

                shutil.copy(sample_gif, input_gif)

                with patch("giflab.lossy.subprocess.run") as mock_run, patch(
                    "giflab.lossy.validate_path_security"
                ) as mock_validate, patch(
                    "giflab.external_engines.imagemagick.run_command"
                ) as mock_run_command, patch(
                    "giflab.lossy._extract_frame_timing"
                ) as mock_frame_timing:
                    mock_result = Mock()
                    mock_result.stderr = None
                    mock_run.return_value = mock_result
                    mock_validate.side_effect = lambda x: x
                    mock_frame_timing.return_value = [100, 100, 100, 100]

                    for i in range(4):
                        (png_sequence_dir / f"frame_{i:04d}.png").touch()

                    def mock_png_export(*args, **kwargs):
                        return {
                            "render_ms": 100,
                            "engine": "imagemagick",
                            "command": ["convert", "input.gif", "output.png"],
                            "kilobytes": 5.0,
                            "frame_count": 4,
                            "frame_pattern": "frame_%04d.png",
                        }

                    mock_run_command.side_effect = mock_png_export

                    def create_output_file(*args, **kwargs):
                        output_gif.touch()
                        return mock_result

                    mock_run.side_effect = create_output_file

                    try:
                        result = compress_with_animately_advanced_lossy(
                            input_path=input_gif,
                            output_path=output_gif,
                            lossy_level=60,
                            color_keep_count=32,
                            png_sequence_dir=png_sequence_dir,
                        )

                        assert result["engine"] == "animately-advanced"
                        assert result["lossy_level"] == 60
                        assert result["color_keep_count"] == 32
                        assert "render_ms" in result
                        assert "png_sequence_metadata" in result
                        assert "json_config_path" in result

                        png_metadata = result["png_sequence_metadata"]
                        assert png_metadata["frame_count"] > 0
                        assert "render_ms" in png_metadata

                        json_path = result["json_config_path"]
                        assert json_path
                        assert "animately_config.json" in str(json_path)

                        with open(json_path) as f:
                            config = json.load(f)

                        assert config["lossy"] == 60
                        assert config["colors"] == 32
                        assert "frames" in config
                        assert len(config["frames"]) > 0

                        for frame in config["frames"]:
                            assert "png" in frame
                            assert "delay" in frame
                            assert frame["delay"] >= 20
                            assert Path(frame["png"]).suffix == ".png"

                    except Exception as e:
                        if "ImageMagick" in str(e) or "convert" in str(e):
                            pytest.skip(
                                "ImageMagick not available for integration test"
                            )
                        else:
                            raise
        else:
            pytest.skip("No sample GIF files available for integration test")


class TestAnimatelyAdvancedLossyFast:
    """Fast tests covering key paths with real GIF fixtures."""

    def test_extract_gif_metadata_with_real_gif(self):
        """Test GIF metadata extraction with a real GIF file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gif_path = Path(tmpdir) / "test.gif"
            create_test_gif(gif_path, frames=3, size=(100, 100))

            total_frames, original_colors = _extract_gif_metadata(gif_path)

            assert isinstance(total_frames, int)
            assert isinstance(original_colors, int)
            assert total_frames == 3
            assert original_colors > 0

    def test_compress_function_delegates_to_process(self, fast_compress):
        """Test the main public function delegates correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.gif"
            output_path = Path(tmpdir) / "output.gif"

            create_test_gif(input_path, frames=3)

            with patch("giflab.lossy._process_advanced_lossy") as mock_process:
                mock_process.return_value = {
                    "render_ms": 150,
                    "engine": "animately-advanced",
                    "command": ["animately", "advanced", "args"],
                    "kilobytes": 25.5,
                    "lossy_level": 70,
                }

                result = compress_with_animately_advanced_lossy(
                    input_path=input_path, output_path=output_path, lossy_level=70
                )

                assert result["engine"] == "animately-advanced"
                assert result["lossy_level"] == 70
                assert result["render_ms"] == 150
                mock_process.assert_called_once()

    def test_animately_execution_failure(self):
        """Test handling of animately execution failure with non-zero exit code."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            output_path = Path(tmpdir) / "output.gif"
            png_dir = Path(tmpdir) / "frames"

            config_path.write_text('{"frames": []}')
            png_dir.mkdir()

            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(
                    returncode=1, stderr="Animately error"
                )

                with pytest.raises(
                    RuntimeError, match="Animately advanced lossy execution failed"
                ):
                    _execute_animately_advanced("animately", config_path, output_path)


if __name__ == "__main__":
    pytest.main([__file__])
