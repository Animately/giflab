"""Tests for giflab.lossy module."""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from giflab.lossy import (
    LossyEngine,
    apply_compression_with_all_params,
    apply_lossy_compression,
    compress_with_animately,
    compress_with_gifsicle,
    get_compression_estimate,
    validate_lossy_level,
)


class TestLossyEngine:
    """Tests for LossyEngine enum."""

    def test_engine_values(self):
        """Test that enum values are correct."""
        assert LossyEngine.GIFSICLE.value == "gifsicle"
        assert LossyEngine.ANIMATELY.value == "animately"


class TestValidateLossyLevel:
    """Tests for validate_lossy_level function."""

    def test_valid_levels(self):
        """Test that valid lossy levels pass validation."""
        for level in [0, 40, 120]:
            # Should not raise any exception
            validate_lossy_level(level, LossyEngine.GIFSICLE)
            validate_lossy_level(level, LossyEngine.ANIMATELY)

    def test_negative_level(self):
        """Test that negative lossy levels raise ValueError."""
        with pytest.raises(ValueError, match="must be non-negative"):
            validate_lossy_level(-1, LossyEngine.GIFSICLE)

    def test_invalid_level(self):
        """Test that invalid lossy levels raise ValueError."""
        with pytest.raises(ValueError, match="not in supported levels"):
            validate_lossy_level(99, LossyEngine.GIFSICLE)


class TestApplyLossyCompression:
    """Tests for apply_lossy_compression function."""

    def test_negative_lossy_level(self):
        """Test that negative lossy level raises ValueError."""
        with pytest.raises(ValueError, match="must be non-negative"):
            apply_lossy_compression(
                Path("input.gif"),
                Path("output.gif"),
                -1,
                1.0,  # frame_keep_ratio
                LossyEngine.GIFSICLE,
            )

    def test_missing_input_file(self):
        """Test that missing input file raises IOError."""
        with pytest.raises(IOError, match="Input file not found"):
            apply_lossy_compression(
                Path("nonexistent.gif"),
                Path("output.gif"),
                0,
                1.0,  # frame_keep_ratio
                LossyEngine.GIFSICLE,
            )

    @patch("giflab.lossy.compress_with_gifsicle")
    @patch("pathlib.Path.exists")
    def test_gifsicle_engine_dispatch(self, mock_exists, mock_compress):
        """Test that gifsicle engine is called correctly."""
        mock_exists.return_value = True
        mock_compress.return_value = {"render_ms": 100}

        input_path = Path("input.gif")
        output_path = Path("output.gif")

        result = apply_lossy_compression(
            input_path, output_path, 40, 1.0, LossyEngine.GIFSICLE  # frame_keep_ratio
        )

        mock_compress.assert_called_once_with(input_path, output_path, 40, 1.0)
        assert result == {"render_ms": 100}

    @patch("giflab.lossy.compress_with_animately")
    @patch("pathlib.Path.exists")
    def test_animately_engine_dispatch(self, mock_exists, mock_compress):
        """Test that animately engine is called correctly."""
        mock_exists.return_value = True
        mock_compress.return_value = {"render_ms": 200}

        input_path = Path("input.gif")
        output_path = Path("output.gif")

        result = apply_lossy_compression(
            input_path, output_path, 120, 0.8, LossyEngine.ANIMATELY  # frame_keep_ratio
        )

        mock_compress.assert_called_once_with(input_path, output_path, 120, 0.8)
        assert result == {"render_ms": 200}

    @patch("pathlib.Path.exists")
    def test_unsupported_engine(self, mock_exists):
        """Test that unsupported engine raises ValueError."""
        mock_exists.return_value = True

        # Create a mock engine that's not in the enum
        with pytest.raises(ValueError, match="Unsupported engine"):
            apply_lossy_compression(
                Path("input.gif"),
                Path("output.gif"),
                0,
                1.0,  # frame_keep_ratio
                "invalid_engine",  # type: ignore[arg-type]  # This will cause the error
            )


class TestCompressWithGifsicle:
    """Tests for compress_with_gifsicle function."""

    @patch("giflab.lossy.get_gifsicle_version")
    @patch("giflab.lossy.extract_gif_metadata")
    @patch("subprocess.run")
    @patch("pathlib.Path.exists")
    @patch("time.time")
    def test_successful_compression_lossless(
        self, mock_time, mock_exists, mock_run, mock_metadata, mock_version
    ):
        """Test successful gifsicle compression with lossless setting."""
        # Mock time progression
        mock_time.side_effect = [1000.0, 1000.5]  # 500ms execution

        # Mock metadata extraction
        mock_meta = MagicMock()
        mock_meta.orig_frames = 10
        mock_meta.orig_n_colors = 128
        mock_metadata.return_value = mock_meta

        # Mock version detection
        mock_version.return_value = "1.94"

        # Mock successful subprocess
        mock_result = MagicMock()
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        # Mock output file exists
        mock_exists.return_value = True

        input_path = Path("input.gif")
        output_path = Path("output.gif")

        result = compress_with_gifsicle(input_path, output_path, 0, 1.0)

        # Verify command construction (no frame reduction for ratio 1.0)
        from giflab.config import DEFAULT_ENGINE_CONFIG

        expected_cmd = [
            DEFAULT_ENGINE_CONFIG.GIFSICLE_PATH,
            "--optimize",
            str(input_path.resolve()),
            "--output",
            str(output_path.resolve()),
        ]

        mock_run.assert_called_once_with(
            expected_cmd, capture_output=True, text=True, check=True, timeout=10
        )

        # Verify result
        assert result["render_ms"] == 500
        assert result["engine"] == "gifsicle"
        assert result["lossy_level"] == 0
        assert result["frame_keep_ratio"] == 1.0
        assert result["original_frames"] == 10
        assert result["command"] == " ".join(expected_cmd)

    @patch("giflab.lossy.get_gifsicle_version")
    @patch("giflab.lossy.extract_gif_metadata")
    @patch("subprocess.run")
    @patch("pathlib.Path.exists")
    @patch("time.time")
    def test_successful_compression_lossy(
        self, mock_time, mock_exists, mock_run, mock_metadata, mock_version
    ):
        """Test successful gifsicle compression with lossy setting."""
        mock_time.side_effect = [1000.0, 1001.0]  # 1000ms execution

        # Mock metadata extraction
        mock_meta = MagicMock()
        mock_meta.orig_frames = 20
        mock_meta.orig_n_colors = 256
        mock_metadata.return_value = mock_meta

        # Mock version detection
        mock_version.return_value = "1.94"

        mock_result = MagicMock()
        mock_result.stderr = ""
        mock_run.return_value = mock_result
        mock_exists.return_value = True

        input_path = Path("input.gif")
        output_path = Path("output.gif")

        result = compress_with_gifsicle(input_path, output_path, 40, 1.0)

        # Verify lossy flag is included
        from giflab.config import DEFAULT_ENGINE_CONFIG

        expected_cmd = [
            DEFAULT_ENGINE_CONFIG.GIFSICLE_PATH,
            "--optimize",
            "--lossy=40",
            str(input_path.resolve()),
            "--output",
            str(output_path.resolve()),
        ]

        mock_run.assert_called_once_with(
            expected_cmd, capture_output=True, text=True, check=True, timeout=10
        )

        assert result["render_ms"] == 1000
        assert result["lossy_level"] == 40
        assert result["frame_keep_ratio"] == 1.0
        assert result["original_frames"] == 20

    @patch("giflab.lossy.get_gifsicle_version")
    @patch("giflab.lossy.build_gifsicle_frame_args")
    @patch("giflab.lossy.extract_gif_metadata")
    @patch("subprocess.run")
    @patch("pathlib.Path.exists")
    @patch("time.time")
    def test_compression_with_frame_reduction(
        self,
        mock_time,
        mock_exists,
        mock_run,
        mock_metadata,
        mock_frame_args,
        mock_version,
    ):
        """Test gifsicle compression with frame reduction."""
        mock_time.side_effect = [1000.0, 1000.8]  # 800ms execution

        # Mock metadata extraction
        mock_meta = MagicMock()
        mock_meta.orig_frames = 10
        mock_meta.orig_n_colors = 128
        mock_metadata.return_value = mock_meta

        # Mock frame reduction arguments (new frame selection syntax)
        mock_frame_args.return_value = ["#0", "#2", "#4", "#6"]

        # Mock version detection
        mock_version.return_value = "1.94"

        mock_result = MagicMock()
        mock_result.stderr = ""
        mock_run.return_value = mock_result
        mock_exists.return_value = True

        input_path = Path("input.gif")
        output_path = Path("output.gif")

        result = compress_with_gifsicle(input_path, output_path, 0, 0.8)

        # Verify frame reduction arguments are included (input file comes first, then frame args)
        from giflab.config import DEFAULT_ENGINE_CONFIG

        expected_cmd = [
            DEFAULT_ENGINE_CONFIG.GIFSICLE_PATH,
            "--optimize",
            str(input_path.resolve()),
            "#0",
            "#2",
            "#4",
            "#6",
            "--output",
            str(output_path.resolve()),
        ]

        mock_run.assert_called_once_with(
            expected_cmd, capture_output=True, text=True, check=True, timeout=10
        )

        # Verify frame args function was called
        mock_frame_args.assert_called_once_with(0.8, 10)

        assert result["frame_keep_ratio"] == 0.8
        assert result["original_frames"] == 10

    @patch("giflab.lossy.extract_gif_metadata")
    @patch("subprocess.run")
    def test_subprocess_error(self, mock_run, mock_metadata):
        """Test handling of subprocess errors."""
        # Mock metadata extraction
        mock_meta = MagicMock()
        mock_meta.orig_frames = 5
        mock_meta.orig_n_colors = 128
        mock_metadata.return_value = mock_meta

        mock_run.side_effect = subprocess.CalledProcessError(
            1, "gifsicle", stderr="Error message"
        )

        with pytest.raises(RuntimeError, match="Gifsicle failed with exit code 1"):
            compress_with_gifsicle(Path("input.gif"), Path("output.gif"), 0, 1.0)

    @patch("giflab.lossy.extract_gif_metadata")
    @patch("subprocess.run")
    def test_timeout_error(self, mock_run, mock_metadata):
        """Test handling of subprocess timeout."""
        # Mock metadata extraction
        mock_meta = MagicMock()
        mock_meta.orig_frames = 5
        mock_meta.orig_n_colors = 128
        mock_metadata.return_value = mock_meta

        mock_run.side_effect = subprocess.TimeoutExpired("gifsicle", 10)

        with pytest.raises(RuntimeError, match="Gifsicle timed out"):
            compress_with_gifsicle(Path("input.gif"), Path("output.gif"), 0, 1.0)

    @patch("giflab.lossy.extract_gif_metadata")
    @patch("subprocess.run")
    @patch("pathlib.Path.exists")
    @patch("time.time")
    def test_missing_output_file(self, mock_time, mock_exists, mock_run, mock_metadata):
        """Test error when output file is not created."""
        mock_time.side_effect = [1000.0, 1000.1]

        # Mock metadata extraction
        mock_meta = MagicMock()
        mock_meta.orig_frames = 5
        mock_meta.orig_n_colors = 128
        mock_metadata.return_value = mock_meta

        mock_run.return_value = MagicMock()
        mock_exists.return_value = False  # Output file doesn't exist

        with pytest.raises(RuntimeError, match="failed to create output file"):
            compress_with_gifsicle(Path("input.gif"), Path("output.gif"), 0, 1.0)


class TestCompressWithAnimately:
    """Tests for compress_with_animately function."""

    @patch("giflab.lossy.get_animately_version")
    @patch("giflab.lossy.extract_gif_metadata")
    @patch("subprocess.run")
    @patch("pathlib.Path.exists")
    @patch("time.time")
    def test_successful_compression_lossless(
        self, mock_time, mock_exists, mock_run, mock_metadata, mock_version
    ):
        """Test successful animately compression with lossless setting."""
        mock_time.side_effect = [1000.0, 1000.3]  # 300ms execution

        # Mock metadata extraction
        mock_meta = MagicMock()
        mock_meta.orig_frames = 15
        mock_meta.orig_n_colors = 128
        mock_metadata.return_value = mock_meta

        # Mock version detection
        mock_version.return_value = "animately-engine"

        mock_result = MagicMock()
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        # Mock animately launcher and output file both exist
        mock_exists.return_value = True

        input_path = Path("input.gif")
        output_path = Path("output.gif")

        result = compress_with_animately(input_path, output_path, 0, 1.0)

        # Verify command construction
        from giflab.config import DEFAULT_ENGINE_CONFIG

        expected_cmd = [
            DEFAULT_ENGINE_CONFIG.ANIMATELY_PATH,
            "--input",
            str(input_path.resolve()),
            "--output",
            str(output_path.resolve()),
        ]

        mock_run.assert_called_once_with(
            expected_cmd, capture_output=True, text=True, check=True, timeout=10
        )

        # Timing should be approximately 300ms (allow for floating point precision)
        assert 298 <= result["render_ms"] <= 301
        assert result["engine"] == "animately"
        assert result["lossy_level"] == 0
        assert result["frame_keep_ratio"] == 1.0
        assert result["original_frames"] == 15

    @patch("giflab.lossy.extract_gif_metadata")
    @patch("subprocess.run")
    @patch("pathlib.Path.exists")
    @patch("time.time")
    def test_successful_compression_lossy(
        self, mock_time, mock_exists, mock_run, mock_metadata
    ):
        """Test successful animately compression with lossy setting."""
        mock_time.side_effect = [1000.0, 1002.0]  # 2000ms execution

        # Mock metadata extraction
        mock_meta = MagicMock()
        mock_meta.orig_frames = 8
        mock_meta.orig_n_colors = 128
        mock_metadata.return_value = mock_meta

        mock_result = MagicMock()
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        # Mock both launcher and output file exist
        mock_exists.return_value = True

        result = compress_with_animately(
            Path("input.gif"), Path("output.gif"), 120, 0.7
        )

        # Should include frame reduction args
        # Note: The exact command will depend on mock_frame_args, but we can't easily test the exact command here
        # without more complex mocking, so we'll just verify the basic structure

        assert result["render_ms"] == 2000
        assert result["lossy_level"] == 120
        assert result["frame_keep_ratio"] == 0.7
        assert result["original_frames"] == 8

    @patch("giflab.lossy._is_executable")
    def test_missing_launcher(self, mock_is_executable):
        """Test error when animately launcher is missing."""
        mock_is_executable.return_value = False

        with pytest.raises(RuntimeError, match="Animately launcher not found"):
            compress_with_animately(Path("input.gif"), Path("output.gif"), 0, 1.0)

    @patch("giflab.lossy.extract_gif_metadata")
    @patch("subprocess.run")
    @patch("pathlib.Path.exists")
    def test_subprocess_error(self, mock_exists, mock_run, mock_metadata):
        """Test handling of subprocess errors."""
        mock_exists.return_value = True  # Launcher exists

        # Mock metadata extraction
        mock_meta = MagicMock()
        mock_meta.orig_frames = 5
        mock_meta.orig_n_colors = 128
        mock_metadata.return_value = mock_meta

        mock_run.side_effect = subprocess.CalledProcessError(
            2, "animately", stderr="Animately error"
        )

        with pytest.raises(RuntimeError, match="Animately failed with exit code 2"):
            compress_with_animately(Path("input.gif"), Path("output.gif"), 0, 1.0)


class TestApplyCompressionWithAllParams:
    """Tests for apply_compression_with_all_params function."""

    @patch("giflab.lossy.compress_with_gifsicle")
    @patch("pathlib.Path.exists")
    def test_gifsicle_with_all_params(self, mock_exists, mock_compress):
        """Test compression with all parameters using gifsicle."""
        mock_exists.return_value = True
        mock_compress.return_value = {
            "render_ms": 300,
            "engine": "gifsicle",
            "lossy_level": 40,
            "frame_keep_ratio": 0.8,
            "color_keep_count": 64,
        }

        result = apply_compression_with_all_params(
            Path("input.gif"), Path("output.gif"), 40, 0.8, 64, LossyEngine.GIFSICLE
        )

        # Should call compress_with_gifsicle with all params
        mock_compress.assert_called_once_with(
            Path("input.gif"), Path("output.gif"), 40, 0.8, 64
        )

        assert result["lossy_level"] == 40
        assert result["frame_keep_ratio"] == 0.8
        assert result["color_keep_count"] == 64

    @patch("giflab.lossy.compress_with_animately")
    @patch("pathlib.Path.exists")
    def test_animately_with_all_params(self, mock_exists, mock_compress):
        """Test compression with all parameters using animately."""
        mock_exists.return_value = True
        mock_compress.return_value = {
            "render_ms": 400,
            "engine": "animately",
            "lossy_level": 120,
            "frame_keep_ratio": 0.7,
            "color_keep_count": 128,
        }

        result = apply_compression_with_all_params(
            Path("input.gif"), Path("output.gif"), 120, 0.7, 128, LossyEngine.ANIMATELY
        )

        mock_compress.assert_called_once_with(
            Path("input.gif"), Path("output.gif"), 120, 0.7, 128
        )

        assert result["engine"] == "animately"

    def test_parameter_validation(self):
        """Test that all parameters are validated."""
        # Test invalid lossy level
        with pytest.raises(ValueError, match="must be non-negative"):
            apply_compression_with_all_params(
                Path("input.gif"), Path("output.gif"), -1, 0.8, 64
            )

        # Test invalid frame ratio
        with pytest.raises(ValueError, match="not in supported ratios"):
            apply_compression_with_all_params(
                Path("input.gif"),
                Path("output.gif"),
                0,
                0.33,  # Not in supported ratios
                64,
            )

        # Test invalid color count
        with pytest.raises(ValueError, match="not in supported counts"):
            apply_compression_with_all_params(
                Path("input.gif"),
                Path("output.gif"),
                0,
                0.8,
                4,  # 4 is not in supported counts [256, 128, 64, 32, 16, 8]
            )


class TestFrameKeepRatioValidation:
    """Tests for frame keep ratio validation in lossy functions."""

    def test_invalid_frame_keep_ratio(self):
        """Test that invalid frame keep ratios are rejected."""
        with pytest.raises(ValueError, match="not in supported ratios"):
            apply_lossy_compression(
                Path("input.gif"),
                Path("output.gif"),
                0,
                0.33,  # Not in supported ratios
                LossyEngine.GIFSICLE,
            )

    def test_valid_frame_keep_ratios(self):
        """Test that all configured frame keep ratios are accepted."""
        valid_ratios = [1.0, 0.9, 0.8, 0.7, 0.5]

        with patch("pathlib.Path.exists") as mock_exists:
            mock_exists.return_value = (
                False  # Will fail with file not found, but validation should pass
            )

            for ratio in valid_ratios:
                with pytest.raises(IOError, match="Input file not found"):
                    # Should pass validation but fail on file not found
                    apply_lossy_compression(
                        Path("nonexistent.gif"),
                        Path("output.gif"),
                        0,
                        ratio,
                        LossyEngine.GIFSICLE,
                    )


class TestColorIntegration:
    """Tests for color reduction integration in compression functions."""

    @patch("giflab.lossy.get_gifsicle_version")
    @patch("giflab.lossy.build_gifsicle_color_args")
    @patch("giflab.lossy.extract_gif_metadata")
    @patch("subprocess.run")
    @patch("pathlib.Path.exists")
    @patch("time.time")
    def test_gifsicle_with_color_reduction(
        self,
        mock_time,
        mock_exists,
        mock_run,
        mock_metadata,
        mock_color_args,
        mock_version,
    ):
        """Test gifsicle compression with color reduction."""
        mock_time.side_effect = [1000.0, 1000.5]  # 500ms execution

        # Mock metadata extraction
        mock_meta = MagicMock()
        mock_meta.orig_frames = 10
        mock_meta.orig_n_colors = 256
        mock_metadata.return_value = mock_meta

        # Mock color reduction arguments (includes dithering parameter)
        mock_color_args.return_value = ["--colors", "64", "--no-dither"]

        # Mock version detection
        mock_version.return_value = "1.94"

        mock_result = MagicMock()
        mock_result.stderr = ""
        mock_run.return_value = mock_result
        mock_exists.return_value = True

        input_path = Path("input.gif")
        output_path = Path("output.gif")

        result = compress_with_gifsicle(input_path, output_path, 0, 1.0, 64)

        # Verify color reduction arguments are included
        from giflab.config import DEFAULT_ENGINE_CONFIG

        expected_cmd = [
            DEFAULT_ENGINE_CONFIG.GIFSICLE_PATH,
            "--optimize",
            "--colors",
            "64",
            "--no-dither",
            str(input_path.resolve()),
            "--output",
            str(output_path.resolve()),
        ]

        mock_run.assert_called_once_with(
            expected_cmd, capture_output=True, text=True, check=True, timeout=10
        )

        # Verify color args function was called with dithering parameter
        mock_color_args.assert_called_once_with(64, 256, dithering=False)

        assert result["color_keep_count"] == 64
        assert result["original_colors"] == 256

    @patch("giflab.lossy.build_animately_color_args")
    @patch("giflab.lossy.extract_gif_metadata")
    @patch("subprocess.run")
    @patch("pathlib.Path.exists")
    @patch("time.time")
    def test_animately_with_color_reduction(
        self, mock_time, mock_exists, mock_run, mock_metadata, mock_color_args
    ):
        """Test animately compression with color reduction."""
        mock_time.side_effect = [1000.0, 1001.0]  # 1000ms execution

        # Mock metadata extraction
        mock_meta = MagicMock()
        mock_meta.orig_frames = 15
        mock_meta.orig_n_colors = 128
        mock_metadata.return_value = mock_meta

        # Mock color reduction arguments
        mock_color_args.return_value = ["--colors", "64"]

        mock_result = MagicMock()
        mock_result.stderr = ""
        mock_run.return_value = mock_result
        mock_exists.return_value = True

        result = compress_with_animately(
            Path("input.gif"), Path("output.gif"), 40, 0.8, 64
        )

        # Verify color args function was called
        mock_color_args.assert_called_once_with(64, 128)

        assert result["color_keep_count"] == 64
        assert result["original_colors"] == 128


class TestGetCompressionEstimate:
    """Tests for get_compression_estimate function."""

    @patch("giflab.lossy.extract_gif_metadata")
    @patch("pathlib.Path.exists")
    def test_compression_estimate(self, mock_exists, mock_metadata):
        """Test compression estimation."""
        mock_exists.return_value = True

        # Mock metadata
        mock_meta = MagicMock()
        mock_meta.orig_frames = 20
        mock_meta.orig_n_colors = 256
        mock_meta.orig_kilobytes = 1000.0
        mock_metadata.return_value = mock_meta

        estimate = get_compression_estimate(
            Path("test.gif"),
            40,  # lossy level
            0.8,  # frame keep ratio (80% of frames)
            128,  # color keep count
        )

        assert estimate["original_size_kb"] == 1000.0
        assert estimate["estimated_size_kb"] > 0
        assert estimate["estimated_compression_ratio"] > 1.0
        assert estimate["frame_reduction_percent"] == 20.0  # 1 - 0.8 = 0.2 = 20%
        assert estimate["color_reduction_percent"] == 50.0  # (256 - 128) / 256 = 50%
        assert estimate["target_frames"] == 16  # 80% of 20
        assert estimate["target_colors"] == 128
        assert estimate["lossy_level"] == 40
        assert estimate["quality_loss_estimate"] >= 0

    def test_parameter_validation_in_estimate(self):
        """Test parameter validation in compression estimate."""
        with pytest.raises(ValueError, match="not in supported ratios"):
            get_compression_estimate(Path("test.gif"), 0, 0.33, 128)

        with pytest.raises(ValueError, match="not in supported counts"):
            get_compression_estimate(
                Path("test.gif"), 0, 0.8, 4
            )  # 4 is not in supported counts

    @patch("pathlib.Path.exists")
    def test_missing_file_estimate(self, mock_exists):
        """Test error when input file doesn't exist."""
        mock_exists.return_value = False

        with pytest.raises(IOError, match="Input file not found"):
            get_compression_estimate(Path("missing.gif"), 0, 0.8, 128)
