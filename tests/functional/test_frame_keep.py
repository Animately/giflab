"""Tests for giflab.frame_keep module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from giflab.frame_keep import (
    build_animately_frame_args,
    build_gifsicle_frame_args,
    calculate_frame_indices,
    calculate_target_frame_count,
    get_frame_reduction_info,
    validate_frame_keep_ratio,
)


class TestCalculateFrameIndices:
    """Tests for calculate_frame_indices function."""

    def test_keep_all_frames(self):
        """Test keeping all frames (ratio 1.0)."""
        indices = calculate_frame_indices(10, 1.0)
        assert indices == list(range(10))

    def test_keep_half_frames(self):
        """Test keeping half frames (ratio 0.5)."""
        indices = calculate_frame_indices(10, 0.5)
        assert len(indices) == 5
        assert indices == [0, 2, 4, 6, 8]

    def test_keep_zero_frames(self):
        """Test keeping zero frames (ratio 0.0) - should keep at least first frame."""
        indices = calculate_frame_indices(10, 0.0)
        assert indices == [0]

    def test_keep_80_percent(self):
        """Test keeping 80% of frames."""
        indices = calculate_frame_indices(10, 0.8)
        assert len(indices) == 8
        # Should be evenly distributed
        expected = [0, 1, 2, 3, 5, 6, 7, 8]
        assert indices == expected

    def test_keep_70_percent(self):
        """Test keeping 70% of frames."""
        indices = calculate_frame_indices(20, 0.7)
        assert len(indices) == 14

    def test_single_frame_gif(self):
        """Test with single frame GIF."""
        indices = calculate_frame_indices(1, 0.5)
        assert indices == [0]

    def test_invalid_ratio_negative(self):
        """Test invalid negative ratio."""
        with pytest.raises(ValueError, match="keep_ratio must be between 0.0 and 1.0"):
            calculate_frame_indices(10, -0.1)

    def test_invalid_ratio_too_large(self):
        """Test invalid ratio greater than 1.0."""
        with pytest.raises(ValueError, match="keep_ratio must be between 0.0 and 1.0"):
            calculate_frame_indices(10, 1.1)

    def test_invalid_frame_count_zero(self):
        """Test invalid frame count (zero)."""
        with pytest.raises(ValueError, match="total_frames must be positive"):
            calculate_frame_indices(0, 0.5)

    def test_invalid_frame_count_negative(self):
        """Test invalid frame count (negative)."""
        with pytest.raises(ValueError, match="total_frames must be positive"):
            calculate_frame_indices(-5, 0.5)


class TestCalculateTargetFrameCount:
    """Tests for calculate_target_frame_count function."""

    def test_keep_all_frames(self):
        """Test target count when keeping all frames."""
        count = calculate_target_frame_count(10, 1.0)
        assert count == 10

    def test_keep_half_frames(self):
        """Test target count when keeping half frames."""
        count = calculate_target_frame_count(10, 0.5)
        assert count == 5

    def test_keep_zero_frames(self):
        """Test target count when keeping zero frames (minimum 1)."""
        count = calculate_target_frame_count(10, 0.0)
        assert count == 1

    def test_fractional_result(self):
        """Test target count with fractional result."""
        count = calculate_target_frame_count(10, 0.33)
        assert count == 3  # int(10 * 0.33) = 3

    def test_invalid_ratio(self):
        """Test invalid ratio."""
        with pytest.raises(ValueError, match="keep_ratio must be between 0.0 and 1.0"):
            calculate_target_frame_count(10, 1.5)

    def test_invalid_frame_count(self):
        """Test invalid frame count."""
        with pytest.raises(ValueError, match="total_frames must be positive"):
            calculate_target_frame_count(0, 0.5)


class TestBuildGifsicleFrameArgs:
    """Tests for build_gifsicle_frame_args function."""

    def test_no_reduction_needed(self):
        """Test no arguments when keeping all frames."""
        args = build_gifsicle_frame_args(1.0, 10)
        assert args == []

    def test_keep_every_other_frame(self):
        """Test keeping every other frame."""
        args = build_gifsicle_frame_args(0.5, 10)
        # Should keep frames 0, 2, 4, 6, 8 (frame selection syntax)
        expected_frames = ["#0", "#2", "#4", "#6", "#8"]
        assert args == expected_frames

    def test_keep_first_and_last(self):
        """Test keeping only first few frames."""
        args = build_gifsicle_frame_args(0.2, 10)
        # Should keep 2 frames: 0, 5 (frame selection syntax)
        assert "#0" in args
        assert "#5" in args

    def test_single_frame_gif(self):
        """Test with single frame GIF."""
        args = build_gifsicle_frame_args(0.5, 1)
        assert args == ["#0"]  # Keep the single frame

    def test_keep_80_percent(self):
        """Test keeping 80% of frames."""
        args = build_gifsicle_frame_args(0.8, 10)
        # Should keep 8 frames with frame selection syntax
        assert len(args) == 8
        assert "#0" in args


class TestBuildAnimatelyFrameArgs:
    """Tests for build_animately_frame_args function."""

    def test_no_reduction_needed(self):
        """Test no arguments when keeping all frames."""
        args = build_animately_frame_args(1.0, 10)
        assert args == []

    def test_frame_reduction_args(self):
        """Test frame reduction arguments for animately."""
        args = build_animately_frame_args(0.5, 10)
        # Should include frame reduction parameter with --reduce
        assert "--reduce" in args
        assert "0.50" in args

    def test_different_ratios(self):
        """Test different frame reduction ratios."""
        args_80 = build_animately_frame_args(0.8, 10)
        args_70 = build_animately_frame_args(0.7, 10)

        assert "--reduce" in args_80
        assert "--reduce" in args_70
        assert "0.80" in args_80
        assert "0.70" in args_70

    def test_single_frame_gif(self):
        """Test with single frame GIF."""
        args = build_animately_frame_args(0.5, 1)
        assert args == []  # No reduction needed for single frame


class TestValidateFrameKeepRatio:
    """Tests for validate_frame_keep_ratio function."""

    def test_valid_ratios(self):
        """Test that configured valid ratios pass validation."""
        valid_ratios = [1.0, 0.9, 0.8, 0.7, 0.5]
        for ratio in valid_ratios:
            # Should not raise any exception
            validate_frame_keep_ratio(ratio)

    def test_invalid_ratio_out_of_range(self):
        """Test invalid ratios outside 0.0-1.0 range."""
        with pytest.raises(ValueError, match="must be between 0.0 and 1.0"):
            validate_frame_keep_ratio(-0.1)

        with pytest.raises(ValueError, match="must be between 0.0 and 1.0"):
            validate_frame_keep_ratio(1.1)

    def test_invalid_ratio_not_configured(self):
        """Test ratio not in configured valid ratios."""
        with pytest.raises(ValueError, match="not in supported ratios"):
            validate_frame_keep_ratio(0.33)

    def test_floating_point_tolerance(self):
        """Test that small floating point differences are tolerated."""
        # Should pass due to tolerance
        validate_frame_keep_ratio(0.8000001)
        validate_frame_keep_ratio(0.7999999)


class TestGetFrameReductionInfo:
    """Tests for get_frame_reduction_info function."""

    @patch("pathlib.Path.exists")
    @patch("PIL.Image.open")
    def test_valid_gif_analysis(self, mock_open, mock_exists):
        """Test frame reduction analysis for valid GIF."""
        mock_exists.return_value = True

        # Mock PIL Image
        mock_img = MagicMock()
        mock_img.format = "GIF"
        mock_img.n_frames = 3  # Explicitly set n_frames to an integer
        mock_img.seek.side_effect = [
            None,
            None,
            EOFError(),
        ]  # 3 frames (2 seeks, then EOFError)
        mock_img.tell.return_value = 0
        mock_open.return_value.__enter__.return_value = mock_img

        info = get_frame_reduction_info(Path("test.gif"), 0.8)

        assert info["original_frames"] == 3
        assert info["keep_ratio"] == 0.8
        assert info["target_frames"] == 2  # 80% of 3 frames
        assert info["frames_kept"] == 2
        assert "frame_indices" in info
        assert abs(info["reduction_percent"] - 20.0) < 1e-10

    @patch("pathlib.Path.exists")
    def test_missing_file(self, mock_exists):
        """Test error when input file doesn't exist."""
        mock_exists.return_value = False

        with pytest.raises(IOError, match="Input file not found"):
            get_frame_reduction_info(Path("missing.gif"), 0.8)

    def test_invalid_ratio(self):
        """Test error with invalid frame keep ratio."""
        with pytest.raises(ValueError, match="not in supported ratios"):
            get_frame_reduction_info(Path("test.gif"), 0.33)

    @patch("pathlib.Path.exists")
    @patch("PIL.Image.open")
    def test_non_gif_file(self, mock_open, mock_exists):
        """Test error when file is not a GIF."""
        mock_exists.return_value = True

        mock_img = MagicMock()
        mock_img.format = "PNG"  # Not a GIF
        mock_open.return_value.__enter__.return_value = mock_img

        with pytest.raises(ValueError, match="File is not a GIF"):
            get_frame_reduction_info(Path("test.png"), 0.8)

    @patch("pathlib.Path.exists")
    @patch("PIL.Image.open")
    def test_pil_error_handling(self, mock_open, mock_exists):
        """Test handling of PIL errors."""
        mock_exists.return_value = True
        mock_open.side_effect = Exception("PIL error")

        with pytest.raises(IOError, match="Error reading GIF"):
            get_frame_reduction_info(Path("test.gif"), 0.8)

    @patch("pathlib.Path.exists")
    @patch("PIL.Image.open")
    def test_single_frame_gif(self, mock_open, mock_exists):
        """Test analysis of single frame GIF."""
        mock_exists.return_value = True

        mock_img = MagicMock()
        mock_img.format = "GIF"
        mock_img.n_frames = 1  # Explicitly set n_frames to 1
        mock_img.seek.side_effect = [EOFError()]  # Only 1 frame
        mock_img.tell.return_value = 0
        mock_open.return_value.__enter__.return_value = mock_img

        info = get_frame_reduction_info(Path("single.gif"), 0.5)

        assert info["original_frames"] == 1
        assert info["target_frames"] == 1
        assert info["frames_kept"] == 1
        assert info["frame_indices"] == [0]
