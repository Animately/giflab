"""Test suite for GIF disposal method analysis functions."""

from pathlib import Path
from unittest.mock import Mock, patch

from giflab.lossy import (
    _analyze_gif_disposal_complexity,
    _select_optimal_disposal_method,
)


class TestAnalyzeGifDisposalComplexity:
    """Test the _analyze_gif_disposal_complexity function."""

    @patch("giflab.lossy.discover_tool")
    def test_imagemagick_not_available(self, mock_discover_tool):
        """Test fallback when ImageMagick is not available."""
        # Mock ImageMagick as unavailable
        mock_tool = Mock()
        mock_tool.available = False
        mock_discover_tool.return_value = mock_tool

        result = _analyze_gif_disposal_complexity(Path("test.gif"))

        # Should return low complexity fallback
        expected = {
            "complexity_score": 0.0,
            "has_variable_frames": False,
            "has_offsets": False,
            "frame_size_variance": 0.0,
            "disposal_risk": "low",
        }
        assert result == expected
        mock_discover_tool.assert_called_once_with("imagemagick")

    @patch("giflab.lossy.discover_tool")
    @patch("subprocess.run")
    def test_identify_command_fails(self, mock_run, mock_discover_tool):
        """Test fallback when identify command fails."""
        # Mock ImageMagick as available
        mock_tool = Mock()
        mock_tool.available = True
        mock_discover_tool.return_value = mock_tool

        # Mock identify command failure
        mock_result = Mock()
        mock_result.returncode = 1
        mock_run.return_value = mock_result

        result = _analyze_gif_disposal_complexity(Path("test.gif"))

        # Should return low complexity fallback
        expected = {
            "complexity_score": 0.0,
            "has_variable_frames": False,
            "has_offsets": False,
            "frame_size_variance": 0.0,
            "disposal_risk": "low",
        }
        assert result == expected

    @patch("giflab.lossy.discover_tool")
    @patch("subprocess.run")
    def test_simple_gif_analysis(self, mock_run, mock_discover_tool):
        """Test analysis of simple GIF with uniform frames."""
        # Mock ImageMagick as available
        mock_tool = Mock()
        mock_tool.available = True
        mock_discover_tool.return_value = mock_tool

        # Mock identify output for simple GIF (all frames same size, no offsets)
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = """test.gif[0] GIF 100x100 100x100+0+0 8-bit sRGB 256c 1024B 0.100s
test.gif[1] GIF 100x100 100x100+0+0 8-bit sRGB 256c 1024B 0.100s
test.gif[2] GIF 100x100 100x100+0+0 8-bit sRGB 256c 1024B 0.100s"""
        mock_run.return_value = mock_result

        result = _analyze_gif_disposal_complexity(Path("test.gif"))

        # Should detect low complexity
        assert result["complexity_score"] == 0.0
        assert result["has_variable_frames"] is False
        assert result["has_offsets"] is False
        assert result["disposal_risk"] == "low"

    @patch("giflab.lossy.discover_tool")
    @patch("subprocess.run")
    def test_complex_gif_analysis(self, mock_run, mock_discover_tool):
        """Test analysis of complex GIF with variable frames and offsets."""
        # Mock ImageMagick as available
        mock_tool = Mock()
        mock_tool.available = True
        mock_discover_tool.return_value = mock_tool

        # Mock identify output for complex GIF (different sizes, offsets)
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = """test.gif[0] GIF 100x100 100x100+0+0 8-bit sRGB 256c 1024B 0.100s
test.gif[1] GIF 50x50 100x100+25+25 8-bit sRGB 256c 512B 0.100s
test.gif[2] GIF 80x60 100x100+10+20 8-bit sRGB 256c 768B 0.100s
test.gif[3] GIF 90x90 100x100+5+5 8-bit sRGB 256c 972B 0.100s"""
        mock_run.return_value = mock_result

        result = _analyze_gif_disposal_complexity(Path("test.gif"))

        # Should detect high complexity
        assert result["complexity_score"] > 0.5
        assert result["has_variable_frames"] is True
        assert result["has_offsets"] is True
        assert result["disposal_risk"] in ["medium", "high"]

    @patch("giflab.lossy.discover_tool")
    @patch("subprocess.run")
    def test_single_frame_gif(self, mock_run, mock_discover_tool):
        """Test analysis of single frame GIF."""
        # Mock ImageMagick as available
        mock_tool = Mock()
        mock_tool.available = True
        mock_discover_tool.return_value = mock_tool

        # Mock identify output for single frame GIF
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "test.gif GIF 100x100 100x100+0+0 8-bit sRGB 256c 1024B"
        mock_run.return_value = mock_result

        result = _analyze_gif_disposal_complexity(Path("test.gif"))

        # Single frame should always be low complexity
        assert result["complexity_score"] == 0.0
        assert result["disposal_risk"] == "low"

    @patch("giflab.lossy.discover_tool")
    @patch("subprocess.run")
    def test_timeout_handling(self, mock_run, mock_discover_tool):
        """Test that timeout is properly configured."""
        # Mock ImageMagick as available
        mock_tool = Mock()
        mock_tool.available = True
        mock_discover_tool.return_value = mock_tool

        # Mock subprocess call
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = ""
        mock_run.return_value = mock_result

        _analyze_gif_disposal_complexity(Path("test.gif"))

        # Verify timeout is set to 30 seconds
        mock_run.assert_called_once()
        call_args = mock_run.call_args
        assert call_args[1]["timeout"] == 30

    @patch("giflab.lossy.discover_tool")
    @patch("subprocess.run")
    def test_exception_handling(self, mock_run, mock_discover_tool):
        """Test graceful handling of unexpected exceptions."""
        # Mock ImageMagick as available
        mock_tool = Mock()
        mock_tool.available = True
        mock_discover_tool.return_value = mock_tool

        # Mock subprocess to raise exception
        mock_run.side_effect = Exception("Unexpected error")

        result = _analyze_gif_disposal_complexity(Path("test.gif"))

        # Should return low complexity fallback on any exception
        expected = {
            "complexity_score": 0.0,
            "has_variable_frames": False,
            "has_offsets": False,
            "frame_size_variance": 0.0,
            "disposal_risk": "low",
        }
        assert result == expected


class TestSelectOptimalDisposalMethod:
    """Test the _select_optimal_disposal_method function."""

    @patch("giflab.lossy._analyze_gif_disposal_complexity")
    def test_high_complexity_gif(self, mock_analyze):
        """Test disposal method selection for high complexity GIFs."""
        # Mock high complexity analysis
        mock_analyze.return_value = {
            "complexity_score": 0.8,
            "disposal_risk": "high",
            "has_variable_frames": True,
            "has_offsets": True,
        }

        result = _select_optimal_disposal_method(
            Path("test.gif"), frame_keep_ratio=0.7, total_frames=20
        )

        # High complexity should force background disposal
        assert result == "background"

    @patch("giflab.lossy._analyze_gif_disposal_complexity")
    def test_medium_complexity_gif_with_frame_reduction(self, mock_analyze):
        """Test medium complexity GIF with frame reduction."""
        # Mock medium complexity analysis
        mock_analyze.return_value = {
            "complexity_score": 0.5,
            "disposal_risk": "medium",
            "has_variable_frames": True,
            "has_offsets": False,
        }

        result = _select_optimal_disposal_method(
            Path("test.gif"), frame_keep_ratio=0.6, total_frames=20  # Frame reduction
        )

        # Medium complexity with frame reduction should use background
        assert result == "background"

    @patch("giflab.lossy._analyze_gif_disposal_complexity")
    def test_medium_complexity_gif_no_frame_reduction(self, mock_analyze):
        """Test medium complexity GIF without frame reduction."""
        # Mock medium complexity analysis
        mock_analyze.return_value = {
            "complexity_score": 0.5,
            "disposal_risk": "medium",
            "has_variable_frames": True,
            "has_offsets": False,
        }

        result = _select_optimal_disposal_method(
            Path("test.gif"),
            frame_keep_ratio=1.0,  # No frame reduction
            total_frames=20,
        )

        # Medium complexity without frame reduction should preserve original
        assert result is None

    @patch("giflab.lossy._analyze_gif_disposal_complexity")
    def test_low_complexity_aggressive_reduction(self, mock_analyze):
        """Test low complexity GIF with aggressive frame reduction."""
        # Mock low complexity analysis
        mock_analyze.return_value = {
            "complexity_score": 0.1,
            "disposal_risk": "low",
            "has_variable_frames": False,
            "has_offsets": False,
        }

        result = _select_optimal_disposal_method(
            Path("test.gif"),
            frame_keep_ratio=0.3,  # Aggressive reduction
            total_frames=20,
        )

        # Low complexity with aggressive reduction should preserve original
        assert result is None

    @patch("giflab.lossy._analyze_gif_disposal_complexity")
    def test_low_complexity_moderate_reduction(self, mock_analyze):
        """Test low complexity GIF with moderate frame reduction."""
        # Mock low complexity analysis
        mock_analyze.return_value = {
            "complexity_score": 0.1,
            "disposal_risk": "low",
            "has_variable_frames": False,
            "has_offsets": False,
        }

        result = _select_optimal_disposal_method(
            Path("test.gif"),
            frame_keep_ratio=0.6,  # Moderate reduction
            total_frames=20,
        )

        # Low complexity with moderate reduction should use background
        assert result == "background"

    @patch("giflab.lossy._analyze_gif_disposal_complexity")
    def test_low_complexity_light_reduction(self, mock_analyze):
        """Test low complexity GIF with light frame reduction."""
        # Mock low complexity analysis
        mock_analyze.return_value = {
            "complexity_score": 0.1,
            "disposal_risk": "low",
            "has_variable_frames": False,
            "has_offsets": False,
        }

        result = _select_optimal_disposal_method(
            Path("test.gif"), frame_keep_ratio=0.9, total_frames=20  # Light reduction
        )

        # Low complexity with light reduction should use none
        assert result == "none"

    @patch("giflab.lossy._analyze_gif_disposal_complexity")
    def test_analysis_exception_fallback(self, mock_analyze):
        """Test fallback logic when complexity analysis fails."""
        # Mock analysis to raise exception
        mock_analyze.side_effect = Exception("Analysis failed")

        result = _select_optimal_disposal_method(
            Path("test.gif"), frame_keep_ratio=0.6, total_frames=20
        )

        # Should fall back to ratio-based logic
        assert result == "background"

    def test_edge_case_ratios(self):
        """Test edge cases for frame keep ratios."""
        # Test exact boundary values
        with patch("giflab.lossy._analyze_gif_disposal_complexity") as mock_analyze:
            mock_analyze.return_value = {
                "complexity_score": 0.1,
                "disposal_risk": "low",
                "has_variable_frames": False,
                "has_offsets": False,
            }

            # Test exactly 0.5 ratio
            result = _select_optimal_disposal_method(
                Path("test.gif"), frame_keep_ratio=0.5, total_frames=20
            )
            assert result is None  # Boundary condition

            # Test exactly 0.8 ratio
            result = _select_optimal_disposal_method(
                Path("test.gif"), frame_keep_ratio=0.8, total_frames=20
            )
            assert result == "background"
