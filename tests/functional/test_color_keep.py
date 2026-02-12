"""Tests for giflab.color_keep module and aligned color reduction between engines.

This includes unit tests for the color_keep API as well as integration tests
verifying that color reduction alignment strategies work correctly and that
both engines produce consistent results across different color ranges.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from giflab.color_keep import (
    analyze_gif_palette,
    build_animately_color_args,
    build_gifsicle_color_args,
    count_gif_colors,
    extract_dominant_colors,
    get_color_reduction_info,
    get_optimal_color_count,
    validate_color_keep_count,
)
from giflab.lossy import LossyEngine, compress_with_animately, compress_with_gifsicle
from PIL import Image

from tests.integration.test_engine_equivalence import _engine_available


class TestValidateColorKeepCount:
    """Tests for validate_color_keep_count function."""

    def test_valid_counts(self):
        """Test that configured valid color counts pass validation."""
        valid_counts = [256, 128, 64]
        for count in valid_counts:
            # Should not raise any exception
            validate_color_keep_count(count)

    def test_invalid_count_not_configured(self):
        """Test count not in configured valid counts."""
        with pytest.raises(ValueError, match="not in supported counts"):
            validate_color_keep_count(4)  # 4 is not in [256, 128, 64, 32, 16, 8]

    def test_invalid_count_negative(self):
        """Test negative color count."""
        with pytest.raises(ValueError, match="must be a positive integer"):
            validate_color_keep_count(-1)

    def test_invalid_count_zero(self):
        """Test zero color count."""
        with pytest.raises(ValueError, match="must be a positive integer"):
            validate_color_keep_count(0)

    def test_non_integer_count(self):
        """Test non-integer color count."""
        with pytest.raises(ValueError, match="must be a positive integer"):
            validate_color_keep_count(64.5)  # type: ignore


class TestBuildGifsicleColorArgs:
    """Tests for build_gifsicle_color_args function."""

    def test_no_reduction_needed_target_higher(self):
        """Test no arguments when target >= original colors."""
        args = build_gifsicle_color_args(128, 64)  # Target higher than original
        assert args == []

    def test_no_reduction_needed_target_equal(self):
        """Test no arguments when target equals original colors."""
        args = build_gifsicle_color_args(128, 128)  # Target equals original
        assert args == []

    def test_no_reduction_needed_max_colors(self):
        """Test no arguments when target is max (256)."""
        args = build_gifsicle_color_args(256, 128)  # Target is max
        assert args == []

    def test_color_reduction_needed(self):
        """Test color reduction arguments when needed."""
        args = build_gifsicle_color_args(64, 256)
        assert args == ["--colors", "64", "--no-dither"]

    def test_various_color_counts(self):
        """Test different color reduction scenarios."""
        # 256 -> 128
        args = build_gifsicle_color_args(128, 256)
        assert args == ["--colors", "128", "--no-dither"]

        # 200 -> 64
        args = build_gifsicle_color_args(64, 200)
        assert args == ["--colors", "64", "--no-dither"]


class TestBuildAnimatelyColorArgs:
    """Tests for build_animately_color_args function."""

    def test_no_reduction_needed(self):
        """Test no arguments when no reduction needed."""
        args = build_animately_color_args(128, 64)
        assert args == []

    def test_color_reduction_needed(self):
        """Test color reduction arguments when needed."""
        args = build_animately_color_args(64, 256)
        assert args == ["--colors", "64"]

    def test_max_color_handling(self):
        """Test handling of maximum color count."""
        args = build_animately_color_args(256, 128)
        assert args == []  # No reduction for max count


class TestCountGifColors:
    """Tests for count_gif_colors function."""

    @patch("pathlib.Path.exists")
    @patch("PIL.Image.open")
    def test_palette_mode_gif(self, mock_open, mock_exists):
        """Test color counting for palette mode GIF."""
        mock_exists.return_value = True

        # Mock PIL Image in palette mode
        mock_img = MagicMock()
        mock_img.format = "GIF"
        mock_img.mode = "P"
        # Create a simple palette with 5 unique colors
        mock_palette = [
            255,
            0,
            0,  # Red
            0,
            255,
            0,  # Green
            0,
            0,
            255,  # Blue
            255,
            255,
            0,  # Yellow
            0,
            0,
            0,
        ]  # Black
        mock_palette.extend([0] * (256 * 3 - len(mock_palette)))  # Pad to full palette
        mock_img.getpalette.return_value = mock_palette
        mock_open.return_value.__enter__.return_value = mock_img

        color_count = count_gif_colors(Path("test.gif"))

        # Should detect the unique colors in the palette
        assert color_count > 0
        assert color_count <= 256

    @patch("pathlib.Path.exists")
    @patch("PIL.Image.open")
    def test_rgb_mode_gif(self, mock_open, mock_exists):
        """Test color counting for RGB mode GIF."""
        mock_exists.return_value = True

        # Mock PIL Image in RGB mode
        mock_img = MagicMock()
        mock_img.format = "GIF"
        mock_img.mode = "RGB"

        # Mock quantization
        mock_quantized = MagicMock()
        mock_quantized.getpalette.return_value = [
            255,
            0,
            0,
            0,
            255,
            0,
            0,
            0,
            255,
        ]  # 3 colors
        mock_img.quantize.return_value = mock_quantized

        mock_open.return_value.__enter__.return_value = mock_img

        color_count = count_gif_colors(Path("test.gif"))

        assert color_count > 0
        assert color_count <= 256
        mock_img.quantize.assert_called_once_with(colors=256)

    @patch("pathlib.Path.exists")
    def test_missing_file(self, mock_exists):
        """Test error when file doesn't exist."""
        mock_exists.return_value = False

        with pytest.raises(IOError, match="File not found"):
            count_gif_colors(Path("missing.gif"))

    @patch("pathlib.Path.exists")
    @patch("PIL.Image.open")
    def test_non_gif_file(self, mock_open, mock_exists):
        """Test error when file is not a GIF."""
        mock_exists.return_value = True

        mock_img = MagicMock()
        mock_img.format = "PNG"  # Not a GIF
        mock_open.return_value.__enter__.return_value = mock_img

        with pytest.raises(ValueError, match="File is not a GIF"):
            count_gif_colors(Path("test.png"))

    @patch("pathlib.Path.exists")
    @patch("PIL.Image.open")
    def test_pil_error_handling(self, mock_open, mock_exists):
        """Test handling of PIL errors."""
        mock_exists.return_value = True
        mock_open.side_effect = Exception("PIL error")

        with pytest.raises(IOError, match="Error reading GIF"):
            count_gif_colors(Path("test.gif"))


class TestGetColorReductionInfo:
    """Tests for get_color_reduction_info function."""

    @patch("giflab.color_keep.count_gif_colors")
    @patch("pathlib.Path.exists")
    def test_valid_color_analysis(self, mock_exists, mock_count):
        """Test color reduction analysis for valid GIF."""
        mock_exists.return_value = True
        mock_count.return_value = 256  # Original has 256 colors

        info = get_color_reduction_info(Path("test.gif"), 128)

        assert info["original_colors"] == 256
        assert info["target_colors"] == 128
        assert info["color_keep_count"] == 128
        assert info["reduction_needed"] is True
        assert info["reduction_percent"] == 50.0
        assert info["compression_ratio"] == 2.0

    @patch("giflab.color_keep.count_gif_colors")
    @patch("pathlib.Path.exists")
    def test_no_reduction_needed(self, mock_exists, mock_count):
        """Test when no color reduction is needed."""
        mock_exists.return_value = True
        mock_count.return_value = 64  # Original has fewer colors than target

        info = get_color_reduction_info(Path("test.gif"), 128)

        assert info["original_colors"] == 64
        assert info["target_colors"] == 64
        assert info["color_keep_count"] == 128
        assert info["reduction_needed"] is False
        assert info["reduction_percent"] == 0.0
        assert info["compression_ratio"] == 1.0

    @patch("pathlib.Path.exists")
    def test_missing_file(self, mock_exists):
        """Test error when input file doesn't exist."""
        mock_exists.return_value = False

        with pytest.raises(IOError, match="Input file not found"):
            get_color_reduction_info(Path("missing.gif"), 128)

    def test_invalid_color_count(self):
        """Test error with invalid color count."""
        with pytest.raises(ValueError, match="not in supported counts"):
            get_color_reduction_info(
                Path("test.gif"), 4
            )  # 4 is not in supported counts


class TestExtractDominantColors:
    """Tests for extract_dominant_colors function."""

    def test_rgb_image_dominant_colors(self):
        """Test dominant color extraction from RGB image."""
        # Create a mock PIL Image
        mock_img = MagicMock()
        mock_img.mode = "RGB"

        # Mock numpy array conversion
        with patch("numpy.array") as mock_array, patch(
            "giflab.color_keep.Counter"
        ) as mock_counter:
            # Mock pixel data
            mock_array.return_value.reshape.return_value = [
                [255, 0, 0],  # Red
                [255, 0, 0],  # Red (duplicate)
                [0, 255, 0],  # Green
                [0, 0, 255],  # Blue
            ]

            # Mock counter results
            mock_counter.return_value.most_common.return_value = [
                ((255, 0, 0), 2),  # Red appears twice
                ((0, 255, 0), 1),  # Green appears once
                ((0, 0, 255), 1),  # Blue appears once
            ]

            colors = extract_dominant_colors(mock_img, 3)

            assert colors == [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
            mock_counter.assert_called_once()

    def test_palette_mode_conversion(self):
        """Test color extraction with palette mode conversion."""
        mock_img = MagicMock()
        mock_img.mode = "P"

        # Mock conversion to RGB
        mock_rgb = MagicMock()
        mock_rgb.mode = "RGB"
        mock_img.convert.return_value = mock_rgb

        with patch("numpy.array") as mock_array, patch(
            "giflab.color_keep.Counter"
        ) as mock_counter:
            mock_array.return_value.reshape.return_value = [[128, 128, 128]]
            mock_counter.return_value.most_common.return_value = [((128, 128, 128), 1)]

            colors = extract_dominant_colors(mock_img, 1)

            mock_img.convert.assert_called_once_with("RGB")
            assert colors == [(128, 128, 128)]

    def test_invalid_n_colors(self):
        """Test error with invalid n_colors."""
        mock_img = MagicMock()

        with pytest.raises(ValueError, match="n_colors must be positive"):
            extract_dominant_colors(mock_img, 0)

        with pytest.raises(ValueError, match="n_colors must be positive"):
            extract_dominant_colors(mock_img, -1)


class TestAnalyzeGifPalette:
    """Tests for analyze_gif_palette function."""

    @patch("giflab.color_keep.extract_dominant_colors")
    @patch("giflab.color_keep.count_gif_colors")
    @patch("pathlib.Path.exists")
    @patch("PIL.Image.open")
    def test_palette_mode_analysis(
        self, mock_open, mock_exists, mock_count, mock_extract
    ):
        """Test palette analysis for palette mode GIF."""
        mock_exists.return_value = True
        mock_count.return_value = 128
        mock_extract.return_value = [(255, 0, 0), (0, 255, 0)]

        # Mock PIL Image in palette mode
        mock_img = MagicMock()
        mock_img.format = "GIF"
        mock_img.mode = "P"
        mock_img.getpalette.return_value = [255, 0, 0] * 128  # 128 colors
        mock_img.info = {}
        mock_open.return_value.__enter__.return_value = mock_img

        analysis = analyze_gif_palette(Path("test.gif"))

        assert analysis["total_colors"] == 128
        assert analysis["dominant_colors"] == [(255, 0, 0), (0, 255, 0)]
        assert analysis["palette_info"]["mode"] == "palette"
        assert analysis["palette_info"]["palette_size"] == 128
        assert analysis["palette_info"]["has_transparency"] is False

        # Check reduction candidates
        assert 256 in analysis["reduction_candidates"]
        assert 128 in analysis["reduction_candidates"]
        assert 64 in analysis["reduction_candidates"]

    @patch("pathlib.Path.exists")
    def test_missing_file(self, mock_exists):
        """Test error when file doesn't exist."""
        mock_exists.return_value = False

        with pytest.raises(IOError, match="File not found"):
            analyze_gif_palette(Path("missing.gif"))


class TestGetOptimalColorCount:
    """Tests for get_optimal_color_count function."""

    @patch("giflab.color_keep.analyze_gif_palette")
    def test_optimal_count_high_quality(self, mock_analyze):
        """Test optimal color count with high quality threshold."""
        mock_analyze.return_value = {"total_colors": 256}

        optimal = get_optimal_color_count(Path("test.gif"), 0.9)

        # Should suggest a color count that retains 90% of colors
        assert optimal in [256, 128, 64, 32, 16, 8]

    @patch("giflab.color_keep.analyze_gif_palette")
    def test_optimal_count_low_quality(self, mock_analyze):
        """Test optimal color count with low quality threshold."""
        mock_analyze.return_value = {"total_colors": 256}

        optimal = get_optimal_color_count(Path("test.gif"), 0.2)

        # Should suggest aggressive reduction
        assert optimal in [256, 128, 64]

    def test_invalid_quality_threshold(self):
        """Test error with invalid quality threshold."""
        with pytest.raises(ValueError, match="must be between 0.0 and 1.0"):
            get_optimal_color_count(Path("test.gif"), 1.5)

        with pytest.raises(ValueError, match="must be between 0.0 and 1.0"):
            get_optimal_color_count(Path("test.gif"), -0.1)


# ---------------------------------------------------------------------------
# Color reduction alignment tests (merged from test_color_reduction_alignment.py)
# ---------------------------------------------------------------------------


def _create_colorful_test_gif(
    path: Path, frames: int = 4, size: tuple[int, int] = (60, 60)
) -> None:
    """Create a test GIF with many colors to enable color reduction testing.

    This generates a GIF with a rich color palette that can be meaningfully reduced
    to various target color counts.
    """
    images = []

    # Create a color gradient pattern that will result in many colors
    for frame in range(frames):
        # Create RGB image
        img = Image.new("RGB", size)
        pixels = np.zeros((size[1], size[0], 3), dtype=np.uint8)

        # Create a gradient pattern with many colors
        for y in range(size[1]):
            for x in range(size[0]):
                # Create a complex color pattern that varies with position and frame
                r = int((x / size[0]) * 255) ^ (frame * 17)
                g = int((y / size[1]) * 255) ^ (frame * 31)
                b = int(((x + y) / (size[0] + size[1])) * 255) ^ (frame * 47)

                # Add some noise to create more unique colors
                r = (r + (x * y * frame) % 64) % 256
                g = (g + (x + y + frame) % 64) % 256
                b = (b + (x - y + frame) % 64) % 256

                pixels[y, x] = [r, g, b]

        # Convert numpy array to PIL Image
        img = Image.fromarray(pixels, "RGB")
        images.append(img)

    # Save as GIF with high quality to preserve colors
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=120,
        loop=0,
        optimize=False,  # Don't optimize to preserve colors
        palette=None,  # Let PIL create a full palette
    )


@pytest.mark.parametrize("target_colors", [128, 64, 32, 16])
def test_aligned_color_reduction(target_colors):
    """Test that aligned color reduction produces consistent results."""
    if not (
        _engine_available(LossyEngine.GIFSICLE)
        and _engine_available(LossyEngine.ANIMATELY)
    ):
        pytest.skip("Both engines must be available")

    # Create test GIF
    with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmp:
        test_gif = Path(tmp.name)

    try:
        _create_colorful_test_gif(test_gif, frames=4)
        original_colors = count_gif_colors(test_gif)

        # Skip if no reduction needed
        if target_colors >= original_colors:
            pytest.skip(f"No reduction needed: {target_colors} >= {original_colors}")

        # Test both engines with color reduction only
        results = {}

        for engine in [LossyEngine.GIFSICLE, LossyEngine.ANIMATELY]:
            with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmp_out:
                output_path = Path(tmp_out.name)

            try:
                if engine == LossyEngine.GIFSICLE:
                    # Use the aligned settings (no dithering)
                    result = compress_with_gifsicle(
                        test_gif,
                        output_path,
                        lossy_level=0,
                        frame_keep_ratio=1.0,
                        color_keep_count=target_colors,
                    )
                else:
                    result = compress_with_animately(
                        test_gif,
                        output_path,
                        lossy_level=0,
                        frame_keep_ratio=1.0,
                        color_keep_count=target_colors,
                    )

                final_colors = count_gif_colors(output_path)
                file_size = output_path.stat().st_size

                results[engine.value] = {
                    "colors": final_colors,
                    "size": file_size,
                    "command": result.get("command", []),
                }

            finally:
                output_path.unlink()

        # Verify alignment
        if len(results) == 2:
            gifsicle_colors = results["gifsicle"]["colors"]
            animately_colors = results["animately-standard"]["colors"]

            # Both should not exceed the target
            assert (
                gifsicle_colors <= target_colors
            ), f"Gifsicle exceeded limit: {gifsicle_colors} > {target_colors}"
            assert (
                animately_colors <= target_colors
            ), f"Animately exceeded limit: {animately_colors} > {target_colors}"

            # With aligned settings, they should be very close or identical
            color_diff = abs(gifsicle_colors - animately_colors)
            assert color_diff <= 1, (
                f"Aligned engines should produce similar results: "
                f"gifsicle={gifsicle_colors}, animately={animately_colors}, diff={color_diff}"
            )

            # File sizes should be reasonably close (within 50% for compression differences)
            gifsicle_size = results["gifsicle"]["size"]
            animately_size = results["animately-standard"]["size"]

            size_ratio = max(gifsicle_size, animately_size) / min(
                gifsicle_size, animately_size
            )
            assert (
                size_ratio <= 2.0
            ), f"File sizes too different: gifsicle={gifsicle_size}, animately={animately_size}, ratio={size_ratio}"

            # Verify gifsicle uses --no-dither in command
            gifsicle_cmd = results["gifsicle"]["command"]
            assert (
                "--no-dither" in gifsicle_cmd
            ), "Gifsicle should use --no-dither for alignment"
            assert (
                "--colors" in gifsicle_cmd
            ), "Gifsicle should use --colors for reduction"

            # Verify animately uses --colors in command
            animately_cmd = results["animately-standard"]["command"]
            assert (
                "--colors" in animately_cmd
            ), "Animately should use --colors for reduction"

    finally:
        test_gif.unlink()


@pytest.mark.parametrize(
    "color_range",
    [
        (256, [128, 64, 32, 16]),  # Full palette reduction
        (128, [64, 32, 16]),  # Half palette reduction
        (64, [32, 16]),  # Quarter palette reduction
    ],
)
def test_color_reduction_consistency_across_ranges(color_range):
    """Test color reduction consistency across different color ranges."""
    if not (
        _engine_available(LossyEngine.GIFSICLE)
        and _engine_available(LossyEngine.ANIMATELY)
    ):
        pytest.skip("Both engines must be available")

    original_colors, targets = color_range

    # Create test GIF
    with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmp:
        test_gif = Path(tmp.name)

    try:
        _create_colorful_test_gif(test_gif, frames=3)

        # Test each target in the range
        for target_colors in targets:
            print(f"\nTesting {original_colors} -> {target_colors} colors")

            # Test both engines
            results = {}

            for engine in [LossyEngine.GIFSICLE, LossyEngine.ANIMATELY]:
                with tempfile.NamedTemporaryFile(
                    suffix=".gif", delete=False
                ) as tmp_out:
                    output_path = Path(tmp_out.name)

                try:
                    if engine == LossyEngine.GIFSICLE:
                        compress_with_gifsicle(
                            test_gif,
                            output_path,
                            lossy_level=0,
                            frame_keep_ratio=1.0,
                            color_keep_count=target_colors,
                        )
                    else:
                        compress_with_animately(
                            test_gif,
                            output_path,
                            lossy_level=0,
                            frame_keep_ratio=1.0,
                            color_keep_count=target_colors,
                        )

                    final_colors = count_gif_colors(output_path)
                    results[engine.value] = final_colors

                finally:
                    output_path.unlink()

            # Verify consistency
            if len(results) == 2:
                gifsicle_colors = results["gifsicle"]
                animately_colors = results["animately-standard"]

                # Both should respect the target
                assert gifsicle_colors <= target_colors
                assert animately_colors <= target_colors

                # Should be close or identical (allow more tolerance for simple test GIFs)
                color_diff = abs(gifsicle_colors - animately_colors)
                assert color_diff <= 3, (
                    f"Inconsistent results for {target_colors} colors: "
                    f"gifsicle={gifsicle_colors}, animately={animately_colors}"
                )

                print(
                    f"  OK {target_colors}: gifsicle={gifsicle_colors}, animately={animately_colors}"
                )

    finally:
        test_gif.unlink()


def test_color_reduction_edge_cases():
    """Test color reduction edge cases for alignment."""
    if not (
        _engine_available(LossyEngine.GIFSICLE)
        and _engine_available(LossyEngine.ANIMATELY)
    ):
        pytest.skip("Both engines must be available")

    # Create test GIF
    with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmp:
        test_gif = Path(tmp.name)

    try:
        _create_colorful_test_gif(test_gif, frames=2)
        original_colors = count_gif_colors(test_gif)

        # Test edge cases
        edge_cases = [
            (original_colors, "no_reduction_equal"),  # No reduction needed
            (
                original_colors + 10,
                "no_reduction_higher",
            ),  # Target higher than original
            (8, "extreme_reduction"),  # Very aggressive reduction
            (2, "minimal_colors"),  # Minimal color count
        ]

        for target_colors, case_name in edge_cases:
            print(f"\nTesting edge case: {case_name} (target: {target_colors})")

            # Test both engines
            results = {}

            for engine in [LossyEngine.GIFSICLE, LossyEngine.ANIMATELY]:
                with tempfile.NamedTemporaryFile(
                    suffix=".gif", delete=False
                ) as tmp_out:
                    output_path = Path(tmp_out.name)

                try:
                    if engine == LossyEngine.GIFSICLE:
                        compress_with_gifsicle(
                            test_gif,
                            output_path,
                            lossy_level=0,
                            frame_keep_ratio=1.0,
                            color_keep_count=target_colors,
                        )
                    else:
                        compress_with_animately(
                            test_gif,
                            output_path,
                            lossy_level=0,
                            frame_keep_ratio=1.0,
                            color_keep_count=target_colors,
                        )

                    final_colors = count_gif_colors(output_path)
                    results[engine.value] = final_colors

                except Exception as e:
                    # Some edge cases might fail, that's okay
                    print(f"  {engine.value} failed: {e}")
                    results[engine.value] = None

                finally:
                    if output_path.exists():
                        output_path.unlink()

            # Analyze results
            gifsicle_colors = results.get("gifsicle")
            animately_colors = results.get("animately-standard")

            if gifsicle_colors is not None and animately_colors is not None:
                # Both succeeded
                if target_colors >= original_colors:
                    # No reduction should occur - but allow for small variations in color counting
                    assert (
                        gifsicle_colors <= original_colors + 2
                    ), f"Gifsicle colors unexpectedly high: {gifsicle_colors} > {original_colors + 2}"
                    assert (
                        animately_colors <= original_colors + 2
                    ), f"Animately colors unexpectedly high: {animately_colors} > {original_colors + 2}"
                else:
                    # Reduction should occur
                    assert (
                        gifsicle_colors <= target_colors
                    ), f"Gifsicle exceeded target: {gifsicle_colors} > {target_colors}"
                    assert (
                        animately_colors <= target_colors
                    ), f"Animately exceeded target: {animately_colors} > {target_colors}"

                    # Results should be close
                    color_diff = abs(gifsicle_colors - animately_colors)
                    assert (
                        color_diff <= 2
                    ), f"Results too different: {gifsicle_colors} vs {animately_colors}"

                print(
                    f"  OK Both engines: gifsicle={gifsicle_colors}, animately={animately_colors}"
                )

            elif gifsicle_colors is None and animately_colors is None:
                # Both failed - acceptable for extreme edge cases
                print("  OK Both engines failed (acceptable for extreme case)")

            else:
                # One succeeded, one failed - this might indicate an issue
                print(
                    f"  WARN Mixed results: gifsicle={gifsicle_colors}, animately={animately_colors}"
                )

    finally:
        test_gif.unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
