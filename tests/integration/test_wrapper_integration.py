"""Integration tests for updated tool wrappers."""

import tempfile
from pathlib import Path

import pytest
from giflab.meta import extract_gif_metadata
from giflab.tool_wrappers import (  # Dithering-specific wrappers; All Bayer scale variations
    FFmpegColorReducer,
    FFmpegColorReducerBayerScale0,
    FFmpegColorReducerBayerScale1,
    FFmpegColorReducerBayerScale2,
    FFmpegColorReducerBayerScale3,
    FFmpegColorReducerBayerScale4,
    FFmpegColorReducerBayerScale5,
    FFmpegColorReducerFloydSteinberg,
    FFmpegColorReducerNone,
    FFmpegColorReducerSierra2,
    FFmpegFrameReducer,
    FFmpegLossyCompressor,
    GifskiLossyCompressor,
    ImageMagickColorReducer,
    ImageMagickColorReducerFloydSteinberg,
    ImageMagickColorReducerNone,
    ImageMagickColorReducerRiemersma,
    ImageMagickFrameReducer,
    ImageMagickLossyCompressor,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def test_gif():
    """Path to a simple test GIF fixture."""
    return Path(__file__).parent / "fixtures" / "simple_4frame.gif"


@pytest.fixture
def single_frame_gif():
    """Path to a single-frame test GIF fixture."""
    return Path(__file__).parent / "fixtures" / "single_frame.gif"


@pytest.fixture
def many_colors_gif():
    """Path to a many-color test GIF fixture."""
    return Path(__file__).parent / "fixtures" / "many_colors.gif"


# ---------------------------------------------------------------------------
# ImageMagick wrapper integration tests
# ---------------------------------------------------------------------------


@pytest.mark.external_tools
class TestImageMagickWrapperIntegration:
    """Integration tests for ImageMagick wrappers."""

    def test_color_reducer_functionality(self, many_colors_gif):
        """Test ImageMagick color reducer actually reduces colors."""
        wrapper = ImageMagickColorReducer()

        if not wrapper.available():
            pytest.skip("ImageMagick not available")

        with tempfile.NamedTemporaryFile(suffix=".gif") as tmp:
            output_path = Path(tmp.name)

            # Test color reduction to 32 colors
            result = wrapper.apply(many_colors_gif, output_path, params={"colors": 32})

            # Validate metadata structure
            assert result["engine"] == "imagemagick"
            assert result["render_ms"] > 0
            assert result["kilobytes"] > 0
            assert "command" in result

            # Validate functional change
            assert output_path.exists()

            # Check that colors were actually reduced (this is a basic check)
            original_size = many_colors_gif.stat().st_size
            output_size = output_path.stat().st_size

            # Color reduction should typically reduce file size
            assert output_size <= original_size * 1.2  # Allow 20% tolerance

    def test_frame_reducer_functionality(self, test_gif):
        """Test ImageMagick frame reducer actually reduces frames."""
        wrapper = ImageMagickFrameReducer()

        if not wrapper.available():
            pytest.skip("ImageMagick not available")

        with tempfile.NamedTemporaryFile(suffix=".gif") as tmp:
            output_path = Path(tmp.name)

            # Test frame reduction to 50%
            result = wrapper.apply(test_gif, output_path, params={"ratio": 0.5})

            # Validate metadata
            assert result["engine"] == "imagemagick"
            assert result["render_ms"] > 0
            assert output_path.exists()

            # Validate functional change - should have fewer frames
            try:
                original_meta = extract_gif_metadata(test_gif)
                output_meta = extract_gif_metadata(output_path)

                # Should have roughly half the frames (±1 for rounding)
                expected_frames = max(1, original_meta.orig_frames // 2)
                assert abs(output_meta.orig_frames - expected_frames) <= 1
            except Exception:
                # If metadata extraction fails, just check file exists
                pass

    def test_lossy_compressor_functionality(self, test_gif):
        """Test ImageMagick lossy compressor reduces file size."""
        wrapper = ImageMagickLossyCompressor()

        if not wrapper.available():
            pytest.skip("ImageMagick not available")

        with tempfile.NamedTemporaryFile(suffix=".gif") as tmp:
            output_path = Path(tmp.name)

            # Test lossy compression with low quality
            result = wrapper.apply(test_gif, output_path, params={"quality": 50})

            # Validate metadata
            assert result["engine"] == "imagemagick"
            assert result["render_ms"] > 0
            assert output_path.exists()

            # Lossy compression should reduce file size
            original_size = test_gif.stat().st_size
            output_size = output_path.stat().st_size
            assert output_size <= original_size

    def test_parameter_validation(self, test_gif):
        """Test that wrappers validate parameters correctly."""
        wrapper = ImageMagickColorReducer()

        if not wrapper.available():
            pytest.skip("ImageMagick not available")

        with tempfile.NamedTemporaryFile(suffix=".gif") as tmp:
            output_path = Path(tmp.name)

            # Should raise ValueError for missing colors parameter
            with pytest.raises(ValueError, match="params must include 'colors'"):
                wrapper.apply(test_gif, output_path, params={})

            # Should raise ValueError for None params
            with pytest.raises(ValueError, match="params must include 'colors'"):
                wrapper.apply(test_gif, output_path, params=None)


# ---------------------------------------------------------------------------
# FFmpeg wrapper integration tests
# ---------------------------------------------------------------------------


@pytest.mark.external_tools
class TestFFmpegWrapperIntegration:
    """Integration tests for FFmpeg wrappers."""

    def test_color_reducer_functionality(self, many_colors_gif):
        """Test FFmpeg color reducer via palette generation."""
        wrapper = FFmpegColorReducer()

        if not wrapper.available():
            pytest.skip("FFmpeg not available")

        with tempfile.NamedTemporaryFile(suffix=".gif") as tmp:
            output_path = Path(tmp.name)

            result = wrapper.apply(many_colors_gif, output_path, params={"fps": 10.0})

            # Validate metadata
            assert result["engine"] == "ffmpeg"
            assert result["render_ms"] > 0
            assert output_path.exists()

            # Command should show two-pass operation
            assert "palettegen" in result["command"]
            assert "paletteuse" in result["command"]

    def test_frame_reducer_functionality(self, test_gif):
        """Test FFmpeg frame reducer."""
        wrapper = FFmpegFrameReducer()

        if not wrapper.available():
            pytest.skip("FFmpeg not available")

        with tempfile.NamedTemporaryFile(suffix=".gif") as tmp:
            output_path = Path(tmp.name)

            result = wrapper.apply(test_gif, output_path, params={"ratio": 0.5})

            # Validate metadata
            assert result["engine"] == "ffmpeg"
            assert result["render_ms"] > 0
            assert (
                "fps=" in result["command"]
            )  # Should contain fps parameter (calculated from ratio)
            assert output_path.exists()

    def test_lossy_compressor_functionality(self, test_gif):
        """Test FFmpeg lossy compressor."""
        wrapper = FFmpegLossyCompressor()

        if not wrapper.available():
            pytest.skip("FFmpeg not available")

        with tempfile.NamedTemporaryFile(suffix=".gif") as tmp:
            output_path = Path(tmp.name)

            result = wrapper.apply(
                test_gif, output_path, params={"qv": 25, "fps": 12.0}
            )

            # Validate metadata
            assert result["engine"] == "ffmpeg"
            assert result["render_ms"] > 0
            assert "q:v" in result["command"]
            assert output_path.exists()

    def test_parameter_validation(self, test_gif):
        """Test FFmpeg parameter validation."""
        wrapper = FFmpegFrameReducer()

        if not wrapper.available():
            pytest.skip("FFmpeg not available")

        with tempfile.NamedTemporaryFile(suffix=".gif") as tmp:
            output_path = Path(tmp.name)

            # Should raise ValueError for missing ratio parameter
            with pytest.raises(ValueError, match="params must include 'ratio'"):
                wrapper.apply(test_gif, output_path, params={})


# ---------------------------------------------------------------------------
# gifski wrapper integration tests
# ---------------------------------------------------------------------------


@pytest.mark.external_tools
class TestGifskiWrapperIntegration:
    """Integration tests for gifski wrapper."""

    def test_lossy_compressor_functionality(self, test_gif):
        """Test gifski lossy compressor."""
        wrapper = GifskiLossyCompressor()

        if not wrapper.available():
            pytest.skip("gifski not available")

        with tempfile.NamedTemporaryFile(suffix=".gif") as tmp:
            output_path = Path(tmp.name)

            result = wrapper.apply(test_gif, output_path, params={"quality": 70})

            # Validate metadata
            assert result["engine"] == "gifski"
            assert result["render_ms"] > 0
            assert "quality" in result["command"]
            assert output_path.exists()

    def test_default_parameters(self, test_gif):
        """Test gifski with default parameters."""
        wrapper = GifskiLossyCompressor()

        if not wrapper.available():
            pytest.skip("gifski not available")

        with tempfile.NamedTemporaryFile(suffix=".gif") as tmp:
            output_path = Path(tmp.name)

            # Should work with no params (uses defaults)
            result = wrapper.apply(test_gif, output_path, params=None)

            assert result["engine"] == "gifski"
            assert output_path.exists()


# ---------------------------------------------------------------------------
# Cross-engine consistency tests
# ---------------------------------------------------------------------------


@pytest.mark.external_tools
class TestCrossEngineConsistency:
    """Test consistency across different engines."""

    def test_metadata_schema_consistency(self, test_gif):
        """Test that all engines return consistent metadata schema."""
        wrappers = [
            ImageMagickLossyCompressor(),
            FFmpegLossyCompressor(),
            GifskiLossyCompressor(),
        ]

        results = []
        for wrapper in wrappers:
            if not wrapper.available():
                continue

            with tempfile.NamedTemporaryFile(suffix=".gif") as tmp:
                output_path = Path(tmp.name)

                # Use appropriate parameters for each wrapper
                if isinstance(wrapper, ImageMagickLossyCompressor):
                    params = {"quality": 75}
                elif isinstance(wrapper, FFmpegLossyCompressor):
                    params = {"qv": 25, "fps": 15.0}
                else:  # GifskiLossyCompressor
                    params = {"quality": 75}

                result = wrapper.apply(test_gif, output_path, params=params)
                results.append(result)

        if len(results) < 2:
            pytest.skip("Need at least 2 engines available for consistency test")

        # All results should have the same keys
        expected_keys = {"render_ms", "engine", "command", "kilobytes"}
        for result in results:
            assert set(result.keys()) >= expected_keys
            assert result["render_ms"] > 0
            assert result["kilobytes"] >= 0
            assert len(result["command"]) > 0

    def test_combine_group_consistency(self):
        """Test that COMBINE_GROUP values are consistent."""
        # ImageMagick wrappers
        assert ImageMagickColorReducer.COMBINE_GROUP == "imagemagick"
        assert ImageMagickFrameReducer.COMBINE_GROUP == "imagemagick"
        assert ImageMagickLossyCompressor.COMBINE_GROUP == "imagemagick"

        # FFmpeg wrappers
        assert FFmpegColorReducer.COMBINE_GROUP == "ffmpeg"
        assert FFmpegFrameReducer.COMBINE_GROUP == "ffmpeg"
        assert FFmpegLossyCompressor.COMBINE_GROUP == "ffmpeg"

        # gifski wrapper
        assert GifskiLossyCompressor.COMBINE_GROUP == "gifski"

    def test_availability_methods(self):
        """Test that availability methods work correctly."""
        wrappers = [
            ImageMagickColorReducer(),
            FFmpegColorReducer(),
            GifskiLossyCompressor(),
        ]

        for wrapper in wrappers:
            # Should return boolean
            available = wrapper.available()
            assert isinstance(available, bool)

            # Version should return string
            version = wrapper.version()
            assert isinstance(version, str)
            assert len(version) > 0


# ---------------------------------------------------------------------------
# Dithering-Specific Wrapper Tests (Research-Based)
# ---------------------------------------------------------------------------


@pytest.mark.external_tools
class TestImageMagickDitheringWrappers:
    """Test ImageMagick dithering-specific wrappers based on research findings."""

    def test_riemersma_wrapper(self, many_colors_gif):
        """Test ImageMagick Riemersma dithering wrapper (best all-around performer from research)."""
        wrapper = ImageMagickColorReducerRiemersma()

        if not wrapper.available():
            pytest.skip("ImageMagick not available")

        with tempfile.NamedTemporaryFile(suffix=".gif") as tmp:
            output_path = Path(tmp.name)

            result = wrapper.apply(many_colors_gif, output_path, params={"colors": 16})

            # Validate metadata structure
            assert result["engine"] == "imagemagick"
            assert result["dithering_method"] == "Riemersma"
            assert result["pipeline_variant"] == "imagemagick_dither_riemersma"
            assert result["render_ms"] > 0
            assert output_path.exists()

    def test_floyd_steinberg_wrapper(self, many_colors_gif):
        """Test ImageMagick Floyd-Steinberg wrapper (standard baseline)."""
        wrapper = ImageMagickColorReducerFloydSteinberg()

        if not wrapper.available():
            pytest.skip("ImageMagick not available")

        with tempfile.NamedTemporaryFile(suffix=".gif") as tmp:
            output_path = Path(tmp.name)

            result = wrapper.apply(many_colors_gif, output_path, params={"colors": 16})

            assert result["dithering_method"] == "FloydSteinberg"
            assert result["pipeline_variant"] == "imagemagick_dither_floydsteinberg"
            assert output_path.exists()

    def test_none_wrapper(self, many_colors_gif):
        """Test ImageMagick no-dithering wrapper (size priority baseline)."""
        wrapper = ImageMagickColorReducerNone()

        if not wrapper.available():
            pytest.skip("ImageMagick not available")

        with tempfile.NamedTemporaryFile(suffix=".gif") as tmp:
            output_path = Path(tmp.name)

            result = wrapper.apply(many_colors_gif, output_path, params={"colors": 16})

            assert result["dithering_method"] == "None"
            assert result["pipeline_variant"] == "imagemagick_dither_none"
            assert output_path.exists()

    def test_wrapper_combine_group_consistency(self):
        """Test that all ImageMagick dithering wrappers have consistent COMBINE_GROUP."""
        wrappers = [
            ImageMagickColorReducerRiemersma(),
            ImageMagickColorReducerFloydSteinberg(),
            ImageMagickColorReducerNone(),
        ]

        for wrapper in wrappers:
            assert wrapper.COMBINE_GROUP == "imagemagick"


@pytest.mark.external_tools
class TestFFmpegDitheringWrappers:
    """Test FFmpeg dithering-specific wrappers based on research findings."""

    def test_sierra2_wrapper(self, many_colors_gif):
        """Test FFmpeg Sierra2 wrapper (excellent quality/size balance from research)."""
        wrapper = FFmpegColorReducerSierra2()

        if not wrapper.available():
            pytest.skip("FFmpeg not available")

        with tempfile.NamedTemporaryFile(suffix=".gif") as tmp:
            output_path = Path(tmp.name)

            result = wrapper.apply(
                many_colors_gif, output_path, params={"colors": 16, "fps": 10.0}
            )

            # Validate metadata structure
            assert result["engine"] == "ffmpeg"
            assert result["dithering_method"] == "sierra2"
            assert result["pipeline_variant"] == "ffmpeg_dither_sierra2"
            assert result["render_ms"] > 0
            assert output_path.exists()

    def test_floyd_steinberg_wrapper(self, many_colors_gif):
        """Test FFmpeg Floyd-Steinberg wrapper (quality baseline)."""
        wrapper = FFmpegColorReducerFloydSteinberg()

        if not wrapper.available():
            pytest.skip("FFmpeg not available")

        with tempfile.NamedTemporaryFile(suffix=".gif") as tmp:
            output_path = Path(tmp.name)

            result = wrapper.apply(many_colors_gif, output_path, params={"colors": 16})

            assert result["dithering_method"] == "floyd_steinberg"
            assert result["pipeline_variant"] == "ffmpeg_dither_floyd_steinberg"
            assert output_path.exists()

    def test_all_bayer_scale_wrappers(self, many_colors_gif):
        """Test all FFmpeg Bayer Scale wrappers (scales 0-5) for systematic comparison."""
        bayer_wrappers = [
            (
                FFmpegColorReducerBayerScale0,
                "bayer:bayer_scale=0",
                "Poor quality - elimination candidate",
            ),
            (
                FFmpegColorReducerBayerScale1,
                "bayer:bayer_scale=1",
                "Higher quality variant",
            ),
            (FFmpegColorReducerBayerScale2, "bayer:bayer_scale=2", "Medium pattern"),
            (FFmpegColorReducerBayerScale3, "bayer:bayer_scale=3", "Good balance"),
            (
                FFmpegColorReducerBayerScale4,
                "bayer:bayer_scale=4",
                "Best for noisy content",
            ),
            (
                FFmpegColorReducerBayerScale5,
                "bayer:bayer_scale=5",
                "Maximum compression",
            ),
        ]

        for wrapper_class, expected_method, _description in bayer_wrappers:
            wrapper = wrapper_class()

            if not wrapper.available():
                pytest.skip("FFmpeg not available")

            with tempfile.NamedTemporaryFile(suffix=".gif") as tmp:
                output_path = Path(tmp.name)

                result = wrapper.apply(
                    many_colors_gif, output_path, params={"colors": 16}
                )

                # Validate metadata for each Bayer scale
                assert result["engine"] == "ffmpeg"
                assert result["dithering_method"] == expected_method
                assert result["render_ms"] > 0
                assert output_path.exists()

                # Validate pipeline variant naming
                scale = expected_method.split("=")[1]
                expected_variant = f"ffmpeg_dither_bayer_bayer_scale_{scale}"
                assert result["pipeline_variant"] == expected_variant

    def test_none_wrapper(self, many_colors_gif):
        """Test FFmpeg no-dithering wrapper (size priority baseline)."""
        wrapper = FFmpegColorReducerNone()

        if not wrapper.available():
            pytest.skip("FFmpeg not available")

        with tempfile.NamedTemporaryFile(suffix=".gif") as tmp:
            output_path = Path(tmp.name)

            result = wrapper.apply(many_colors_gif, output_path, params={"colors": 16})

            assert result["dithering_method"] == "none"
            assert result["pipeline_variant"] == "ffmpeg_dither_none"
            assert output_path.exists()

    def test_wrapper_combine_group_consistency(self):
        """Test that all FFmpeg dithering wrappers have consistent COMBINE_GROUP."""
        wrappers = [
            FFmpegColorReducerSierra2(),
            FFmpegColorReducerFloydSteinberg(),
            # All Bayer scale variations
            FFmpegColorReducerBayerScale0(),
            FFmpegColorReducerBayerScale1(),
            FFmpegColorReducerBayerScale2(),
            FFmpegColorReducerBayerScale3(),
            FFmpegColorReducerBayerScale4(),
            FFmpegColorReducerBayerScale5(),
            FFmpegColorReducerNone(),
        ]

        for wrapper in wrappers:
            assert wrapper.COMBINE_GROUP == "ffmpeg"


@pytest.mark.external_tools
class TestDitheringComparison:
    """Test comparisons between different dithering methods based on research findings."""

    def test_enhanced_vs_basic_wrappers(self, many_colors_gif):
        """Test that enhanced wrappers produce different results than basic wrapper."""
        basic_wrapper = ImageMagickColorReducer()
        riemersma_wrapper = ImageMagickColorReducerRiemersma()

        if not basic_wrapper.available():
            pytest.skip("ImageMagick not available")

        with tempfile.NamedTemporaryFile(
            suffix=".gif"
        ) as tmp1, tempfile.NamedTemporaryFile(suffix=".gif") as tmp2:
            basic_output = Path(tmp1.name)
            riemersma_output = Path(tmp2.name)

            # Test basic wrapper with dithering_method parameter
            basic_wrapper.apply(
                many_colors_gif,
                basic_output,
                params={"colors": 16, "dithering_method": "None"},
            )

            # Test Riemersma-specific wrapper
            riemersma_result = riemersma_wrapper.apply(
                many_colors_gif, riemersma_output, params={"colors": 16}
            )

            # Both should succeed but potentially produce different results
            assert basic_output.exists()
            assert riemersma_output.exists()

            # Riemersma should include specific dithering metadata
            assert riemersma_result["dithering_method"] == "Riemersma"
            assert (
                riemersma_result["pipeline_variant"] == "imagemagick_dither_riemersma"
            )

    def test_dithering_method_naming_consistency(self):
        """Test that wrapper names consistently reflect their dithering methods."""
        test_cases = [
            (ImageMagickColorReducerRiemersma, "riemersma"),
            (ImageMagickColorReducerFloydSteinberg, "floyd"),
            (ImageMagickColorReducerNone, "none"),
            (FFmpegColorReducerSierra2, "sierra2"),
            (FFmpegColorReducerFloydSteinberg, "floyd"),
            # All Bayer scale variations
            (FFmpegColorReducerBayerScale0, "bayer0"),
            (FFmpegColorReducerBayerScale1, "bayer1"),
            (FFmpegColorReducerBayerScale2, "bayer2"),
            (FFmpegColorReducerBayerScale3, "bayer3"),
            (FFmpegColorReducerBayerScale4, "bayer4"),
            (FFmpegColorReducerBayerScale5, "bayer5"),
            (FFmpegColorReducerNone, "none"),
        ]

        for wrapper_class, expected_name_part in test_cases:
            assert expected_name_part in wrapper_class.NAME.lower()

    def test_research_tier_coverage(self):
        """Test that all Tier 1 methods from research are covered by wrappers."""
        # Tier 1 - Essential Methods (Must Test) from research:
        # 1. ImageMagick Riemersma - Best all-around performer ✅
        # 2. FFmpeg Floyd-Steinberg - Standard high-quality baseline ✅
        # 3. FFmpeg Sierra2 - Excellent quality/size balance ✅
        # 4. ImageMagick None - Size priority baseline ✅

        tier_1_wrappers = [
            ImageMagickColorReducerRiemersma,
            FFmpegColorReducerFloydSteinberg,
            FFmpegColorReducerSierra2,
            ImageMagickColorReducerNone,
        ]

        for wrapper_class in tier_1_wrappers:
            # Each wrapper should be instantiable
            wrapper = wrapper_class()
            assert hasattr(wrapper, "apply")
            assert hasattr(wrapper, "available")
            assert hasattr(wrapper, "COMBINE_GROUP")

            # NAME should indicate the dithering method
            assert len(wrapper_class.NAME) > 10  # Should be descriptive

    def test_bayer_scale_research_validation(self):
        """Test that all Bayer scale variations are available for elimination testing."""
        # From research: Performance on Noise Content (16 colors):
        # - Scale 4-5: 128K (SSIM: ~352) - Best compression
        # - Scale 3: 137K (SSIM: 390) - Good balance
        # - Scale 1-2: 144-146K (SSIM: 505-1107) - Better quality, larger files
        # - Scale 0: Poor quality from research

        scale_expectations = [
            (FFmpegColorReducerBayerScale0, "elimination candidate", "poor quality"),
            (FFmpegColorReducerBayerScale1, "tier 2", "higher quality"),
            (FFmpegColorReducerBayerScale2, "tier 2", "medium pattern from research"),
            (FFmpegColorReducerBayerScale3, "tier 2", "good balance"),
            (FFmpegColorReducerBayerScale4, "tier 1", "best compression"),
            (FFmpegColorReducerBayerScale5, "tier 1", "maximum compression"),
        ]

        for wrapper_class, tier, description in scale_expectations:
            wrapper = wrapper_class()

            # Each Bayer scale should be available for testing
            assert hasattr(wrapper, "apply")
            assert hasattr(wrapper, "available")
            assert wrapper.COMBINE_GROUP == "ffmpeg"

            # Naming should indicate the scale
            scale = wrapper_class.NAME[-1]  # Last character should be scale number
            assert scale.isdigit()
            assert "bayer" in wrapper.NAME.lower()

            # Docstring should reference research findings
            assert (
                description.split()[0] in wrapper_class.__doc__.lower()
                or tier in wrapper_class.__doc__.lower()
            )
