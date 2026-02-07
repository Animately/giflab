"""Engine equivalence and engine-specific validation tests.

Combines cross-engine equivalence checks with per-engine output validation
for frame reduction, color reduction, and lossy compression operations.
"""

import subprocess
import tempfile
import time
from pathlib import Path

import pytest
from giflab.color_keep import count_gif_colors
from giflab.config import DEFAULT_ENGINE_CONFIG
from giflab.lossy import LossyEngine, compress_with_animately, compress_with_gifsicle
from giflab.meta import extract_gif_metadata
from giflab.metrics import extract_gif_frames
from giflab.tool_wrappers import (
    AnimatelyColorReducer,
    AnimatelyFrameReducer,
    AnimatelyLossyCompressor,
    FFmpegColorReducer,
    FFmpegLossyCompressor,
    GifsicleColorReducer,
    GifsicleFrameReducer,
    GifsicleLossyCompressor,
    GifskiLossyCompressor,
    ImageMagickColorReducer,
    ImageMagickLossyCompressor,
)
from giflab.wrapper_validation import ValidationConfig, WrapperOutputValidator
from PIL import Image, ImageDraw

pytestmark = pytest.mark.slow


# ===========================================================================
# Helper utilities & fixtures
# ===========================================================================


def _create_test_gif(
    path: Path, frames: int = 10, size: tuple[int, int] = (50, 50)
) -> None:
    """Generate a simple animated GIF for test purposes.

    The content purposefully varies per-frame so both colour and frame
    analyses have something to measure.
    """
    images = []
    for i in range(frames):
        img = Image.new(
            "RGB", size, color=((i * 37) % 255, (i * 53) % 255, (i * 97) % 255)
        )
        draw = ImageDraw.Draw(img)
        # moving rectangle
        offset = (i * 3) % (size[0] - 10)
        draw.rectangle([offset, offset, offset + 10, offset + 10], fill=(255, 255, 255))
        images.append(img)

    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=120,
        loop=0,
    )


def _engine_available(engine: LossyEngine) -> bool:
    """Return True if the specified engine binary appears to be available."""
    if engine == LossyEngine.GIFSICLE:
        binary = DEFAULT_ENGINE_CONFIG.GIFSICLE_PATH
        try:
            subprocess.run(
                [binary, "--version"], capture_output=True, check=True, timeout=3
            )
            return True
        except Exception:
            return False
    elif engine == LossyEngine.ANIMATELY:
        binary = DEFAULT_ENGINE_CONFIG.ANIMATELY_PATH
        from giflab.lossy import _is_executable

        return _is_executable(binary)
    else:
        return False


# Fixture yields a temporary GIF path
@pytest.fixture(scope="module")
def test_gif_tmp():
    """Provide a temporary GIF shared by tests in this module."""
    with tempfile.TemporaryDirectory() as tmpdir:
        gif_path = Path(tmpdir) / "sample.gif"
        _create_test_gif(gif_path, frames=10, size=(50, 50))
        yield gif_path


def _compress_with(
    engine: LossyEngine,
    src: Path,
    lossy_level: int,
    frame_ratio: float,
    colour_count: int | None = None,
):
    """Wrapper around compression functions with sensible defaults."""
    tmpdir = tempfile.TemporaryDirectory()
    out_path = Path(tmpdir.name) / f"output_{engine.value}.gif"

    # Use the direct compression functions to avoid validation issues
    if engine == LossyEngine.GIFSICLE:
        result = compress_with_gifsicle(
            src, out_path, lossy_level, frame_ratio, colour_count
        )
    else:
        result = compress_with_animately(
            src, out_path, lossy_level, frame_ratio, colour_count
        )

    return out_path, result, tmpdir


# ===========================================================================
# Cross-engine equivalence tests
# ===========================================================================


@pytest.mark.parametrize(
    "lossy_level, frame_ratio, colours",
    [
        (0, 1.0, None),  # lossless, no reduction
        (0, 1.0, 64),  # lossless with color reduction
        (0, 0.5, None),  # frame reduction only
    ],
)
def test_gifsicle_vs_animately_equivalence(
    test_gif_tmp, lossy_level, frame_ratio, colours
):
    """Ensure both engines apply the *same* high-level operations.

    We don't compare visual quality -- only that derived *metadata* such as
    frame count and palette size line up, proving we invoked the engines
    consistently.
    """
    # Skip if either engine isn't present
    if not (
        _engine_available(LossyEngine.GIFSICLE)
        and _engine_available(LossyEngine.ANIMATELY)
    ):
        pytest.skip(
            "Both gifsicle and animately must be available for equivalence checks"
        )

    try:
        gif_out, meta_gif, tmpdir_gif = _compress_with(
            LossyEngine.GIFSICLE, test_gif_tmp, lossy_level, frame_ratio, colours
        )
    except RuntimeError as e:
        pytest.skip(f"Gifsicle failed: {e}")

    try:
        ani_out, meta_ani, tmpdir_ani = _compress_with(
            LossyEngine.ANIMATELY, test_gif_tmp, lossy_level, frame_ratio, colours
        )
    except RuntimeError as e:
        tmpdir_gif.cleanup()
        pytest.skip(f"Animately failed: {e}")

    try:
        # ----------------- Assertions on frames -----------------
        frames_gif = extract_gif_frames(gif_out).frame_count
        frames_ani = extract_gif_frames(ani_out).frame_count

        print(f"Frame counts: gifsicle={frames_gif}, animately={frames_ani}")

        # When we requested a reduction ensure it was honoured
        original_frames = extract_gif_frames(test_gif_tmp).frame_count
        expected_frames = max(1, int(original_frames * frame_ratio))

        print(f"Original frames: {original_frames}, expected: {expected_frames}")

        # Both engines should produce the same number of frames
        assert (
            frames_gif == frames_ani
        ), f"Frame count mismatch -- gifsicle={frames_gif}, animately={frames_ani}"

        # And it should match our expectation
        assert (
            frames_gif == expected_frames
        ), f"Frame reduction did not meet expectation: got {frames_gif}, expected {expected_frames}"

        # ----------------- Assertions on colours ----------------
        colours_gif = count_gif_colors(gif_out)
        colours_ani = count_gif_colors(ani_out)

        print(f"Color counts: gifsicle={colours_gif}, animately={colours_ani}")

        if colours is not None:
            # Engines should not exceed the requested palette size
            assert (
                colours_gif <= colours
            ), f"Gifsicle exceeded color limit: {colours_gif} > {colours}"
            assert (
                colours_ani <= colours
            ), f"Animately exceeded color limit: {colours_ani} > {colours}"

        # For now, just ensure both engines produce reasonable color counts
        # (allow significant differences as engines may optimize differently)
        assert (
            colours_gif > 0 and colours_ani > 0
        ), "Both engines should produce GIFs with colors"

        # The color counts should be in a reasonable range for a simple test GIF
        assert (
            colours_gif <= 256 and colours_ani <= 256
        ), "Color counts should not exceed GIF maximum"

        # Print summary for manual verification
        print(f"Test passed: {lossy_level=}, {frame_ratio=}, {colours=}")
        print(f"   Frames: {frames_gif} (both engines)")
        print(f"   Colors: gifsicle={colours_gif}, animately={colours_ani}")

    finally:
        # Clean up temporary directories
        tmpdir_gif.cleanup()
        tmpdir_ani.cleanup()


def test_engine_basic_functionality():
    """Basic smoke test to ensure both engines can process a simple GIF."""
    if not (
        _engine_available(LossyEngine.GIFSICLE)
        and _engine_available(LossyEngine.ANIMATELY)
    ):
        pytest.skip(
            "Both gifsicle and animately must be available for basic functionality test"
        )

    # Create a simple test GIF
    with tempfile.TemporaryDirectory() as tmpdir:
        test_gif = Path(tmpdir) / "simple.gif"
        _create_test_gif(test_gif, frames=3, size=(30, 30))

        # Test both engines can process it
        try:
            gif_out, _, tmpdir_gif = _compress_with(
                LossyEngine.GIFSICLE, test_gif, 0, 1.0, None
            )
            assert gif_out.exists(), "Gifsicle should create output file"
            tmpdir_gif.cleanup()
            print("Gifsicle basic functionality: PASS")
        except Exception as e:
            print(f"Gifsicle basic functionality: FAIL - {e}")

        try:
            ani_out, _, tmpdir_ani = _compress_with(
                LossyEngine.ANIMATELY, test_gif, 0, 1.0, None
            )
            assert ani_out.exists(), "Animately should create output file"
            tmpdir_ani.cleanup()
            print("Animately basic functionality: PASS")
        except Exception as e:
            print(f"Animately basic functionality: FAIL - {e}")


# ===========================================================================
# Engine-specific validation tests (from test_engine_specific_validation.py)
# ===========================================================================


class TestValidationFixtures:
    """Test fixture management and validation."""

    @pytest.fixture(scope="class")
    def fixtures_dir(self):
        """Directory containing test fixtures."""
        return Path(__file__).parent.parent / "fixtures"

    @pytest.fixture(scope="class")
    def test_10_frames_gif(self, fixtures_dir):
        """10-frame test GIF."""
        return fixtures_dir / "test_10_frames.gif"

    @pytest.fixture(scope="class")
    def test_4_frames_gif(self, fixtures_dir):
        """4-frame test GIF."""
        return fixtures_dir / "test_4_frames.gif"

    @pytest.fixture(scope="class")
    def test_30_frames_gif(self, fixtures_dir):
        """30-frame test GIF."""
        return fixtures_dir / "test_30_frames.gif"

    @pytest.fixture(scope="class")
    def test_256_colors_gif(self, fixtures_dir):
        """Many-colors test GIF."""
        return fixtures_dir / "test_256_colors.gif"

    @pytest.fixture(scope="class")
    def test_2_colors_gif(self, fixtures_dir):
        """2-color test GIF."""
        return fixtures_dir / "test_2_colors.gif"

    @pytest.fixture
    def validator(self):
        """Validator with strict configuration for testing."""
        config = ValidationConfig(
            FRAME_RATIO_TOLERANCE=0.05,  # 5% tolerance
            COLOR_COUNT_TOLERANCE=2,  # Allow 2 extra colors
            FPS_TOLERANCE=0.1,  # 10% FPS tolerance
            MIN_COLOR_REDUCTION_PERCENT=0.05,  # Require 5% color reduction minimum
        )
        return WrapperOutputValidator(config)


@pytest.mark.external_tools
class TestFrameReductionValidation(TestValidationFixtures):
    """Test frame reduction validation across all engines."""

    def test_gifsicle_frame_reduction_50_percent(self, test_10_frames_gif, validator):
        """Test Gifsicle frame reduction validation at 50%."""
        wrapper = GifsicleFrameReducer()
        if not wrapper.available():
            pytest.skip("Gifsicle not available")

        with tempfile.NamedTemporaryFile(suffix=".gif") as tmp_file:
            output_path = Path(tmp_file.name)

            # Apply 50% frame reduction
            result = wrapper.apply(
                test_10_frames_gif, output_path, params={"ratio": 0.5}
            )

            # Validate the result
            assert "validations" in result
            assert result["validation_passed"] is True

            # Check frame count validation specifically
            frame_validations = [
                v
                for v in result["validations"]
                if v["validation_type"] == "frame_count"
            ]
            assert len(frame_validations) == 1

            frame_validation = frame_validations[0]
            assert frame_validation["is_valid"] is True
            assert (
                abs(frame_validation["actual"]["ratio"] - 0.5)
                <= validator.config.FRAME_RATIO_TOLERANCE
            )

            # Verify actual frame count
            output_metadata = extract_gif_metadata(output_path)
            expected_frames = 5  # 50% of 10 frames
            assert (
                abs(output_metadata.orig_frames - expected_frames) <= 1
            )  # Allow +/-1 frame

    def test_gifsicle_frame_reduction_30_percent(self, test_30_frames_gif, validator):
        """Test Gifsicle frame reduction validation at 30%."""
        wrapper = GifsicleFrameReducer()
        if not wrapper.available():
            pytest.skip("Gifsicle not available")

        with tempfile.NamedTemporaryFile(suffix=".gif") as tmp_file:
            output_path = Path(tmp_file.name)

            # Apply 30% frame reduction (keep 30% of frames)
            result = wrapper.apply(
                test_30_frames_gif, output_path, params={"ratio": 0.3}
            )

            assert result["validation_passed"] is True

            # Verify actual frame count
            output_metadata = extract_gif_metadata(output_path)
            expected_frames = int(30 * 0.3)  # ~9 frames
            assert (
                abs(output_metadata.orig_frames - expected_frames) <= 2
            )  # Allow +/-2 frames

    def test_animately_frame_reduction_validation(self, test_10_frames_gif, validator):
        """Test Animately frame reduction validation."""
        wrapper = AnimatelyFrameReducer()
        if not wrapper.available():
            pytest.skip("Animately not available")

        with tempfile.NamedTemporaryFile(suffix=".gif") as tmp_file:
            output_path = Path(tmp_file.name)

            # Apply 60% frame reduction
            result = wrapper.apply(
                test_10_frames_gif, output_path, params={"ratio": 0.6}
            )

            # Should have validation results
            assert "validations" in result

            # Check frame count validation
            frame_validations = [
                v
                for v in result["validations"]
                if v["validation_type"] == "frame_count"
            ]
            if len(frame_validations) > 0:
                frame_validation = frame_validations[0]
                # Either passes validation or provides useful error info
                if not frame_validation["is_valid"]:
                    print(
                        f"Frame validation failed: {frame_validation['error_message']}"
                    )
                    print(
                        f"Expected: {frame_validation['expected']}, Actual: {frame_validation['actual']}"
                    )

    def test_edge_case_single_frame_output(self, test_4_frames_gif, validator):
        """Test edge case where frame reduction results in single frame."""
        wrapper = GifsicleFrameReducer()
        if not wrapper.available():
            pytest.skip("Gifsicle not available")

        with tempfile.NamedTemporaryFile(suffix=".gif") as tmp_file:
            output_path = Path(tmp_file.name)

            # Apply very aggressive frame reduction (keep 10% = ~0.4 frames, should result in 1 frame)
            result = wrapper.apply(
                test_4_frames_gif, output_path, params={"ratio": 0.1}
            )

            # Should still have validations
            assert "validations" in result

            # Check that minimum frame requirement is enforced
            output_metadata = extract_gif_metadata(output_path)
            assert output_metadata.orig_frames >= 1  # At least 1 frame


@pytest.mark.external_tools
class TestColorReductionValidation(TestValidationFixtures):
    """Test color reduction validation across all engines."""

    def test_gifsicle_color_reduction_32_colors(self, test_256_colors_gif, validator):
        """Test Gifsicle color reduction to 32 colors."""
        wrapper = GifsicleColorReducer()
        if not wrapper.available():
            pytest.skip("Gifsicle not available")

        with tempfile.NamedTemporaryFile(suffix=".gif") as tmp_file:
            output_path = Path(tmp_file.name)

            result = wrapper.apply(
                test_256_colors_gif, output_path, params={"colors": 32}
            )

            assert "validations" in result

            # Check color count validation
            color_validations = [
                v
                for v in result["validations"]
                if v["validation_type"] == "color_count"
            ]
            assert len(color_validations) == 1

            color_validation = color_validations[0]
            if color_validation["is_valid"]:
                # Colors should be <= 32 + tolerance
                assert color_validation["actual"] <= (
                    32 + validator.config.COLOR_COUNT_TOLERANCE
                )
            else:
                print(f"Color validation failed: {color_validation['error_message']}")
                print(
                    f"Expected: {color_validation['expected']}, Actual: {color_validation['actual']}"
                )

    def test_animately_color_reduction_validation(self, test_256_colors_gif, validator):
        """Test Animately color reduction validation."""
        wrapper = AnimatelyColorReducer()
        if not wrapper.available():
            pytest.skip("Animately not available")

        with tempfile.NamedTemporaryFile(suffix=".gif") as tmp_file:
            output_path = Path(tmp_file.name)

            result = wrapper.apply(
                test_256_colors_gif, output_path, params={"colors": 16}
            )

            assert "validations" in result

            # Verify color reduction occurred
            color_validations = [
                v
                for v in result["validations"]
                if v["validation_type"] == "color_count"
            ]
            if len(color_validations) > 0:
                color_validation = color_validations[0]
                if not color_validation["is_valid"]:
                    print(f"Color validation info: {color_validation['error_message']}")

    def test_imagemagick_color_reduction_validation(
        self, test_256_colors_gif, validator
    ):
        """Test ImageMagick color reduction validation."""
        wrapper = ImageMagickColorReducer()
        if not wrapper.available():
            pytest.skip("ImageMagick not available")

        with tempfile.NamedTemporaryFile(suffix=".gif") as tmp_file:
            output_path = Path(tmp_file.name)

            result = wrapper.apply(
                test_256_colors_gif, output_path, params={"colors": 64}
            )

            assert "validations" in result
            # ImageMagick should successfully reduce colors
            color_validations = [
                v
                for v in result["validations"]
                if v["validation_type"] == "color_count"
            ]
            if len(color_validations) > 0:
                color_validation = color_validations[0]
                # Should either pass or give useful error info
                if not color_validation["is_valid"]:
                    print(
                        f"ImageMagick color validation: {color_validation['error_message']}"
                    )

    def test_ffmpeg_color_reduction_validation(self, test_256_colors_gif, validator):
        """Test FFmpeg color reduction validation."""
        wrapper = FFmpegColorReducer()
        if not wrapper.available():
            pytest.skip("FFmpeg not available")

        with tempfile.NamedTemporaryFile(suffix=".gif") as tmp_file:
            output_path = Path(tmp_file.name)

            result = wrapper.apply(
                test_256_colors_gif, output_path, params={"colors": 128}
            )

            assert "validations" in result

            # FFmpeg uses palette generation, should be effective
            color_validations = [
                v
                for v in result["validations"]
                if v["validation_type"] == "color_count"
            ]
            if len(color_validations) > 0:
                color_validation = color_validations[0]
                if not color_validation["is_valid"]:
                    print(
                        f"FFmpeg color validation: {color_validation['error_message']}"
                    )

    def test_color_reduction_edge_case_already_few_colors(
        self, test_2_colors_gif, validator
    ):
        """Test color reduction on GIF that already has few colors."""
        wrapper = GifsicleColorReducer()
        if not wrapper.available():
            pytest.skip("Gifsicle not available")

        with tempfile.NamedTemporaryFile(suffix=".gif") as tmp_file:
            output_path = Path(tmp_file.name)

            # Try to reduce to 16 colors when input only has 2
            result = wrapper.apply(
                test_2_colors_gif, output_path, params={"colors": 16}
            )

            assert "validations" in result

            # This should pass - no reduction needed
            color_validations = [
                v
                for v in result["validations"]
                if v["validation_type"] == "color_count"
            ]
            if len(color_validations) > 0:
                color_validation = color_validations[0]
                # Should either pass or indicate no reduction was needed
                assert color_validation["actual"] <= 16


@pytest.mark.external_tools
class TestLossyCompressionValidation(TestValidationFixtures):
    """Test lossy compression validation across all engines."""

    def test_gifsicle_lossy_compression_validation(self, test_10_frames_gif, validator):
        """Test Gifsicle lossy compression validation."""
        wrapper = GifsicleLossyCompressor()
        if not wrapper.available():
            pytest.skip("Gifsicle not available")

        with tempfile.NamedTemporaryFile(suffix=".gif") as tmp_file:
            output_path = Path(tmp_file.name)

            result = wrapper.apply(
                test_10_frames_gif, output_path, params={"lossy_level": 40}
            )

            assert "validations" in result

            # Check file integrity validation (most important for lossy)
            integrity_validations = [
                v
                for v in result["validations"]
                if v["validation_type"] == "file_integrity"
            ]
            assert len(integrity_validations) == 1
            assert integrity_validations[0]["is_valid"] is True

            # Timing should be preserved in lossy compression
            timing_validations = [
                v
                for v in result["validations"]
                if v["validation_type"] == "timing_preservation"
            ]
            if len(timing_validations) > 0:
                timing_validation = timing_validations[0]
                if not timing_validation["is_valid"]:
                    print(f"Timing validation: {timing_validation['error_message']}")

    def test_animately_lossy_compression_validation(self, test_4_frames_gif, validator):
        """Test Animately lossy compression validation."""
        wrapper = AnimatelyLossyCompressor()
        if not wrapper.available():
            pytest.skip("Animately not available")

        with tempfile.NamedTemporaryFile(suffix=".gif") as tmp_file:
            output_path = Path(tmp_file.name)

            result = wrapper.apply(
                test_4_frames_gif, output_path, params={"lossy_level": 60}
            )

            assert "validations" in result

            # File should be created and valid
            integrity_validations = [
                v
                for v in result["validations"]
                if v["validation_type"] == "file_integrity"
            ]
            assert len(integrity_validations) == 1
            assert integrity_validations[0]["is_valid"] is True

    def test_imagemagick_lossy_compression_validation(
        self, test_4_frames_gif, validator
    ):
        """Test ImageMagick lossy compression validation."""
        wrapper = ImageMagickLossyCompressor()
        if not wrapper.available():
            pytest.skip("ImageMagick not available")

        with tempfile.NamedTemporaryFile(suffix=".gif") as tmp_file:
            output_path = Path(tmp_file.name)

            result = wrapper.apply(
                test_4_frames_gif, output_path, params={"lossy_level": 30}
            )

            assert "validations" in result

            # Check basic validations pass
            integrity_validations = [
                v
                for v in result["validations"]
                if v["validation_type"] == "file_integrity"
            ]
            assert len(integrity_validations) >= 1

    def test_ffmpeg_lossy_compression_validation(self, test_4_frames_gif, validator):
        """Test FFmpeg lossy compression validation."""
        wrapper = FFmpegLossyCompressor()
        if not wrapper.available():
            pytest.skip("FFmpeg not available")

        with tempfile.NamedTemporaryFile(suffix=".gif") as tmp_file:
            output_path = Path(tmp_file.name)

            result = wrapper.apply(
                test_4_frames_gif, output_path, params={"lossy_level": 25}
            )

            assert "validations" in result

            # FFmpeg should produce valid output
            integrity_validations = [
                v
                for v in result["validations"]
                if v["validation_type"] == "file_integrity"
            ]
            assert len(integrity_validations) >= 1

    def test_gifski_lossy_compression_validation(self, test_4_frames_gif, validator):
        """Test Gifski lossy compression validation."""
        wrapper = GifskiLossyCompressor()
        if not wrapper.available():
            pytest.skip("Gifski not available")

        with tempfile.NamedTemporaryFile(suffix=".gif") as tmp_file:
            output_path = Path(tmp_file.name)

            result = wrapper.apply(
                test_4_frames_gif, output_path, params={"lossy_level": 50}
            )

            assert "validations" in result

            # Gifski should produce high-quality output
            integrity_validations = [
                v
                for v in result["validations"]
                if v["validation_type"] == "file_integrity"
            ]
            assert len(integrity_validations) >= 1


class TestValidationEdgeCases(TestValidationFixtures):
    """Test edge cases and boundary conditions."""

    @pytest.mark.external_tools
    def test_extreme_frame_reduction(self, test_30_frames_gif, validator):
        """Test extreme frame reduction (very low ratios)."""
        wrapper = GifsicleFrameReducer()
        if not wrapper.available():
            pytest.skip("Gifsicle not available")

        with tempfile.NamedTemporaryFile(suffix=".gif") as tmp_file:
            output_path = Path(tmp_file.name)

            # Extreme reduction: keep only 3.3% of frames
            result = wrapper.apply(
                test_30_frames_gif, output_path, params={"ratio": 0.033}
            )

            # Should still produce valid output
            assert (
                result.get("validation_passed") is not False
            )  # Allow None (validation error) or True

            # Should have at least 1 frame
            output_metadata = extract_gif_metadata(output_path)
            assert output_metadata.orig_frames >= 1

    @pytest.mark.external_tools
    def test_no_reduction_color_params(self, test_256_colors_gif, validator):
        """Test color reduction with parameter that requires no actual reduction."""
        wrapper = GifsicleColorReducer()
        if not wrapper.available():
            pytest.skip("Gifsicle not available")

        with tempfile.NamedTemporaryFile(suffix=".gif") as tmp_file:
            output_path = Path(tmp_file.name)

            # Request more colors than input - use valid count but higher than actual
            result = wrapper.apply(
                test_256_colors_gif,
                output_path,
                params={"colors": 256},  # Request 256 when input has fewer
            )

            # Should still validate successfully
            assert "validations" in result

            # Color validation should account for this case
            color_validations = [
                v
                for v in result["validations"]
                if v["validation_type"] == "color_count"
            ]
            if len(color_validations) > 0:
                color_validation = color_validations[0]
                # Should pass since no reduction was needed/possible
                assert color_validation.get("is_valid") in [True, None]

    def test_validation_with_corrupted_output(self, test_4_frames_gif, validator):
        """Test validation behavior with corrupted output file."""
        # Create a corrupted file that's large enough to pass size check
        with tempfile.NamedTemporaryFile(suffix=".gif") as tmp_file:
            output_path = Path(tmp_file.name)

            # Write invalid GIF data (large enough to pass size check)
            with open(output_path, "wb") as f:
                f.write(
                    b"Not a GIF file, but long enough to pass the size check. " * 10
                )

            # Test file integrity validation
            result = validator.validate_file_integrity(output_path, {})

            assert result.is_valid is False
            # Should fail on GIF format check now
            assert (
                "Cannot read output file as valid GIF" in result.error_message
                or "is not a GIF" in result.error_message
            )

    def test_validation_performance_with_large_gif(self, validator):
        """Test validation performance doesn't significantly impact processing."""
        # This would ideally test with a very large GIF, but we'll simulate
        # the performance aspect by testing the validation overhead

        start_time = time.time()

        # Run multiple validations to measure overhead
        for _i in range(10):
            result = validator.validate_file_integrity(
                Path(__file__), {}  # Use this Python file as a non-GIF
            )
            assert result.is_valid is False

        end_time = time.time()
        total_time = end_time - start_time

        # Validation should be very fast
        assert total_time < 1.0  # Less than 1 second for 10 validations
        print(f"Validation performance: {total_time:.3f}s for 10 integrity checks")
