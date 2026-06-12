"""Functional tests for the ffmpeg/imagemagick lossy palette mapping.

The public ``lossy_level`` (0-100) for the ffmpeg and imagemagick lossy
wrappers maps onto a real, engine-native GIF lossy axis: palette size +
dithering (``tool_wrappers._lossy_level_to_palette_size``). These tests pin
the mapping, the wrapper-side range validation, and the exact commands the
engine helpers construct -- all with ``run_command`` mocked, so no engine
binaries are needed.

Regression context: before 2026-06-12 both wrappers routed ``lossy_level``
to knobs that do not affect GIF pixels (ffmpeg ``-q:v``, a video-DCT knob;
imagemagick ``-quality``, a PNG/JPEG zlib knob), producing byte-identical
output at every level (2026-06-09 finding in
``scripts/audit/engine_lossy_calibration.py``). The real-engine companion
test is ``tests/integration/test_lossy_axis_real.py``.
"""

from pathlib import Path
from unittest.mock import patch

import pytest
from giflab.external_engines import ffmpeg as ffmpeg_engine
from giflab.external_engines import imagemagick as imagemagick_engine
from giflab.tool_wrappers import (
    FFmpegLossyCompressor,
    ImageMagickLossyCompressor,
    _lossy_level_to_palette_size,
)

# Minimal metadata dict in the shape run_command / the engine helpers return.
_META = {"render_ms": 1, "engine": "stub", "command": "stub-cmd", "kilobytes": 1}


# ---------------------------------------------------------------------------
# Mapping: lossy_level -> palette size
# ---------------------------------------------------------------------------


class TestLossyLevelToPaletteSize:
    def test_endpoints_and_quartiles(self):
        # Geometric 256 -> 16: palette halves every 25 levels.
        assert _lossy_level_to_palette_size(0) == 256
        assert _lossy_level_to_palette_size(25) == 128
        assert _lossy_level_to_palette_size(50) == 64
        assert _lossy_level_to_palette_size(75) == 32
        assert _lossy_level_to_palette_size(100) == 16

    def test_monotone_non_increasing(self):
        sizes = [_lossy_level_to_palette_size(level) for level in range(101)]
        assert all(
            a >= b for a, b in zip(sizes, sizes[1:], strict=False)
        ), "palette size must never grow as lossy_level increases"

    def test_bounds(self):
        for level in range(101):
            assert 16 <= _lossy_level_to_palette_size(level) <= 256


# ---------------------------------------------------------------------------
# Wrapper-side validation (loud errors, no silent clamp)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "wrapper_cls",
    [FFmpegLossyCompressor, ImageMagickLossyCompressor],
    ids=["ffmpeg", "imagemagick"],
)
class TestWrapperValidation:
    def test_missing_lossy_level_raises(self, wrapper_cls, tmp_path):
        with pytest.raises(ValueError, match="lossy_level"):
            wrapper_cls().apply(tmp_path / "in.gif", tmp_path / "out.gif", params={})

    def test_none_params_raises(self, wrapper_cls, tmp_path):
        with pytest.raises(ValueError, match="lossy_level"):
            wrapper_cls().apply(tmp_path / "in.gif", tmp_path / "out.gif", params=None)

    @pytest.mark.parametrize("bad_level", [-1, 101, 120])
    def test_out_of_range_level_raises(self, wrapper_cls, bad_level, tmp_path):
        # Replaces ffmpeg's old silent clamp and imagemagick's old confusing
        # "quality must be in 0-100" crash at level 120.
        with pytest.raises(ValueError, match="between 0 and 100"):
            wrapper_cls().apply(
                tmp_path / "in.gif",
                tmp_path / "out.gif",
                params={"lossy_level": bad_level},
            )


# ---------------------------------------------------------------------------
# Wrapper -> engine-helper wiring (mapped ``colors`` kwarg)
# ---------------------------------------------------------------------------


_LEVEL_TO_COLORS = [(0, 256), (50, 64), (100, 16)]


class TestWrapperPassesMappedColors:
    @pytest.mark.parametrize("level,expected_colors", _LEVEL_TO_COLORS)
    def test_ffmpeg_wrapper_maps_level_to_colors(
        self, level, expected_colors, tmp_path
    ):
        with patch(
            "giflab.tool_wrappers.ffmpeg_lossy_compress", return_value=dict(_META)
        ) as engine_fn, patch(
            "giflab.tool_wrappers.validate_wrapper_apply_result",
            side_effect=lambda wrapper, i, o, p, result: result,
        ):
            FFmpegLossyCompressor().apply(
                tmp_path / "in.gif",
                tmp_path / "out.gif",
                params={"lossy_level": level},
            )
        assert engine_fn.call_args.kwargs["colors"] == expected_colors

    @pytest.mark.parametrize("level,expected_colors", _LEVEL_TO_COLORS)
    def test_imagemagick_wrapper_maps_level_to_colors(
        self, level, expected_colors, tmp_path
    ):
        with patch(
            "giflab.tool_wrappers.imagemagick_lossy_compress",
            return_value=dict(_META),
        ) as engine_fn, patch(
            "giflab.tool_wrappers.validate_wrapper_apply_result",
            side_effect=lambda wrapper, i, o, p, result: result,
        ):
            ImageMagickLossyCompressor().apply(
                tmp_path / "in.gif",
                tmp_path / "out.gif",
                params={"lossy_level": level},
            )
        assert engine_fn.call_args.kwargs["colors"] == expected_colors


# ---------------------------------------------------------------------------
# Engine command construction (run_command mocked, no binaries)
# ---------------------------------------------------------------------------


class TestFFmpegLossyCommand:
    def _run(self, tmp_path: Path, **kwargs):
        calls: list[list[str]] = []

        def fake_run(cmd, *, engine, output_path, **kw):
            calls.append(list(cmd))
            # Mirror the real run_command contract: command is the joined cmd.
            return dict(_META, engine=engine, command=" ".join(cmd))

        with patch.object(
            ffmpeg_engine, "_ffmpeg_binary", return_value="ffmpeg"
        ), patch.object(ffmpeg_engine, "run_command", side_effect=fake_run):
            result = ffmpeg_engine.lossy_compress(
                tmp_path / "in.gif", tmp_path / "out.gif", **kwargs
            )
        return calls, result

    def test_two_pass_palette_command(self, tmp_path):
        calls, result = self._run(tmp_path, colors=64)

        assert len(calls) == 2, "expected palettegen + paletteuse passes"
        pass1, pass2 = calls
        assert "palettegen=max_colors=64" in " ".join(pass1)
        assert "paletteuse=dither=sierra2_4a" in " ".join(pass2)
        # The inert video-DCT knob must be gone.
        assert "-q:v" not in pass1 + pass2

        # Combined-metadata shape (same contract as ffmpeg.color_reduce).
        assert set(result) >= {"render_ms", "engine", "command", "kilobytes"}
        assert result["engine"] == "ffmpeg"
        assert "palettegen" in result["command"]
        assert "paletteuse" in result["command"]

    def test_dithering_method_passthrough(self, tmp_path):
        calls, _ = self._run(tmp_path, colors=32, dithering_method="floyd_steinberg")
        assert "paletteuse=dither=floyd_steinberg" in " ".join(calls[1])

    @pytest.mark.parametrize("bad_colors", [3, 0, 257, 300])
    def test_colors_out_of_range_raises(self, tmp_path, bad_colors):
        with pytest.raises(ValueError, match="colors"):
            self._run(tmp_path, colors=bad_colors)


class TestImageMagickLossyCommand:
    def _run(self, tmp_path: Path, **kwargs):
        calls: list[list[str]] = []

        def fake_run(cmd, *, engine, output_path, **kw):
            calls.append(list(cmd))
            return dict(_META, engine=engine)

        with patch.object(
            imagemagick_engine, "_magick_binary", return_value="magick"
        ), patch.object(imagemagick_engine, "run_command", side_effect=fake_run):
            result = imagemagick_engine.lossy_compress(
                tmp_path / "in.gif", tmp_path / "out.gif", **kwargs
            )
        return calls, result

    def test_quantise_command(self, tmp_path):
        calls, _ = self._run(tmp_path, colors=64)

        assert len(calls) == 1
        cmd = calls[0]
        assert "-dither" in cmd
        assert cmd[cmd.index("-dither") + 1] == "Riemersma"
        assert "-colors" in cmd
        assert cmd[cmd.index("-colors") + 1] == "64"
        # The inert PNG/JPEG zlib knob must be gone.
        assert "-quality" not in cmd

    def test_256_colors_is_plain_resave(self, tmp_path):
        # colors >= 256: nothing to quantise away -- plain re-save keeps
        # lossy_level=0 honestly untouched (per-frame palettes preserved).
        calls, _ = self._run(tmp_path, colors=256)

        assert len(calls) == 1
        cmd = calls[0]
        assert "-colors" not in cmd
        assert "-dither" not in cmd
        assert "-quality" not in cmd
        assert cmd == ["magick", str(tmp_path / "in.gif"), str(tmp_path / "out.gif")]

    def test_dithering_method_passthrough(self, tmp_path):
        calls, _ = self._run(tmp_path, colors=32, dithering_method="FloydSteinberg")
        cmd = calls[0]
        assert cmd[cmd.index("-dither") + 1] == "FloydSteinberg"

    @pytest.mark.parametrize("bad_colors", [0, -1, 257])
    def test_colors_out_of_range_raises(self, tmp_path, bad_colors):
        with pytest.raises(ValueError, match="colors"):
            self._run(tmp_path, colors=bad_colors)
