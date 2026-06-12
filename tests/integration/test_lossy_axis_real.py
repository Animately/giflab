"""Regression test: ffmpeg/imagemagick ``lossy_level`` must actually change GIF output.

The 2026-06-09 multi-engine calibration (PR #58,
``scripts/audit/engine_lossy_calibration.py``) found both engines produced
BYTE-IDENTICAL output at every public lossy level: their wrappers routed
``lossy_level`` to ``-q:v`` (ffmpeg, a video-DCT knob) / ``-quality``
(imagemagick, a PNG/JPEG zlib knob), neither of which touches GIF pixels.
Any caller asking these engines for a lossy GIF silently got the lossless
result.

This test pins the fix with real engines: the lossy axis is palette-size
reduction + dithering, so increasing ``lossy_level`` must change the bytes,
shrink the file, and reduce the colour count. It uses colour-rich content
generated in-test (mirroring the calibration harness's ``_rich_gradient``)
because the small checked-in fixtures are too colour-poor to exercise a
palette-reduction axis -- on a 16-colour GIF, reducing to 16 colours is
honestly a no-op.
"""

import hashlib
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from giflab.tool_wrappers import FFmpegLossyCompressor, ImageMagickLossyCompressor


def _make_rich_gradient_gif(path: Path, n_frames: int = 4) -> None:
    """4-frame 160x96 2D RGB gradient (many colours; mirrors the calibration
    harness's ``_rich_gradient`` content archetype)."""
    yy, xx = np.mgrid[0:96, 0:160]
    frames = []
    for k in range(n_frames):
        r = xx / 160 * 255
        g = yy / 96 * 255
        b = ((xx + yy) / 256 * 255 + k * 3) % 256
        frames.append(np.clip(np.stack([r, g, b], -1), 0, 255).astype(np.uint8))
    imgs = [
        Image.fromarray(f, "RGB").convert("P", palette=Image.ADAPTIVE, colors=256)
        for f in frames
    ]
    imgs[0].save(
        path,
        save_all=True,
        append_images=imgs[1:],
        duration=100,
        loop=0,
        optimize=False,
    )


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def _unique_colors_first_frame(path: Path) -> int:
    with Image.open(path) as img:
        arr = np.asarray(img.convert("RGB")).reshape(-1, 3)
    return len(np.unique(arr, axis=0))


@pytest.mark.external_tools
@pytest.mark.parametrize(
    "wrapper_cls",
    [FFmpegLossyCompressor, ImageMagickLossyCompressor],
    ids=["ffmpeg", "imagemagick"],
)
class TestLossyAxisIsReal:
    def test_lossy_level_changes_output(self, wrapper_cls, tmp_path):
        wrapper = wrapper_cls()
        if not wrapper.available():
            pytest.skip(f"{wrapper.NAME} engine not available")

        src = tmp_path / "rich_gradient.gif"
        _make_rich_gradient_gif(src)

        outputs: dict[int, Path] = {}
        for level in (0, 40, 100):
            out = tmp_path / f"lossy_{level}.gif"
            result = wrapper.apply(src, out, params={"lossy_level": level})
            assert out.exists(), f"no output at lossy_level={level}"
            assert result["render_ms"] >= 0
            outputs[level] = out

        # (a) The exact 2026-06-09 inertness regression: on the broken
        # wrappers every level produced the same bytes (same md5).
        assert _md5(outputs[0]) != _md5(outputs[100]), (
            f"{wrapper.NAME}: lossy_level is INERT -- byte-identical output at "
            "levels 0 and 100 (the 2026-06-09 calibration failure mode)"
        )

        # (b) More lossy must mean a smaller file on colour-rich content.
        size_0 = outputs[0].stat().st_size
        size_100 = outputs[100].stat().st_size
        assert size_100 < size_0, (
            f"{wrapper.NAME}: expected size(L100) < size(L0), "
            f"got {size_100} >= {size_0}"
        )

        # (c) The axis is palette reduction: max lossy must use fewer colours.
        colors_0 = _unique_colors_first_frame(outputs[0])
        colors_100 = _unique_colors_first_frame(outputs[100])
        assert colors_100 < colors_0, (
            f"{wrapper.NAME}: expected fewer unique colours at L100, "
            f"got {colors_100} >= {colors_0}"
        )
