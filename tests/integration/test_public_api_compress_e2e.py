"""End-to-end tests for ``giflab.compress`` using real engine binaries.

Skipped per-engine when the binary is unavailable on PATH.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from giflab import CompressResult, compress
from giflab.tool_wrappers import (
    AnimatelyLossyCompressor,
    GifsicleLossyCompressor,
)
from PIL import Image, ImageDraw


def _build_real_gif(
    path: Path, frames: int = 5, size: tuple[int, int] = (64, 64)
) -> None:
    """Create a small but real multi-frame GIF that engines can actually compress."""
    images = []
    for i in range(frames):
        img = Image.new("RGB", size, color=(i * 50 % 255, 100, 150))
        draw = ImageDraw.Draw(img)
        draw.rectangle([i * 5, i * 5, i * 5 + 20, i * 5 + 20], fill=(255, 255, 255))
        images.append(img)
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=200,
        loop=0,
    )


@pytest.fixture
def real_gif(tmp_path: Path) -> Path:
    path = tmp_path / "real.gif"
    _build_real_gif(path)
    return path


@pytest.mark.skipif(
    not GifsicleLossyCompressor.available(), reason="gifsicle binary not on PATH"
)
def test_compress_gifsicle_e2e(tmp_path: Path, real_gif: Path) -> None:
    out_path = tmp_path / "out_gifsicle.gif"
    result = compress(
        input_path=real_gif,
        output_path=out_path,
        engine="gifsicle",
        params={"lossy_level": 40},
    )

    assert isinstance(result, CompressResult)
    assert out_path.exists()
    assert result.output_bytes > 0
    assert result.output_bytes == out_path.stat().st_size
    assert result.render_ms >= 0
    assert result.engine == "gifsicle"
    assert result.engine_version  # non-empty
    assert result.params == {"lossy_level": 40}


@pytest.mark.skipif(
    not AnimatelyLossyCompressor.available(), reason="animately binary not on PATH"
)
def test_compress_animately_e2e(tmp_path: Path, real_gif: Path) -> None:
    out_path = tmp_path / "out_animately.gif"
    result = compress(
        input_path=real_gif,
        output_path=out_path,
        engine="animately",
        params={"lossy_level": 40},
    )

    assert isinstance(result, CompressResult)
    assert out_path.exists()
    assert result.output_bytes > 0
    assert result.engine == "animately"
    assert result.engine_version  # non-empty


@pytest.mark.skipif(
    not GifsicleLossyCompressor.available(), reason="gifsicle binary not on PATH"
)
def test_compress_overwrites_existing_output(tmp_path: Path, real_gif: Path) -> None:
    """Per contract: compress() overwrites output_path if it exists."""
    out_path = tmp_path / "exists.gif"
    out_path.write_bytes(b"stale data")
    stale_size = out_path.stat().st_size

    result = compress(real_gif, out_path, engine="gifsicle", params={"lossy_level": 40})

    # New file replaces the stale bytes; size will differ.
    assert result.output_bytes != stale_size
    assert result.output_path == out_path


# ---------------------------------------------------------------------------
# Content-aware lossy ceiling (audit-fix
# [[giflab-content-classifier-lossy-ceiling]]) — real animately.
# ---------------------------------------------------------------------------


def _build_photographic_gradient_gif(
    path: Path, frames: int = 8, size: tuple[int, int] = (300, 300)
) -> None:
    """Render a real smooth-gradient GIF (near-256 palette) that classifies
    PHOTOGRAPHIC. Larger than the functional synthetics so the real engine and
    the classifier both have meaningful structure to work with."""
    from giflab.synthetic_gifs import SyntheticFrameGenerator

    gen = SyntheticFrameGenerator()
    images = [
        gen.create_frame("gradient", size, frame=i, total_frames=frames)
        for i in range(frames)
    ]
    images[0].save(path, save_all=True, append_images=images[1:], duration=120, loop=0)


def _build_flat_chart_gif(
    path: Path, frames: int = 8, size: tuple[int, int] = (300, 300)
) -> None:
    """Render a real flat-colour animated chart GIF that classifies
    DATA_VIZ_ANIMATION."""
    from giflab.synthetic_gifs import SyntheticFrameGenerator

    gen = SyntheticFrameGenerator()
    images = [
        gen.create_frame("charts", size, frame=i, total_frames=frames)
        for i in range(frames)
    ]
    images[0].save(path, save_all=True, append_images=images[1:], duration=120, loop=0)


def _dominant_hues(path: Path, top_n: int = 6) -> set[tuple[int, int, int]]:
    """Return the ``top_n`` most common RGB colours in the first frame, snapped
    to a coarse 32-level grid so near-identical hues compare equal.

    Used to verify a flat chart's *categorical* fill hues survive compression —
    a posterised / banded output drops or shifts these dominant flat colours.
    """
    from PIL import Image, ImageSequence

    with Image.open(path) as img:
        frame = next(ImageSequence.Iterator(img)).convert("RGB")
        arr = np.asarray(frame).reshape(-1, 3)

    # Snap to a 32-level grid so anti-aliasing jitter does not fragment a hue.
    snapped = (arr // 32) * 32
    colours, counts = np.unique(snapped, axis=0, return_counts=True)
    order = np.argsort(counts)[::-1][:top_n]
    return {tuple(int(c) for c in colours[i]) for i in order}


@pytest.mark.skipif(
    not AnimatelyLossyCompressor.available(), reason="animately binary not on PATH"
)
def test_compress_photographic_gradient_respects_ceiling(tmp_path: Path) -> None:
    """Acceptance: compress() on a photographic-gradient GIF clamps a high
    requested lossy level down to the photographic ceiling and surfaces a
    warning (real animately).

    Asserts the engine *actually ran* (output written, non-trivial size) so the
    test cannot pass vacuously if animately ever no-ops, in addition to the
    clamp assertions.
    """
    from giflab.config import ClassifierConfig

    gif = tmp_path / "gradient_xlarge.gif"
    _build_photographic_gradient_gif(gif)
    out_path = tmp_path / "gradient_out.gif"

    ceiling = ClassifierConfig().MAX_LOSSY_PHOTOGRAPHIC
    result = compress(gif, out_path, engine="animately", params={"lossy_level": 80})

    # The clamp is reflected on the result params and announced in warnings.
    assert result.params["lossy_level"] == ceiling
    assert any("clamp" in w.lower() for w in result.warnings)
    # The engine genuinely produced an output (not a vacuous pass).
    assert out_path.exists() and result.output_bytes > 0


@pytest.mark.skipif(
    not AnimatelyLossyCompressor.available(), reason="animately binary not on PATH"
)
def test_compress_flat_chart_hues_survive_high_lossy(tmp_path: Path) -> None:
    """Acceptance: a flat-colour animated chart's categorical hues survive a
    high requested lossy level (real animately).

    Stronger than asserting the clamp merely fired: this compresses at a high
    requested level and checks the dominant flat fill hues of the *real engine
    output* still match the source's, i.e. they were not posterised away. With
    the conservative data-viz ceiling the high request is clamped to a
    flat-content-safe level; the hue check verifies that level actually preserves
    the categorical palette.
    """
    from giflab.config import ClassifierConfig

    gif = tmp_path / "charts.gif"
    _build_flat_chart_gif(gif)
    out_path = tmp_path / "charts_out.gif"

    ceiling = ClassifierConfig().MAX_LOSSY_DATA_VIZ
    source_hues = _dominant_hues(gif)

    result = compress(gif, out_path, engine="animately", params={"lossy_level": 120})

    # Clamped down to the conservative data-viz ceiling, with a warning.
    assert result.params["lossy_level"] == ceiling
    assert any("clamp" in w.lower() for w in result.warnings)
    # The engine actually produced an output.
    assert out_path.exists() and result.output_bytes > 0
    # The categorical fill hues survived — most of the source's dominant flat
    # colours are still present in the compressed output (not banded away).
    output_hues = _dominant_hues(out_path)
    preserved = source_hues & output_hues
    assert len(preserved) >= max(1, len(source_hues) // 2), (
        "flat chart hues were destroyed by lossy compression: "
        f"source dominant hues {sorted(source_hues)} vs "
        f"output dominant hues {sorted(output_hues)}"
    )


@pytest.mark.skipif(
    not AnimatelyLossyCompressor.available(), reason="animately binary not on PATH"
)
def test_compress_ceiling_bypass_leaves_level_intact(tmp_path: Path) -> None:
    """With apply_content_ceiling=False the requested level passes through to
    real animately untouched and no warning is emitted — the audit path."""
    gif = tmp_path / "gradient_bypass.gif"
    _build_photographic_gradient_gif(gif)
    out_path = tmp_path / "gradient_bypass_out.gif"

    result = compress(
        gif,
        out_path,
        engine="animately",
        params={"lossy_level": 80},
        apply_content_ceiling=False,
    )

    assert result.params["lossy_level"] == 80
    assert result.warnings == ()
