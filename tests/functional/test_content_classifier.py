"""Functional tests for the pre-compression content classifier and the
per-content-type lossy ceiling enforced inside ``giflab.compress``.

These tests build synthetic GIFs from ``SyntheticFrameGenerator`` frame types
and use a MOCKED animately wrapper (the same pattern as
``test_public_api_compress.py``) so no real engine subprocess runs. The
classifier itself runs real cv2/numpy work on the synthetic frames.

Audit-fix [[giflab-content-classifier-lossy-ceiling]].
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from giflab import CompressResult, compress
from giflab.content_classifier import (
    ContentClass,
    ContentClassification,
    classify_content,
)
from giflab.metrics import extract_gif_frames
from giflab.synthetic_gifs import SyntheticFrameGenerator
from PIL import Image, ImageDraw

# ---------------------------------------------------------------------------
# Synthetic GIF builders
# ---------------------------------------------------------------------------


def _save_synthetic_gif(
    path: Path,
    content_type: str,
    *,
    frames: int = 12,
    size: tuple[int, int] = (120, 120),
) -> Path:
    """Render a multi-frame GIF of the given synthetic content type to ``path``."""
    gen = SyntheticFrameGenerator()
    images = [
        gen.create_frame(content_type, size, frame=i, total_frames=frames)
        for i in range(frames)
    ]
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=80,
        loop=0,
    )
    return path


def _frames_of(path: Path) -> list:
    return extract_gif_frames(path).frames


def _save_drawn_gif(
    path: Path,
    draw_frame: Callable[[ImageDraw.ImageDraw, int], None],
    *,
    background: tuple[int, int, int],
    frames: int = 12,
    size: tuple[int, int] = (200, 200),
) -> Path:
    """Render a full-size multi-frame GIF whose frames are drawn by a callback.

    Used to build *realistic* non-chart limited-palette content (a solid fill, a
    text/UI frame, a simple flat cartoon) at a meaningful size, so the classifier
    actually reaches its scoring path instead of being short-circuited by the
    ``_MIN_ANALYSABLE_DIM`` dimension guard (the blind spot the old 10x10
    ``tiny_gif`` test hid — PR #40 round-3 review).
    """
    images = []
    for i in range(frames):
        img = Image.new("RGB", size, background)
        draw_frame(ImageDraw.Draw(img), i)
        images.append(img)
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=80,
        loop=0,
    )
    return path


def _draw_solid_fill(d: ImageDraw.ImageDraw, i: int) -> None:
    """A single flat colour that shifts slightly per frame (animated solid)."""
    d.rectangle([0, 0, 200, 200], fill=(40 + (i * 3) % 60, 120, 200))


def _draw_text_ui(d: ImageDraw.ImageDraw, i: int) -> None:
    """A text/UI-style frame: dark text bars on white + a coloured button."""
    for y in range(20, 160, 14):
        d.rectangle([20, y, 20 + (100 + (i * 3) % 40), y + 6], fill=(20, 20, 20))
    d.rectangle([20, 170, 90, 190], fill=(0, 120, 220))


def _draw_flat_cartoon(d: ImageDraw.ImageDraw, i: int) -> None:
    """A simple flat-colour cartoon: a moving blob, ground, and sun."""
    x = 40 + (i * 4) % 80
    d.ellipse([x, 60, x + 70, 130], fill=(220, 60, 60))
    d.rectangle([20, 150, 180, 180], fill=(60, 140, 80))
    d.ellipse([130, 30, 170, 70], fill=(250, 240, 90))


# ---------------------------------------------------------------------------
# Mocked animately wrapper plumbing (mirrors test_public_api_compress.py)
# ---------------------------------------------------------------------------


def _mock_animately(
    out_path: Path, *, out_size: int = 512, out_frames: int | None = None
):
    """Build a mock AnimatelyLossyCompressor class whose apply() writes a file.

    If ``out_frames`` is set, the mock writes a real multi-frame GIF with that
    many frames (used to exercise the frame-drop warning); otherwise it writes
    ``out_size`` opaque bytes.
    """
    instance = MagicMock()

    def _apply(*_a, **_kw):
        if out_frames is not None:
            imgs = [
                Image.new("RGB", (16, 16), (i * 7 % 255, 0, 0))
                for i in range(out_frames)
            ]
            imgs[0].save(
                out_path, save_all=True, append_images=imgs[1:], duration=80, loop=0
            )
        else:
            out_path.write_bytes(b"\x00" * out_size)
        return {"render_ms": 5, "engine": "animately", "command": "x", "kilobytes": 1}

    instance.apply.side_effect = _apply
    cls = MagicMock(return_value=instance)
    cls.available = MagicMock(return_value=True)
    cls.version = MagicMock(return_value="animately-test")
    return cls, instance


# ---------------------------------------------------------------------------
# Classifier-level tests
# ---------------------------------------------------------------------------


def test_data_viz_chart_classified_and_high_level_clamped(tmp_path: Path) -> None:
    """A flat-colour animated chart classifies DATA_VIZ_ANIMATION; an extreme
    requested lossy level (above the conservative data-viz ceiling) clamps DOWN
    to that ceiling and surfaces a warning.

    The ceiling is deliberately NON-lossless (``MAX_LOSSY_DATA_VIZ`` defaults to
    40, not 0): single-frame primitives cannot isolate a categorical chart from
    a flat logo / cartoon, so forcing lossless on the whole flat-content
    population was withdrawn (PR #40 round-3 review). What survives is a
    conservative guard against *extreme* lossy on flat content.
    """
    gif = _save_synthetic_gif(tmp_path / "charts.gif", "charts", frames=16)
    classification = classify_content(_frames_of(gif))

    assert isinstance(classification, ContentClassification)
    assert classification.content_class is ContentClass.DATA_VIZ_ANIMATION
    from giflab.config import ClassifierConfig

    ceiling = ClassifierConfig().MAX_LOSSY_DATA_VIZ
    assert classification.lossy_max == ceiling
    assert ceiling > 0, "data-viz ceiling must NOT force lossless (round-3 fix)"

    # A request well above the ceiling clamps down to it.
    out_path = tmp_path / "charts_out.gif"
    cls, instance = _mock_animately(out_path)
    with patch("giflab.public_api.AnimatelyLossyCompressor", cls):
        result = compress(gif, out_path, engine="animately", params={"lossy_level": 80})

    assert instance.apply.call_args.kwargs["params"]["lossy_level"] == ceiling
    assert result.warnings  # ceiling-applied warning surfaced


def test_photographic_gradient_classified(tmp_path: Path) -> None:
    """A smooth-gradient GIF classifies PHOTOGRAPHIC with the 20 ceiling."""
    gif = _save_synthetic_gif(tmp_path / "gradient.gif", "gradient", frames=12)
    classification = classify_content(_frames_of(gif))

    assert classification.content_class is ContentClass.PHOTOGRAPHIC
    from giflab.config import ClassifierConfig

    assert classification.lossy_max == ClassifierConfig().MAX_LOSSY_PHOTOGRAPHIC


def test_film_grain_classified(tmp_path: Path) -> None:
    """A photographic-noise GIF classifies FILM_GRAIN with the 30 ceiling."""
    gif = _save_synthetic_gif(tmp_path / "noise.gif", "noise", frames=12)
    classification = classify_content(_frames_of(gif))

    assert classification.content_class is ContentClass.FILM_GRAIN
    from giflab.config import ClassifierConfig

    assert classification.lossy_max == ClassifierConfig().MAX_LOSSY_FILM_GRAIN


def test_film_grain_low_frame_count(tmp_path: Path) -> None:
    """A 5-frame grainy GIF still lands FILM_GRAIN via the mean-grain-energy
    fallback (cross-frame std is statistically noisy below ~6 frames)."""
    gif = _save_synthetic_gif(tmp_path / "noise_tiny.gif", "noise", frames=5)
    classification = classify_content(_frames_of(gif))

    assert classification.content_class is ContentClass.FILM_GRAIN


# ---------------------------------------------------------------------------
# Ceiling-enforcement tests (through compress())
# ---------------------------------------------------------------------------


def test_compress_ceiling_clamps_down_not_up(tmp_path: Path) -> None:
    """Photographic content: a level already below the ceiling is untouched
    (and emits no warning); a level above the ceiling clamps down to it."""
    gif = _save_synthetic_gif(tmp_path / "gradient.gif", "gradient", frames=12)
    from giflab.config import ClassifierConfig

    ceiling = ClassifierConfig().MAX_LOSSY_PHOTOGRAPHIC

    # Below ceiling: stays put, no warning.
    out1 = tmp_path / "low.gif"
    cls1, inst1 = _mock_animately(out1)
    with patch("giflab.public_api.AnimatelyLossyCompressor", cls1):
        res1 = compress(gif, out1, engine="animately", params={"lossy_level": 10})
    assert inst1.apply.call_args.kwargs["params"]["lossy_level"] == 10
    assert res1.warnings == ()

    # Above ceiling: clamps down and warns.
    out2 = tmp_path / "high.gif"
    cls2, inst2 = _mock_animately(out2)
    with patch("giflab.public_api.AnimatelyLossyCompressor", cls2):
        res2 = compress(gif, out2, engine="animately", params={"lossy_level": 60})
    assert inst2.apply.call_args.kwargs["params"]["lossy_level"] == ceiling
    assert res2.warnings


def test_non_special_content_no_ceiling_no_warning(
    tmp_path: Path, tiny_gif: Path
) -> None:
    """OTHER content (tiny single-frame GIF) is never clamped and emits no
    warning; the requested level passes through untouched."""
    out_path = tmp_path / "tiny_out.gif"
    cls, instance = _mock_animately(out_path)
    with patch("giflab.public_api.AnimatelyLossyCompressor", cls):
        result = compress(
            tiny_gif, out_path, engine="animately", params={"lossy_level": 60}
        )

    assert instance.apply.call_args.kwargs["params"]["lossy_level"] == 60
    assert result.warnings == ()


# ---------------------------------------------------------------------------
# False-positive guards (PR #40 round-3 review).
#
# The BLOCKING bug was the data-viz scorer being dominated by "low palette", so
# *any* limited-palette flat content (solid fills, text/UI, flat cartoons) was
# classified DATA_VIZ_ANIMATION and forced to lossless (ceiling 0). These tests
# pin the user-visible guarantee that ordinary full-size limited-palette content
# at a TYPICAL lossy request is NOT forced lossless — i.e. a normal lossy
# compression passes through untouched. They use full-size (200x200) multi-frame
# content (not the 10x10 ``tiny_gif`` that tripped the dimension guard and never
# reached the scorer — the exact blind spot the old suite hid).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "background", "draw_fn"),
    [
        ("solid_fill", (40, 120, 200), _draw_solid_fill),
        ("text_ui", (255, 255, 255), _draw_text_ui),
        ("flat_cartoon", (245, 225, 150), _draw_flat_cartoon),
    ],
)
def test_ordinary_flat_content_not_forced_lossless(
    tmp_path: Path,
    name: str,
    background: tuple[int, int, int],
    draw_fn: Callable[[ImageDraw.ImageDraw, int], None],
) -> None:
    """Full-size non-chart limited-palette content (solid fill / text-UI / flat
    cartoon) at a typical lossy request is NEVER forced lossless: a request at
    or below the conservative data-viz ceiling passes through to the engine
    untouched, with no clamp warning.

    This is the regression guard for the round-3 BLOCKING bug — these inputs were
    previously clamped to ``lossy_level=0`` (lossless), silently defeating lossy
    compression for the single most common GIF category.
    """
    from giflab.config import ClassifierConfig

    ceiling = ClassifierConfig().MAX_LOSSY_DATA_VIZ
    gif = _save_drawn_gif(
        tmp_path / f"{name}.gif", draw_fn, background=background, frames=12
    )

    # A typical lossy request (at the ceiling) must pass through untouched: no
    # clamp, no warning. (At the ceiling the clamp is a no-op by definition; the
    # point is that the requested level is preserved, never driven to lossless.)
    requested = ceiling
    out_path = tmp_path / f"{name}_out.gif"
    cls, instance = _mock_animately(out_path)
    with patch("giflab.public_api.AnimatelyLossyCompressor", cls):
        result = compress(
            gif, out_path, engine="animately", params={"lossy_level": requested}
        )

    sent = instance.apply.call_args.kwargs["params"]["lossy_level"]
    assert sent == requested, (
        f"{name}: lossy_level {requested} must pass through untouched, "
        f"not be driven down to {sent}"
    )
    assert sent > 0, f"{name}: must NEVER be forced lossless"
    assert not any("clamp" in w.lower() for w in result.warnings), (
        f"{name}: no clamp warning expected at or below the ceiling; "
        f"got {result.warnings}"
    )


def test_audit_optout_bypasses_ceiling(tmp_path: Path) -> None:
    """With apply_content_ceiling=False, photographic content is NOT clamped
    and no warning is emitted — the audit monotonicity grid stays intact."""
    gif = _save_synthetic_gif(tmp_path / "gradient.gif", "gradient", frames=12)
    out_path = tmp_path / "bypass.gif"
    cls, instance = _mock_animately(out_path)
    with patch("giflab.public_api.AnimatelyLossyCompressor", cls):
        result = compress(
            gif,
            out_path,
            engine="animately",
            params={"lossy_level": 60},
            apply_content_ceiling=False,
        )

    assert instance.apply.call_args.kwargs["params"]["lossy_level"] == 60
    assert result.warnings == ()


def test_frame_drop_warning_emitted(tmp_path: Path) -> None:
    """When the engine output has fewer frames than the input, compress()
    surfaces a frame-drop warning string."""
    gif = _save_synthetic_gif(tmp_path / "gradient.gif", "gradient", frames=12)
    out_path = tmp_path / "dropped.gif"
    # Output has only 4 frames vs 12 input frames.
    cls, _ = _mock_animately(out_path, out_frames=4)
    with patch("giflab.public_api.AnimatelyLossyCompressor", cls):
        result = compress(gif, out_path, engine="animately", params={"lossy_level": 10})

    assert any("frame" in w.lower() for w in result.warnings)


def test_frame_drop_warning_skipped_when_frame_count_unreadable(
    tmp_path: Path,
) -> None:
    """When the engine output is not a readable GIF, ``_safe_frame_count``
    returns ``None`` and the frame-drop warning is silently skipped (best-effort
    behaviour — an unreadable output must never block a legitimate compress).

    Uses a non-photographic single-frame input so the content ceiling does not
    fire either, isolating the frame-drop branch: the result must carry NO
    warnings.
    """
    from giflab.public_api import _safe_frame_count

    gif = _save_synthetic_gif(tmp_path / "gradient.gif", "gradient", frames=12)
    out_path = tmp_path / "garbage_out.gif"
    # out_frames=None → the mock writes opaque non-GIF bytes, so
    # _safe_frame_count(out_path) cannot determine a frame count.
    cls, _ = _mock_animately(out_path, out_size=512)
    with patch("giflab.public_api.AnimatelyLossyCompressor", cls):
        # lossy_level 10 is below the photographic ceiling → no clamp warning.
        result = compress(gif, out_path, engine="animately", params={"lossy_level": 10})

    # The output is unreadable, so the frame-count probe returns None and the
    # frame-drop comparison is short-circuited — no frame-drop warning.
    assert (
        _safe_frame_count(out_path) is None
    ), "test precondition: the opaque-bytes output must be unreadable as a GIF"
    assert not any("frame" in w.lower() for w in result.warnings), (
        "frame-drop warning must be suppressed when the output frame count "
        f"cannot be determined; got warnings: {result.warnings}"
    )


def test_frame_drop_and_clamp_warnings_co_occur(tmp_path: Path) -> None:
    """A single compress() can surface BOTH a content-ceiling clamp warning
    and a frame-drop warning on the same result — they are independent and
    must not suppress each other.

    Photographic input (clamps from 60 → ceiling) + an output with fewer
    frames than the input (frame drop) → two warnings.
    """
    from giflab.config import ClassifierConfig

    ceiling = ClassifierConfig().MAX_LOSSY_PHOTOGRAPHIC
    gif = _save_synthetic_gif(tmp_path / "gradient.gif", "gradient", frames=12)
    out_path = tmp_path / "clamped_dropped.gif"
    # Output has 4 frames vs 12 input frames → frame drop, AND it's a readable
    # GIF so the frame-count probe succeeds.
    cls, instance = _mock_animately(out_path, out_frames=4)
    with patch("giflab.public_api.AnimatelyLossyCompressor", cls):
        result = compress(gif, out_path, engine="animately", params={"lossy_level": 60})

    # Clamp happened (photographic ceiling well below 60).
    assert instance.apply.call_args.kwargs["params"]["lossy_level"] == ceiling
    assert any(
        "clamp" in w.lower() for w in result.warnings
    ), f"expected a clamp warning; got: {result.warnings}"
    # Frame drop also surfaced.
    assert any(
        "frame" in w.lower() for w in result.warnings
    ), f"expected a frame-drop warning; got: {result.warnings}"
    # Both warnings present simultaneously.
    assert len(result.warnings) >= 2, (
        f"expected both clamp and frame-drop warnings to co-occur; "
        f"got: {result.warnings}"
    )


# engine → the wrapper symbol looked up inside ``giflab.public_api`` (the patch
# target). Every non-animately lossy engine is calibrated
# (scripts/audit/engine_lossy_calibration.py) and currently has NO content
# ceiling:
#   - gifsicle (2026-06-05) / gifski (2026-06-09) / ffmpeg (2026-06-12, on its
#     now-real palette/dither lossy axis): measured GRADUAL, banding-free
#     degradation (no posterisation cliff — the failure mode the ceiling
#     guards against).
#   - imagemagick (2026-06-12, also a real axis now): gradual after a one-off
#     quantiser entry-step at L0→10 (~0.24 composite on gradients, banding 0);
#     whether that step warrants a ceiling is deferred to the follow-up task
#     giflab-imagemagick-lossy-entry-step-ceiling-calibration — no ceiling
#     until that calibration lands.
# Either way the ceiling must never fire for them today. This freezes that
# invariant.
_NON_ANIMATELY_WRAPPER_SYMBOLS = {
    "gifsicle": "GifsicleLossyCompressor",
    "gifski": "GifskiLossyCompressor",
    "imagemagick": "ImageMagickLossyCompressor",
    "ffmpeg": "FFmpegLossyCompressor",
}


@pytest.mark.parametrize(
    "engine", sorted(_NON_ANIMATELY_WRAPPER_SYMBOLS), ids=lambda e: e
)
def test_ceiling_skipped_for_non_animately_engine(engine: str, tmp_path: Path) -> None:
    """The ceiling is animately-calibrated only — every other lossy engine skips
    classification entirely (no clamp, no warning), even at a ``lossy_level``
    (60) well above animately's photographic ceiling (20).

    Characterization lock: the ``engine == "animately"`` gate already restricts
    the ceiling to animately, so this passes immediately for all four engines.
    It freezes the data-backed verdict that none of them needs a ceiling — a
    future widening of the gate to ``engine in {...}`` would silently start
    clamping these engines and break this test.
    """
    gif = _save_synthetic_gif(tmp_path / "gradient.gif", "gradient", frames=12)
    out_path = tmp_path / f"{engine}_out.gif"
    instance = MagicMock()
    instance.apply.side_effect = lambda *a, **kw: (
        out_path.write_bytes(b"\x00" * 256),
        {"render_ms": 1, "engine": engine, "command": "x", "kilobytes": 1},
    )[1]
    cls = MagicMock(return_value=instance)
    cls.available = MagicMock(return_value=True)
    cls.version = MagicMock(return_value=f"{engine}-test")

    symbol = _NON_ANIMATELY_WRAPPER_SYMBOLS[engine]
    with patch(f"giflab.public_api.{symbol}", cls):
        result = compress(gif, out_path, engine=engine, params={"lossy_level": 60})

    # lossy_level passed through UNCLAMPED (60, not animately's photographic 20).
    assert instance.apply.call_args.kwargs["params"]["lossy_level"] == 60
    # No clamp warning (and no other warning) emitted.
    assert result.warnings == ()
