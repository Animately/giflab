"""Functional tests for the pre-compression content classifier and the
per-content-type lossy ceiling enforced inside ``giflab.compress``.

These tests build synthetic GIFs from ``SyntheticFrameGenerator`` frame types
and use a MOCKED animately wrapper (the same pattern as
``test_public_api_compress.py``) so no real engine subprocess runs. The
classifier itself runs real cv2/numpy work on the synthetic frames.

Audit-fix [[giflab-content-classifier-lossy-ceiling]].
"""

from __future__ import annotations

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
from PIL import Image

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


def test_data_viz_chart_classified_and_ceiling_applied(tmp_path: Path) -> None:
    """A flat-colour animated chart classifies DATA_VIZ_ANIMATION and the
    data-viz ceiling clamps a high requested lossy level all the way down."""
    gif = _save_synthetic_gif(tmp_path / "charts.gif", "charts", frames=16)
    classification = classify_content(_frames_of(gif))

    assert isinstance(classification, ContentClassification)
    assert classification.content_class is ContentClass.DATA_VIZ_ANIMATION
    from giflab.config import ClassifierConfig

    assert classification.lossy_max == ClassifierConfig().MAX_LOSSY_DATA_VIZ

    out_path = tmp_path / "charts_out.gif"
    cls, instance = _mock_animately(out_path)
    with patch("giflab.public_api.AnimatelyLossyCompressor", cls):
        result = compress(gif, out_path, engine="animately", params={"lossy_level": 60})

    sent = instance.apply.call_args.kwargs["params"]["lossy_level"]
    assert sent <= classification.lossy_max
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


def test_ceiling_skipped_for_non_animately_engine(tmp_path: Path) -> None:
    """The ceiling is animately-calibrated only — other lossy engines skip
    classification entirely (no clamp, no warning)."""
    gif = _save_synthetic_gif(tmp_path / "gradient.gif", "gradient", frames=12)
    out_path = tmp_path / "gifsicle_out.gif"
    instance = MagicMock()
    instance.apply.side_effect = lambda *a, **kw: (
        out_path.write_bytes(b"\x00" * 256),
        {"render_ms": 1, "engine": "gifsicle", "command": "x", "kilobytes": 1},
    )[1]
    cls = MagicMock(return_value=instance)
    cls.available = MagicMock(return_value=True)
    cls.version = MagicMock(return_value="gifsicle-test")

    with patch("giflab.public_api.GifsicleLossyCompressor", cls):
        result = compress(gif, out_path, engine="gifsicle", params={"lossy_level": 60})

    assert instance.apply.call_args.kwargs["params"]["lossy_level"] == 60
    assert result.warnings == ()
