"""Pre-compression content classification for the lossy ceiling.

This module classifies a GIF's *original* frames into a small set of content
classes and maps each class to a maximum animately ``lossy_level`` (a "ceiling").
``giflab.compress`` consults the classifier before dispatching to the engine and
clamps the requested lossy level DOWN to the per-class ceiling, surfacing a
warning when it does so. The motivation is the 2026-05-26 outlier deep-dive
(``docs/metrics-audit/outlier-deep-dive-2026-05-26.md``): flat categorical
charts band badly under any lossy level, and near-256-colour photographic /
film-grain content posterises above modest ceilings.

Design constraints (see ``CLAUDE.md`` "Metric accuracy is load-bearing"):

- **Continuous over discrete.** The classifier runs PRE-compression and can
  only read single-stream original-frame primitives — pair metrics like
  ``texture_similarity`` do not exist yet. The per-class SCORES blend several
  fractional signals (flat-area fraction, gradient-area fraction,
  palette-fullness, grain energy) into a smooth confidence so a GIF one pixel
  either side of a boundary does not flip class. The ceiling *value* is
  necessarily discrete (it is an engine parameter) — that discreteness is
  unavoidable; the classification is not razor-edged.
- **Fail soft.** A classification problem must never block a legitimate
  compress: frame-extraction failure / corrupt input returns ``OTHER`` with no
  ceiling, logged, never raised.
- **Deterministic.** Frame sampling uses ``np.linspace`` indices (mirroring
  ``conditional_metrics.detect_content_profile``), never random, so the clamp is
  a deterministic function of (input, engine, params).

Lightweight-import contract: this module imports cv2/numpy at module top
(``gradient_color_artifacts`` does the same) and lazily imports
``giflab.metrics.extract_gif_frames`` *inside* ``classify_content_from_path`` so
importing ``content_classifier`` never eagerly drags metrics' heavy import
graph (torch/lpips). ``compress`` in turn imports this module lazily, which is
what keeps the no-torch smoke contract green.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import cv2
import numpy as np

from giflab.config import ClassifierConfig
from giflab.gradient_color_artifacts import (
    DitherQualityAnalyzer,
    GradientBandingDetector,
)

logger = logging.getLogger(__name__)

# Deterministic frame-sampling budget. Five frames keeps the cv2 Sobel/
# Laplacian work well under the >2x runtime budget (Risk #2) while still giving
# the cross-frame grain-density std a couple of samples above the tiny-frame
# threshold.
_SAMPLE_SIZE = 5

# Below this frame count the cross-frame grain-density std has too few samples to
# be reliable (Outlier 4 vintage portrait is 5 frames), so film-grain detection
# falls back to the MEAN per-frame grain density (see ``_score_film_grain``).
_TINY_FRAME_THRESHOLD = 6

# Sliding-window patch size for the flat / gradient region detectors. The
# detectors default to 64px, which fits almost no windows in a realistic
# email-GIF frame; 32px (the smallest the dither analyser allows is 16) gives a
# meaningful region count across the 100-1000px range our corpus spans.
_PATCH_SIZE = 32

# Magnitude (per-pixel grayscale Laplacian) above which a pixel counts as
# "high-frequency". Film grain lights up most pixels; sparse chart line-edges
# light up few. Distinguishes dense texture noise from sparse sharp edges far
# more reliably than a global Laplacian variance (which a chart's antialiased
# lines spike just as hard as real grain).
_HIGHPASS_PIXEL_THRESHOLD = 8.0

# A frame smaller than this in BOTH dimensions carries too little structure to
# classify (e.g. a 10x10 swatch) — treat as OTHER, no ceiling.
_MIN_ANALYSABLE_DIM = _PATCH_SIZE


class ContentClass(Enum):
    """Coarse content classes that drive the lossy ceiling."""

    DATA_VIZ_ANIMATION = "data_viz_animation"
    PHOTOGRAPHIC = "photographic"
    FILM_GRAIN = "film_grain"
    OTHER = "other"


@dataclass(frozen=True)
class ContentClassification:
    """Result of :func:`classify_content`.

    Attributes:
        content_class: the winning :class:`ContentClass`.
        lossy_max: the maximum animately ``lossy_level`` permitted for this
            class, or ``None`` when no ceiling applies (``OTHER``).
        reason: a short human-readable explanation, suitable for a warning.
        confidence: the winning class's blended score in ``[0.0, 1.0]``.
    """

    content_class: ContentClass
    lossy_max: int | None
    reason: str
    confidence: float


# ---------------------------------------------------------------------------
# Single-frame primitive helpers
# ---------------------------------------------------------------------------


def _to_uint8_rgb(frame: np.ndarray) -> np.ndarray:
    """Coerce a frame to an HxWx3 uint8 RGB array."""
    arr = frame
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]
    return arr


def _patch_grid_count(frame: np.ndarray, patch_size: int = 64) -> int:
    """Number of non-overlapping ``patch_size`` patches that fit in ``frame``.

    Used to normalise region counts into fractions in ``[0, 1]`` so the scores
    are size-invariant (a 1000x1000 gradient and a 120x120 gradient land in the
    same place). Matches the sliding-window stride used by the detectors
    (step = patch_size // 2), so it is an upper bound on the regions they can
    report.
    """
    h, w = frame.shape[:2]
    step = max(1, patch_size // 2)
    nx = max(0, (w - patch_size) // step)
    ny = max(0, (h - patch_size) // step)
    return max(1, nx * ny)


def _palette_fullness(frame: np.ndarray) -> float:
    """Fraction of the 256-colour GIF palette the frame occupies, in ``[0, 1]``.

    Counts unique RGB triples and divides by 256. Near 1.0 means a rich
    photographic palette; a low value means a limited categorical palette
    (charts, flat UI).
    """
    rgb = _to_uint8_rgb(frame)
    flat = rgb.reshape(-1, 3)
    n_unique = int(np.unique(flat, axis=0).shape[0])
    return min(1.0, n_unique / 256.0)


def _grain_density(frame: np.ndarray) -> float:
    """Spatial density of high-frequency content, in ``[0, 1]``.

    The fraction of pixels whose grayscale Laplacian magnitude exceeds
    ``_HIGHPASS_PIXEL_THRESHOLD``. Film grain / sensor noise lights up *most*
    pixels (~0.9); smooth gradients and flat fills light up ~none (~0.0); a
    chart or UI lights up only its sparse line-edge pixels (~0.05). This is a
    smooth continuous fraction — far more discriminating than a global Laplacian
    variance, which a chart's sharp antialiased lines spike as hard as grain.
    """
    rgb = _to_uint8_rgb(frame)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float64)
    highpass = np.abs(cv2.Laplacian(gray, cv2.CV_64F))
    return float(np.mean(highpass > _HIGHPASS_PIXEL_THRESHOLD))


# ---------------------------------------------------------------------------
# Per-class smooth scoring
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Features:
    """Size-invariant, fractional frame features blended by the scorers."""

    flat_fraction: float  # mean fraction of patches that are flat
    gradient_fraction: float  # mean fraction of patches that are gradients
    palette_fullness: float  # mean fraction of the 256-colour palette used
    grain_mean: float  # mean per-frame grain density (fraction of HF pixels)
    grain_std: float  # cross-frame std of grain density (noisy below ~6 frames)
    frame_count: int


def _extract_features(frames: list[np.ndarray]) -> _Features:
    """Compute blended features from a deterministic sample of ``frames``."""
    sample_size = min(_SAMPLE_SIZE, len(frames))
    indices = np.linspace(0, len(frames) - 1, sample_size, dtype=int)
    sampled = [frames[i] for i in indices]

    banding = GradientBandingDetector(patch_size=_PATCH_SIZE)
    dither = DitherQualityAnalyzer(patch_size=_PATCH_SIZE)

    flat_fracs: list[float] = []
    grad_fracs: list[float] = []
    palette_fracs: list[float] = []
    grain_densities: list[float] = []

    for frame in sampled:
        rgb = _to_uint8_rgb(frame)
        denom = _patch_grid_count(rgb, _PATCH_SIZE)
        flat_fracs.append(min(1.0, len(dither.detect_flat_regions(rgb)) / denom))
        grad_fracs.append(min(1.0, len(banding.detect_gradient_regions(rgb)) / denom))
        palette_fracs.append(_palette_fullness(rgb))
        grain_densities.append(_grain_density(rgb))

    return _Features(
        flat_fraction=float(np.mean(flat_fracs)),
        gradient_fraction=float(np.mean(grad_fracs)),
        palette_fullness=float(np.mean(palette_fracs)),
        grain_mean=float(np.mean(grain_densities)),
        grain_std=float(np.std(grain_densities)) if len(grain_densities) > 1 else 0.0,
        frame_count=len(frames),
    )


def _score_data_viz(f: _Features) -> float:
    """Confidence that the content is a flat-colour categorical animation.

    The workhorse signal is a LIMITED categorical palette (charts use a handful
    of fill colours, not a 256-colour photographic ramp). It is reinforced by
    low grain (flat fills, not texture) and a large flat-area fraction. No
    single razor-edge threshold — every term is a smooth fraction.
    """
    limited_palette = 1.0 - f.palette_fullness
    low_grain = 1.0 - f.grain_mean
    flat = f.flat_fraction
    # Palette weighted highest — it is what separates data-viz from a smooth
    # photographic gradient (both are low-grain).
    return float(
        np.average([limited_palette, low_grain, flat], weights=[2.0, 1.0, 1.0])
    )


def _score_photographic(f: _Features) -> float:
    """Confidence that the content is a smooth photographic gradient/image.

    The workhorse signal is a NEAR-FULL palette (photographic content spans the
    colour space), reinforced by low-to-moderate grain (high grain belongs to
    FILM_GRAIN) and the presence of detected gradient regions. Smooth fractions
    throughout.
    """
    full_palette = f.palette_fullness
    not_grainy = 1.0 - f.grain_mean
    gradient = f.gradient_fraction
    # Palette weighted highest — separates photographic from data-viz; gradient
    # regions are a weak supporting signal (the detector is conservative).
    return float(
        np.average([full_palette, not_grainy, gradient], weights=[2.0, 1.0, 0.5])
    )


def _score_film_grain(f: _Features) -> float:
    """Confidence that the content is film grain / sensor noise.

    The primary signal is grain DENSITY — the fraction of pixels carrying
    high-frequency content. Genuine grain saturates this near 1.0; charts and
    gradients sit near 0. Above ``_TINY_FRAME_THRESHOLD`` frames the cross-frame
    std (grain that *moves* frame to frame) reinforces it; below that threshold
    the std has too few samples to be reliable (Outlier 4 vintage portrait =
    5 frames), so we fall back to the per-frame density alone, which a genuinely
    grainy GIF still exhibits strongly.
    """
    grain = f.grain_mean
    if f.frame_count >= _TINY_FRAME_THRESHOLD:
        moving = float(np.clip(f.grain_std * 4.0, 0.0, 1.0))
        return float(max(grain, 0.5 * (grain + moving)))
    return grain


# ---------------------------------------------------------------------------
# Public classification
# ---------------------------------------------------------------------------


def classify_content(frames: list[np.ndarray]) -> ContentClassification:
    """Classify original ``frames`` into a :class:`ContentClass` + ceiling.

    Returns ``OTHER`` (no ceiling) for empty input or when no class scores above
    the configured minimum confidence. Never raises on well-formed frame lists;
    callers that start from a path should use
    :func:`classify_content_from_path`, which additionally fails soft on
    extraction errors.
    """
    cfg = ClassifierConfig()

    if not frames:
        return ContentClassification(
            ContentClass.OTHER, None, "no frames to classify", 0.0
        )

    # Frames too small to carry analysable structure (e.g. a 10x10 swatch)
    # cannot be classified meaningfully — treat as OTHER, no ceiling. Guards the
    # degenerate single-frame / tiny-GIF edge case from a spurious low-palette
    # data-viz match.
    h, w = frames[0].shape[:2]
    if h < _MIN_ANALYSABLE_DIM and w < _MIN_ANALYSABLE_DIM:
        return ContentClassification(
            ContentClass.OTHER,
            None,
            f"frame too small to classify ({w}x{h})",
            0.0,
        )

    f = _extract_features(frames)

    scores = {
        ContentClass.DATA_VIZ_ANIMATION: _score_data_viz(f),
        ContentClass.PHOTOGRAPHIC: _score_photographic(f),
        ContentClass.FILM_GRAIN: _score_film_grain(f),
    }
    winner = max(scores, key=lambda k: scores[k])
    confidence = scores[winner]

    if confidence < cfg.MIN_CONFIDENCE:
        return ContentClassification(
            ContentClass.OTHER,
            None,
            f"no class above confidence {cfg.MIN_CONFIDENCE:.2f} "
            f"(best {winner.value}={confidence:.2f})",
            confidence,
        )

    lossy_max = cfg.lossy_max_for(winner)
    reason = (
        f"classified as {winner.value} (confidence {confidence:.2f}; "
        f"flat={f.flat_fraction:.2f} gradient={f.gradient_fraction:.2f} "
        f"palette={f.palette_fullness:.2f} grain={f.grain_mean:.2f}); "
        f"lossy ceiling {lossy_max}"
    )
    return ContentClassification(winner, lossy_max, reason, confidence)


def classify_content_from_path(gif_path: Path) -> ContentClassification:
    """Extract frames from ``gif_path`` and classify them; fail soft on error.

    Frame extraction is the only place this can blow up (corrupt GIF, IO error).
    A classification problem must never block a legitimate compress, so any
    failure returns ``OTHER`` (no ceiling), logged at debug level.
    """
    try:
        # Lazy import: keeps importing this module free of metrics' heavy graph.
        from giflab.metrics import extract_gif_frames

        result = extract_gif_frames(gif_path)
        frames = result.frames
    except Exception as exc:  # noqa: BLE001 — fail soft, never block compress
        logger.debug(
            "content classification skipped: frame extraction failed for %s: %s",
            gif_path,
            exc,
        )
        return ContentClassification(
            ContentClass.OTHER, None, f"frame extraction failed: {exc}", 0.0
        )

    return classify_content(frames)
