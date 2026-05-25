"""Regression tests for structure-based metrics on flat-content (solid colour) pairs.

Audit context: ``docs/metrics-audit/2026-05-22/report.md`` flagged that ``fsim``,
``gmsd``, ``edge_similarity``, and ``sharpness_similarity`` returned
identity-equivalent values when fed two maximally different solid-colour
frames (e.g. white vs black). Cause: every metric is structure-derived, and
on flat content every derivative is exactly zero, so the zero/zero stability
fallback (or the explicit empty-edge / zero-variance early return) silently
reports "perfect".

Fix decision (see task note ``giflab-fsim-flat-content-returns-1``): the four
metrics now use a *content-aware honest fallback* that distinguishes:

* both frames flat AND mean colours match → identity value (1.0 / 0.0)
* both frames flat AND mean colours differ → worst-case value
* exactly one frame flat → worst-case value (unambiguous mismatch)
* otherwise → unchanged structure-based computation

These tests pin that behaviour for the four affected metrics.
"""

from __future__ import annotations

import numpy as np
import pytest
from giflab.metrics import (
    edge_similarity,
    fsim,
    gmsd,
    sharpness_similarity,
)

# ---------------------------------------------------------------------------
# Fixtures (kept small; flat-content metrics are content-independent of size)
# ---------------------------------------------------------------------------


SIZE = (96, 96, 3)


def _solid(colour: tuple[int, int, int]) -> np.ndarray:
    arr = np.empty(SIZE, dtype=np.uint8)
    arr[:] = colour
    return arr


def _textured(base_colour: tuple[int, int, int]) -> np.ndarray:
    """Solid background with a contrasting rectangle — has real structure."""
    arr = _solid(base_colour)
    inv = tuple(255 - c for c in base_colour)
    arr[20:60, 20:60] = inv
    return arr


# ---------------------------------------------------------------------------
# Identity preservation — flat-vs-itself must still report identity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("colour", [(0, 0, 0), (255, 255, 255), (128, 128, 128)])
def test_fsim_identity_on_flat_self_pair_returns_one(
    colour: tuple[int, int, int]
) -> None:
    frame = _solid(colour)
    assert fsim(frame, frame.copy()) == pytest.approx(1.0)


@pytest.mark.parametrize("colour", [(0, 0, 0), (255, 255, 255), (128, 128, 128)])
def test_gmsd_identity_on_flat_self_pair_returns_zero(
    colour: tuple[int, int, int]
) -> None:
    frame = _solid(colour)
    assert gmsd(frame, frame.copy()) == pytest.approx(0.0)


@pytest.mark.parametrize("colour", [(0, 0, 0), (255, 255, 255), (128, 128, 128)])
def test_edge_similarity_identity_on_flat_self_pair_returns_one(
    colour: tuple[int, int, int]
) -> None:
    frame = _solid(colour)
    assert edge_similarity(frame, frame.copy()) == pytest.approx(1.0)


@pytest.mark.parametrize("colour", [(0, 0, 0), (255, 255, 255), (128, 128, 128)])
def test_sharpness_similarity_identity_on_flat_self_pair_returns_one(
    colour: tuple[int, int, int],
) -> None:
    frame = _solid(colour)
    assert sharpness_similarity(frame, frame.copy()) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Pathological discrimination — two different flat colours must NOT score 1.0
# ---------------------------------------------------------------------------


def test_fsim_white_vs_black_is_not_identity() -> None:
    """Audit regression: white vs black previously returned 1.0 (perfect)."""
    score = fsim(_solid((255, 255, 255)), _solid((0, 0, 0)))
    assert score < 0.5, (
        f"fsim(white, black) returned {score:.4f} — should report worst-case, "
        "not identity. See audit: docs/metrics-audit/2026-05-22/report.md"
    )


def test_gmsd_white_vs_black_is_not_zero() -> None:
    """Audit regression: white vs black previously returned 0.0 (no distortion)."""
    score = gmsd(_solid((255, 255, 255)), _solid((0, 0, 0)))
    assert score > 0.1, (
        f"gmsd(white, black) returned {score:.4f} — should report distortion, "
        "not 0.0 (no distortion). See audit."
    )


def test_edge_similarity_white_vs_black_is_not_identity() -> None:
    """Audit regression: white vs black previously returned 1.0 (perfect)."""
    score = edge_similarity(_solid((255, 255, 255)), _solid((0, 0, 0)))
    assert score < 0.5, (
        f"edge_similarity(white, black) returned {score:.4f} — should report "
        "worst-case, not identity. See audit."
    )


def test_sharpness_similarity_white_vs_black_is_not_identity() -> None:
    """Audit regression: white vs black previously returned 1.0 (perfect)."""
    score = sharpness_similarity(_solid((255, 255, 255)), _solid((0, 0, 0)))
    assert score < 0.5, (
        f"sharpness_similarity(white, black) returned {score:.4f} — should "
        "report worst-case, not identity. See audit."
    )


# ---------------------------------------------------------------------------
# Two-flat near-match — slightly-different greys should NOT collapse to worst
# ---------------------------------------------------------------------------


def test_fsim_near_identical_flats_stay_high() -> None:
    """grey-128 vs grey-129 differs by 1/255 — should be near-identity, not worst."""
    score = fsim(_solid((128, 128, 128)), _solid((129, 129, 129)))
    assert score == pytest.approx(1.0)


def test_gmsd_near_identical_flats_stay_low() -> None:
    score = gmsd(_solid((128, 128, 128)), _solid((129, 129, 129)))
    assert score == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Asymmetric (one flat, one textured) — unambiguous mismatch
# ---------------------------------------------------------------------------


def test_fsim_flat_vs_textured_is_not_identity() -> None:
    score = fsim(_solid((255, 255, 255)), _textured((255, 255, 255)))
    assert score < 0.95


def test_edge_similarity_flat_vs_textured_is_zero() -> None:
    """Flat side has no edges; textured side has edges. Intersection must be 0."""
    score = edge_similarity(_solid((255, 255, 255)), _textured((255, 255, 255)))
    assert score == pytest.approx(0.0)


def test_sharpness_similarity_flat_vs_textured_is_zero() -> None:
    """Flat side has zero variance; textured has positive. Must report worst-case."""
    score = sharpness_similarity(_solid((255, 255, 255)), _textured((255, 255, 255)))
    assert score == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Structural-content unchanged — fix must NOT regress non-flat behaviour
# ---------------------------------------------------------------------------


def test_fsim_textured_self_pair_unchanged() -> None:
    frame = _textured((255, 255, 255))
    score = fsim(frame, frame.copy())
    assert score == pytest.approx(1.0)


def test_gmsd_textured_self_pair_unchanged() -> None:
    frame = _textured((255, 255, 255))
    score = gmsd(frame, frame.copy())
    assert score == pytest.approx(0.0, abs=1e-6)


def test_edge_similarity_textured_self_pair_unchanged() -> None:
    frame = _textured((255, 255, 255))
    score = edge_similarity(frame, frame.copy())
    assert score == pytest.approx(1.0)


def test_sharpness_similarity_textured_self_pair_unchanged() -> None:
    frame = _textured((255, 255, 255))
    score = sharpness_similarity(frame, frame.copy())
    assert score == pytest.approx(1.0)
