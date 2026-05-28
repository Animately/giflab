"""Regression tests for structure-based metrics on flat-content (solid colour) pairs.

Audit context: ``docs/metrics-audit/2026-05-22/report.md`` flagged that ``fsim``,
``gmsd``, ``edge_similarity``, and ``sharpness_similarity`` returned
identity-equivalent values when fed two maximally different solid-colour
frames (e.g. white vs black). Cause: every metric is structure-derived, and
on flat content every derivative is exactly zero, so the zero/zero stability
fallback (or the explicit empty-edge / zero-variance early return) silently
reports "perfect".

Fix decision (see task note ``giflab-fsim-flat-content-returns-1`` and
``giflab-flat-mean-tol-recalibration``): the four metrics now use a
*content-aware honest fallback* with smooth degradation:

* both frames flat AND L2 colour distance < FLAT_IDENTITY_FLOOR  → identity
* both frames flat AND L2 distance between floor and floor+scale  → smooth blend
* both frames flat AND L2 >= floor + scale                        → worst-case
* exactly one frame flat → worst-case value (unambiguous mismatch)
* otherwise → unchanged structure-based computation

These tests pin that behaviour for the four affected metrics, including the
sub-DN cliff case that PR #13's hard FLAT_MEAN_TOL = 1.0 missed.

Rollout: [[giflab-rollout-2026-05-26]] Wave 2.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from giflab.metrics import (
    FLAT_DEGRADATION_SCALE,
    FLAT_IDENTITY_FLOOR,
    FLAT_STD_THRESHOLD,
    _flat_content_fallback,
    _flat_mean_distance,
    _is_flat_frame,
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
# Module-level constants sanity check
# ---------------------------------------------------------------------------


def test_flat_identity_floor_is_positive() -> None:
    """FLAT_IDENTITY_FLOOR must be > 0 to tolerate floating-point round-trips."""
    assert FLAT_IDENTITY_FLOOR > 0.0


def test_flat_degradation_scale_is_larger_than_floor() -> None:
    """FLAT_DEGRADATION_SCALE must be > FLAT_IDENTITY_FLOOR for the blend formula."""
    assert FLAT_DEGRADATION_SCALE > FLAT_IDENTITY_FLOOR


def test_flat_std_threshold_is_positive() -> None:
    assert FLAT_STD_THRESHOLD > 0.0


# ---------------------------------------------------------------------------
# _is_flat_frame helper
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("colour", [(0, 0, 0), (255, 255, 255), (128, 128, 128)])
def test_is_flat_frame_detects_solid(colour: tuple[int, int, int]) -> None:
    assert _is_flat_frame(_solid(colour)) is True


def test_is_flat_frame_rejects_textured() -> None:
    assert _is_flat_frame(_textured((255, 255, 255))) is False


# ---------------------------------------------------------------------------
# _flat_mean_distance helper
# ---------------------------------------------------------------------------


def test_flat_mean_distance_identical_is_zero() -> None:
    frame = _solid((128, 128, 128))
    assert _flat_mean_distance(frame, frame.copy()) == pytest.approx(0.0, abs=1e-6)


def test_flat_mean_distance_white_vs_black() -> None:
    """sqrt((255**2)*3) = 255*sqrt(3) ≈ 441.67."""
    dist = _flat_mean_distance(_solid((255, 255, 255)), _solid((0, 0, 0)))
    assert dist == pytest.approx(255 * math.sqrt(3), rel=1e-4)


def test_flat_mean_distance_sub_dn() -> None:
    """(255,255,255) vs (254,254,254) has L2 = sqrt(3) ≈ 1.73."""
    dist = _flat_mean_distance(_solid((255, 255, 255)), _solid((254, 254, 254)))
    assert dist == pytest.approx(math.sqrt(3), rel=1e-4)


# ---------------------------------------------------------------------------
# _flat_content_fallback — cliff test (THIS MUST FAIL BEFORE THE FIX)
# ---------------------------------------------------------------------------


def test_flat_content_fallback_sub_dn_is_identity_not_worst() -> None:
    """PR #13 cliff: (255,255,255) vs (253,253,253) has L2≈3.46 which is
    above the old FLAT_MEAN_TOL threshold (sqrt(3)≈1.73). With the hard
    threshold it returned worst_value=0.0, but the pair is perceptually
    indistinguishable — it MUST return near identity_value.

    This test demonstrates the cliff bug. After the smooth-degradation fix
    the fallback must NOT return worst_value (0.0) for this sub-DN pair.
    """
    white = _solid((255, 255, 255))
    near_white = _solid((253, 253, 253))  # L2 ≈ sqrt(12) ≈ 3.46
    # L2 ≈ 3.46 is below FLAT_IDENTITY_FLOOR (≈1.0) + FLAT_DEGRADATION_SCALE (≈50)
    # so result must be well above 0.0 (should be near-identity)
    result = _flat_content_fallback(white, near_white, identity_value=1.0, worst_value=0.0)
    assert result is not None
    # With smooth degradation at L2≈3.46, floor=1.0, scale=50.0:
    # blend = (3.46 - 1.0) / 50.0 ≈ 0.049
    # score = 1.0 * (1 - 0.049) + 0.0 * 0.049 ≈ 0.951
    assert result > 0.5, (
        f"_flat_content_fallback returned {result:.4f} for sub-DN pair "
        f"(255,255,255) vs (253,253,253) — this is the PR #13 cliff bug. "
        f"The smooth-degradation fix should return near-identity (>0.5), not worst-case."
    )


def test_flat_content_fallback_exact_identity() -> None:
    """Same colour → exactly identity_value."""
    frame = _solid((128, 128, 128))
    result = _flat_content_fallback(frame, frame.copy(), identity_value=1.0, worst_value=0.0)
    assert result == pytest.approx(1.0)


def test_flat_content_fallback_catastrophe_is_worst() -> None:
    """White vs black (L2≈441) is far past the saturation point → worst_value."""
    result = _flat_content_fallback(
        _solid((255, 255, 255)), _solid((0, 0, 0)),
        identity_value=1.0, worst_value=0.0
    )
    assert result == pytest.approx(0.0)


def test_flat_content_fallback_one_flat_is_worst() -> None:
    """Asymmetric case: one flat, one textured → worst."""
    result = _flat_content_fallback(
        _solid((255, 255, 255)), _textured((255, 255, 255)),
        identity_value=1.0, worst_value=0.0
    )
    assert result == pytest.approx(0.0)


def test_flat_content_fallback_two_non_flat_returns_none() -> None:
    """Neither flat → None (caller falls through to structure-based computation)."""
    result = _flat_content_fallback(
        _textured((255, 255, 255)), _textured((0, 0, 0)),
        identity_value=1.0, worst_value=0.0
    )
    assert result is None


# ---------------------------------------------------------------------------
# Monotonicity — smooth curve must be strictly monotonic (no rebounds)
# ---------------------------------------------------------------------------


def test_flat_content_fallback_monotonic_over_colour_sweep() -> None:
    """Verify smooth degradation: as L2 grows from 0 to 255, score monotonically
    decreases from identity to worst. Uses fsim (identity=1, worst=0).

    Sweeps (255,255,255) vs (255-k, 255-k, 255-k) for k = 0..255.
    """
    scores: list[float] = []
    for k in range(0, 256, 5):  # step=5 to keep test fast
        frame1 = _solid((255, 255, 255))
        frame2 = _solid((255 - k, 255 - k, 255 - k))
        result = _flat_content_fallback(frame1, frame2, identity_value=1.0, worst_value=0.0)
        assert result is not None, f"Expected non-None fallback for k={k}"
        scores.append(result)

    for i in range(1, len(scores)):
        assert scores[i] <= scores[i - 1] + 1e-9, (
            f"Monotonicity violation at step {i}: "
            f"score[{i-1}]={scores[i-1]:.6f} < score[{i}]={scores[i]:.6f}"
        )


def test_flat_content_fallback_saturates_to_worst() -> None:
    """Past floor+scale the score must saturate at worst_value (no undershoot)."""
    # L2 well past FLAT_IDENTITY_FLOOR + FLAT_DEGRADATION_SCALE
    # (255,255,255) vs (0,0,0): L2 ≈ 441 >> 1 + 50 = 51
    result = _flat_content_fallback(
        _solid((255, 255, 255)), _solid((0, 0, 0)),
        identity_value=1.0, worst_value=0.0
    )
    assert result == pytest.approx(0.0, abs=1e-9)


def test_flat_content_fallback_below_floor_is_exact_identity() -> None:
    """Below FLAT_IDENTITY_FLOOR the score must be exactly identity_value."""
    # (128,128,128) vs (128,128,128): L2 = 0 < FLAT_IDENTITY_FLOOR
    result = _flat_content_fallback(
        _solid((128, 128, 128)), _solid((128, 128, 128)),
        identity_value=1.0, worst_value=0.0
    )
    assert result == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# gmsd — separate worst_value (0.5 not 0.0)
# ---------------------------------------------------------------------------


def test_flat_content_fallback_gmsd_identity_is_zero() -> None:
    """gmsd identity is 0.0 (lower is better)."""
    frame = _solid((128, 128, 128))
    result = _flat_content_fallback(frame, frame.copy(), identity_value=0.0, worst_value=0.5)
    assert result == pytest.approx(0.0)


def test_flat_content_fallback_gmsd_catastrophe_is_half() -> None:
    """gmsd worst-case for flat content is 0.5."""
    result = _flat_content_fallback(
        _solid((255, 255, 255)), _solid((0, 0, 0)),
        identity_value=0.0, worst_value=0.5
    )
    assert result == pytest.approx(0.5)


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
# Pathological discrimination — white vs black must NOT score as identity
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
# Sub-DN cliff regression — the PR #13 regression this PR fixes
# ---------------------------------------------------------------------------


def test_fsim_sub_dn_drift_stays_near_identity() -> None:
    """(255,255,255) vs (253,253,253) — 2DN per channel, sub-pixel drift.

    PR #13 cliff bug: FLAT_MEAN_TOL=1.0 * sqrt(3)≈1.73 < L2≈3.46 → returned
    worst-case 0.0. Smooth degradation must keep this near identity.
    """
    score = fsim(_solid((255, 255, 255)), _solid((253, 253, 253)))
    assert score > 0.5, (
        f"fsim for 2DN sub-pixel drift returned {score:.4f}. "
        f"PR #13 cliff bug: this perceptually-invisible drift should stay near "
        f"identity, not collapse to worst-case."
    )


def test_fsim_small_drift_not_worst_case() -> None:
    """(255,255,255) vs (250,250,250) — 5DN per channel.
    Must be degraded but NOT worst-case (0.0).
    """
    score = fsim(_solid((255, 255, 255)), _solid((250, 250, 250)))
    assert score > 0.0, (
        f"fsim for 5DN drift returned {score:.4f} — should be modestly degraded, "
        f"not worst-case."
    )


def test_fsim_medium_drift_mid_band() -> None:
    """(255,255,255) vs (200,200,200) — 55DN per channel, L2≈95.
    With scale=50 this is well past saturation → near worst-case.
    """
    score = fsim(_solid((255, 255, 255)), _solid((200, 200, 200)))
    # L2 ≈ 55 * sqrt(3) ≈ 95 >> floor+scale ≈ 51 → should be worst-case (0.0)
    assert score <= 0.1, (
        f"fsim for 55DN drift returned {score:.4f} — should be at/near worst-case."
    )


def test_gmsd_sub_dn_drift_stays_near_identity() -> None:
    """gmsd version of the PR #13 cliff regression test."""
    score = gmsd(_solid((255, 255, 255)), _solid((253, 253, 253)))
    assert score < 0.4, (
        f"gmsd for 2DN drift returned {score:.4f} — PR #13 cliff returned "
        f"worst-case (0.5). Should stay near identity (0.0)."
    )


def test_edge_similarity_sub_dn_drift_stays_near_identity() -> None:
    """edge_similarity version of the PR #13 cliff regression test."""
    score = edge_similarity(_solid((255, 255, 255)), _solid((253, 253, 253)))
    assert score > 0.5, (
        f"edge_similarity for 2DN drift returned {score:.4f} — should stay near "
        f"identity (1.0), not collapse to 0.0."
    )


def test_sharpness_similarity_sub_dn_drift_stays_near_identity() -> None:
    """sharpness_similarity version of the PR #13 cliff regression test."""
    score = sharpness_similarity(_solid((255, 255, 255)), _solid((253, 253, 253)))
    assert score > 0.5, (
        f"sharpness_similarity for 2DN drift returned {score:.4f} — should stay "
        f"near identity (1.0), not collapse to 0.0."
    )


# ---------------------------------------------------------------------------
# Asymmetric (one flat, one textured) — unambiguous mismatch → worst-case
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
