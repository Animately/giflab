"""Correctness tests for the ptp() fast-reject in ``_is_flat_frame``.

The optimisation adds a peak-to-peak (max−min) guard before the expensive
``astype(float32).std`` call.  The fast-reject may only return ``False``
when the frame is provably non-flat; it must never suppress a ``True``
(flat) result.

Layer: smoke — pure logic, no GIF I/O, no engines.
"""

from __future__ import annotations

import numpy as np
import pytest
from giflab.metrics import _is_flat_frame

# ---------------------------------------------------------------------------
# Flat frames must still return True
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "colour",
    [(0, 0, 0), (255, 255, 255), (128, 128, 128), (64, 127, 200)],
)
def test_flat_rgb_frame_returns_true(colour: tuple[int, int, int]) -> None:
    """Solid-colour 480×480 RGB frame should always be detected as flat."""
    frame = np.full((480, 480, 3), colour, dtype=np.uint8)
    assert _is_flat_frame(frame) is True, (
        f"Solid colour {colour} should be flat but _is_flat_frame returned False. "
        "Fast-reject guard may be too aggressive."
    )


@pytest.mark.parametrize("value", [0, 128, 255])
def test_flat_grayscale_frame_returns_true(value: int) -> None:
    """Solid-value 480×480 grayscale frame should be detected as flat."""
    frame = np.full((480, 480), value, dtype=np.uint8)
    assert _is_flat_frame(frame) is True


def test_near_flat_frame_within_threshold_returns_true() -> None:
    """Frame with ptp ≤ 1 and std < 1.0 must not be rejected by the fast path."""
    frame = np.full((480, 480, 3), 100, dtype=np.uint8)
    # Set a single pixel one DN higher — ptp becomes 1, std ≈ 0.001 → still flat
    frame[0, 0] = (101, 101, 101)
    assert _is_flat_frame(frame) is True


# ---------------------------------------------------------------------------
# Non-flat frames must return False
# ---------------------------------------------------------------------------


def test_textured_frame_returns_false() -> None:
    """Real-content 480×480 frame (contrasting rectangle) should be non-flat."""
    frame = np.full((480, 480, 3), 255, dtype=np.uint8)
    frame[100:380, 100:380] = 0  # large black rectangle → high std
    assert _is_flat_frame(frame) is False


def test_high_ptp_frame_returns_false() -> None:
    """Frame with ptp > 4 in at least one channel should fast-reject to False."""
    frame = np.full((480, 480, 3), 100, dtype=np.uint8)
    # Scattered pixels with value 110 → ptp = 10, well above 4
    rng = np.random.default_rng(42)
    idx = rng.integers(0, 480, size=(5_000, 2))
    frame[idx[:, 0], idx[:, 1]] = 110
    assert _is_flat_frame(frame) is False


def test_greyscale_frame_with_wide_spread_returns_false() -> None:
    """Greyscale frame with many pixels above threshold is non-flat."""
    frame = np.full((64, 64), 50, dtype=np.uint8)
    # Set half the pixels to 60 (ptp=10, std >> threshold) — unambiguously non-flat
    frame[:32, :] = 60
    assert _is_flat_frame(frame) is False


# ---------------------------------------------------------------------------
# Boundary: ptp exactly at the guard boundary does not break accuracy
# ---------------------------------------------------------------------------


def test_ptp_equals_4_still_accurate() -> None:
    """ptp = 4 sits just at the guard boundary; std < 1.0 → still flat."""
    # 480*480 = 230400 pixels; one pixel at +4, rest at 0.
    # std ≈ 4 / sqrt(230400) ≈ 0.008 — well below FLAT_STD_THRESHOLD=1.0.
    frame = np.zeros((480, 480, 3), dtype=np.uint8)
    frame[0, 0] = (4, 4, 4)  # ptp == 4 on each channel
    # With the guard "ptp > 4", this falls through to the accurate std check.
    assert _is_flat_frame(frame) is True


def test_large_frame_high_ptp_returns_false() -> None:
    """A 480×480 frame with ptp >> 4 and many pixels at the high value is non-flat.

    Note: ptp > 4 alone does NOT guarantee std >= threshold when the high-value
    pixels are rare outliers (e.g., 1 pixel in 230400 gives std ≈ 0.01).  This
    test uses a substantial minority (1/4 of all pixels) to ensure the frame is
    genuinely non-flat (std >> 1.0).
    """
    frame = np.zeros((480, 480, 3), dtype=np.uint8)
    # 120×480 rows at value 5 → 1/4 of all pixels, ptp = 5, std >> threshold
    frame[:120, :] = (5, 5, 5)
    assert _is_flat_frame(frame) is False


# ---------------------------------------------------------------------------
# Non-flat result consistent with original logic (regression)
# ---------------------------------------------------------------------------


def test_result_matches_reference_for_random_frame() -> None:
    """Optimised path must agree with the unoptimised reference implementation."""

    def _reference(frame: np.ndarray, threshold: float = 1.0) -> bool:
        arr = frame.astype(np.float32)
        if arr.ndim == 2:
            return float(np.std(arr)) < threshold
        return bool(np.all(np.std(arr.reshape(-1, arr.shape[-1]), axis=0) < threshold))

    rng = np.random.default_rng(0)
    for _ in range(50):
        frame = rng.integers(0, 256, size=(48, 48, 3), dtype=np.uint8)
        assert _is_flat_frame(frame) == _reference(
            frame
        ), "Optimised _is_flat_frame disagrees with reference implementation."
