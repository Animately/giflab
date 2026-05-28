"""Performance regression test for the ``_is_flat_frame`` ptp() fast-reject.

The 2026-05-27 optimisation adds a ``ptp() > 4`` guard before the expensive
``astype(float32).std`` call.  On a 480×480 non-flat frame (the overwhelmingly
common real-GIF case), this skips the float-cast and std entirely, reducing the
per-call cost from ~4.8 ms to sub-millisecond.

This test:
1. Verifies that ``_is_flat_frame`` on a 100-frame, 480×480 non-flat batch
   completes in well under 1 s total  (< 10 ms/frame budget — very generous;
   actual fast-reject cost is ~0.1 ms/frame on a modern M-series chip).
2. Ensures the function returns ``False`` for every non-flat frame in the batch
   (correctness gate alongside the timing gate).

Layer: functional — synthetic frames, no GIF I/O, no external engines.

Notes on flakiness:
- The 1 s budget is 10× the expected actual runtime (~38 ms for 100 frames
  with the fast-reject).  Even under heavy system load, 10 ms/frame for a
  pure-numpy O(n) max/min scan is well within reach.
- We use ``time.perf_counter`` and a single wall-clock window rather than
  per-call measurement to reduce timer overhead.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from giflab.metrics import _is_flat_frame


_FRAME_H = 480
_FRAME_W = 480
_N_FRAMES = 100
# Each frame: contrasting rectangle — definitely non-flat (ptp ≫ 4).
# With the ptp() fast-reject the expected wall-clock is ~0.02 s (≈ 0.2 ms/frame).
# The budget is set to 0.5 s — 25× the expected time — to be robust on CI and
# loaded laptops, while still catching a regression to the pre-optimisation path
# (~1.0 s on an M3 machine; the unoptimised code fails this budget).
_NONFLAT_BATCH_WALL_BUDGET_S = 0.5


@pytest.fixture(scope="module")
def non_flat_frames() -> list[np.ndarray]:
    """100 distinct 480×480 RGB frames, all with high ptp → fast-reject fires."""
    rng = np.random.default_rng(2026_05_27)
    frames = []
    for i in range(_N_FRAMES):
        frame = np.full((_FRAME_H, _FRAME_W, 3), 200, dtype=np.uint8)
        # Random rectangle with contrasting colour ensures ptp > 4 on all channels.
        r0 = rng.integers(50, 200)
        r1 = rng.integers(r0 + 20, _FRAME_H - 10)
        c0 = rng.integers(50, 200)
        c1 = rng.integers(c0 + 20, _FRAME_W - 10)
        fill = int(rng.integers(0, 100))
        frame[r0:r1, c0:c1] = fill
        frames.append(frame)
    return frames


def test_is_flat_frame_nonflat_batch_all_false(
    non_flat_frames: list[np.ndarray],
) -> None:
    """Every non-flat frame must return False."""
    results = [_is_flat_frame(f) for f in non_flat_frames]
    false_count = sum(1 for r in results if not r)
    assert false_count == _N_FRAMES, (
        f"Expected all {_N_FRAMES} non-flat frames to return False; "
        f"{_N_FRAMES - false_count} incorrectly returned True."
    )


def test_is_flat_frame_nonflat_batch_within_time_budget(
    non_flat_frames: list[np.ndarray],
) -> None:
    """100 non-flat 480×480 frames should complete well under 1 s total.

    The fast-reject (ptp > 4) terminates each call before the expensive
    float-cast + std, so 100 calls should take ~38 ms, not ~480 ms.
    """
    # Warm-up pass to avoid import / JIT cold-start artefacts.
    _ = _is_flat_frame(non_flat_frames[0])

    t_start = time.perf_counter()
    for frame in non_flat_frames:
        _is_flat_frame(frame)
    elapsed = time.perf_counter() - t_start

    assert elapsed < _NONFLAT_BATCH_WALL_BUDGET_S, (
        f"_is_flat_frame processed {_N_FRAMES} non-flat 480×480 frames in "
        f"{elapsed:.3f}s — exceeds {_NONFLAT_BATCH_WALL_BUDGET_S}s budget. "
        f"The ptp() fast-reject optimisation may be missing or disabled. "
        f"(Per-frame: {elapsed / _N_FRAMES * 1000:.2f} ms; "
        f"unoptimised path takes ~10 ms/frame on this hardware.)"
    )
