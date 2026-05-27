"""Smoke tests for `scripts/audit/sanity.py` pure helpers.

These tests cover the small pure utilities used by the audit sanity script —
no GIF I/O, no metric calculation. Specifically the monotonicity check and
the byte-identical coalescing helper added to guard against the
smooth_gradient SSIM bump-up false positive at lossy saturation.

See ~/repos/obsidian/Work/Tasks/giflab-ssim-smooth-gradient-bump.md for
context — animately's --lossy parameter saturates around level ~125 on
low-complexity content, so consecutive lossy grid points past the knee
produce byte-identical compressed outputs (or near-identical ones) whose
metric values trivially "bump up" by floating-point noise.
"""

from __future__ import annotations

import sys
from pathlib import Path

# scripts/audit/ is not a Python package; load sanity.py directly.
_SCRIPTS_AUDIT = (
    Path(__file__).resolve().parents[2] / "scripts" / "audit"
)
if str(_SCRIPTS_AUDIT) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_AUDIT))

import sanity  # type: ignore[import-not-found]  # noqa: E402


class TestMonotonicityCheck:
    def test_higher_better_monotonic(self) -> None:
        # Values strictly non-increasing: no inversions.
        assert sanity._monotonicity_check([1.0, 0.9, 0.8, 0.7], "higher_better") == []

    def test_higher_better_inversion(self) -> None:
        # 0.7 -> 0.75 is an inversion (got better when worse expected).
        invs = sanity._monotonicity_check([1.0, 0.9, 0.7, 0.75], "higher_better")
        assert len(invs) == 1
        assert invs[0][0] == 2  # index of the from-level
        assert invs[0][1] == 0.7
        assert invs[0][2] == 0.75

    def test_lower_better_monotonic(self) -> None:
        assert sanity._monotonicity_check([0.0, 1.0, 5.0, 10.0], "lower_better") == []

    def test_lower_better_inversion(self) -> None:
        # 5.0 -> 4.9 is an inversion (got smaller when larger expected).
        invs = sanity._monotonicity_check([0.0, 1.0, 5.0, 4.9], "lower_better")
        assert len(invs) == 1

    def test_flat_direction_returns_empty(self) -> None:
        # Direction "flat" means the metric doesn't discriminate; never flag.
        assert sanity._monotonicity_check([1.0, 0.9, 1.1, 0.8], "flat") == []

    def test_tolerance_absorbs_noise(self) -> None:
        # Default tol is 1e-4; a sub-tolerance bump is not flagged.
        invs = sanity._monotonicity_check(
            [0.9, 0.85, 0.85 + 5e-5, 0.84], "higher_better"
        )
        assert invs == []


class TestCoalesceByteIdenticalLevels:
    """The lossy arm of sanity.py should drop consecutive byte-identical
    compressions before running the monotonicity check, because saturated
    animately output produces literally identical bytes (and therefore
    identical metric values) — testing monotonicity on such a sequence is
    spurious.
    """

    def test_no_duplicates_passthrough(self, tmp_path: Path) -> None:
        # Each path has unique bytes -> nothing coalesced.
        p1 = tmp_path / "a.gif"
        p2 = tmp_path / "b.gif"
        p3 = tmp_path / "c.gif"
        p1.write_bytes(b"AAAA")
        p2.write_bytes(b"BBBB")
        p3.write_bytes(b"CCCC")

        per_metric = {"ssim": [0.9, 0.8, 0.7]}
        kept_paths, kept_metrics = sanity._coalesce_byte_identical_levels(
            [p1, p2, p3], per_metric
        )
        assert kept_paths == [p1, p2, p3]
        assert kept_metrics == {"ssim": [0.9, 0.8, 0.7]}

    def test_drops_consecutive_byte_identical(self, tmp_path: Path) -> None:
        # Saturation case: levels 100, 130, 160, 200 might produce
        # bytes A, B, B, B. Want to keep [A, B] only.
        p_a = tmp_path / "lossy_100.gif"
        p_b1 = tmp_path / "lossy_130.gif"
        p_b2 = tmp_path / "lossy_160.gif"
        p_b3 = tmp_path / "lossy_200.gif"
        p_a.write_bytes(b"unique-pre-saturation")
        p_b1.write_bytes(b"saturated-bytes")
        p_b2.write_bytes(b"saturated-bytes")
        p_b3.write_bytes(b"saturated-bytes")

        per_metric = {
            "ssim": [0.85, 0.83, 0.83, 0.83],
            "mse_max": [50.0, 48.0, 48.0, 48.0],
        }
        kept_paths, kept_metrics = sanity._coalesce_byte_identical_levels(
            [p_a, p_b1, p_b2, p_b3], per_metric
        )
        assert kept_paths == [p_a, p_b1]
        assert kept_metrics == {"ssim": [0.85, 0.83], "mse_max": [50.0, 48.0]}

    def test_handles_empty_input(self, tmp_path: Path) -> None:
        kept_paths, kept_metrics = sanity._coalesce_byte_identical_levels([], {})
        assert kept_paths == []
        assert kept_metrics == {}

    def test_only_drops_consecutive_duplicates(self, tmp_path: Path) -> None:
        # A, B, A, A: positions 0,2,3 share bytes "A" but only consecutive
        # duplicates are coalesced (positions 2-3 collapse, position 0
        # stays separate from position 2).
        p_a1 = tmp_path / "a1.gif"
        p_b = tmp_path / "b.gif"
        p_a2 = tmp_path / "a2.gif"
        p_a3 = tmp_path / "a3.gif"
        p_a1.write_bytes(b"X")
        p_b.write_bytes(b"Y")
        p_a2.write_bytes(b"X")
        p_a3.write_bytes(b"X")

        per_metric = {"ssim": [0.9, 0.5, 0.9, 0.9]}
        kept_paths, kept_metrics = sanity._coalesce_byte_identical_levels(
            [p_a1, p_b, p_a2, p_a3], per_metric
        )
        # Only a3 (consecutive duplicate of a2) is dropped.
        assert kept_paths == [p_a1, p_b, p_a2]
        assert kept_metrics == {"ssim": [0.9, 0.5, 0.9]}
