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

See ~/repos/obsidian/Work/Tasks/giflab-audit-monotonicity-tolerance-content-aware.md
for context on the range-derived tolerance fix — the old fixed tol=1e-4 was
too tight for metrics with wider observed ranges, flagging noise-floor flips
as SUSPICIOUS even when they were well within the metric's resolution.
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
        # 5.0 -> 3.0 is a clear inversion (40% of range — well above the 5%
        # noise floor). With range-derived tol = max(1e-4, 5.0*0.05) = 0.25,
        # the drop of 2.0 is flagged as a real inversion.
        invs = sanity._monotonicity_check([0.0, 1.0, 5.0, 3.0], "lower_better")
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

    def test_range_derived_tol_absorbs_noise_floor_flip(self) -> None:
        """A flip well within 5% of the observed range should not be flagged.

        Scenario: a metric with observed range 0.1 (values 0.8–0.9) has a
        small bump of +0.004 at the last level — this is a noise-floor flip
        (4% of the range) that the old fixed tol=1e-4 incorrectly flagged as
        SUSPICIOUS. The range-derived tol (max(1e-4, 0.1 * 0.05) = 0.005)
        should absorb it.

        This is the motivating case from
        ~/repos/obsidian/Work/Tasks/giflab-audit-monotonicity-tolerance-content-aware.md
        and the chist investigation (PR #11): metrics saturated near their
        ceiling have an effective resolution set by their observed range, not
        by a universal fixed epsilon.
        """
        # Observed range: 0.9 - 0.8 = 0.1
        # Flip size at index 2→3: 0.804 - 0.800 = 0.004
        # Old tol=1e-4: 0.004 > 1e-4 → would be flagged
        # New tol=max(1e-4, 0.1 * 0.05)=0.005: 0.004 < 0.005 → absorbed
        invs = sanity._monotonicity_check(
            [0.9, 0.85, 0.800, 0.804], "higher_better"
        )
        assert invs == [], (
            f"Expected no inversions for a noise-floor flip (0.004) within "
            f"5% of the observed range (0.1), but got: {invs}"
        )

    def test_range_derived_tol_still_flags_real_inversions(self) -> None:
        """A large inversion (>5% of range) must still be flagged.

        Sanity check: the range-derived tolerance does not suppress real
        monotonicity violations, only noise-floor flips.
        """
        # Observed range: 0.9 - 0.6 = 0.3
        # Flip size: 0.75 - 0.6 = 0.15 (50% of range — clearly real)
        invs = sanity._monotonicity_check(
            [0.9, 0.75, 0.6, 0.75], "higher_better"
        )
        assert len(invs) == 1, (
            "Expected exactly one inversion for a large flip (50% of range); "
            f"got {invs}"
        )

    def test_range_derived_tol_lower_better(self) -> None:
        """Range-derived tolerance works symmetrically for lower_better metrics."""
        # Observed range: 10.0 - 8.0 = 2.0
        # Flip size at index 2→3: 8.2 - 8.3 = -0.1 (went down when should go up)
        # Old tol=1e-4: 0.1 > 1e-4 → flagged
        # New tol=max(1e-4, 2.0*0.05)=0.1: 0.1 <= 0.1 → absorbed (boundary case)
        # Use 0.09 to be safely inside:
        # values: [8.0, 8.8, 9.0, 8.91] — flip from 9.0 to 8.91 = -0.09
        # range: 9.0 - 8.0 = 1.0; tol = max(1e-4, 1.0*0.05)=0.05
        # flip 0.09 > 0.05 → still flagged. Use smaller flip:
        # values: [8.0, 8.8, 9.0, 8.96] — flip = -0.04, tol=0.05 → absorbed
        invs = sanity._monotonicity_check(
            [8.0, 8.8, 9.0, 8.96], "lower_better"
        )
        assert invs == [], (
            f"Expected no inversions for lower_better noise flip (0.04) "
            f"within 5% of observed range (1.0), but got: {invs}"
        )


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


def _make_multiframe_gif(path: Path, color: tuple[int, int, int] = (128, 128, 128)) -> None:
    """Create a 2-frame solid-colour GIF that ``read_gif_frames`` can read."""
    from PIL import Image

    frames = [Image.new("RGB", (16, 16), color=color) for _ in range(2)]
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=100,
        loop=0,
        optimize=False,
    )


class _FakeSpec:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeMetricsConfig:
    """Settable-attribute stand-in for ``giflab.config.MetricsConfig``."""

    def __init__(self) -> None:
        self.ENABLE_DEEP_PERCEPTUAL = False
        self.ENABLE_TEMPORAL_ARTIFACTS = False


def _build_fake_gl(metric_dict_factory):
    """Build a fake ``gl`` dict for ``_import_giflab``.

    ``metric_dict_factory`` is called once per ``run_metrics`` invocation
    and returns the dict of metric values for that call.
    """
    class _FakeGenerator:
        def __init__(self, out_dir: Path) -> None:
            self.out_dir = out_dir
            # Cover both IDENTITY_SAMPLE and MONOTONICITY_BASES with a
            # single base name; tests below pin both constants to a single
            # shared name to keep the fixture minimal.
            self.synthetic_specs = [_FakeSpec("base_under_test")]

        def generate_gifs(self, *, use_targeted_set: bool = False):
            p = self.out_dir / "base_under_test.gif"
            _make_multiframe_gif(p)
            return [p]

    return {
        "SyntheticGifGenerator": _FakeGenerator,
        "calculate_comprehensive_metrics": lambda original, compressed, **kwargs: metric_dict_factory(),
        "compress": None,  # filled in per-test
        "MetricsConfig": _FakeMetricsConfig,
    }


class TestCoalesceIntegratedIntoRunSanity:
    """The coalesce helper must actually be invoked by ``run_sanity()``'s lossy
    arm — otherwise the smooth_gradient SSIM bump-up bug it exists to fix is
    still present at runtime. These tests guard the wiring, not just the
    helper in isolation.
    """

    def _pin_constants(self, monkeypatch) -> None:
        """Shrink all the degradation grids and pin IDENTITY_SAMPLE +
        MONOTONICITY_BASES to a single shared name so the test runs in a
        few milliseconds.
        """
        monkeypatch.setattr(sanity, "MONOTONICITY_BASES", ["base_under_test"])
        monkeypatch.setattr(sanity, "IDENTITY_SAMPLE", ["base_under_test"])
        monkeypatch.setattr(sanity, "NOISE_SIGMAS", [5])
        monkeypatch.setattr(sanity, "BLUR_SIGMAS", [0.5])
        monkeypatch.setattr(sanity, "QUANTIZE_COLORS", [64])
        # Use 4 lossy levels: first 2 unique, last 2 byte-identical with #2.
        monkeypatch.setattr(sanity, "LOSSY_LEVELS", [30, 60, 120, 200])

    def test_lossy_arm_calls_coalesce_with_paths_and_metrics(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """Spy on ``_coalesce_byte_identical_levels`` and assert ``run_sanity()``
        invokes it from the lossy arm exactly once with the ordered paths and
        the per-metric levels collected during the lossy loop.
        """
        gl = _build_fake_gl(lambda: {"ssim": 0.5, "psnr": 30.0})

        # First two lossy levels write unique bytes; levels 3 and 4 saturate
        # and produce byte-identical outputs that should be coalesced.
        call_counter = {"n": 0}

        def _fake_compress(input_path, output_path, engine, params):
            call_counter["n"] += 1
            n = call_counter["n"]
            if n == 1:
                Path(output_path).write_bytes(b"unique-level-1")
            elif n == 2:
                Path(output_path).write_bytes(b"unique-level-2")
            else:
                Path(output_path).write_bytes(b"SATURATED")
            return output_path

        gl["compress"] = _fake_compress
        monkeypatch.setattr(sanity, "_import_giflab", lambda: gl)
        self._pin_constants(monkeypatch)

        real_coalesce = sanity._coalesce_byte_identical_levels
        calls: list[tuple[list[Path], dict[str, list[float]]]] = []

        def _spy(paths, per_metric):
            calls.append(
                (list(paths), {k: list(v) for k, v in per_metric.items()})
            )
            return real_coalesce(paths, per_metric)

        monkeypatch.setattr(sanity, "_coalesce_byte_identical_levels", _spy)

        sanity.run_sanity(tmp_path)

        # Coalesce must be invoked exactly once — only the lossy arm calls it.
        assert len(calls) == 1, (
            "Expected exactly one call to _coalesce_byte_identical_levels "
            f"from the lossy arm; got {len(calls)}"
        )
        passed_paths, passed_metrics = calls[0]
        # 4 lossy levels => 4 paths and 4 metric values per metric pre-coalesce.
        assert len(passed_paths) == 4
        for vals in passed_metrics.values():
            assert len(vals) == 4

        # And the coalesced result drops the trailing duplicate, leaving 3.
        kept_paths, kept_metrics = real_coalesce(passed_paths, passed_metrics)
        assert len(kept_paths) == 3
        for vals in kept_metrics.values():
            assert len(vals) == 3

    def test_lossy_arm_stores_coalesced_not_raw(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """Direct integration assertion: replace ``_coalesce_byte_identical_levels``
        with a sentinel-returning stub and confirm the SENTINEL ends up in the
        stored monotonicity entry (i.e. is what ``_monotonicity_check`` sees
        downstream). Proves ``run_sanity()`` uses the helper's output and
        doesn't silently fall back to ``per_metric_levels``.
        """
        gl = _build_fake_gl(lambda: {"ssim": 0.5})

        def _fake_compress(input_path, output_path, engine, params):
            Path(output_path).write_bytes(b"saturated")
            return output_path

        gl["compress"] = _fake_compress
        monkeypatch.setattr(sanity, "_import_giflab", lambda: gl)
        self._pin_constants(monkeypatch)
        # Use 2 lossy levels for this test — content doesn't matter
        # because the stub coalesce returns a fixed sentinel.
        monkeypatch.setattr(sanity, "LOSSY_LEVELS", [30, 60])

        # Sentinel must have 2+ entries to clear the `len(vals) < 2`
        # skip-guard inside the monotonicity-check loop.
        sentinel_values = [99.0, 99.5]  # Distinctive — couldn't naturally arise.

        def _stub_coalesce(paths, per_metric):
            return list(paths), {"ssim": list(sentinel_values)}

        monkeypatch.setattr(
            sanity, "_coalesce_byte_identical_levels", _stub_coalesce
        )

        # Capture the value-lists passed to _monotonicity_check. The lossy
        # arm's stored entry should appear as [99.0] (the sentinel),
        # confirming run_sanity() reads from the coalesced result.
        captured: dict[str, list[list[float]]] = {"seen": []}
        real_check = sanity._monotonicity_check

        def _check_spy(values, direction, tol=1e-4):
            captured["seen"].append(list(values))
            return real_check(values, direction, tol)

        monkeypatch.setattr(sanity, "_monotonicity_check", _check_spy)

        sanity.run_sanity(tmp_path)

        assert sentinel_values in captured["seen"], (
            "Expected the sentinel value-list returned by the stubbed "
            "_coalesce_byte_identical_levels to be passed to "
            "_monotonicity_check via the stored monotonicity entry — "
            "proving run_sanity() uses the coalesced result, not the raw "
            f"per_metric_levels. Got: {captured['seen']}"
        )
