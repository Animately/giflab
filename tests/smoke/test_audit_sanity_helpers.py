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

import math
import sys
from pathlib import Path

# scripts/audit/ is not a Python package; load sanity.py directly.
_SCRIPTS_AUDIT = Path(__file__).resolve().parents[2] / "scripts" / "audit"
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
        # Range-derived tol on [0.9, 0.85, 0.85+5e-5, 0.84]:
        # observed_range = 0.9 - 0.84 = 0.06
        # tol = max(1e-4, 0.06 * 0.05) = max(1e-4, 0.003) = 0.003
        # The sub-tolerance bump of 5e-5 (≈0.08% of range) is well within
        # tol and is not flagged.
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
        invs = sanity._monotonicity_check([0.9, 0.85, 0.800, 0.804], "higher_better")
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
        invs = sanity._monotonicity_check([0.9, 0.75, 0.6, 0.75], "higher_better")
        assert len(invs) == 1, (
            "Expected exactly one inversion for a large flip (50% of range); "
            f"got {invs}"
        )

    def test_range_derived_tol_lower_better(self) -> None:
        """Range-derived tolerance works symmetrically for lower_better metrics."""
        # Values [8.0, 8.8, 9.0, 8.96]:
        # observed_range = 9.0 - 8.0 = 1.0
        # tol = max(1e-4, 1.0 * 0.05) = 0.05
        # Flip at index 2→3: 8.96 - 9.0 = -0.04 (went down when should go up)
        # |flip| = 0.04 < tol = 0.05 → absorbed as noise-floor
        # (Old fixed tol=1e-4 would have flagged this.)
        invs = sanity._monotonicity_check([8.0, 8.8, 9.0, 8.96], "lower_better")
        assert invs == [], (
            f"Expected no inversions for lower_better noise flip (0.04) "
            f"within 5% of observed range (1.0), but got: {invs}"
        )


class TestMetricClassification:
    """`_classify_metric` triages a metric name into one of four classes so the
    verdict loop can segregate structurally-non-monotonic-by-design metrics
    (dispersion siblings, single-stream `_compressed` keys, diagnostic/system
    metrics) from the SUSPICIOUS verdict — otherwise the ~1 genuine
    pairwise-quality signal (composite_quality) is buried under ~70 structural
    false positives. See docs/metrics-audit/2026-06-03/post-fix-verdict.md.
    """

    def test_primary_pairwise_quality_metrics(self) -> None:
        for m in (
            "composite_quality",
            "ssim",
            "ssim_mean",
            "ms_ssim",
            "psnr_mean",
            "edge_similarity",
            "texture_similarity",
            "fsim",
            "gmsd",
            "chist",
            "lpips_quality_mean",
            "ssimulacra2_mean",
            "deltae_mean",
            "banding_score_mean",
            "deltae_pct_gt2",
        ):
            assert sanity._classify_metric(m) == "pairwise_quality", m

    def test_dispersion_and_positional_siblings(self) -> None:
        for m in (
            "ssim_std",
            "edge_similarity_std",
            "gmsd_min",
            "fsim_max",
            "mse_first",
            "mse_last",
            "chist_middle",
            "chist_positional_variance",
            "deltae_max",
            "ssimulacra2_min",
        ):
            assert sanity._classify_metric(m) == "dispersion", m

    def test_single_stream_compressed_metrics(self) -> None:
        for m in (
            "temporal_consistency_compressed",
            "temporal_consistency_compressed_max",
            "temporal_consistency_post",
            "disposal_artifacts_compressed",
            "disposal_artifacts_post",
            "flicker_excess_compressed",
            "flicker_frame_ratio_compressed",
            "flat_flicker_ratio_compressed",
            "quality_oscillation_frequency_compressed",
            "temporal_pumping_score_compressed",
            "lpips_t_mean_compressed",
            "lpips_t_p95_compressed",
        ):
            assert sanity._classify_metric(m) == "single_stream", m

    def test_temporal_delta_is_pairwise_not_single_stream(self) -> None:
        # _delta is the genuine original-vs-compressed change signal — it must
        # stay eligible for SUSPICIOUS, unlike the single-stream siblings.
        assert (
            sanity._classify_metric("temporal_consistency_delta") == "pairwise_quality"
        )
        assert sanity._classify_metric("disposal_artifacts_delta") == "pairwise_quality"

    def test_diagnostic_system_metrics(self) -> None:
        for m in (
            "render_ms",
            "kilobytes",
            "efficiency",
            "frame_count",
            "compressed_frame_count",
            "color_count_compressed",
            "color_count_original",
            "compression_ratio",
        ):
            assert sanity._classify_metric(m) == "diagnostic", m


class TestSingleStreamKeysInSync:
    """Guard that sanity.py's locally-mirrored single-stream family stays a
    superset of giflab.metrics._SINGLE_STREAM_TEMPORAL_KEYS. The mirror exists
    so the pure classification helpers don't import the heavy metrics stack;
    this test (which DOES import giflab, lazily) keeps the two from drifting.
    """

    def test_mirror_covers_source_of_truth(self) -> None:
        from giflab.metrics import _SINGLE_STREAM_TEMPORAL_KEYS

        missing = set(_SINGLE_STREAM_TEMPORAL_KEYS) - sanity._SINGLE_STREAM_FAMILY_KEYS
        assert not missing, (
            "sanity._SINGLE_STREAM_FAMILY_KEYS has drifted from "
            f"giflab.metrics._SINGLE_STREAM_TEMPORAL_KEYS — missing: {missing}. "
            "Add the new key(s) so the audit verdict triage keeps segregating them."
        )


class TestDecideVerdict:
    """`_decide_verdict` returns (verdict, note, classification). Structural
    metrics with monotonicity failures must be downgraded from SUSPICIOUS to
    DIAGNOSTIC so the SUSPICIOUS list only contains genuine pairwise-quality
    inversions.
    """

    _FAIL = [
        {
            "kind": "lossy",
            "base": "smooth_gradient",
            "values": [1.0, 0.5],
            "inversions": [],
        }
    ]

    def test_pairwise_failure_is_suspicious(self) -> None:
        verdict, _note, classification = sanity._decide_verdict(
            "composite_quality", "higher_better", 1.0, 0.1858, self._FAIL, "solid"
        )
        assert verdict == "SUSPICIOUS"
        assert classification == "pairwise_quality"

    def test_dispersion_failure_is_diagnostic_not_suspicious(self) -> None:
        verdict, _note, classification = sanity._decide_verdict(
            "ssim_std", "lower_better", 0.0, 0.0099, self._FAIL, "structural"
        )
        assert verdict == "DIAGNOSTIC"
        assert classification == "dispersion"

    def test_single_stream_failure_is_diagnostic(self) -> None:
        verdict, _note, _c = sanity._decide_verdict(
            "temporal_consistency_compressed",
            "higher_better",
            0.97,
            0.89,
            self._FAIL,
            "structural",
        )
        assert verdict == "DIAGNOSTIC"

    def test_diagnostic_metric_failure_is_diagnostic(self) -> None:
        verdict, _note, _c = sanity._decide_verdict(
            "render_ms", "higher_better", 3952.0, 65.0, self._FAIL, "solid"
        )
        assert verdict == "DIAGNOSTIC"

    def test_no_failures_is_pass(self) -> None:
        verdict, _note, _c = sanity._decide_verdict(
            "ssim", "higher_better", 1.0, 0.0001, [], "solid"
        )
        assert verdict == "PASS"

    def test_flat_identity_equals_pathological_is_inconclusive(self) -> None:
        verdict, _note, _c = sanity._decide_verdict(
            "compression_ratio", "flat", 1.0, 1.0, [], "solid"
        )
        assert verdict == "INCONCLUSIVE"


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


def _make_multiframe_gif(
    path: Path, color: tuple[int, int, int] = (128, 128, 128)
) -> None:
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

        FIXTURE_MONOTONICITY_BASES is cleared so the fixture-based loop does
        not run a second coalesce call — tests in this class are asserting on
        the synthetic-bases lossy arm in isolation.
        """
        monkeypatch.setattr(sanity, "MONOTONICITY_BASES", ["base_under_test"])
        monkeypatch.setattr(sanity, "IDENTITY_SAMPLE", ["base_under_test"])
        monkeypatch.setattr(sanity, "NOISE_SIGMAS", [5])
        monkeypatch.setattr(sanity, "BLUR_SIGMAS", [0.5])
        monkeypatch.setattr(sanity, "QUANTIZE_COLORS", [64])
        # Use 4 lossy levels: first 2 unique, last 2 byte-identical with #2.
        monkeypatch.setattr(sanity, "LOSSY_LEVELS", [30, 60, 120, 200])
        # Clear fixture bases so they don't trigger a second coalesce call.
        monkeypatch.setattr(sanity, "FIXTURE_MONOTONICITY_BASES", {})

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

        def _fake_compress(
            input_path, output_path, engine, params, *, apply_content_ceiling=True
        ):
            # The lossy arm MUST bypass the content ceiling — otherwise the
            # photographic bases in MONOTONICITY_BASES clamp and the curve
            # degenerates. See test_lossy_arm_bypasses_content_ceiling below.
            assert apply_content_ceiling is False, (
                "run_sanity() lossy arm called compress without "
                "apply_content_ceiling=False"
            )
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
            calls.append((list(paths), {k: list(v) for k, v in per_metric.items()}))
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

        def _fake_compress(
            input_path, output_path, engine, params, *, apply_content_ceiling=True
        ):
            assert apply_content_ceiling is False, (
                "run_sanity() lossy arm called compress without "
                "apply_content_ceiling=False"
            )
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

        monkeypatch.setattr(sanity, "_coalesce_byte_identical_levels", _stub_coalesce)

        # Capture the value-lists passed to _monotonicity_check. The lossy
        # arm's stored entry should appear as [99.0] (the sentinel),
        # confirming run_sanity() reads from the coalesced result.
        captured: dict[str, list[list[float]]] = {"seen": []}
        real_check = sanity._monotonicity_check

        # tol defaults to None so the spy exercises the new range-derived
        # code path inside _monotonicity_check rather than silently pinning
        # the wiring test to the old fixed-tol behaviour.
        def _check_spy(values, direction, tol=None):
            captured["seen"].append(list(values))
            return real_check(values, direction, tol=tol)

        monkeypatch.setattr(sanity, "_monotonicity_check", _check_spy)

        sanity.run_sanity(tmp_path)

        assert sentinel_values in captured["seen"], (
            "Expected the sentinel value-list returned by the stubbed "
            "_coalesce_byte_identical_levels to be passed to "
            "_monotonicity_check via the stored monotonicity entry — "
            "proving run_sanity() uses the coalesced result, not the raw "
            f"per_metric_levels. Got: {captured['seen']}"
        )


class TestAuditLossySweepBypassesContentCeiling:
    """Lock the audit-bypass contract at the call sites, not just through the
    public ``compress()`` API.

    The content-aware lossy ceiling (PR #40 / [[giflab-content-classifier-lossy-ceiling]])
    clamps photographic / data-viz inputs DOWN to a per-class ceiling. The
    monotonicity sweep's whole job is to drive the requested lossy grid
    (LOSSY_LEVELS spans well above those ceilings) and measure animately's
    response — so it MUST pass ``apply_content_ceiling=False`` on EVERY
    ``compress`` call. If a sweep call site forgot the kwarg, photographic
    bases (smooth_gradient, photographic_noise) would clamp to 20/30 and
    levels 60/100/160 would collapse into byte-identical outputs that
    ``_coalesce_byte_identical_levels`` swallows — degenerating the curve to a
    single point and silently misrepresenting animately's true lossy response.

    The earlier ``test_audit_optout_bypasses_ceiling`` only proves the public
    API *honours* the kwarg. These tests prove the audit *uses* it, capturing
    the kwarg as forwarded from each lossy arm.
    """

    def _capturing_compress(self, captured_kwargs: list):
        """A fake compress() that records the apply_content_ceiling kwarg of
        every call and writes unique bytes per call (so coalescing keeps all
        levels and every grid point produces a compress call)."""
        counter = {"n": 0}

        def _fake(
            input_path, output_path, engine, params, *, apply_content_ceiling=True
        ):
            captured_kwargs.append(apply_content_ceiling)
            counter["n"] += 1
            Path(output_path).write_bytes(f"unique-{counter['n']}".encode())
            return output_path

        return _fake

    def test_synthetic_lossy_arm_passes_bypass_kwarg(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """The synthetic-bases (MONOTONICITY_BASES) lossy arm forwards
        ``apply_content_ceiling=False`` on every grid point."""
        gl = _build_fake_gl(lambda: {"ssim": 0.5})
        captured: list[bool] = []
        gl["compress"] = self._capturing_compress(captured)
        monkeypatch.setattr(sanity, "_import_giflab", lambda: gl)

        monkeypatch.setattr(sanity, "MONOTONICITY_BASES", ["base_under_test"])
        monkeypatch.setattr(sanity, "IDENTITY_SAMPLE", ["base_under_test"])
        monkeypatch.setattr(sanity, "NOISE_SIGMAS", [5])
        monkeypatch.setattr(sanity, "BLUR_SIGMAS", [0.5])
        monkeypatch.setattr(sanity, "QUANTIZE_COLORS", [64])
        # Multiple levels spanning above the photographic ceiling (20).
        monkeypatch.setattr(sanity, "LOSSY_LEVELS", [20, 60, 100, 160])
        monkeypatch.setattr(sanity, "FIXTURE_MONOTONICITY_BASES", {})

        sanity.run_sanity(tmp_path)

        assert captured, "lossy arm produced no compress calls — test is inert"
        assert all(v is False for v in captured), (
            "synthetic-bases lossy sweep called compress with "
            f"apply_content_ceiling != False: {captured}. Photographic/"
            "data-viz bases would clamp and the monotonicity curve degenerates."
        )
        # One compress call per lossy level.
        assert len(captured) == 4, (
            f"expected 4 compress calls (one per LOSSY_LEVELS entry); got "
            f"{len(captured)}"
        )

    def test_fixture_lossy_arm_passes_bypass_kwarg(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """The fixture-bases (FIXTURE_MONOTONICITY_BASES) lossy arm forwards
        ``apply_content_ceiling=False`` on every grid point."""
        from PIL import Image

        fixture_dir = tmp_path / "fixtures"
        fixture_dir.mkdir()
        fixture_path = fixture_dir / "guard.gif"
        frames_list = [Image.new("P", (16, 16), color=i) for i in (100, 120)]
        for fl in frames_list:
            fl.putpalette([i % 256 for i in range(768)])
        frames_list[0].save(
            fixture_path,
            save_all=True,
            append_images=frames_list[1:],
            duration=100,
            loop=0,
            optimize=False,
        )

        gl = _build_fake_gl(lambda: {"ssim": 0.5})
        captured: list[bool] = []
        gl["compress"] = self._capturing_compress(captured)
        monkeypatch.setattr(sanity, "_import_giflab", lambda: gl)

        # No synthetic bases — isolate the fixture lossy arm.
        monkeypatch.setattr(sanity, "MONOTONICITY_BASES", [])
        monkeypatch.setattr(sanity, "IDENTITY_SAMPLE", [])
        monkeypatch.setattr(sanity, "NOISE_SIGMAS", [5])
        monkeypatch.setattr(sanity, "BLUR_SIGMAS", [0.5])
        monkeypatch.setattr(sanity, "QUANTIZE_COLORS", [64])
        monkeypatch.setattr(sanity, "LOSSY_LEVELS", [20, 60, 100])
        monkeypatch.setattr(
            sanity, "FIXTURE_MONOTONICITY_BASES", {"guard": "guard.gif"}
        )
        monkeypatch.setattr(sanity, "_FIXTURES_DIR", fixture_dir)

        sanity.run_sanity(tmp_path / "workdir")

        assert captured, "fixture lossy arm produced no compress calls — test is inert"
        assert all(v is False for v in captured), (
            "fixture-bases lossy sweep called compress with "
            f"apply_content_ceiling != False: {captured}."
        )
        assert len(captured) == 3, (
            f"expected 3 compress calls (one per LOSSY_LEVELS entry); got "
            f"{len(captured)}"
        )

    def test_common_compress_animately_forwards_bypass_kwarg(self) -> None:
        """``_common.compress_animately`` — the helper that pilot.py and
        sweep.py route through — forwards ``apply_content_ceiling=False`` to
        the underlying ``compress``."""
        import importlib.util

        common_path = _SCRIPTS_AUDIT / "_common.py"
        spec = importlib.util.spec_from_file_location("audit_common", common_path)
        common = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(common)

        captured: dict[str, object] = {}

        def _fake_compress(
            input_path, output_path, engine, params, *, apply_content_ceiling=True
        ):
            captured["apply_content_ceiling"] = apply_content_ceiling
            captured["engine"] = engine
            captured["params"] = dict(params)
            return output_path

        gl = {"compress": _fake_compress}
        ok, err = common.compress_animately(Path("in.gif"), Path("out.gif"), 100, gl)

        assert ok, f"compress_animately reported failure: {err}"
        assert captured.get("apply_content_ceiling") is False, (
            "compress_animately did not forward apply_content_ceiling=False — "
            f"pilot.py / sweep.py lossy grids would be clamped. Got: {captured}"
        )
        assert captured["params"] == {"lossy_level": 100}


# ---------------------------------------------------------------------------
# Transparency-bearing fixture tests
# ---------------------------------------------------------------------------
#
# These tests guard the alpha-compositing regression class surfaced in PR #8:
# ``extract_gif_frames`` previously called ``Image.convert('RGB')`` directly
# on palette+transparency GIFs.  PIL resolves transparent pixels through the
# file's declared background palette colour — which is UNSTABLE across
# re-encoding because compressors rearrange the palette.  Original and
# compressed copies of the same visible content therefore had transparent
# regions filled with DIFFERENT RGB values, and every pair-comparison metric
# (ssim, ms_ssim, chist, lpips, …) treated that as real content disagreement.
#
# The test class below:
#  1. Loads the ``transparency_bearing_monotonicity.gif`` fixture from
#     ``tests/fixtures/`` (generated by ``make fixtures``).
#  2. Demonstrates the regression path: ``img.convert('RGB')`` fills transparent
#     pixels with the palette colour at the transparency index (dark, non-white),
#     while the correct RGBA-composite path fills them with white (255, 255, 255).
#  3. Verifies that ``sanity.run_sanity()`` picks up FIXTURE_MONOTONICITY_BASES
#     and runs the fixture through the degradation loop when the fixture file
#     exists, and gracefully skips when it does not.
# ---------------------------------------------------------------------------


def _import_fixture_generator():
    """Import scripts/fixtures/generate.py module."""
    import importlib.util

    gen_path = (
        Path(__file__).resolve().parents[2] / "scripts" / "fixtures" / "generate.py"
    )
    spec = importlib.util.spec_from_file_location("generate_fixtures", gen_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestTransparencyBearingGifFixture:
    """Smoke tests for the transparency-bearing monotonicity fixture.

    These tests exercise the FIXTURE specifically — generation, structure, and
    the PR #8 regression demonstration.  They do NOT run the full metrics
    pipeline; they only look at the PIL-level frame extraction behaviour so
    the test completes in milliseconds.
    """

    def test_fixture_generator_creates_transparency_bearing_gif(
        self, tmp_path: Path
    ) -> None:
        """generate_all() must create transparency_bearing_monotonicity.gif."""
        mod = _import_fixture_generator()
        mod.generate_all(output_dir=tmp_path)
        out = tmp_path / "transparency_bearing_monotonicity.gif"
        assert out.exists(), (
            "generate_all() did not create transparency_bearing_monotonicity.gif — "
            "check that create_transparency_bearing_monotonicity_gif is registered "
            "in the generators list inside generate_all()."
        )

    def test_fixture_has_transparency_index(self, tmp_path: Path) -> None:
        """The generated GIF must have a ``transparency`` key in im.info."""
        from PIL import Image

        mod = _import_fixture_generator()
        p = mod.create_transparency_bearing_monotonicity_gif(tmp_path)
        with Image.open(p) as im:
            assert "transparency" in im.info, (
                "transparency_bearing_monotonicity.gif must have a GIF transparency "
                "palette index in im.info — this is the structural property that "
                "triggers the PR #8 bug class."
            )

    def test_fixture_has_transparent_pixels(self, tmp_path: Path) -> None:
        """The transparency index must be used by some pixels in the GIF."""
        import numpy as np
        from PIL import Image

        mod = _import_fixture_generator()
        p = mod.create_transparency_bearing_monotonicity_gif(tmp_path)
        with Image.open(p) as im:
            t_idx = im.info.get("transparency")
            assert t_idx is not None
            arr = np.array(im)
            n_transparent = int((arr == t_idx).sum())
            assert n_transparent > 0, (
                f"Fixture has transparency index {t_idx} but no pixels use it — "
                "the transparent oval was not written into the palette P array."
            )

    def test_fixture_has_four_frames(self, tmp_path: Path) -> None:
        """The fixture must have exactly 4 frames."""
        from PIL import Image

        mod = _import_fixture_generator()
        p = mod.create_transparency_bearing_monotonicity_gif(tmp_path)
        with Image.open(p) as im:
            n = getattr(im, "n_frames", 1)
            assert n == 4, f"Expected 4 frames, got {n}"

    def test_fixture_is_deterministic(self, tmp_path: Path) -> None:
        """Running the generator twice must produce byte-identical output."""
        mod = _import_fixture_generator()
        run1_dir = tmp_path / "run1"
        run2_dir = tmp_path / "run2"
        run1_dir.mkdir()
        run2_dir.mkdir()
        out1 = mod.create_transparency_bearing_monotonicity_gif(run1_dir)
        out2 = mod.create_transparency_bearing_monotonicity_gif(run2_dir)
        assert out1.read_bytes() == out2.read_bytes(), (
            "create_transparency_bearing_monotonicity_gif() produced different "
            "bytes on two consecutive calls — generator must be fully deterministic."
        )

    def test_buggy_path_gives_different_frame_than_correct_path(
        self, tmp_path: Path
    ) -> None:
        """Demonstrates the PR #8 regression class on the saved fixture.

        Mechanism actually observed on the saved fixture (verified empirically
        on Pillow 10.x): Pillow's GIF encoder truncates the saved palette to
        the smallest power of two that covers the in-use *content* indices —
        in this fixture, 128 entries.  Palette index 255 (our chosen
        transparency sentinel) is therefore OUT OF RANGE in the saved file.
        When PIL loads the file and ``convert('RGB')`` tries to resolve
        transparent pixels through that missing palette entry, it falls back
        to ``(0, 0, 0)`` (black).

        The correct RGBA-composite path always yields ``(255, 255, 255)``
        (white) for fully transparent pixels regardless of how the encoder
        compacted the palette.

        So on this fixture:
        - Buggy ``convert('RGB')`` path → transparent pixels = black
        - Correct RGBA composite onto white → transparent pixels = white
        - Per-channel delta ≈ 255 DN — strong enough to collapse any
          pair-comparison metric (ssim, ms_ssim, chist, lpips, …)

        Reverting the compositing fix in PR #8 would cause that 255 DN delta
        to corrupt every metric reading on this fixture, which is exactly the
        regression the sanity monotonicity check is meant to catch.
        """
        import numpy as np
        from PIL import Image

        mod = _import_fixture_generator()
        p = mod.create_transparency_bearing_monotonicity_gif(tmp_path)

        with Image.open(p) as im:
            t_idx = im.info.get("transparency")
            assert t_idx is not None

            # RGBA path: which pixels are transparent?
            rgba = im.convert("RGBA")
            rgba_arr = np.array(rgba)
            alpha_zero_mask = rgba_arr[:, :, 3] == 0
            assert alpha_zero_mask.sum() > 0, (
                "RGBA conversion did not produce any alpha=0 pixels — "
                "the GIF transparency index is not being respected."
            )

            # Correct path: alpha-composite onto white → transparent pixels become white.
            bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
            composited_arr = np.array(Image.alpha_composite(bg, rgba).convert("RGB"))
            transparent_composited = composited_arr[alpha_zero_mask]
            assert (transparent_composited == 255).all(), (
                "RGBA-composite path did not yield (255,255,255) for transparent "
                f"pixels — got values: {np.unique(transparent_composited, axis=0)}"
            )

            # Buggy path: convert('RGB') directly — must NOT yield white on these pixels.
            # This is the load-bearing assertion: if it DID yield white, the buggy
            # path and the correct path would coincide on the fixture, the
            # regression-class would not be demonstrated, and the sanity check
            # would have no signal to flag a reverted compositing fix.
            buggy_arr = np.array(im.convert("RGB"))
            transparent_buggy = buggy_arr[alpha_zero_mask]
            n_white_pixels = int(np.all(transparent_buggy == 255, axis=-1).sum())
            n_total = int(alpha_zero_mask.sum())
            assert n_white_pixels < n_total, (
                "Buggy convert('RGB') path produced white (255,255,255) for "
                f"{n_white_pixels}/{n_total} transparent pixels — same value as "
                "the correct RGBA-composite path.  If buggy and correct paths "
                "coincide, this fixture does not demonstrate the PR #8 "
                "alpha-compositing regression and offers no protection if the "
                "compositing fix is reverted.  Inspect the fixture's saved "
                "palette: it likely now includes white at the transparency "
                "index (the original failure mode depended on Pillow truncating "
                "the saved palette so index 255 was out of range)."
            )

            # And the per-channel delta on transparent pixels must be large
            # (~255 DN in practice) — strong enough to collapse any pair-
            # comparison metric if the compositing fix is reverted.
            mean_delta = float(
                np.abs(
                    transparent_buggy.astype(float)
                    - transparent_composited.astype(float)
                ).mean()
            )
            assert mean_delta > 64.0, (
                f"Buggy-vs-correct per-channel delta on transparent pixels = "
                f"{mean_delta:.1f} DN — too small to corrupt metrics.  PR #8's "
                "regression class is only demonstrated by a large delta "
                "(~255 DN in practice — black-vs-white)."
            )


class TestFixtureMonotonicityBasesIntegration:
    """Verify that run_sanity() picks up and runs FIXTURE_MONOTONICITY_BASES.

    These tests use the same mock-GL infrastructure as
    TestCoalesceIntegratedIntoRunSanity to stay fast (no real metrics, no
    real GIFs from the synthetic set).  They inject a pre-generated fixture
    into a tmp fixtures directory and assert that the fixture's logical name
    appears in the monotonicity results.
    """

    def _pin_constants(self, monkeypatch) -> None:
        monkeypatch.setattr(sanity, "MONOTONICITY_BASES", [])
        monkeypatch.setattr(sanity, "IDENTITY_SAMPLE", [])
        monkeypatch.setattr(sanity, "NOISE_SIGMAS", [5])
        monkeypatch.setattr(sanity, "BLUR_SIGMAS", [0.5])
        monkeypatch.setattr(sanity, "QUANTIZE_COLORS", [64])
        monkeypatch.setattr(sanity, "LOSSY_LEVELS", [])

    def _make_fake_gl(self) -> dict:
        """Minimal gl dict: SyntheticGifGenerator returns no GIFs, metrics always 0.5."""

        class _FakeGen:
            def __init__(self, d: Path) -> None:
                self.synthetic_specs = []

            def generate_gifs(self, *, use_targeted_set: bool = False):
                return []

        class _FakeCfg:
            def __init__(self) -> None:
                self.ENABLE_DEEP_PERCEPTUAL = False
                self.ENABLE_TEMPORAL_ARTIFACTS = False

        return {
            "SyntheticGifGenerator": _FakeGen,
            "calculate_comprehensive_metrics": lambda *a, **kw: {"ssim": 0.5},
            "compress": None,
            "MetricsConfig": _FakeCfg,
        }

    def test_fixture_base_appears_in_monotonicity_results(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """When fixture file exists, its logical name must appear in monotonicity keys."""
        from PIL import Image

        # Create a minimal 2-frame GIF as the fixture
        fixture_dir = tmp_path / "fixtures"
        fixture_dir.mkdir()
        fixture_path = fixture_dir / "transparency_bearing_monotonicity.gif"
        frames_list = [Image.new("P", (16, 16), color=i) for i in (100, 120)]
        frames_list[0].putpalette([i % 256 for i in range(768)])
        frames_list[1].putpalette([i % 256 for i in range(768)])
        frames_list[0].save(
            fixture_path,
            save_all=True,
            append_images=frames_list[1:],
            duration=100,
            loop=0,
            optimize=False,
        )

        self._pin_constants(monkeypatch)
        gl = self._make_fake_gl()
        monkeypatch.setattr(sanity, "_import_giflab", lambda: gl)
        monkeypatch.setattr(
            sanity,
            "FIXTURE_MONOTONICITY_BASES",
            {"transparency_bearing": "transparency_bearing_monotonicity.gif"},
        )
        monkeypatch.setattr(sanity, "_FIXTURES_DIR", fixture_dir)

        results = sanity.run_sanity(tmp_path / "workdir", skip_lossy=True)

        mono_keys = list(results["monotonicity"].keys())
        fixture_keys = [k for k in mono_keys if "transparency_bearing" in k]
        assert len(fixture_keys) > 0, (
            "Expected monotonicity keys containing 'transparency_bearing' but got: "
            f"{mono_keys}. run_sanity() is not processing FIXTURE_MONOTONICITY_BASES."
        )
        # Each of noise, blur, quantize arms should produce an entry
        kinds_found = {k.split("::")[0] for k in fixture_keys}
        assert "noise" in kinds_found, f"noise arm missing; found: {kinds_found}"
        assert "blur" in kinds_found, f"blur arm missing; found: {kinds_found}"
        assert "quantize" in kinds_found, f"quantize arm missing; found: {kinds_found}"

    def test_missing_fixture_skipped_gracefully(
        self, tmp_path: Path, monkeypatch, capsys
    ) -> None:
        """When the fixture file does not exist inside an existing fixtures dir,
        skip without error AND record the skip in ``skipped_fixture_checks``."""
        empty_fixtures_dir = tmp_path / "fixtures_dir_exists_but_empty"
        empty_fixtures_dir.mkdir()

        self._pin_constants(monkeypatch)
        gl = self._make_fake_gl()
        monkeypatch.setattr(sanity, "_import_giflab", lambda: gl)
        monkeypatch.setattr(
            sanity,
            "FIXTURE_MONOTONICITY_BASES",
            {"transparency_bearing": "transparency_bearing_monotonicity.gif"},
        )
        monkeypatch.setattr(sanity, "_FIXTURES_DIR", empty_fixtures_dir)

        # Must not raise
        results = sanity.run_sanity(tmp_path / "workdir", skip_lossy=True)

        mono_keys = list(results["monotonicity"].keys())
        fixture_keys = [k for k in mono_keys if "transparency_bearing" in k]
        assert fixture_keys == [], (
            "Expected no transparency_bearing entries when fixture is missing, "
            f"got: {fixture_keys}"
        )

        captured = capsys.readouterr()
        assert "skip fixture monotonicity" in captured.out, (
            "Expected a 'skip fixture monotonicity' message in stdout when the "
            "fixture file is not found."
        )

        # The skipped fixture must be surfaced in the results — otherwise a
        # silent skip looks identical to "all PASS" to downstream consumers.
        skipped = results.get("skipped_fixture_checks", [])
        assert any(
            entry.get("logical_name") == "transparency_bearing" for entry in skipped
        ), (
            "Expected skipped_fixture_checks to record 'transparency_bearing' "
            f"when the fixture file is absent — got: {skipped}.  Without this "
            "the verdict loop's empty-iteration produces a silent PASS that's "
            "indistinguishable from 'all monotonicity checks passed'."
        )

    def test_missing_fixtures_dir_records_all_skips(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """When tests/fixtures/ does not exist at all, every configured fixture
        base must appear in ``skipped_fixture_checks``.

        This is the silent-PASS guard: on a fresh clone where neither the
        committed fixture nor `make fixtures` has produced anything, the
        sanity check must NOT report all-green for the alpha-compositing
        regression class.  Downstream CI gates can read skipped_fixture_checks
        and fail the build.
        """
        self._pin_constants(monkeypatch)
        gl = self._make_fake_gl()
        monkeypatch.setattr(sanity, "_import_giflab", lambda: gl)
        monkeypatch.setattr(
            sanity,
            "FIXTURE_MONOTONICITY_BASES",
            {
                "transparency_bearing": "transparency_bearing_monotonicity.gif",
                "another_guard": "another_fixture.gif",
            },
        )
        # _FIXTURES_DIR itself does not exist
        monkeypatch.setattr(sanity, "_FIXTURES_DIR", tmp_path / "no_such_dir")

        results = sanity.run_sanity(tmp_path / "workdir", skip_lossy=True)

        skipped = results.get("skipped_fixture_checks", [])
        skipped_names = {entry["logical_name"] for entry in skipped}
        assert skipped_names == {"transparency_bearing", "another_guard"}, (
            "Expected ALL configured fixture bases in skipped_fixture_checks "
            f"when tests/fixtures/ is missing — got: {skipped_names}"
        )

    def test_present_fixture_does_not_record_skip(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """Happy path: when the fixture is present, skipped_fixture_checks must be empty."""
        from PIL import Image

        fixture_dir = tmp_path / "fixtures"
        fixture_dir.mkdir()
        fixture_path = fixture_dir / "transparency_bearing_monotonicity.gif"
        frames_list = [Image.new("P", (16, 16), color=i) for i in (100, 120)]
        for fl in frames_list:
            fl.putpalette([i % 256 for i in range(768)])
        frames_list[0].save(
            fixture_path,
            save_all=True,
            append_images=frames_list[1:],
            duration=100,
            loop=0,
            optimize=False,
        )

        self._pin_constants(monkeypatch)
        gl = self._make_fake_gl()
        monkeypatch.setattr(sanity, "_import_giflab", lambda: gl)
        monkeypatch.setattr(
            sanity,
            "FIXTURE_MONOTONICITY_BASES",
            {"transparency_bearing": "transparency_bearing_monotonicity.gif"},
        )
        monkeypatch.setattr(sanity, "_FIXTURES_DIR", fixture_dir)

        results = sanity.run_sanity(tmp_path / "workdir", skip_lossy=True)

        assert results.get("skipped_fixture_checks", []) == [], (
            "Expected empty skipped_fixture_checks when the fixture is present, "
            f"got: {results.get('skipped_fixture_checks')}"
        )

    def test_config_output_includes_fixture_bases(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """The 'config' key in run_sanity() output must list fixture_monotonicity_bases."""
        self._pin_constants(monkeypatch)
        gl = self._make_fake_gl()
        monkeypatch.setattr(sanity, "_import_giflab", lambda: gl)
        monkeypatch.setattr(
            sanity,
            "FIXTURE_MONOTONICITY_BASES",
            {"transparency_bearing": "transparency_bearing_monotonicity.gif"},
        )
        # _FIXTURES_DIR doesn't exist → fixture loop skips gracefully
        monkeypatch.setattr(sanity, "_FIXTURES_DIR", tmp_path / "nonexistent")

        results = sanity.run_sanity(tmp_path / "workdir", skip_lossy=True)

        cfg = results.get("config", {})
        assert (
            "fixture_monotonicity_bases" in cfg
        ), "run_sanity() output['config'] must contain 'fixture_monotonicity_bases' key."
        assert "transparency_bearing" in cfg["fixture_monotonicity_bases"], (
            "Expected 'transparency_bearing' in config.fixture_monotonicity_bases; "
            f"got: {cfg['fixture_monotonicity_bases']}"
        )


class TestDirectionNaNGuard:
    """NaN-over-sentinels contract for `_direction`.

    See ~/repos/obsidian/Work/Tasks/giflab-audit-disagreement-tie-aware-ranks.md
    (validate-edge findings §3): if any identity/pathological aggregate is NaN
    (possible post-#54 for genuinely edgeless identity content), `_direction`
    must return "flat" — never silently fall through to "lower_better" via
    `NaN < 0 == False`.
    """

    def test_nan_identity_returns_flat(self) -> None:
        assert sanity._direction(float("nan"), 1.0) == "flat"

    def test_nan_pathological_returns_flat(self) -> None:
        assert sanity._direction(1.0, float("nan")) == "flat"

    def test_both_nan_returns_flat(self) -> None:
        assert sanity._direction(float("nan"), float("nan")) == "flat"

    def test_finite_directions_unchanged(self) -> None:
        assert sanity._direction(1.0, 0.0) == "higher_better"
        assert sanity._direction(0.0, 1.0) == "lower_better"
        assert sanity._direction(0.5, 0.5) == "flat"


class TestNanAggregate:
    """`_nan_aggregate` backs the identity mean/std (run_sanity verdict loop):
    NaN identity samples must be excluded instead of poisoning the aggregate."""

    def test_ignores_nan_values(self) -> None:
        import numpy as np

        assert sanity._nan_aggregate([1.0, float("nan"), 3.0], np.mean) == 2.0

    def test_all_nan_returns_nan_without_warning(self) -> None:
        import warnings

        import numpy as np

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            out = sanity._nan_aggregate([float("nan"), float("nan")], np.mean)
        assert math.isnan(out)

    def test_empty_returns_nan(self) -> None:
        import numpy as np

        assert math.isnan(sanity._nan_aggregate([], np.mean))

    def test_std_aggregate(self) -> None:
        import numpy as np

        assert sanity._nan_aggregate([1.0, float("nan"), 1.0], np.std) == 0.0


class TestDecideVerdictNaNIsInconclusive:
    def test_flat_with_nan_identity_is_inconclusive_not_pass(self) -> None:
        # A metric whose identity aggregate is NaN could not be measured on
        # the reference pairs — that must surface as INCONCLUSIVE, never as a
        # quiet PASS.
        verdict, note, _ = sanity._decide_verdict(
            "ssim", "flat", float("nan"), 0.5, [], "solid"
        )
        assert verdict == "INCONCLUSIVE"
        assert "NaN" in note

    def test_flat_with_nan_pathological_is_inconclusive(self) -> None:
        verdict, _, _ = sanity._decide_verdict(
            "ssim", "flat", 1.0, float("nan"), [], "solid"
        )
        assert verdict == "INCONCLUSIVE"
