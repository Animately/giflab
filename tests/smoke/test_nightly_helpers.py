"""Unit tests for the shared nightly-tier environment helpers.

``tests/nightly/helpers.py`` is stdlib-only (os, math) by design so these
pure-logic tests can live in the smoke layer without violating its
no-heavy-imports rule. Cross-package test imports are established practice
(see tests/functional/test_color_keep.py importing from tests.integration).
"""

import math

import pytest

from tests.nightly.helpers import (
    analyze_memory_samples,
    effective_cpu_count,
    is_ci,
    nan_aware_close,
    performance_threshold_multiplier,
)

_CI_VARS = [
    "CI",
    "CONTINUOUS_INTEGRATION",
    "GITHUB_ACTIONS",
    "TRAVIS",
    "JENKINS_URL",
    "BUILDKITE",
    "CIRCLECI",
]


def _clear_ci_env(monkeypatch):
    for var in _CI_VARS:
        monkeypatch.delenv(var, raising=False)


@pytest.mark.fast
class TestIsCi:
    def test_false_when_no_ci_vars(self, monkeypatch):
        _clear_ci_env(monkeypatch)
        assert is_ci() is False

    def test_true_when_github_actions_set(self, monkeypatch):
        _clear_ci_env(monkeypatch)
        monkeypatch.setenv("GITHUB_ACTIONS", "true")
        assert is_ci() is True

    def test_empty_value_does_not_count(self, monkeypatch):
        _clear_ci_env(monkeypatch)
        monkeypatch.setenv("CI", "")
        assert is_ci() is False


@pytest.mark.fast
class TestPerformanceThresholdMultiplier:
    def test_local_default(self, monkeypatch):
        _clear_ci_env(monkeypatch)
        assert performance_threshold_multiplier() == 1.0

    def test_ci_default(self, monkeypatch):
        _clear_ci_env(monkeypatch)
        monkeypatch.setenv("CI", "true")
        assert performance_threshold_multiplier() == 2.0

    def test_caller_calibrated_ci_factor(self, monkeypatch):
        # test_vectorization_performance.py keeps its calibrated 3.0x factor
        _clear_ci_env(monkeypatch)
        monkeypatch.setenv("CI", "true")
        assert performance_threshold_multiplier(ci=3.0) == 3.0

    def test_caller_calibrated_local_factor(self, monkeypatch):
        _clear_ci_env(monkeypatch)
        assert performance_threshold_multiplier(local=1.5, ci=3.0) == 1.5


@pytest.mark.fast
class TestEffectiveCpuCount:
    def test_returns_positive_int(self):
        count = effective_cpu_count()
        assert isinstance(count, int)
        assert count >= 1


@pytest.mark.fast
class TestNanAwareClose:
    def test_both_nan_is_close(self):
        # Both sides honestly failed the same way (e.g. ssimulacra2
        # binary unavailable) — that IS equivalence.
        assert nan_aware_close(float("nan"), float("nan")) is True

    def test_one_sided_nan_is_not_close(self):
        # One side fabricating a number where the other honestly NaNs
        # is a real divergence and must fail loudly.
        assert nan_aware_close(float("nan"), 0.5) is False
        assert nan_aware_close(0.5, float("nan")) is False
        assert nan_aware_close(float("nan"), 0.0) is False

    def test_within_tolerance(self):
        assert nan_aware_close(1.0, 1.0 + 1e-12) is True
        assert nan_aware_close(100.0, 100.0 * (1 + 1e-7), rel_tol=1e-6) is True

    def test_outside_tolerance(self):
        assert nan_aware_close(1.0, 1.001) is False

    def test_abs_tol_near_zero(self):
        assert nan_aware_close(0.0, 5e-7, rel_tol=0.0, abs_tol=1e-6) is True
        assert nan_aware_close(0.0, 5e-6, rel_tol=0.0, abs_tol=1e-6) is False

    def test_equal_infinities_are_close(self):
        assert nan_aware_close(float("inf"), float("inf")) is True
        assert nan_aware_close(float("-inf"), float("-inf")) is True

    def test_opposite_or_mixed_infinities_are_not_close(self):
        assert nan_aware_close(float("inf"), float("-inf")) is False
        assert nan_aware_close(float("inf"), 1e308) is False

    def test_bool_inputs_are_valid_numbers(self):
        # bool is a subclass of int and passes isinstance(val, int | float)
        # guards in the metric-comparison loops; math.isnan(True) is valid.
        assert nan_aware_close(True, 1.0) is True
        assert nan_aware_close(False, 0.0) is True
        assert math.isnan(True) is False  # documents the property relied on


@pytest.mark.fast
class TestAnalyzeMemorySamples:
    def test_plateau_after_ramp_is_not_a_leak(self):
        # One-time ramp (LPIPS model load) then a flat plateau with small
        # wobble: the old detector flagged this as monotonic leak-shaped
        # growth because the ramp sat inside its regression window.
        samples = [(0, 100.0), (10, 600.0)] + [
            (i, 600.0 + (i % 3)) for i in range(20, 101, 10)
        ]
        result = analyze_memory_samples(samples, warmup_samples=2)
        assert result["potential_leak"] is False
        assert result["growth_rate_mb_per_iteration"] < 0.5

    def test_sustained_growth_is_a_leak(self):
        # 1 MB/iteration sustained growth sampled every 10 iterations.
        samples = [(i, 100.0 + i * 1.0) for i in range(0, 101, 10)]
        result = analyze_memory_samples(samples, warmup_samples=2)
        assert result["potential_leak"] is True
        assert result["growth_rate_mb_per_iteration"] == pytest.approx(1.0)

    def test_rate_is_per_iteration_not_per_sample(self):
        # The same 1 MB/iteration leak sampled at different cadences must
        # report the same per-iteration rate — the units bug this helper
        # exists to fix (slope over sample positions was 10x off for
        # every-10-iterations sampling).
        every_10 = [(i, 100.0 + i) for i in range(0, 101, 10)]
        every_1 = [(i, 100.0 + i) for i in range(0, 101)]
        rate_10 = analyze_memory_samples(every_10)["growth_rate_mb_per_iteration"]
        rate_1 = analyze_memory_samples(every_1)["growth_rate_mb_per_iteration"]
        assert rate_10 == pytest.approx(rate_1)
        assert rate_10 == pytest.approx(1.0)

    def test_too_few_samples_reports_no_leak(self):
        cases = [
            [],
            [(0, 100.0)],
            [(0, 100.0), (1, 200.0)],
            [(0, 100.0), (1, 200.0), (2, 300.0)],  # only 1 post-warm-up point
        ]
        for samples in cases:
            result = analyze_memory_samples(samples, warmup_samples=2)
            assert result["growth_rate_mb_per_iteration"] == 0.0
            assert result["potential_leak"] is False

    def test_memory_drop_breaks_monotonic_growth(self):
        # A >10% drop inside the post-warm-up window means GC reclaimed
        # memory — not a monotonic leak.
        samples = [
            (0, 100.0),
            (10, 500.0),
            (20, 800.0),
            (30, 850.0),
            (40, 600.0),
            (50, 900.0),
        ]
        result = analyze_memory_samples(samples, warmup_samples=2)
        assert result["is_monotonic_growth"] is False
        assert result["potential_leak"] is False

    def test_warmup_exclusion_does_not_hide_sustained_leak(self):
        # A leak that starts at iteration zero still shows in the
        # post-warm-up window (90 of 100 iterations remain).
        samples = [(i, 100.0 + 2.0 * i) for i in range(0, 101, 10)]
        result = analyze_memory_samples(samples, warmup_samples=2)
        assert result["potential_leak"] is True
        assert result["growth_rate_mb_per_iteration"] == pytest.approx(2.0)

    def test_identical_iteration_indices_do_not_divide_by_zero(self):
        samples = [(5, 100.0), (5, 110.0), (5, 120.0), (5, 130.0)]
        result = analyze_memory_samples(samples, warmup_samples=2)
        assert result["growth_rate_mb_per_iteration"] == 0.0
        assert result["potential_leak"] is False
