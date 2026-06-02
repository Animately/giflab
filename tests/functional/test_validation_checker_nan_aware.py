"""Tests for NaN-aware guards in ValidationChecker.

Bug reproduced: when all SSIMULACRA2 (or deep-perceptual / temporal-artifact)
scores are NaN, ``any([nan, nan, nan])`` evaluates to True in Python because NaN
is truthy.  The early-return guard therefore does NOT trigger, control falls
through to the per-score comparisons, every ``is not None and score < threshold``
evaluates False (NaN comparisons are always False in Python), and NO warning or
issue gets appended.  The result is a silent PASS for a measurement that failed
end-to-end.

This test suite exercises the exact failure path before the production fix is
applied (so all tests must fail red-first) and the expected corrected
behaviour after the fix.

Layer: functional  — pure logic, GifMetadata constructed inline, no real I/O.
"""

import math
from unittest.mock import patch

import pytest
from giflab.meta import GifMetadata
from giflab.optimization_validation.validation_checker import ValidationChecker
from giflab.optimization_validation.data_structures import ValidationStatus


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

NAN = float("nan")


def _make_metadata(
    *,
    orig_frames: int = 10,
    orig_fps: float = 10.0,
    orig_kilobytes: float = 512.0,
) -> GifMetadata:
    """Construct minimal GifMetadata for validation tests."""
    return GifMetadata(
        gif_sha="deadbeef" * 8,
        orig_filename="test.gif",
        orig_kilobytes=orig_kilobytes,
        orig_width=64,
        orig_height=64,
        orig_frames=orig_frames,
        orig_fps=orig_fps,
        orig_n_colors=256,
        entropy=5.0,
    )


def _make_checker() -> ValidationChecker:
    return ValidationChecker(None)  # default config


def _base_metrics() -> dict:
    """Baseline compression_metrics with all optional sections absent."""
    return {
        "composite_quality": 0.75,
        "efficiency": 0.6,
        "compression_ratio": 2.0,
        "compressed_frame_count": 10,
        "orig_fps": 10.0,
        "kilobytes": 256.0,
    }


# ---------------------------------------------------------------------------
# _is_missing helper (tested directly once implemented)
# ---------------------------------------------------------------------------

class TestIsMissingHelper:
    """The fix must expose (or inline) a consistent _is_missing predicate."""

    def test_none_is_missing(self):
        """None values must be considered missing."""
        from giflab.optimization_validation.validation_checker import _is_missing  # noqa: F401
        assert _is_missing(None) is True

    def test_nan_is_missing(self):
        """NaN float values must be considered missing."""
        from giflab.optimization_validation.validation_checker import _is_missing  # noqa: F401
        assert _is_missing(float("nan")) is True

    def test_valid_float_is_not_missing(self):
        """Real floats (including 0.0 and negative) must NOT be missing."""
        from giflab.optimization_validation.validation_checker import _is_missing  # noqa: F401
        assert _is_missing(0.0) is False
        assert _is_missing(0.5) is False
        assert _is_missing(-1.0) is False
        assert _is_missing(1.0) is False

    def test_zero_float_is_not_missing(self):
        """0.0 is a valid measurement — must not be treated as missing."""
        from giflab.optimization_validation.validation_checker import _is_missing  # noqa: F401
        assert _is_missing(0.0) is False


# ---------------------------------------------------------------------------
# Core bug: SSIMULACRA2 all-NaN silent PASS
# ---------------------------------------------------------------------------

class TestSsimulacra2AllNanSilentPass:
    """
    When triggered=1.0 but all three scores are NaN (the all-frames-failed
    path introduced by PR #10), the validator must NOT silently PASS.

    Prior to the fix:
      any([nan, nan, nan])  → True   (NaN is truthy)
      guard check: not True or False → False
      falls through; each `is not None and score < threshold` → False
      no issues / warnings appended → silent PASS

    After the fix, an all-NaN score set with triggered=1.0 must produce a
    warning (ssimulacra2_unavailable category) rather than passing silently.
    """

    def _run(self, scores: dict) -> object:
        checker = _make_checker()
        metrics = {**_base_metrics(), **scores}
        return checker.validate_compression_result(
            original_metadata=_make_metadata(),
            compression_metrics=metrics,
            pipeline_id="pipe_test",
            gif_name="test.gif",
        )

    def test_all_nan_scores_with_triggered_one_is_not_silent_pass(self):
        """All-NaN SSIMULACRA2 scores with triggered=1.0 must NOT produce PASS."""
        result = self._run(
            {
                "ssimulacra2_mean": NAN,
                "ssimulacra2_p95": NAN,
                "ssimulacra2_min": NAN,
                "ssimulacra2_triggered": 1.0,
                # composite_quality above 0.7 so the borderline branch is inactive
                "composite_quality": 0.75,
            }
        )
        # Silent PASS is the bug; after the fix this must produce at least a warning
        all_categories = [w.category for w in result.warnings] + [
            i.category for i in result.issues
        ]
        assert "ssimulacra2_unavailable" in all_categories, (
            f"Expected ssimulacra2_unavailable warning; got status={result.status} "
            f"warnings={[w.category for w in result.warnings]} "
            f"issues={[i.category for i in result.issues]}"
        )

    def test_all_nan_scores_with_triggered_zero_skips_gracefully(self):
        """When triggered=0.0 the section is explicitly disabled — no spurious warning."""
        result = self._run(
            {
                "ssimulacra2_mean": NAN,
                "ssimulacra2_p95": NAN,
                "ssimulacra2_min": NAN,
                "ssimulacra2_triggered": 0.0,
                "composite_quality": 0.75,
            }
        )
        # triggered=0.0 → disabled; no ssimulacra2_unavailable warning expected
        warning_categories = [w.category for w in result.warnings]
        assert "ssimulacra2_unavailable" not in warning_categories, (
            f"ssimulacra2_unavailable should not appear when triggered=0.0, "
            f"got warnings={warning_categories}"
        )

    def test_all_none_scores_with_triggered_one_is_not_silent_pass(self):
        """
        Same bug shape but with Python None rather than NaN.
        ``any([None, None, None])`` is also False (None is falsy) — this path
        was accidentally correct before, but must stay correct after the fix.
        """
        result = self._run(
            {
                "ssimulacra2_mean": None,
                "ssimulacra2_p95": None,
                "ssimulacra2_min": None,
                "ssimulacra2_triggered": 1.0,
                "composite_quality": 0.75,
            }
        )
        # None scores + triggered=1.0 → ssimulacra2_unavailable warning
        all_categories = [w.category for w in result.warnings] + [
            i.category for i in result.issues
        ]
        assert "ssimulacra2_unavailable" in all_categories, (
            f"Expected ssimulacra2_unavailable; got status={result.status}"
        )

    def test_mixed_nan_none_with_triggered_one_is_not_silent_pass(self):
        """Mixed NaN/None in scores is also all-missing and must not silent PASS."""
        result = self._run(
            {
                "ssimulacra2_mean": NAN,
                "ssimulacra2_p95": None,
                "ssimulacra2_min": NAN,
                "ssimulacra2_triggered": 1.0,
                "composite_quality": 0.75,
            }
        )
        all_categories = [w.category for w in result.warnings] + [
            i.category for i in result.issues
        ]
        assert "ssimulacra2_unavailable" in all_categories

    def test_valid_scores_still_pass_when_good(self):
        """Positive control: valid high-quality scores must still produce PASS."""
        result = self._run(
            {
                "ssimulacra2_mean": 0.85,
                "ssimulacra2_p95": 0.80,
                "ssimulacra2_min": 0.70,
                "ssimulacra2_triggered": 1.0,
                "composite_quality": 0.80,
            }
        )
        # Good scores → no ssimulacra2 issues
        ssim2_categories = [i.category for i in result.issues if "ssimulacra2" in i.category]
        assert ssim2_categories == [], (
            f"Unexpected SSIMULACRA2 issues for valid scores: {ssim2_categories}"
        )

    def test_partial_nan_still_flags_available_bad_scores(self):
        """
        If only some scores are NaN but at least one is valid and poor,
        the valid poor score must still trigger an issue.
        """
        result = self._run(
            {
                "ssimulacra2_mean": 0.1,   # valid and poor → should flag
                "ssimulacra2_p95": NAN,    # missing
                "ssimulacra2_min": NAN,    # missing
                "ssimulacra2_triggered": 1.0,
                "composite_quality": 0.75,
            }
        )
        issue_categories = [i.category for i in result.issues]
        assert "ssimulacra2_poor_quality" in issue_categories, (
            f"Expected ssimulacra2_poor_quality issue; got issues={issue_categories}"
        )


# ---------------------------------------------------------------------------
# Deep-perceptual guard: same any([...]) pattern
# ---------------------------------------------------------------------------

class TestDeepPerceptualAllNanSilentPass:
    """
    ``_validate_deep_perceptual_metrics`` uses ``not any([lpips_quality_mean,
    lpips_quality_p95, lpips_quality_max])`` — identical NaN-truthy bug.

    When all three LPIPS-quality values are NaN AND deep_perceptual_used=True,
    the guard evaluates to ``not any([nan, nan, nan]) or not True``
    → ``not True or False`` → ``False``, control falls through, no checks fire.
    """

    def _run(self, scores: dict) -> object:
        checker = _make_checker()
        metrics = {**_base_metrics(), **scores}
        return checker.validate_compression_result(
            original_metadata=_make_metadata(),
            compression_metrics=metrics,
            pipeline_id="pipe_test",
            gif_name="test.gif",
        )

    def test_all_nan_lpips_with_device_not_fallback_should_warn_or_issue(self):
        """
        All-NaN LPIPS-quality values with a real device signals measurement
        failure; the validator should not silently pass when the device is not
        'fallback' (i.e., deep_perceptual_used=True).
        """
        result = self._run(
            {
                "lpips_quality_mean": NAN,
                "lpips_quality_p95": NAN,
                "lpips_quality_max": NAN,
                "deep_perceptual_device": "cpu",   # not 'fallback' → used=True
                "composite_quality": 0.75,
            }
        )
        # Should produce deep_perceptual_unavailable OR some lpips warning
        all_categories = [w.category for w in result.warnings] + [
            i.category for i in result.issues
        ]
        perceptual_categories = [c for c in all_categories if "perceptual" in c or "lpips" in c]
        assert perceptual_categories, (
            f"Expected a perceptual/lpips warning when all scores NaN + device='cpu'; "
            f"got status={result.status}, warnings={[w.category for w in result.warnings]}, "
            f"issues={[i.category for i in result.issues]}"
        )

    def test_valid_lpips_scores_still_flag_poor_mean(self):
        """Positive control: valid high LPIPS mean must still flag the issue."""
        result = self._run(
            {
                "lpips_quality_mean": 0.45,   # above lpips_quality_threshold (0.3)
                "lpips_quality_p95": 0.4,
                "lpips_quality_max": 0.5,
                "deep_perceptual_device": "cpu",
                "composite_quality": 0.75,
            }
        )
        issue_categories = [i.category for i in result.issues]
        assert "perceptual_quality_degradation" in issue_categories, (
            f"Expected perceptual_quality_degradation; got {issue_categories}"
        )


# ---------------------------------------------------------------------------
# Temporal-artifact guard: same any([...]) NaN-truthy bug
# ---------------------------------------------------------------------------

class TestTemporalArtifactAllNanSilentPass:
    """
    ``_validate_temporal_artifacts`` uses ``not any([flicker_excess,
    flat_flicker_ratio, temporal_pumping, lpips_t_mean])`` — same pattern.

    When all four are NaN the ``any()`` returns True (truthy NaN), guard fires,
    early-return happens — but actually this path leads to a *warning* which is
    the CORRECT behaviour (data unavailable).  However if only some are NaN and
    the rest are None the guard may accidentally return without checking the
    non-None values.  The test below pins the expected behaviour.
    """

    def _run(self, scores: dict) -> object:
        checker = _make_checker()
        metrics = {**_base_metrics(), **scores}
        return checker.validate_compression_result(
            original_metadata=_make_metadata(),
            compression_metrics=metrics,
            pipeline_id="pipe_test",
            gif_name="test.gif",
        )

    def test_all_nan_temporal_metrics_produces_unavailable_warning(self):
        """All-NaN temporal metrics must produce the 'unavailable' warning, not silently pass."""
        result = self._run(
            {
                "flicker_excess": NAN,
                "flat_flicker_ratio": NAN,
                "temporal_pumping_score": NAN,
                "lpips_t_mean": NAN,
            }
        )
        warning_categories = [w.category for w in result.warnings]
        # NaN is truthy → any([nan,...]) is True → NOT True = False → guard fails to fire
        # Fix must treat NaN as absent so the guard DOES fire and warns
        assert "temporal_artifacts" in warning_categories, (
            f"Expected temporal_artifacts unavailable warning; "
            f"got warnings={warning_categories}"
        )

    def test_all_none_temporal_metrics_produces_unavailable_warning(self):
        """All-None temporal metrics (the original pre-NaN path) also must warn."""
        result = self._run(
            {
                "flicker_excess": None,
                "flat_flicker_ratio": None,
                "temporal_pumping_score": None,
                "lpips_t_mean": None,
            }
        )
        warning_categories = [w.category for w in result.warnings]
        assert "temporal_artifacts" in warning_categories

    def test_valid_high_flicker_excess_flags_issue(self):
        """Positive control: a real high flicker_excess must still produce an issue."""
        result = self._run(
            {
                "flicker_excess": 0.10,    # well above default threshold 0.02
                "flat_flicker_ratio": None,
                "temporal_pumping_score": None,
                "lpips_t_mean": None,
            }
        )
        issue_categories = [i.category for i in result.issues]
        assert "flicker_excess" in issue_categories, (
            f"Expected flicker_excess issue; got {issue_categories}"
        )


# ---------------------------------------------------------------------------
# Frame-drop alignment warning ([[giflab-alignment-warning-threshold]])
# ---------------------------------------------------------------------------


class TestAlignmentAccuracyWarning:
    """``_validate_alignment_accuracy`` appends a soft ``alignment_uncertain``
    warning when a REAL alignment_accuracy lands below the threshold, and is
    NaN/None/absent-honest (the missing-data sentinel must NOT fire it).

    Mirrors _validate_temporal_consistency: a WARNING, never an ERROR — the
    overall verdict is still produced.
    """

    def _run(self, scores: dict) -> object:
        checker = _make_checker()
        metrics = {**_base_metrics(), **scores}
        return checker.validate_compression_result(
            original_metadata=_make_metadata(),
            compression_metrics=metrics,
            pipeline_id="pipe_test",
            gif_name="test.gif",
        )

    def test_sub_threshold_alignment_warns_not_errors(self):
        """A real alignment below threshold produces an alignment_uncertain
        WARNING (status WARNING), not an ERROR."""
        result = self._run({"alignment_accuracy": 0.847})
        warning_categories = [w.category for w in result.warnings]
        issue_categories = [i.category for i in result.issues]
        assert "alignment_uncertain" in warning_categories, (
            f"Expected alignment_uncertain warning; got warnings={warning_categories} "
            f"issues={issue_categories}"
        )
        # Soft warning only — must not be an issue / error.
        assert "alignment_uncertain" not in issue_categories
        assert result.status == ValidationStatus.WARNING

    def test_at_or_above_threshold_no_warning(self):
        """alignment_accuracy >= threshold must NOT produce the warning."""
        result = self._run({"alignment_accuracy": 1.0})
        warning_categories = [w.category for w in result.warnings]
        assert "alignment_uncertain" not in warning_categories

    def test_nan_alignment_no_warning(self):
        """NaN alignment (failure sentinel via NaN) must NOT fire the rule."""
        result = self._run({"alignment_accuracy": NAN})
        warning_categories = [w.category for w in result.warnings]
        assert "alignment_uncertain" not in warning_categories

    def test_none_alignment_no_warning(self):
        """None alignment (never computed) must NOT fire the rule."""
        result = self._run({"alignment_accuracy": None})
        warning_categories = [w.category for w in result.warnings]
        assert "alignment_uncertain" not in warning_categories

    def test_absent_alignment_no_warning(self):
        """Absent alignment key (most common) must NOT fire the rule."""
        result = self._run({})
        warning_categories = [w.category for w in result.warnings]
        assert "alignment_uncertain" not in warning_categories
