"""Tests for the single-stream metric clarification work (Wave 3 audit-fix).

Covers two concerns surfaced by docs/metrics-audit/2026-05-22/report.md:

1. ``composite_quality`` previously absorbed the bare ``temporal_consistency``
   key, which is set to the post-compression value of a metric computed *only*
   on the compressed stream. A perfectly static black compressed output scores
   ``temporal_consistency_post = 1.0`` regardless of what the original looked
   like, so it was lifting composite_quality by ``ENHANCED_TEMPORAL_WEIGHT *
   1.0 = 0.10``. The fix switches the temporal contribution to
   ``temporal_consistency_delta = |post - pre|`` (a true pair signal). The
   legacy behaviour stays available via the ``USE_TEMPORAL_DELTA_FOR_COMPOSITE``
   config flag.

2. Several metric output keys describe a property of the compressed stream
   alone but are named as if they were pair-comparison quality signals. We
   now also emit ``_compressed`` / ``_original`` suffixed aliases so callers
   can opt into the unambiguous names. Legacy bare keys are retained for one
   release cycle.
"""

from __future__ import annotations

import numpy as np

from giflab.config import MetricsConfig
from giflab.enhanced_metrics import (
    calculate_composite_quality,
    process_metrics_with_enhanced_quality,
)
from giflab.metrics import calculate_comprehensive_metrics_from_frames


def _solid_frames(value: int, n: int = 4, size: tuple[int, int] = (32, 32)) -> list[np.ndarray]:
    """Build *n* identical solid-colour frames of the given grey level."""
    h, w = size
    frame = np.full((h, w, 3), value, dtype=np.uint8)
    return [frame.copy() for _ in range(n)]


def _gradient_frames(n: int = 4, size: tuple[int, int] = (32, 32)) -> list[np.ndarray]:
    """Build *n* identical smooth-gradient frames (varies enough to give SSIM ~0)."""
    h, w = size
    grad = np.tile(np.linspace(0, 255, w, dtype=np.uint8), (h, 1))
    frame = np.stack([grad, grad, grad], axis=-1)
    return [frame.copy() for _ in range(n)]


class TestCompositeIgnoresSingleStreamTemporalConsistency:
    """Composite quality must not be lifted by single-stream temporal consistency."""

    def test_animated_vs_static_composite_penalises_high_temporal_delta(self) -> None:
        """Audit-fix regression: when original is animated but compressed is
        static, ``temporal_consistency_delta`` is large (animation destroyed
        by compression). The new composite contribution must DROP for this
        case, not stay flat or rise.

        Compare against the previously-passing case where original and
        compressed are both static (delta = 0). Under the legacy formula
        both cases would have produced the same temporal contribution
        (post = 1.0 for both) — the static-output bug. After the fix the
        animated→static case scores lower because delta is high.
        """
        config = MetricsConfig(USE_TEMPORAL_DELTA_FOR_COMPOSITE=True)

        # Frames where the original has INCONSISTENT inter-frame differences
        # (forces temporal_consistency well below 1.0), but the compressed
        # output collapses to a single static colour
        # (temporal_consistency = 1.0). ``calculate_temporal_consistency``
        # rewards uniform inter-frame diffs and penalises variable ones, so
        # a jagged value sequence like (0, 0, 250, 0) produces ~0.37.
        original_animated = [
            np.full((32, 32, 3), v, dtype=np.uint8) for v in (0, 0, 250, 0)
        ]
        compressed_static = _solid_frames(128, n=4)

        metrics_animated_to_static = calculate_comprehensive_metrics_from_frames(
            original_animated, compressed_static, config=config
        )

        # Same colour both sides → no animation, no delta.
        metrics_static_to_static = calculate_comprehensive_metrics_from_frames(
            _solid_frames(128), _solid_frames(128), config=config
        )

        # Sanity: deltas are as expected.
        assert (
            metrics_animated_to_static["temporal_consistency_delta"]
            > metrics_static_to_static["temporal_consistency_delta"]
        ), "Sanity: animated→static should have higher temporal_consistency_delta"

        # Under the audit-fix formula, the animated→static composite must be
        # lower because the high delta now penalises the temporal term.
        assert (
            metrics_animated_to_static["composite_quality"]
            < metrics_static_to_static["composite_quality"]
        ), (
            f"Composite should penalise destroyed animation: "
            f"animated→static={metrics_animated_to_static['composite_quality']} "
            f"vs static→static={metrics_static_to_static['composite_quality']}"
        )

    def test_composite_uses_temporal_delta_not_post(self) -> None:
        """When USE_TEMPORAL_DELTA_FOR_COMPOSITE is True, the temporal
        contribution is computed from the |pre - post| delta, not from the
        post value alone."""
        config = MetricsConfig(USE_TEMPORAL_DELTA_FOR_COMPOSITE=True)

        # Both have the same temporal_consistency_post=0.8 but differ in
        # temporal_consistency_pre. Under the legacy formula they would
        # produce the same composite contribution; under the delta formula
        # they differ.
        metrics_a = {
            "ssim_mean": 0.5,
            "temporal_consistency": 0.8,
            "temporal_consistency_pre": 0.8,
            "temporal_consistency_post": 0.8,
            "temporal_consistency_delta": 0.0,
        }
        metrics_b = {
            "ssim_mean": 0.5,
            "temporal_consistency": 0.8,
            "temporal_consistency_pre": 0.2,
            "temporal_consistency_post": 0.8,
            "temporal_consistency_delta": 0.6,
        }

        composite_a = calculate_composite_quality(metrics_a, config)
        composite_b = calculate_composite_quality(metrics_b, config)

        # B has a large delta (compression changed temporal behaviour);
        # composite_b should be lower than composite_a.
        assert composite_b < composite_a, (
            f"composite with high temporal delta ({composite_b}) should be "
            f"lower than composite with zero delta ({composite_a})"
        )

    def test_legacy_temporal_path_still_available(self) -> None:
        """USE_TEMPORAL_DELTA_FOR_COMPOSITE=False restores legacy
        composite behaviour for backward compatibility."""
        legacy_config = MetricsConfig(USE_TEMPORAL_DELTA_FOR_COMPOSITE=False)

        metrics_high_post = {
            "ssim_mean": 0.5,
            "temporal_consistency": 0.95,
            "temporal_consistency_pre": 0.10,
            "temporal_consistency_post": 0.95,
            "temporal_consistency_delta": 0.85,
        }
        metrics_low_post = {
            "ssim_mean": 0.5,
            "temporal_consistency": 0.10,
            "temporal_consistency_pre": 0.95,
            "temporal_consistency_post": 0.10,
            "temporal_consistency_delta": 0.85,
        }

        c_high = calculate_composite_quality(metrics_high_post, legacy_config)
        c_low = calculate_composite_quality(metrics_low_post, legacy_config)

        # Under legacy formula, the higher post value gives the higher
        # composite (the bug: ignores pre/delta).
        assert c_high > c_low, (
            f"Legacy formula should reward higher temporal_consistency_post: "
            f"high={c_high} vs low={c_low}"
        )


class TestSingleStreamMetricAliases:
    """Renamed _compressed / _original alias keys must be emitted alongside
    the legacy bare keys."""

    def _run(self) -> dict[str, float | str]:
        config = MetricsConfig()
        original = _gradient_frames(n=4)
        compressed = _solid_frames(128, n=4)
        return calculate_comprehensive_metrics_from_frames(
            original, compressed, config=config
        )

    def test_temporal_consistency_aliases_present(self) -> None:
        result = self._run()
        assert "temporal_consistency_compressed" in result
        assert "temporal_consistency_original" in result
        # Aliases mirror the legacy keys.
        assert (
            result["temporal_consistency_compressed"]
            == result["temporal_consistency_post"]
        )
        assert (
            result["temporal_consistency_original"]
            == result["temporal_consistency_pre"]
        )
        # Legacy bare key still present for backward compatibility.
        assert "temporal_consistency" in result

    def test_disposal_artifacts_aliases_present(self) -> None:
        result = self._run()
        assert "disposal_artifacts_compressed" in result
        assert "disposal_artifacts_original" in result
        assert (
            result["disposal_artifacts_compressed"]
            == result["disposal_artifacts_post"]
        )
        assert (
            result["disposal_artifacts_original"]
            == result["disposal_artifacts_pre"]
        )
        # Legacy bare key still present.
        assert "disposal_artifacts" in result

    def test_temporal_artifact_aliases_present(self) -> None:
        # ENABLE_TEMPORAL_ARTIFACTS defaults to True in MetricsConfig.
        result = self._run()
        # Only assert when the legacy keys actually made it into the dict
        # (the temporal-artifacts module is gated and may emit the
        # zero-fallback shape, which still includes these keys).
        for legacy, alias in [
            ("flicker_excess", "flicker_excess_compressed"),
            ("flicker_frame_ratio", "flicker_frame_ratio_compressed"),
            ("flat_flicker_ratio", "flat_flicker_ratio_compressed"),
            ("temporal_pumping_score", "temporal_pumping_score_compressed"),
            ("lpips_t_mean", "lpips_t_mean_compressed"),
            ("lpips_t_p95", "lpips_t_p95_compressed"),
        ]:
            assert legacy in result, f"legacy key {legacy} missing"
            assert alias in result, f"alias {alias} missing"
            assert result[alias] == result[legacy], (
                f"{alias} should mirror {legacy}: {result[alias]} vs {result[legacy]}"
            )


class TestSingleStreamMetricsDocumented:
    """The two pair-comparison-but-invariant metrics (texture_similarity,
    banding_score_*) should now have docstrings calling out the invariance,
    so future readers don't mistake invariance for single-stream behaviour.
    """

    def test_texture_similarity_docstring_calls_out_intensity_invariance(self) -> None:
        from giflab.metrics import texture_similarity

        doc = (texture_similarity.__doc__ or "").lower()
        assert "lbp" in doc or "binary pattern" in doc, (
            "texture_similarity docstring should mention LBP"
        )
        # Either 'invariant' or 'invariance' wording.
        assert "invarian" in doc, (
            "texture_similarity docstring should explain its invariances "
            "(white↔black collision)"
        )

    def test_detect_banding_artifacts_docstring_calls_out_no_gradient_case(self) -> None:
        from giflab.gradient_color_artifacts import GradientBandingDetector

        doc = (GradientBandingDetector.detect_banding_artifacts.__doc__ or "").lower()
        assert "no gradient" in doc or "no smooth gradient" in doc or "gradient region" in doc, (
            "detect_banding_artifacts docstring should explain the 0.0 = "
            "no-gradient-region-detected case"
        )
