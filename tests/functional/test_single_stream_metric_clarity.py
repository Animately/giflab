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


def _solid_frames(
    value: int, n: int = 4, size: tuple[int, int] = (32, 32)
) -> list[np.ndarray]:
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


def _outlier3_isolation_row() -> dict[str, float]:
    """Build the Outlier-3 temporal-isolation metrics row.

    This pins EVERY non-temporal metric to a perfect value, so the ONLY
    thing the composite can respond to is the temporal contribution. It
    carries the exact Outlier-3 temporal signature from the deep-dive
    (``docs/metrics-audit/outlier-deep-dive-2026-05-26.md``, theatre GIF
    ``f19df0a2``, lines ~132-179):

        temporal_consistency_pre  == 2.28e-8   (original animation is erratic)
        temporal_consistency_post == 2.28e-8   (compression changed nothing)
        temporal_consistency_delta == |post - pre| ≈ 0.0  (no temporal impact)
        temporal_consistency       == 2.28e-8  (the near-zero single-stream
                                                 value the LEGACY path reads)

    Input-convention note (must build the row with the right convention per
    key, otherwise the perfect-contribution assumption breaks):
    ``calculate_composite_quality`` runs ``normalize_metric`` ONLY on the
    ``_mean``-style keys, so ``psnr_mean`` is fed RAW dB (50.0 → 1.0 via
    PSNR_MAX_DB=50). The pre-normalised keys are read directly: feed
    ``ssimulacra2_mean=1.0`` and ``lpips_quality_mean=0.0`` (inverted in-block
    to contribute 1.0), and ``temporal_consistency_delta`` in [0, 1].
    """
    return {
        # --- pinned-perfect non-temporal block -------------------------------
        # _mean keys → routed through normalize_metric (raw values).
        "ssim_mean": 0.99999,
        "ms_ssim_mean": 1.0,
        "psnr_mean": 50.0,  # RAW dB; normalises to 1.0 (PSNR_MAX_DB=50)
        "mse_mean": 0.0,
        "fsim_mean": 1.0,
        "edge_similarity_mean": 0.999,
        "gmsd_mean": 0.0,
        "chist_mean": 1.0,
        "sharpness_similarity_mean": 1.0,
        "texture_similarity_mean": 1.0,
        "banding_score_mean": 0.0,
        "deltae_mean": 0.0,
        # pre-normalised keys → read directly (NO normalize_metric).
        "lpips_quality_mean": 0.0,  # inverted in-block → contributes 1.0
        "ssimulacra2_mean": 1.0,
        # --- Outlier-3 temporal signature -----------------------------------
        "temporal_consistency_pre": 2.28e-8,
        "temporal_consistency_post": 2.28e-8,
        "temporal_consistency_delta": abs(2.28e-8 - 2.28e-8),
        "temporal_consistency": 2.28e-8,
    }


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

    def test_erratic_source_lossless_compression_not_penalised_delta_vs_legacy(
        self,
    ) -> None:
        """Outlier-3 regression deliverable #1 (task TDD test 1).

        The reported bug: a high-motion animation whose original frame-to-frame
        differences are extremely erratic (``temporal_consistency_pre ≈ 0``)
        gets compressed essentially losslessly (every other metric perfect,
        ``temporal_consistency_delta ≈ 0``). The LEGACY post-only formula reads
        the near-zero single-stream ``temporal_consistency`` and wrongly drags
        the composite down by the full temporal weight (~0.10). The fixed delta
        formula reads ``temporal_consistency_delta ≈ 0`` and applies NO penalty,
        because compression preserved the (erratic but faithful) animation.

        BUG-DETECTION is the RELATIVE flag-on vs flag-off comparison, not the
        absolute threshold: with temporal capped at ENHANCED_TEMPORAL_WEIGHT
        (0.10) and every other metric pinned perfect, the worst the legacy
        formula can produce on this row is ``1.0 - 0.10 = 0.90`` — which still
        clears 0.8. So a bare ``composite > 0.8`` assertion would PASS under
        BOTH the fixed and buggy formula (verified: on=0.999939, off=0.899939)
        and would NOT catch a regression that flipped the flag off. The
        ``on - off`` margin is what earns this test its keep.
        """
        row = _outlier3_isolation_row()

        on = calculate_composite_quality(
            row, MetricsConfig(USE_TEMPORAL_DELTA_FOR_COMPOSITE=True)
        )
        off = calculate_composite_quality(
            row, MetricsConfig(USE_TEMPORAL_DELTA_FOR_COMPOSITE=False)
        )

        # Sanity: the delta is effectively zero (erratic-but-preserved).
        assert row["temporal_consistency_delta"] < 1e-6

        # Bug-detection half: the delta formula must score MATERIALLY higher
        # than the legacy post-only formula on this row. The measured gap is
        # the full temporal weight (~0.10); require a material margin so a
        # silent flag-flip regression is caught.
        assert on - off > 0.05, (
            f"Erratic-source/lossless-compression row must score higher under "
            f"the delta formula than under the legacy post-only formula "
            f"(on={on}, off={off}, gap={on - off}). A small gap means the "
            f"near-zero single-stream temporal_consistency is leaking back "
            f"into the composite."
        )

        # Secondary sanity: high-motion lossless content is not penalised.
        # NOTE: this absolute assertion alone is NOT a bug-detector — the
        # legacy off-formula (0.899939) also clears 0.8. Keep it only as a
        # sanity check; the on - off comparison above is the real guard.
        assert (
            on > 0.8
        ), f"High-motion lossless content should not be penalised: on={on}"

    def test_outlier3_full_metric_set_temporal_isolation(self) -> None:
        """Outlier-3 regression deliverable #2 (task TDD test 2).

        Same isolation row as deliverable #1, framed explicitly as the
        lossy-40 Outlier-3 row from the deep-dive
        (``docs/metrics-audit/outlier-deep-dive-2026-05-26.md`` lines ~132-145:
        ssim≈1.0, psnr→perfect, ssimulacra2=1.0, banding=0, edge=0.999,
        temporal delta≈0).

        PROVENANCE — this is a DELIBERATE ISOLATION of the temporal
        contribution with every other metric pinned perfect, so the 0.10-band
        temporal swing is the only thing the composite can respond to. It is
        explicitly NOT:
          * a reproduction of the reported 0.412 / 0.243 composites — those are
            Outlier 1 (Christmas stocking) and Outlier 2 (data-viz), driven by
            ssimulacra2/ssim/MSE collapse, NOT temporal (deep-dive lines 42, 88);
          * the lossy-60 Outlier-3 banding failure (banding 0→100,
            ssimulacra2 1.0→0.319) — that is a banding regression, not temporal.

        ASSERTION — the SAME flag-on vs flag-off relative comparison as
        deliverable #1, for the SAME reason: temporal is capped at 0.10 of the
        composite, so an absolute threshold cannot distinguish the fixed
        formula from the buggy one (both clear 0.8). Only the ``on - off``
        margin detects a regression that reverts to the legacy single-stream
        path.
        """
        row = _outlier3_isolation_row()

        on = calculate_composite_quality(
            row, MetricsConfig(USE_TEMPORAL_DELTA_FOR_COMPOSITE=True)
        )
        off = calculate_composite_quality(
            row, MetricsConfig(USE_TEMPORAL_DELTA_FOR_COMPOSITE=False)
        )

        assert on - off > 0.05, (
            f"Outlier-3 lossy-40 isolation row must score materially higher "
            f"under the delta formula than under the legacy post-only formula "
            f"(on={on}, off={off}, gap={on - off})."
        )

        # Secondary sanity only (passes under both formulas — see docstring).
        assert on > 0.8, f"Outlier-3 lossy-40 should not be penalised: on={on}"


class TestCompositeQualityDefensiveClamps:
    """Defensive unit tests of ``calculate_composite_quality`` — NOT Outlier-3
    deliverables.

    In production ``temporal_consistency_delta`` is already clamped to [0, 1]
    by ``calculate_temporal_consistency`` (metrics.py: ``|post - pre| ≤ 1``),
    so ``delta > 1.0`` is unreachable via the metrics pipeline. The clamp at
    ``enhanced_metrics.py`` (``max(0.0, min(1.0, delta))``) is nonetheless real,
    and this test pins that it behaves identically for an out-of-range delta.
    """

    def test_out_of_range_delta_clamps_to_one(self) -> None:
        row = _outlier3_isolation_row()

        config = MetricsConfig(USE_TEMPORAL_DELTA_FOR_COMPOSITE=True)

        row_15 = dict(row)
        row_15["temporal_consistency_delta"] = 1.5
        c_15 = calculate_composite_quality(row_15, config)

        row_10 = dict(row)
        row_10["temporal_consistency_delta"] = 1.0
        c_10 = calculate_composite_quality(row_10, config)

        # An out-of-range delta must clamp to the same worst-case score as
        # delta == 1.0 (verified: both → 0.8999385 on the all-perfect row),
        # never overshoot below it.
        assert c_15 == c_10, (
            f"delta=1.5 should clamp identically to delta=1.0 "
            f"(c_15={c_15}, c_10={c_10})"
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
            result["disposal_artifacts_compressed"] == result["disposal_artifacts_post"]
        )
        assert result["disposal_artifacts_original"] == result["disposal_artifacts_pre"]
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
            assert (
                result[alias] == result[legacy]
            ), f"{alias} should mirror {legacy}: {result[alias]} vs {result[legacy]}"


class TestSingleStreamMetricsDocumented:
    """The two pair-comparison-but-invariant metrics (texture_similarity,
    banding_score_*) should now have docstrings calling out the invariance,
    so future readers don't mistake invariance for single-stream behaviour.
    """

    def test_texture_similarity_docstring_calls_out_intensity_invariance(self) -> None:
        from giflab.metrics import texture_similarity

        doc = (texture_similarity.__doc__ or "").lower()
        assert (
            "lbp" in doc or "binary pattern" in doc
        ), "texture_similarity docstring should mention LBP"
        # Either 'invariant' or 'invariance' wording.
        assert "invarian" in doc, (
            "texture_similarity docstring should explain its invariances "
            "(white↔black collision)"
        )

    def test_detect_banding_artifacts_docstring_calls_out_no_gradient_case(
        self,
    ) -> None:
        from giflab.gradient_color_artifacts import GradientBandingDetector

        doc = (GradientBandingDetector.detect_banding_artifacts.__doc__ or "").lower()
        assert (
            "no gradient" in doc
            or "no smooth gradient" in doc
            or "gradient region" in doc
        ), (
            "detect_banding_artifacts docstring should explain the 0.0 = "
            "no-gradient-region-detected case"
        )
