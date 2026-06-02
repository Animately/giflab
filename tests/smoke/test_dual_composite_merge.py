"""Contract + unit tests for the dual-composite (white + black) merge.

These pin the load-bearing invariants of the worst-of merge introduced for
[[giflab-alpha-background-configurability-dark-content]]:

* polarity is derived by probing the END-TO-END ``calculate_composite_quality``
  (NOT ``normalize_metric``), which is the only way to get the right direction
  for ``lpips_quality_mean`` / ``ssimulacra2_mean`` /
  ``temporal_consistency_delta`` (none of which have a ``normalize_metric``
  branch);
* the worst-of pick moves the composite pessimistically;
* the merge copies sibling stats from the chosen pass, is NaN-aware, and
  recomputes composite_quality AND efficiency.
"""

import math

import pytest
from giflab.config import MetricsConfig
from giflab.enhanced_metrics import calculate_composite_quality
from giflab.metrics import (
    _composite_metric_polarity,
    _merge_worst_of_dual_composite,
    _POLARITY_PROBE_SENTINELS,
)


class TestPolarityDirectionContract:
    """Test 4: polarity derived from the end-to-end composite, not normalize_metric."""

    def test_derived_keys_match_composite_inputs_delta_mode(self):
        """Derived keys exactly equal the composite-input literals (delta mode)."""
        config = MetricsConfig()  # USE_TEMPORAL_DELTA_FOR_COMPOSITE defaults True
        polarity = _composite_metric_polarity(config)
        expected = {
            "ssim_mean",
            "ms_ssim_mean",
            "psnr_mean",
            "mse_mean",
            "fsim_mean",
            "edge_similarity_mean",
            "gmsd_mean",
            "chist_mean",
            "sharpness_similarity_mean",
            "texture_similarity_mean",
            "lpips_quality_mean",
            "ssimulacra2_mean",
            "banding_score_mean",
            "deltae_mean",
            "temporal_consistency_delta",
        }
        assert set(polarity) == expected

    def test_derived_keys_match_composite_inputs_legacy_temporal_mode(self):
        """Legacy mode varies bare temporal_consistency, not the delta."""
        config = MetricsConfig(USE_TEMPORAL_DELTA_FOR_COMPOSITE=False)
        polarity = _composite_metric_polarity(config)
        assert "temporal_consistency" in polarity
        assert "temporal_consistency_delta" not in polarity

    def test_three_inline_transformed_metrics_get_correct_direction(self):
        """lpips / ssimulacra2 / temporal_delta direction is correct.

        These three are transformed INLINE in calculate_composite_quality and
        have NO normalize_metric branch — probing normalize_metric would derive
        the OPPOSITE direction for lpips/temporal_delta. Probing the end-to-end
        composite gets them right: lpips and temporal_delta are higher-is-worse
        (-1), ssimulacra2 is higher-is-better (+1).
        """
        polarity = _composite_metric_polarity(MetricsConfig())
        assert polarity["lpips_quality_mean"] == -1
        assert polarity["temporal_consistency_delta"] == -1
        assert polarity["ssimulacra2_mean"] == 1

    @pytest.mark.parametrize(
        "key",
        [
            "ssim_mean",
            "ms_ssim_mean",
            "psnr_mean",
            "mse_mean",
            "fsim_mean",
            "edge_similarity_mean",
            "gmsd_mean",
            "chist_mean",
            "sharpness_similarity_mean",
            "texture_similarity_mean",
            "lpips_quality_mean",
            "ssimulacra2_mean",
            "banding_score_mean",
            "deltae_mean",
            "temporal_consistency_delta",
        ],
    )
    def test_worst_of_pick_moves_composite_pessimistically(self, key):
        """For each composite-input key, the worst-of value yields composite
        <= the other value, measured through the SAME end-to-end path.

        This never assumes WHERE the transform lives, so it catches a future
        transform flip OR a move of a transform between normalize_metric and
        the inline block.
        """
        config = MetricsConfig()
        polarity = _composite_metric_polarity(config)
        pol = polarity[key]
        low, high = _POLARITY_PROBE_SENTINELS[key]

        if pol == 1:
            # worst = MIN raw
            worst_raw, other_raw = low, high
        else:
            # worst = MAX raw
            worst_raw, other_raw = high, low

        composite_worst = calculate_composite_quality({key: worst_raw}, config)
        composite_other = calculate_composite_quality({key: other_raw}, config)
        assert composite_worst <= composite_other + 1e-9, (
            key,
            pol,
            composite_worst,
            composite_other,
        )


class TestMergeWorstOfDualComposite:
    """Test 8: direct unit test of the worst-of merge."""

    def _full_stats(self, base):
        """Build a dict with _mean plus _std/_min/_max for every stem."""
        d = {}
        for key, mean in base.items():
            stem = key[: -len("_mean")] if key.endswith("_mean") else key
            d[key] = mean
            d[f"{stem}_std"] = 0.0
            d[f"{stem}_min"] = mean
            d[f"{stem}_max"] = mean
        return d

    def test_per_key_worst_of_by_polarity(self):
        config = MetricsConfig()
        # ssim higher-better (+1): worst = lower. lpips higher-worse (-1):
        # worst = higher. deltae higher-worse (-1): worst = higher.
        white = self._full_stats(
            {
                "ssim_mean": 0.95,
                "lpips_quality_mean": 0.10,
                "deltae_mean": 1.0,
                "compression_ratio": 4.0,
            }
        )
        black = self._full_stats(
            {
                "ssim_mean": 0.70,  # worse (lower) than white
                "lpips_quality_mean": 0.40,  # worse (higher) than white
                "deltae_mean": 0.5,  # better (lower) than white
                "compression_ratio": 4.0,
            }
        )
        white["compression_ratio"] = 4.0
        black["compression_ratio"] = 4.0

        merged = _merge_worst_of_dual_composite(white, black, config)

        assert merged["ssim_mean"] == 0.70  # from black (worse)
        assert merged["lpips_quality_mean"] == 0.40  # from black (worse)
        assert merged["deltae_mean"] == 1.0  # from white (worse)

    def test_sibling_stats_follow_chosen_pass(self):
        config = MetricsConfig()
        white = {
            "ssim_mean": 0.95,
            "ssim_std": 0.01,
            "ssim_min": 0.94,
            "ssim_max": 0.96,
        }
        black = {
            "ssim_mean": 0.70,
            "ssim_std": 0.05,
            "ssim_min": 0.60,
            "ssim_max": 0.80,
        }
        merged = _merge_worst_of_dual_composite(white, black, config)
        # ssim worst is black's lower mean -> siblings come from black.
        assert merged["ssim_mean"] == 0.70
        assert merged["ssim_min"] == 0.60
        assert merged["ssim_max"] == 0.80
        # Invariant holds for the chosen pass.
        assert merged["ssim_min"] <= merged["ssim_mean"] <= merged["ssim_max"]

    def test_nan_aware_one_finite_one_nan_takes_finite(self):
        config = MetricsConfig()
        white = {"ssim_mean": 0.9}
        black = {"ssim_mean": float("nan")}
        merged = _merge_worst_of_dual_composite(white, black, config)
        # one finite + one NaN -> finite (a real measurement beats a non-one).
        assert merged["ssim_mean"] == 0.9

    def test_nan_aware_both_nan_propagates_nan(self):
        config = MetricsConfig()
        white = {"ssim_mean": float("nan")}
        black = {"ssim_mean": float("nan")}
        merged = _merge_worst_of_dual_composite(white, black, config)
        assert math.isnan(merged["ssim_mean"])

    def test_present_in_one_pass_only_taken_without_fabrication(self):
        config = MetricsConfig()
        white = {"ssim_mean": 0.9}
        black = {
            "ssim_mean": 0.9,
            "deltae_mean": 5.0,
            "deltae_min": 5.0,
            "deltae_max": 5.0,
        }
        merged = _merge_worst_of_dual_composite(white, black, config)
        # deltae present only on black -> taken (with siblings), not fabricated.
        assert merged["deltae_mean"] == 5.0

    def test_recomputed_composite_and_efficiency_are_pessimistic(self):
        """After re-running the enhanced processor on the merged dict, both
        composite_quality AND efficiency reflect the worst-of values.

        compression_ratio is identical across passes (file-size derived), so
        efficiency is consistent and lower (reflecting the lower composite).
        """
        from giflab.enhanced_metrics import process_metrics_with_enhanced_quality

        config = MetricsConfig()
        white = self._full_stats({"ssim_mean": 0.95})
        white["compression_ratio"] = 5.0
        black = self._full_stats({"ssim_mean": 0.50})
        black["compression_ratio"] = 5.0

        merged = _merge_worst_of_dual_composite(white, black, config)
        merged = process_metrics_with_enhanced_quality(merged, config)

        # composite reflects the worst (lower) ssim from black.
        white_proc = process_metrics_with_enhanced_quality(dict(white), config)
        assert merged["composite_quality"] < white_proc["composite_quality"]
        # efficiency present and lower (same compression_ratio, lower composite).
        assert "efficiency" in merged
        assert merged["efficiency"] < white_proc["efficiency"]
