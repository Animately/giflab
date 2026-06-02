"""Enhanced quality metrics calculation with comprehensive 11-metric composite quality.

This module provides enhanced composite quality calculation using all 11 available
quality metrics instead of the traditional 4-metric approach. It also includes
the user-requested efficiency metric calculation.
"""

import math
from typing import Any

import numpy as np

from .config import DEFAULT_METRICS_CONFIG, MetricsConfig

# When at least this fraction of the PRESENT (contributing) weight is
# unmeasurable (NaN / None), the composite is "majority-missing": a
# redistributed number would be more misleading than honest, so
# ``calculate_composite_quality`` returns ``float("nan")`` instead.
#
# The ratio is taken over PRESENT weight only — metrics whose keys are simply
# absent from the input never contributed weight, so they don't count toward
# the missing fraction (an input with only two finite metrics is not
# "majority-missing" just because the other thirteen weren't supplied).
#
# The boundary is ``>=``: exactly-half-of-present-weight unmeasurable returns
# NaN. This is a metric-formula constant, not a tunable weight, so it lives
# here rather than in ``config.py`` (placing it there would re-trigger the
# weight-sum-to-1.0 validation in ``MetricsConfig.__post_init__`` for no
# reason).
COMPOSITE_NAN_THRESHOLD = 0.5


def _is_missing(value: Any) -> bool:
    """Return True if *value* is unmeasurable (None or NaN).

    Mirrors ``optimization_validation.validation_checker._is_missing`` — a
    metric that could not be computed is propagated as ``float("nan")`` (or,
    defensively, ``None``) by the upstream metrics paths and must be excluded
    from the composite, not coerced through ``normalize_metric``'s clamp (which
    turns NaN into the upper bound 1.0).
    """
    if value is None:
        return True
    return isinstance(value, float) and math.isnan(value)


def _resolve_composite_from_contributions(
    contributions: list[tuple[float, float]],
) -> float:
    """Collapse per-metric ``(normalized_value, weight)`` pairs into the final
    composite, filtering unmeasurable contributions and redistributing weight.

    ``contributions`` contains an entry for every metric block that was PRESENT
    in the input (key existed), whether or not its value was measurable. A
    measurable contribution carries its finite normalized value; an
    unmeasurable one carries ``float("nan")`` as its value (its weight still
    counts toward "present weight" so the missing-fraction is honest).

    Returns:
        * ``0.0`` when no metric block was present (total present weight 0) —
          preserves the historical "nothing to measure" contract.
        * ``float("nan")`` when the unmeasurable fraction of present weight is
          ``>= COMPOSITE_NAN_THRESHOLD`` (majority-missing).
        * otherwise a finite redistributed score in ``[0.0, 1.0]``.
    """
    total_weight = sum(weight for _, weight in contributions)
    if total_weight <= 0:
        # No quality keys present at all — nothing to measure. Distinct from
        # "keys present but all unmeasurable", which falls into the NaN branch
        # below (that path has total_weight > 0).
        return 0.0

    missing_weight = sum(
        weight for value, weight in contributions if _is_missing(value)
    )

    if missing_weight / total_weight >= COMPOSITE_NAN_THRESHOLD:
        # Majority of the present weight is unmeasurable — an honest NaN beats a
        # confident-looking redistributed number built from a minority signal.
        return float("nan")

    surviving_weight = total_weight - missing_weight
    weighted_sum = sum(
        value * weight for value, weight in contributions if not _is_missing(value)
    )
    # Redistribute: divide by surviving weight so the kept metrics fill the
    # whole [0, 1] range exactly as if the missing keys had been absent.
    composite = weighted_sum / surviving_weight
    return max(0.0, min(1.0, composite))


def normalize_metric(
    metric_name: str, value: float, min_val: float = 0.0, max_val: float = 1.0
) -> float:
    """Normalize a metric value to 0-1 range with appropriate direction.

    Args:
        metric_name: Name of the metric (determines if higher/lower is better)
        value: Raw metric value
        min_val: Minimum possible value for normalization
        max_val: Maximum possible value for normalization

    Returns:
        Normalized value between 0 and 1
    """
    # Handle special cases with known ranges
    if metric_name == "psnr_mean":
        # PSNR: higher is better, typical range 0-50dB
        # PSNR = 1.0dB is poor quality, PSNR = 50dB is excellent
        max_psnr_db = 50.0
        normalized = min(value, max_psnr_db) / max_psnr_db
        return max(0.0, min(1.0, normalized))

    elif metric_name == "mse_mean":
        # MSE can be very large, use log normalization
        if value <= 0:
            return 1.0  # Perfect score for zero MSE
        # Normalize MSE using log scale, then invert (lower MSE is better)
        normalized = 1.0 / (1.0 + np.log10(max(value, 1.0)))
        return float(max(0.0, min(1.0, normalized)))

    elif metric_name == "gmsd_mean":
        # GMSD: lower is better, typical range 0-0.5
        max_val = 0.5
        normalized = 1.0 - min(value, max_val) / max_val
        return max(0.0, min(1.0, normalized))

    elif metric_name == "ms_ssim_mean":
        # MS-SSIM can go negative, handle appropriately
        if value < 0:
            return 0.0  # Negative MS-SSIM indicates very poor quality
        return max(0.0, min(1.0, value))

    elif metric_name == "banding_score_mean":
        # Banding: lower is better, range 0-1
        return max(0.0, min(1.0, 1.0 - value))

    elif metric_name == "deltae_mean":
        # DeltaE: lower is better, typical range 0-20
        normalized = max(0.0, 1.0 - (value / 10.0))
        return max(0.0, min(1.0, normalized))

    else:
        # Standard 0-1 normalization for metrics where higher is better
        if min_val >= max_val:
            return 1.0  # Avoid division by zero
        normalized = (value - min_val) / (max_val - min_val)
        return max(0.0, min(1.0, normalized))


def calculate_composite_quality(
    metrics: dict[str, float], config: MetricsConfig | None = None
) -> float:
    """Calculate enhanced composite quality using all 11 available metrics.

    This function implements the comprehensive approach (Approach B) using
    research-based weights across all quality dimensions.

    Args:
        metrics: Dictionary containing all metric values
        config: Metrics configuration (uses default if None)

    Returns:
        Enhanced composite quality score (0-1)
    """
    if config is None:
        config = DEFAULT_METRICS_CONFIG

    if not config.USE_ENHANCED_COMPOSITE_QUALITY:
        # Fall back to legacy 4-metric calculation
        return calculate_legacy_composite_quality(metrics, config)

    # Build one ``(normalized_value, weight)`` contribution per PRESENT metric
    # block. Unmeasurable values (NaN / None) keep their weight in the list with
    # a NaN ``normalized_value`` — ``_resolve_composite_from_contributions``
    # filters them and redistributes the surviving weight, returning NaN when
    # the unmeasurable fraction reaches COMPOSITE_NAN_THRESHOLD.
    #
    # CRITICAL: a NaN must NOT flow through ``normalize_metric`` and the clamp
    # ``max(0.0, min(1.0, nan))``, which returns the upper bound 1.0 in CPython
    # (every comparison with NaN is False) — that fabricates a PERFECT
    # contribution and inflates the weight. So we test ``_is_missing`` on the
    # RAW value before normalizing and pass NaN straight through when missing.
    contributions: list[tuple[float, float]] = []
    debug_steps: list[str] = []

    def _add(metric_name: str, raw_value: Any, weight: float) -> None:
        if _is_missing(raw_value):
            contributions.append((float("nan"), weight))
            return
        normalized = normalize_metric(metric_name, raw_value)
        contributions.append((normalized, weight))
        debug_steps.append(f"{metric_name}: {raw_value} → {normalized:.3f} × {weight}")

    # Core structural similarity metrics (40% total)
    if "ssim_mean" in metrics:
        _add("ssim_mean", metrics["ssim_mean"], config.ENHANCED_SSIM_WEIGHT)

    if "ms_ssim_mean" in metrics:
        _add("ms_ssim_mean", metrics["ms_ssim_mean"], config.ENHANCED_MS_SSIM_WEIGHT)

    # Signal quality metrics (25% total)
    if "psnr_mean" in metrics:
        _add("psnr_mean", metrics["psnr_mean"], config.ENHANCED_PSNR_WEIGHT)

    if "mse_mean" in metrics:
        _add("mse_mean", metrics["mse_mean"], config.ENHANCED_MSE_WEIGHT)

    # Advanced structural metrics (20% total)
    if "fsim_mean" in metrics:
        _add("fsim_mean", metrics["fsim_mean"], config.ENHANCED_FSIM_WEIGHT)

    if "edge_similarity_mean" in metrics:
        _add(
            "edge_similarity_mean",
            metrics["edge_similarity_mean"],
            config.ENHANCED_EDGE_WEIGHT,
        )

    if "gmsd_mean" in metrics:
        _add("gmsd_mean", metrics["gmsd_mean"], config.ENHANCED_GMSD_WEIGHT)

    # Perceptual quality metrics (10% total)
    if "chist_mean" in metrics:
        _add("chist_mean", metrics["chist_mean"], config.ENHANCED_CHIST_WEIGHT)

    if "sharpness_similarity_mean" in metrics:
        _add(
            "sharpness_similarity_mean",
            metrics["sharpness_similarity_mean"],
            config.ENHANCED_SHARPNESS_WEIGHT,
        )

    if "texture_similarity_mean" in metrics:
        _add(
            "texture_similarity_mean",
            metrics["texture_similarity_mean"],
            config.ENHANCED_TEXTURE_WEIGHT,
        )

    # Temporal consistency (5% total)
    #
    # The ``temporal_consistency_compressed`` key (Wave 7 renamed the bare
    # ``temporal_consistency``) is the post-compression value of a metric
    # computed on the COMPRESSED STREAM ONLY (see ``metrics.py``:
    # ``calculate_temporal_consistency`` runs on ``compressed_frames_resized``).
    # Using it as a composite contribution treats single-stream content
    # stability as if it were a quality-vs-reference signal. A static-black
    # output scores 1.0 here regardless of the original, lifting
    # composite_quality by the full temporal weight.
    #
    # Audit-fix (Wave 3): when ``USE_TEMPORAL_DELTA_FOR_COMPOSITE`` is True,
    # use ``temporal_consistency_delta = |post - pre|`` instead — a true
    # pair signal where 0 means "compression preserved temporal behaviour"
    # and 1 means "temporal behaviour was destroyed".
    #
    # NaN guard: a NaN ``temporal_consistency_delta`` means the pair signal
    # couldn't be computed. We record it as a missing contribution (weight
    # present, value NaN) and DO NOT fall back to the single-stream
    # ``temporal_consistency_compressed`` value — falling back would silently
    # re-introduce the static-black-wins bug the delta was added to fix.
    use_temporal_delta = getattr(config, "USE_TEMPORAL_DELTA_FOR_COMPOSITE", True)
    if use_temporal_delta and "temporal_consistency_delta" in metrics:
        delta = metrics["temporal_consistency_delta"]
        if _is_missing(delta):
            contributions.append((float("nan"), config.ENHANCED_TEMPORAL_WEIGHT))
        else:
            # Smaller delta = higher quality; clamp delta to [0, 1] before
            # inverting so out-of-range values don't blow up the score.
            delta = float(delta)
            normalized = max(0.0, min(1.0, 1.0 - max(0.0, min(1.0, delta))))
            contributions.append((normalized, config.ENHANCED_TEMPORAL_WEIGHT))
    elif "temporal_consistency_compressed" in metrics:
        _add(
            "temporal_consistency_compressed",
            metrics["temporal_consistency_compressed"],
            config.ENHANCED_TEMPORAL_WEIGHT,
        )

    # Deep perceptual metrics (3% total)
    #
    # LPIPS scores are inverted (lower = better). NaN means LPIPS couldn't be
    # computed (model load failure, no scores, subprocess crash) — recorded as
    # a missing contribution and redistributed by
    # ``_resolve_composite_from_contributions`` rather than fabricated to 1.0.
    if "lpips_quality_mean" in metrics:
        lpips_score = metrics["lpips_quality_mean"]
        if _is_missing(lpips_score):
            contributions.append((float("nan"), config.ENHANCED_LPIPS_WEIGHT))
        else:
            # Invert: lower LPIPS = higher quality.
            normalized_lpips = max(0.0, min(1.0, 1.0 - lpips_score))
            contributions.append((normalized_lpips, config.ENHANCED_LPIPS_WEIGHT))

    # SSIMULACRA2 scores are already normalized (0-1, higher = better quality).
    # NaN means the metric couldn't be computed (binary missing, frame-export
    # failure, etc.).
    if "ssimulacra2_mean" in metrics:
        ssimulacra2_score = metrics["ssimulacra2_mean"]
        if _is_missing(ssimulacra2_score):
            contributions.append((float("nan"), config.ENHANCED_SSIMULACRA2_WEIGHT))
        else:
            normalized_ssimulacra2 = max(0.0, min(1.0, ssimulacra2_score))
            contributions.append(
                (normalized_ssimulacra2, config.ENHANCED_SSIMULACRA2_WEIGHT)
            )

    # GIF-specific quality metrics (10% total)
    if "banding_score_mean" in metrics:
        _add(
            "banding_score_mean",
            metrics["banding_score_mean"],
            config.ENHANCED_BANDING_WEIGHT,
        )

    if "deltae_mean" in metrics:
        _add("deltae_mean", metrics["deltae_mean"], config.ENHANCED_DELTAE_WEIGHT)

    final_result = _resolve_composite_from_contributions(contributions)

    # DEBUG: Log final calculation for debugging (only for significant cases).
    ssim_present = metrics.get("ssim_mean")
    if (
        ssim_present is not None
        and not _is_missing(ssim_present)
        and ssim_present > 0.5
    ):
        import logging

        logger = logging.getLogger(__name__)
        logger.info("Enhanced composite quality calculation:")
        for step in debug_steps[:3]:  # Log first few steps
            logger.info(f"  {step}")
        present_weight = sum(w for _, w in contributions)
        missing_weight = sum(w for v, w in contributions if _is_missing(v))
        logger.info(
            f"  Present weight: {present_weight:.3f}, "
            f"Missing weight: {missing_weight:.3f}, "
            f"Final: {final_result}"
        )

    return final_result


def calculate_legacy_composite_quality(
    metrics: dict[str, float], config: MetricsConfig | None = None
) -> float:
    """Calculate legacy 4-metric composite quality for backward compatibility.

    Args:
        metrics: Dictionary containing metric values
        config: Metrics configuration (uses default if None)

    Returns:
        Legacy composite quality score (0-1)
    """
    if config is None:
        config = DEFAULT_METRICS_CONFIG

    # Same NaN anti-pattern as the enhanced path: ``weight * nan = nan`` poisons
    # the sum and ``max(0.0, min(1.0, nan))`` returns 1.0 in CPython — a
    # measurement failure would score as PERFECT quality. Build per-metric
    # contributions and filter/redistribute uniformly via the shared resolver.
    #
    # NOTE: the legacy formula multiplies the RAW metric value by its weight
    # (it does not run ``normalize_metric``), so we preserve that here — only
    # the NaN handling changes. Missing keys default to 0.0 (the historical
    # behaviour) and contribute their weight at the worst-case value; only
    # genuinely-unmeasurable NaN/None values are filtered and redistributed.
    contributions: list[tuple[float, float]] = [
        (metrics.get("ssim_mean", 0.0), config.SSIM_WEIGHT),
        (metrics.get("ms_ssim_mean", 0.0), config.MS_SSIM_WEIGHT),
        (metrics.get("psnr_mean", 0.0), config.PSNR_WEIGHT),
        # Wave 7: bare ``temporal_consistency`` renamed to ``_compressed``.
        (metrics.get("temporal_consistency_compressed", 0.0), config.TEMPORAL_WEIGHT),
    ]

    return _resolve_composite_from_contributions(contributions)


def calculate_efficiency_metric(
    compression_ratio: float, composite_quality: float
) -> float:
    """Calculate balanced efficiency metric with 50% quality, 50% compression weighting.

    This approach provides a more balanced assessment of GIF compression efficiency by:
    - Log-normalizing compression ratio to handle diminishing returns above 20x
    - Using geometric mean with balanced weights (50% quality, 50% compression)
    - Preventing extreme compression ratios from dominating the score
    - Providing equal weighting between quality preservation and compression efficiency

    Args:
        compression_ratio: Compression ratio (original_size / compressed_size)
        composite_quality: Composite quality score (0-1)

    Returns:
        Balanced efficiency score (0-1 range, higher is better)
    """
    if compression_ratio <= 0 or composite_quality < 0:
        return 0.0

    # Log-normalize compression ratio to 0-1 scale
    # Cap at 20x as practical maximum (beyond this has diminishing user value)
    max_practical_compression = 20.0
    normalized_compression = min(
        np.log(1 + compression_ratio) / np.log(1 + max_practical_compression), 1.0
    )

    # Weighted geometric mean: 50% quality, 50% compression
    quality_weight = 0.5
    compression_weight = 0.5

    # Use geometric mean to prevent one-dimensional dominance
    efficiency = (composite_quality**quality_weight) * (
        normalized_compression**compression_weight
    )

    return float(efficiency)


def process_metrics_with_enhanced_quality(
    result: dict[str, Any], config: MetricsConfig | None = None
) -> dict[str, Any]:
    """Process a metrics result dictionary to add enhanced quality metrics.

    Args:
        result: Dictionary containing raw metric values
        config: Metrics configuration (uses default if None)

    Returns:
        Enhanced result dictionary with new metrics added
    """
    if config is None:
        config = DEFAULT_METRICS_CONFIG

    # Calculate enhanced composite quality - now returns single value
    enhanced_composite = calculate_composite_quality(result, config)
    result["composite_quality"] = enhanced_composite

    # Calculate efficiency metric if compression data is available
    if "compression_ratio" in result:
        # Use enhanced composite quality for efficiency calculation
        efficiency = calculate_efficiency_metric(
            result["compression_ratio"], enhanced_composite
        )
        result["efficiency"] = efficiency

    return result


def get_enhanced_weights_info(config: MetricsConfig | None = None) -> dict[str, Any]:
    """Get information about the enhanced weighting scheme.

    Args:
        config: Metrics configuration (uses default if None)

    Returns:
        Dictionary with weight distribution information
    """
    if config is None:
        config = DEFAULT_METRICS_CONFIG

    weights_info = {
        "core_structural": {
            "ssim_mean": config.ENHANCED_SSIM_WEIGHT,
            "ms_ssim_mean": config.ENHANCED_MS_SSIM_WEIGHT,
            "total": config.ENHANCED_SSIM_WEIGHT + config.ENHANCED_MS_SSIM_WEIGHT,
        },
        "signal_quality": {
            "psnr_mean": config.ENHANCED_PSNR_WEIGHT,
            "mse_mean": config.ENHANCED_MSE_WEIGHT,
            "total": config.ENHANCED_PSNR_WEIGHT + config.ENHANCED_MSE_WEIGHT,
        },
        "advanced_structural": {
            "fsim_mean": config.ENHANCED_FSIM_WEIGHT,
            "edge_similarity_mean": config.ENHANCED_EDGE_WEIGHT,
            "gmsd_mean": config.ENHANCED_GMSD_WEIGHT,
            "total": config.ENHANCED_FSIM_WEIGHT
            + config.ENHANCED_EDGE_WEIGHT
            + config.ENHANCED_GMSD_WEIGHT,
        },
        "perceptual_quality": {
            "chist_mean": config.ENHANCED_CHIST_WEIGHT,
            "sharpness_similarity_mean": config.ENHANCED_SHARPNESS_WEIGHT,
            "texture_similarity_mean": config.ENHANCED_TEXTURE_WEIGHT,
            "total": config.ENHANCED_CHIST_WEIGHT
            + config.ENHANCED_SHARPNESS_WEIGHT
            + config.ENHANCED_TEXTURE_WEIGHT,
        },
        "temporal_consistency": {
            "temporal_consistency": config.ENHANCED_TEMPORAL_WEIGHT,
            "total": config.ENHANCED_TEMPORAL_WEIGHT,
        },
        "deep_perceptual": {
            "lpips_mean": config.ENHANCED_LPIPS_WEIGHT,
            "ssimulacra2_mean": config.ENHANCED_SSIMULACRA2_WEIGHT,
            "total": config.ENHANCED_LPIPS_WEIGHT + config.ENHANCED_SSIMULACRA2_WEIGHT,
        },
        "grand_total": (
            config.ENHANCED_SSIM_WEIGHT
            + config.ENHANCED_MS_SSIM_WEIGHT
            + config.ENHANCED_PSNR_WEIGHT
            + config.ENHANCED_MSE_WEIGHT
            + config.ENHANCED_FSIM_WEIGHT
            + config.ENHANCED_EDGE_WEIGHT
            + config.ENHANCED_GMSD_WEIGHT
            + config.ENHANCED_CHIST_WEIGHT
            + config.ENHANCED_SHARPNESS_WEIGHT
            + config.ENHANCED_TEXTURE_WEIGHT
            + config.ENHANCED_TEMPORAL_WEIGHT
            + config.ENHANCED_LPIPS_WEIGHT
            + config.ENHANCED_SSIMULACRA2_WEIGHT
        ),
    }

    return weights_info
