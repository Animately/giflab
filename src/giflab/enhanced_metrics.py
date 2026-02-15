"""Enhanced quality metrics calculation with comprehensive 11-metric composite quality.

This module provides enhanced composite quality calculation using all 11 available
quality metrics instead of the traditional 4-metric approach. It also includes
the user-requested efficiency metric calculation.
"""

from typing import Any

import numpy as np

from .config import DEFAULT_METRICS_CONFIG, MetricsConfig


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

    composite_quality = 0.0
    total_weight = 0.0  # Track actual weights used (for missing metrics)

    # DEBUG: Track calculation steps
    debug_steps = []

    # Core structural similarity metrics (40% total)
    if "ssim_mean" in metrics:
        raw_value = metrics["ssim_mean"]
        normalized = normalize_metric("ssim_mean", raw_value)
        contribution = config.ENHANCED_SSIM_WEIGHT * normalized
        composite_quality += contribution
        total_weight += config.ENHANCED_SSIM_WEIGHT
        debug_steps.append(
            f"SSIM: {raw_value:.3f} → {normalized:.3f} × {config.ENHANCED_SSIM_WEIGHT} = {contribution:.3f}"
        )

    if "ms_ssim_mean" in metrics:
        raw_value = metrics["ms_ssim_mean"]
        normalized = normalize_metric("ms_ssim_mean", raw_value)
        contribution = config.ENHANCED_MS_SSIM_WEIGHT * normalized
        composite_quality += contribution
        total_weight += config.ENHANCED_MS_SSIM_WEIGHT
        debug_steps.append(
            f"MS-SSIM: {raw_value:.3f} → {normalized:.3f} × {config.ENHANCED_MS_SSIM_WEIGHT} = {contribution:.3f}"
        )

    # Signal quality metrics (25% total)
    if "psnr_mean" in metrics:
        normalized = normalize_metric("psnr_mean", metrics["psnr_mean"])
        composite_quality += config.ENHANCED_PSNR_WEIGHT * normalized
        total_weight += config.ENHANCED_PSNR_WEIGHT

    if "mse_mean" in metrics:
        normalized = normalize_metric("mse_mean", metrics["mse_mean"])
        composite_quality += config.ENHANCED_MSE_WEIGHT * normalized
        total_weight += config.ENHANCED_MSE_WEIGHT

    # Advanced structural metrics (20% total)
    if "fsim_mean" in metrics:
        normalized = normalize_metric("fsim_mean", metrics["fsim_mean"])
        composite_quality += config.ENHANCED_FSIM_WEIGHT * normalized
        total_weight += config.ENHANCED_FSIM_WEIGHT

    if "edge_similarity_mean" in metrics:
        normalized = normalize_metric(
            "edge_similarity_mean", metrics["edge_similarity_mean"]
        )
        composite_quality += config.ENHANCED_EDGE_WEIGHT * normalized
        total_weight += config.ENHANCED_EDGE_WEIGHT

    if "gmsd_mean" in metrics:
        normalized = normalize_metric("gmsd_mean", metrics["gmsd_mean"])
        composite_quality += config.ENHANCED_GMSD_WEIGHT * normalized
        total_weight += config.ENHANCED_GMSD_WEIGHT

    # Perceptual quality metrics (10% total)
    if "chist_mean" in metrics:
        normalized = normalize_metric("chist_mean", metrics["chist_mean"])
        composite_quality += config.ENHANCED_CHIST_WEIGHT * normalized
        total_weight += config.ENHANCED_CHIST_WEIGHT

    if "sharpness_similarity_mean" in metrics:
        normalized = normalize_metric(
            "sharpness_similarity_mean", metrics["sharpness_similarity_mean"]
        )
        composite_quality += config.ENHANCED_SHARPNESS_WEIGHT * normalized
        total_weight += config.ENHANCED_SHARPNESS_WEIGHT

    if "texture_similarity_mean" in metrics:
        normalized = normalize_metric(
            "texture_similarity_mean", metrics["texture_similarity_mean"]
        )
        composite_quality += config.ENHANCED_TEXTURE_WEIGHT * normalized
        total_weight += config.ENHANCED_TEXTURE_WEIGHT

    # Temporal consistency (5% total)
    if "temporal_consistency" in metrics:
        normalized = normalize_metric(
            "temporal_consistency", metrics["temporal_consistency"]
        )
        composite_quality += config.ENHANCED_TEMPORAL_WEIGHT * normalized
        total_weight += config.ENHANCED_TEMPORAL_WEIGHT

    # Deep perceptual metrics (3% total)
    if "lpips_quality_mean" in metrics:
        # LPIPS scores are inverted (lower = better), so we need to invert for quality
        # Normalize LPIPS score: 0.0 (identical) -> 1.0, 1.0 (very different) -> 0.0
        lpips_score = metrics["lpips_quality_mean"]
        normalized_lpips = max(
            0.0, min(1.0, 1.0 - lpips_score)
        )  # Invert: lower LPIPS = higher quality
        composite_quality += config.ENHANCED_LPIPS_WEIGHT * normalized_lpips
        total_weight += config.ENHANCED_LPIPS_WEIGHT

    if "ssimulacra2_mean" in metrics:
        # SSIMULACRA2 scores are already normalized (0-1, higher = better quality)
        ssimulacra2_score = metrics["ssimulacra2_mean"]
        normalized_ssimulacra2 = max(0.0, min(1.0, ssimulacra2_score))
        composite_quality += config.ENHANCED_SSIMULACRA2_WEIGHT * normalized_ssimulacra2
        total_weight += config.ENHANCED_SSIMULACRA2_WEIGHT

    # GIF-specific quality metrics (10% total)
    if "banding_score_mean" in metrics:
        normalized = normalize_metric(
            "banding_score_mean", metrics["banding_score_mean"]
        )
        composite_quality += config.ENHANCED_BANDING_WEIGHT * normalized
        total_weight += config.ENHANCED_BANDING_WEIGHT

    if "deltae_mean" in metrics:
        normalized_deltae = normalize_metric("deltae_mean", metrics["deltae_mean"])
        composite_quality += config.ENHANCED_DELTAE_WEIGHT * normalized_deltae
        total_weight += config.ENHANCED_DELTAE_WEIGHT

    # Normalize by actual weights used (handles missing metrics gracefully)
    raw_composite = composite_quality
    if total_weight > 0:
        composite_quality = composite_quality / total_weight

    # DEBUG: Log final calculation for debugging
    final_result = max(0.0, min(1.0, composite_quality))
    if (
        "ssim_mean" in metrics and metrics.get("ssim_mean", 0) > 0.5
    ):  # Only log for significant cases
        import logging

        logger = logging.getLogger(__name__)
        logger.info("Enhanced composite quality calculation:")
        for step in debug_steps[:3]:  # Log first few steps
            logger.info(f"  {step}")
        logger.info(
            f"  Raw total: {raw_composite:.3f}, Total weight: {total_weight:.3f}"
        )
        logger.info(
            f"  Normalized: {composite_quality:.3f}, Final (clamped): {final_result:.3f}"
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

    # Use existing logic from experimental runner
    composite_quality = (
        config.SSIM_WEIGHT * metrics.get("ssim_mean", 0.0)
        + config.MS_SSIM_WEIGHT * metrics.get("ms_ssim_mean", 0.0)
        + config.PSNR_WEIGHT * metrics.get("psnr_mean", 0.0)
        + config.TEMPORAL_WEIGHT * metrics.get("temporal_consistency", 0.0)
    )

    return max(0.0, min(1.0, composite_quality))


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
