"""Quality threshold validation using existing metrics system.

This module integrates with the existing 11-metric quality system to detect
catastrophic quality failures while allowing normal compression degradation.
"""

import logging
from pathlib import Path
from typing import Any

from ..config import DEFAULT_METRICS_CONFIG
from ..enhanced_metrics import calculate_composite_quality
from ..metrics import calculate_comprehensive_metrics
from .types import ValidationResult

logger = logging.getLogger(__name__)


class QualityThresholdValidator:
    """Validates quality degradation using existing metrics system.

    This validator leverages the existing 11-metric comprehensive quality
    system to detect catastrophic quality failures that would indicate
    wrapper corruption rather than normal compression trade-offs.
    """

    def __init__(self, metrics_config: Any = None) -> None:
        """Initialize quality validator.

        Args:
            metrics_config: Metrics configuration to use. Defaults to system config.
        """
        self.metrics_config = metrics_config or DEFAULT_METRICS_CONFIG

        # Quality thresholds for detecting catastrophic failures
        # These are much lower than normal quality targets - we're only
        # trying to catch severe corruption, not optimize compression quality
        self.catastrophic_thresholds = {
            # Composite quality thresholds (0-1 scale)
            "min_composite_quality": 0.1,
            # Individual metric thresholds (detect severe outliers)
            "min_ssim_mean": 0.2,  # SSIM below 20% is very bad
            "max_mse_mean": 10000.0,  # MSE above 10000 indicates major corruption
            "min_psnr_mean": 10.0,  # PSNR below 10dB is extremely poor
            # Compressed-stream stability floor (animation-specific).
            # NB: the underlying metric (temporal_consistency) is computed only
            # on the *compressed* stream (temporal_consistency_post), NOT as an
            # original-vs-compressed pair.  The name reflects that single-stream
            # reality — see CLAUDE.md "Pair-wise over single-stream — and
            # labelled honestly".
            "min_compressed_temporal_consistency": 0.1,  # Severe compressed-stream instability
            # Edge case protections
            "max_quality_variance": 0.9,  # Quality shouldn't vary wildly between frames
        }

    def validate_quality_degradation(
        self,
        input_path: Path,
        output_path: Path,
        wrapper_metadata: dict[str, Any],
        operation_type: str = "unknown",
    ) -> ValidationResult:
        """Validate quality degradation is within acceptable bounds.

        This method uses the existing comprehensive metrics system to detect
        catastrophic quality failures while allowing normal compression trade-offs.

        Args:
            input_path: Original file
            output_path: Compressed file
            wrapper_metadata: Metadata from wrapper execution
            operation_type: Type of operation performed for context

        Returns:
            ValidationResult indicating if quality degradation is acceptable
        """
        try:
            # Determine if this is a frame reduction operation
            is_frame_reduction = operation_type == "frame_reduction"

            # Calculate comprehensive quality metrics
            metrics = calculate_comprehensive_metrics(
                input_path,
                output_path,
                config=self.metrics_config,
                frame_reduction_context=is_frame_reduction,
            )

            # Calculate composite quality scores
            if self.metrics_config.USE_ENHANCED_COMPOSITE_QUALITY:
                # Filter metrics to only numeric values for composite quality calculation
                numeric_metrics = {
                    k: v for k, v in metrics.items() if isinstance(v, int | float)
                }
                composite_quality = calculate_composite_quality(
                    numeric_metrics, self.metrics_config
                )
                quality_type = "enhanced_composite"

                # Use context-aware threshold for frame reduction operations
                if is_frame_reduction:
                    # Frame reduction operations naturally have lower quality due to temporal gaps
                    # Use more lenient threshold based on research findings
                    min_quality_threshold = 0.05  # 5% threshold for frame reduction
                else:
                    min_quality_threshold = self.catastrophic_thresholds[
                        "min_composite_quality"
                    ]
            else:
                # Use legacy 4-metric composite quality
                composite_quality = self._calculate_legacy_composite_quality(metrics)
                quality_type = "legacy_composite"
                min_quality_threshold = self.catastrophic_thresholds[
                    "min_composite_quality"
                ]

            # Check for catastrophic quality failure
            quality_acceptable = composite_quality >= min_quality_threshold

            # Additional checks for severe outliers in individual metrics
            outlier_checks = self._check_metric_outliers(
                metrics, frame_reduction_context=is_frame_reduction
            )
            outliers_acceptable = all(
                check["acceptable"] for check in outlier_checks.values()
            )

            # Overall validation result
            is_valid = quality_acceptable and outliers_acceptable

            # Prepare detailed information
            details = {
                "composite_quality": composite_quality,
                "quality_type": quality_type,
                "min_threshold": min_quality_threshold,
                "operation_type": operation_type,
                "outlier_checks": outlier_checks,
                "metrics_summary": {
                    "ssim_mean": metrics.get("ssim_mean"),
                    "psnr_mean": metrics.get("psnr_mean"),
                    "mse_mean": metrics.get("mse_mean"),
                    "temporal_consistency": metrics.get("temporal_consistency"),
                },
            }

            # Create error message if validation failed
            error_message = None
            if not is_valid:
                error_parts = []
                if not quality_acceptable:
                    error_parts.append(
                        f"{quality_type} quality {composite_quality:.3f} below threshold {min_quality_threshold}"
                    )

                failed_outliers = [
                    name
                    for name, check in outlier_checks.items()
                    if not check["acceptable"]
                ]
                if failed_outliers:
                    error_parts.append(f"Metric outliers detected: {failed_outliers}")

                error_message = "; ".join(error_parts)

            return ValidationResult(
                is_valid=is_valid,
                validation_type="quality_degradation",
                expected={
                    "min_composite_quality": min_quality_threshold,
                    "acceptable_degradation": True,
                },
                actual={
                    "composite_quality": composite_quality,
                    "quality_acceptable": quality_acceptable,
                    "outliers_acceptable": outliers_acceptable,
                },
                error_message=error_message,
                details=details,
            )

        except Exception as e:
            logger.error(f"Quality validation error: {e}")
            return ValidationResult(
                is_valid=False,
                validation_type="quality_degradation",
                expected="quality_metrics_calculation",
                actual="calculation_failed",
                error_message=f"Quality validation failed: {str(e)}",
                details={"exception": str(e)},
            )

    def _calculate_legacy_composite_quality(
        self, metrics: dict[str, float | str]
    ) -> float:
        """Calculate legacy 4-metric composite quality score.

        This is a fallback for when enhanced composite quality is disabled.
        """
        # Use the same weights as the legacy system
        weights = {
            "ssim_mean": self.metrics_config.SSIM_WEIGHT,
            "ms_ssim_mean": self.metrics_config.MS_SSIM_WEIGHT,
            "psnr_mean": self.metrics_config.PSNR_WEIGHT,
            "temporal_consistency": self.metrics_config.TEMPORAL_WEIGHT,
        }

        composite_quality = 0.0
        total_weight = 0.0

        for metric_name, weight in weights.items():
            if metric_name in metrics:
                value = metrics[metric_name]

                # Skip non-numeric values
                if not isinstance(value, int | float):
                    continue

                # Normalize metrics to 0-1 scale
                if metric_name == "psnr_mean":
                    # PSNR: normalize using configured max
                    normalized_value = min(value / self.metrics_config.PSNR_MAX_DB, 1.0)
                elif metric_name in [
                    "ssim_mean",
                    "ms_ssim_mean",
                    "temporal_consistency",
                ]:
                    # These should already be 0-1
                    normalized_value = max(0.0, min(value, 1.0))
                else:
                    normalized_value = value

                composite_quality += weight * normalized_value
                total_weight += weight

        return composite_quality / total_weight if total_weight > 0 else 0.0

    def _check_metric_outliers(
        self, metrics: dict[str, float | str], frame_reduction_context: bool = False
    ) -> dict[str, dict[str, Any]]:
        """Check for severe outliers in individual metrics.

        Returns dict with outlier check results for each metric.
        """
        checks = {}

        # SSIM check
        if "ssim_mean" in metrics:
            ssim_value = metrics["ssim_mean"]
            # Only compare if value is numeric
            is_acceptable = (
                isinstance(ssim_value, int | float)
                and ssim_value >= self.catastrophic_thresholds["min_ssim_mean"]
            )
            checks["ssim"] = {
                "value": ssim_value,
                "threshold": self.catastrophic_thresholds["min_ssim_mean"],
                "acceptable": is_acceptable,
                "description": "Structural similarity",
            }

        # MSE check (higher is worse)
        if "mse_mean" in metrics:
            mse_value = metrics["mse_mean"]
            # Only compare if value is numeric
            is_acceptable = (
                isinstance(mse_value, int | float)
                and mse_value <= self.catastrophic_thresholds["max_mse_mean"]
            )
            checks["mse"] = {
                "value": mse_value,
                "threshold": self.catastrophic_thresholds["max_mse_mean"],
                "acceptable": is_acceptable,
                "description": "Mean squared error",
            }

        # PSNR check
        if "psnr_mean" in metrics:
            psnr_value = metrics["psnr_mean"]
            # Only compare if value is numeric
            is_acceptable = (
                isinstance(psnr_value, int | float)
                and psnr_value >= self.catastrophic_thresholds["min_psnr_mean"]
            )
            checks["psnr"] = {
                "value": psnr_value,
                "threshold": self.catastrophic_thresholds["min_psnr_mean"],
                "acceptable": is_acceptable,
                "description": "Peak signal-to-noise ratio",
            }

        # Temporal consistency check (for animations)
        if "temporal_consistency" in metrics:
            temporal_value = metrics["temporal_consistency"]

            # Use context-aware threshold for frame reduction operations
            if frame_reduction_context:
                # Frame reduction operations naturally have lower temporal consistency
                # Use more lenient threshold based on research findings
                temporal_threshold = (
                    0.05  # 5% threshold for frame reduction (vs 10% normal)
                )
            else:
                temporal_threshold = self.catastrophic_thresholds[
                    "min_compressed_temporal_consistency"
                ]

            # Only compare if value is numeric
            is_acceptable = (
                isinstance(temporal_value, int | float)
                and temporal_value >= temporal_threshold
            )
            checks["temporal"] = {
                "value": temporal_value,
                "threshold": temporal_threshold,
                "acceptable": is_acceptable,
                "description": "Temporal consistency",
            }

        return checks

    def validate_quality_variance(
        self, input_path: Path, output_path: Path, wrapper_metadata: dict[str, Any]
    ) -> ValidationResult:
        """Validate quality doesn't vary wildly between frames.

        This can indicate corruption or processing issues affecting
        some frames more than others.
        """
        try:
            # Calculate metrics - the comprehensive metrics should include per-frame data
            # Note: variance validation doesn't need frame reduction context since it's checking variance, not artifacts
            metrics = calculate_comprehensive_metrics(
                input_path, output_path, config=self.metrics_config
            )

            # Look for quality variance indicators in the metrics
            variance_indicators = []

            # Check positional variance if available
            for metric_name in ["ssim", "mse", "psnr"]:
                variance_key = f"{metric_name}_positional_variance"
                if variance_key in metrics:
                    variance = metrics[variance_key]
                    # Only compare if value is numeric
                    is_acceptable = (
                        isinstance(variance, int | float)
                        and variance
                        <= self.catastrophic_thresholds["max_quality_variance"]
                    )
                    variance_indicators.append(
                        {
                            "metric": metric_name,
                            "variance": variance,
                            "acceptable": is_acceptable,
                        }
                    )

            # Overall variance acceptability
            variance_acceptable = (
                all(indicator["acceptable"] for indicator in variance_indicators)
                if variance_indicators
                else True
            )

            return ValidationResult(
                is_valid=variance_acceptable,
                validation_type="quality_variance",
                expected={
                    "max_variance": self.catastrophic_thresholds["max_quality_variance"]
                },
                actual={"variance_indicators": variance_indicators},
                error_message=None
                if variance_acceptable
                else f"High quality variance detected across frames: {[i for i in variance_indicators if not i['acceptable']]}",
                details={
                    "variance_indicators": variance_indicators,
                    "variance_threshold": self.catastrophic_thresholds[
                        "max_quality_variance"
                    ],
                },
            )

        except Exception as e:
            logger.error(f"Quality variance validation error: {e}")
            return ValidationResult(
                is_valid=False,
                validation_type="quality_variance",
                expected="quality_variance_calculation",
                actual="calculation_failed",
                error_message=f"Quality variance validation failed: {str(e)}",
                details={"exception": str(e)},
            )
