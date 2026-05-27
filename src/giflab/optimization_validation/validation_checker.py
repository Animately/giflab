"""
Core validation system for GifLab compression pipeline.

Provides comprehensive validation of compression results with content-type awareness,
configurable thresholds, and multi-metric validation combinations.
"""

import logging
import math
import time
from pathlib import Path
from typing import Any


def _is_missing(v: object) -> bool:
    """Return True when *v* is absent or unmeasurable.

    A value is considered missing when it is:
    - ``None`` — the metric was never computed, or
    - ``float("nan")`` — the metric was attempted but failed end-to-end.

    Any real float (including 0.0 and negatives) is NOT missing.  This is the
    canonical predicate used throughout the validation guards so that
    ``any([nan, nan])``-style NaN-truthy bugs cannot arise.
    """
    return v is None or (isinstance(v, float) and math.isnan(v))

from ..meta import GifMetadata
from .config import load_validation_config
from .data_structures import (
    ValidationConfig,
    ValidationIssue,
    ValidationMetrics,
    ValidationResult,
    ValidationStatus,
    ValidationWarning,
)

logger = logging.getLogger(__name__)


class ValidationChecker:
    """
    Pipeline-integrated validation system for automated compression issue detection.

    Validates compression results against configurable thresholds with content-type
    awareness and multi-metric combination detection.
    """

    def __init__(self, config_path: Path | ValidationConfig | None = None):
        """
        Initialize ValidationChecker with configuration.

        Args:
            config_path: Optional path to validation configuration file or ValidationConfig object.
                        If Path, loads config from file. If ValidationConfig, uses directly.
                        If None, uses default configuration with content-type adjustments.
        """
        # Handle both Path and ValidationConfig objects
        if isinstance(config_path, ValidationConfig):
            self.config = config_path
        else:
            self.config = load_validation_config(config_path)
        self.validation_results: dict[str, ValidationResult] = {}

        logger.info("ValidationChecker initialized with configuration")
        logger.debug(f"Base quality floor: {self.config.minimum_quality_floor}")
        logger.debug(
            f"Content-type adjustments: {len(self.config.content_type_adjustments)} types"
        )

    def validate_compression_result(
        self,
        original_metadata: GifMetadata,
        compression_metrics: dict[str, Any],
        pipeline_id: str,
        gif_name: str,
        content_type: str = "unknown",
        pipeline_type: str | None = None,
        **kwargs: Any,
    ) -> ValidationResult:
        """
        Main validation method called during pipeline execution.

        Args:
            original_metadata: GifMetadata from original GIF
            compression_metrics: Dict from calculate_comprehensive_metrics()
            pipeline_id: Identifier of the compression pipeline used
            gif_name: Name of the GIF being validated
            content_type: Content type classification (e.g., 'animation_heavy', 'smooth_gradient')
            pipeline_type: Type of pipeline (e.g., 'frame_reduction', 'color_reduction')

        Returns:
            ValidationResult with comprehensive validation status
        """
        start_time = time.perf_counter()

        # Warn about unexpected keyword arguments (for debugging test isolation issues)
        if kwargs:
            logger.warning(
                f"Unexpected keyword arguments to validate_compression_result: {list(kwargs.keys())}"
            )

        # Get effective thresholds for this content type and pipeline
        effective_thresholds = self.config.get_effective_thresholds(
            content_type, pipeline_type
        )

        logger.debug(
            f"Validating {gif_name} ({content_type}) with pipeline {pipeline_id}"
        )

        # Initialize validation result
        result = ValidationResult(
            pipeline_id=pipeline_id,
            gif_name=gif_name,
            content_type=content_type,
            status=ValidationStatus.PASS,
            effective_thresholds=effective_thresholds,
        )

        # Extract and validate metrics
        try:
            # Build validation metrics summary
            result.metrics = self._extract_validation_metrics(
                original_metadata, compression_metrics
            )

            # Perform individual validation checks
            self._validate_frame_reduction(
                result, original_metadata, compression_metrics, effective_thresholds
            )
            self._validate_fps_consistency(
                result, original_metadata, compression_metrics, effective_thresholds
            )
            self._validate_quality_thresholds(
                result, compression_metrics, effective_thresholds
            )
            self._validate_efficiency_thresholds(
                result, compression_metrics, effective_thresholds
            )
            self._validate_disposal_artifacts(
                result, compression_metrics, effective_thresholds
            )
            self._validate_temporal_consistency(
                result, compression_metrics, effective_thresholds
            )
            self._validate_temporal_artifacts(
                result, compression_metrics, effective_thresholds
            )
            self._validate_deep_perceptual_metrics(
                result, compression_metrics, effective_thresholds
            )
            self._validate_ssimulacra2_metrics(
                result, compression_metrics, effective_thresholds
            )

            # Perform multi-metric combination validation
            self._validate_multi_metric_combinations(
                result, original_metadata, compression_metrics
            )

            # Determine final validation status
            if result.issues:
                # Check for artifact-specific issues
                artifact_categories = {
                    "disposal_artifacts",
                    "temporal_corruption",
                    "animation_corruption",
                }
                has_artifacts = any(
                    issue.category in artifact_categories for issue in result.issues
                )
                result.status = (
                    ValidationStatus.ARTIFACT
                    if has_artifacts
                    else ValidationStatus.ERROR
                )
            elif result.warnings:
                result.status = ValidationStatus.WARNING
            else:
                result.status = ValidationStatus.PASS

        except Exception as e:
            logger.error(f"Validation failed for {gif_name}: {e}")
            result.status = ValidationStatus.UNKNOWN
            result.issues.append(
                ValidationIssue(
                    category="validation_error",
                    message=f"Validation system error: {str(e)}",
                    severity="CRITICAL",
                )
            )

        # Record timing and store result
        end_time = time.perf_counter()
        result.validation_time_ms = int((end_time - start_time) * 1000)

        self.validation_results[f"{gif_name}_{pipeline_id}"] = result

        logger.debug(
            f"Validation completed for {gif_name}: {result.status.value} "
            f"({len(result.issues)} issues, {len(result.warnings)} warnings)"
        )

        return result

    def _extract_validation_metrics(
        self, original_metadata: GifMetadata, compression_metrics: dict[str, Any]
    ) -> ValidationMetrics:
        """Extract key metrics for validation from metadata and compression results."""

        # Calculate FPS deviation
        original_fps = original_metadata.orig_fps
        compressed_fps = compression_metrics.get(
            "orig_fps", original_fps
        )  # Fallback to original if not available

        fps_deviation = None
        if original_fps > 0:
            fps_deviation = abs(compressed_fps - original_fps) / original_fps * 100

        # Calculate frame reduction ratio
        original_frames = original_metadata.orig_frames
        compressed_frames = compression_metrics.get(
            "compressed_frame_count", original_frames
        )
        frame_reduction_ratio = (
            compressed_frames / original_frames if original_frames > 0 else None
        )

        return ValidationMetrics(
            # Frame and timing
            original_frame_count=original_frames,
            compressed_frame_count=compressed_frames,
            original_fps=original_fps,
            compressed_fps=compressed_fps,
            frame_reduction_ratio=frame_reduction_ratio,
            fps_deviation_percent=fps_deviation,
            # Quality
            composite_quality=compression_metrics.get("composite_quality"),
            efficiency=compression_metrics.get("efficiency"),
            compression_ratio=compression_metrics.get("compression_ratio"),
            # Temporal and artifacts
            temporal_consistency_pre=compression_metrics.get(
                "temporal_consistency_pre"
            ),
            temporal_consistency_post=compression_metrics.get(
                "temporal_consistency_post"
            ),
            disposal_artifacts_pre=compression_metrics.get("disposal_artifacts_pre"),
            disposal_artifacts_post=compression_metrics.get("disposal_artifacts_post"),
            disposal_artifacts_delta=compression_metrics.get(
                "disposal_artifacts_delta"
            ),
            # Enhanced temporal artifact metrics (Task 1.2)
            flicker_excess=compression_metrics.get("flicker_excess"),
            flicker_frame_ratio=compression_metrics.get("flicker_frame_ratio"),
            flat_flicker_ratio=compression_metrics.get("flat_flicker_ratio"),
            flat_region_count=compression_metrics.get("flat_region_count"),
            temporal_pumping_score=compression_metrics.get("temporal_pumping_score"),
            quality_oscillation_frequency=compression_metrics.get(
                "quality_oscillation_frequency"
            ),
            lpips_t_mean=compression_metrics.get("lpips_t_mean"),
            lpips_t_p95=compression_metrics.get("lpips_t_p95"),
            # Deep perceptual metrics (Task 2.2)
            lpips_quality_mean=compression_metrics.get("lpips_quality_mean"),
            lpips_quality_p95=compression_metrics.get("lpips_quality_p95"),
            lpips_quality_max=compression_metrics.get("lpips_quality_max"),
            deep_perceptual_frame_count=compression_metrics.get(
                "deep_perceptual_frame_count"
            ),
            deep_perceptual_downscaled=compression_metrics.get(
                "deep_perceptual_downscaled"
            ),
            # Size
            original_size_kb=original_metadata.orig_kilobytes,
            compressed_size_kb=compression_metrics.get("kilobytes"),
            # Phase 3: Text/UI validation metrics
            text_ui_edge_density=compression_metrics.get("text_ui_edge_density"),
            text_ui_component_count=compression_metrics.get("text_ui_component_count"),
            ocr_conf_delta_mean=compression_metrics.get("ocr_conf_delta_mean"),
            ocr_conf_delta_min=compression_metrics.get("ocr_conf_delta_min"),
            ocr_regions_analyzed=compression_metrics.get("ocr_regions_analyzed"),
            mtf50_ratio_mean=compression_metrics.get("mtf50_ratio_mean"),
            mtf50_ratio_min=compression_metrics.get("mtf50_ratio_min"),
            edge_sharpness_score=compression_metrics.get("edge_sharpness_score"),
            has_text_ui_content=compression_metrics.get("has_text_ui_content"),
            # Phase 3: SSIMULACRA2 perceptual quality metrics
            ssimulacra2_mean=compression_metrics.get("ssimulacra2_mean"),
            ssimulacra2_p95=compression_metrics.get("ssimulacra2_p95"),
            ssimulacra2_min=compression_metrics.get("ssimulacra2_min"),
            ssimulacra2_frame_count=compression_metrics.get("ssimulacra2_frame_count"),
            ssimulacra2_triggered=compression_metrics.get("ssimulacra2_triggered"),
        )

    def _validate_frame_reduction(
        self,
        result: ValidationResult,
        original_metadata: GifMetadata,
        compression_metrics: dict[str, Any],
        thresholds: dict[str, float],
    ) -> None:
        """Validate frame count reduction accuracy."""

        original_frames = original_metadata.orig_frames
        compressed_frames = compression_metrics.get(
            "compressed_frame_count", original_frames
        )

        if not original_frames or not compressed_frames:
            result.warnings.append(
                ValidationWarning(
                    category="frame_count",
                    message="Frame count data unavailable for validation",
                )
            )
            return

        actual_ratio = compressed_frames / original_frames
        tolerance = thresholds["frame_reduction_tolerance"]

        # Check if we have an expected ratio from pipeline parameters
        expected_ratio = compression_metrics.get("expected_frame_ratio")

        if expected_ratio and abs(actual_ratio - expected_ratio) > tolerance:
            expected_frames = int(original_frames * expected_ratio)
            result.issues.append(
                ValidationIssue(
                    category="frame_reduction",
                    message=f"Frame count mismatch: Expected {expected_frames} frames "
                    f"({expected_ratio*100:.0f}%), got {compressed_frames} ({actual_ratio*100:.1f}%)",
                    expected_value=expected_frames,
                    actual_value=compressed_frames,
                    threshold=tolerance,
                )
            )

        # Warn about significant frame reduction without explicit expectation
        elif expected_ratio is None and actual_ratio < 0.5:
            result.warnings.append(
                ValidationWarning(
                    category="frame_reduction",
                    message=f"Significant frame reduction detected: {original_frames} → {compressed_frames} "
                    f"({actual_ratio*100:.1f}% retained)",
                    actual_value=actual_ratio,
                    recommendation="Verify frame reduction is intentional",
                )
            )

    def _validate_fps_consistency(
        self,
        result: ValidationResult,
        original_metadata: GifMetadata,
        compression_metrics: dict[str, Any],
        thresholds: dict[str, float],
    ) -> None:
        """Validate FPS consistency between original and compressed GIFs."""

        original_fps = original_metadata.orig_fps
        compressed_fps = compression_metrics.get("orig_fps", original_fps)

        if not original_fps or not compressed_fps or original_fps <= 0:
            result.warnings.append(
                ValidationWarning(
                    category="fps_consistency",
                    message="FPS data unavailable for validation",
                )
            )
            return

        deviation = abs(compressed_fps - original_fps) / original_fps
        error_threshold = thresholds["fps_deviation_tolerance"]
        warning_threshold = thresholds["fps_warning_threshold"]

        if deviation > error_threshold:
            percent_change = (compressed_fps - original_fps) / original_fps * 100
            result.issues.append(
                ValidationIssue(
                    category="fps_consistency",
                    message=f"FPS deviation too large: {original_fps:.1f}fps → {compressed_fps:.1f}fps "
                    f"({percent_change:+.1f}% change)",
                    expected_value=original_fps,
                    actual_value=compressed_fps,
                    threshold=error_threshold,
                )
            )
        elif deviation > warning_threshold:
            percent_change = (compressed_fps - original_fps) / original_fps * 100
            result.warnings.append(
                ValidationWarning(
                    category="fps_consistency",
                    message=f"FPS change detected: {original_fps:.1f}fps → {compressed_fps:.1f}fps "
                    f"({percent_change:+.1f}% change)",
                    expected_value=original_fps,
                    actual_value=compressed_fps,
                    threshold=warning_threshold,
                    recommendation="Verify FPS change is intentional",
                )
            )

    def _validate_quality_thresholds(
        self,
        result: ValidationResult,
        compression_metrics: dict[str, Any],
        thresholds: dict[str, float],
    ) -> None:
        """Validate composite quality against thresholds."""

        composite_quality = compression_metrics.get("composite_quality")

        if _is_missing(composite_quality):
            result.warnings.append(
                ValidationWarning(
                    category="quality_threshold",
                    message="Quality data unavailable for validation",
                )
            )
            return

        minimum_floor = thresholds["minimum_quality_floor"]
        warning_threshold = thresholds["quality_warning_threshold"]

        if composite_quality < minimum_floor:
            result.issues.append(
                ValidationIssue(
                    category="quality_threshold",
                    message=f"Quality below minimum threshold: {composite_quality:.3f} < {minimum_floor:.3f}",
                    expected_value=minimum_floor,
                    actual_value=composite_quality,
                    threshold=minimum_floor,
                )
            )
        elif composite_quality < warning_threshold:
            result.warnings.append(
                ValidationWarning(
                    category="quality_threshold",
                    message=f"Quality below warning threshold: {composite_quality:.3f} < {warning_threshold:.3f}",
                    expected_value=warning_threshold,
                    actual_value=composite_quality,
                    threshold=warning_threshold,
                    recommendation="Consider adjusting compression parameters for better quality",
                )
            )

    def _validate_efficiency_thresholds(
        self,
        result: ValidationResult,
        compression_metrics: dict[str, Any],
        thresholds: dict[str, float],
    ) -> None:
        """Validate efficiency metrics against thresholds."""

        efficiency = compression_metrics.get("efficiency")

        if _is_missing(efficiency):
            result.warnings.append(
                ValidationWarning(
                    category="efficiency_threshold",
                    message="Efficiency data unavailable for validation",
                )
            )
            return

        # Check for unreasonably high efficiency (may indicate calculation errors)
        max_reasonable_efficiency = 0.95
        if efficiency > max_reasonable_efficiency:
            result.warnings.append(
                ValidationWarning(
                    category="efficiency_threshold",
                    message=f"Efficiency unexpectedly high: {efficiency:.3f} > {max_reasonable_efficiency:.3f}",
                    expected_value=max_reasonable_efficiency,
                    actual_value=efficiency,
                    threshold=max_reasonable_efficiency,
                    recommendation="Verify quality metrics and compression calculations are correct",
                )
            )

        # Note: No minimum efficiency validation - efficiency is purely informational

    def _validate_disposal_artifacts(
        self,
        result: ValidationResult,
        compression_metrics: dict[str, Any],
        thresholds: dict[str, float],
    ) -> None:
        """Validate disposal artifact detection scores."""

        disposal_pre = compression_metrics.get("disposal_artifacts_pre")
        disposal_post = compression_metrics.get("disposal_artifacts_post")
        disposal_delta = compression_metrics.get("disposal_artifacts_delta")

        if _is_missing(disposal_pre) or _is_missing(disposal_post):
            result.warnings.append(
                ValidationWarning(
                    category="disposal_artifacts",
                    message="Disposal artifact data unavailable for validation",
                )
            )
            return

        artifact_threshold = thresholds["disposal_artifact_threshold"]
        delta_threshold = thresholds["disposal_artifact_delta_threshold"]

        if (
            disposal_pre < artifact_threshold
            or disposal_post < artifact_threshold
            or (not _is_missing(disposal_delta) and abs(disposal_delta) > delta_threshold)
        ):
            result.issues.append(
                ValidationIssue(
                    category="disposal_artifacts",
                    message=f"Disposal artifacts detected: Pre={disposal_pre:.3f}, Post={disposal_post:.3f}"
                    f"{f', Delta={disposal_delta:.3f}' if disposal_delta is not None else ''}",
                    expected_value=artifact_threshold,
                    actual_value=min(disposal_pre, disposal_post),
                    threshold=artifact_threshold,
                )
            )

    def _validate_temporal_consistency(
        self,
        result: ValidationResult,
        compression_metrics: dict[str, Any],
        thresholds: dict[str, float],
    ) -> None:
        """Validate temporal consistency scores."""

        temporal_score = compression_metrics.get("temporal_consistency_post")

        if _is_missing(temporal_score):
            result.warnings.append(
                ValidationWarning(
                    category="temporal_consistency",
                    message="Temporal consistency data unavailable for validation",
                )
            )
            return

        threshold = thresholds["temporal_consistency_threshold"]

        if temporal_score < threshold:
            result.warnings.append(
                ValidationWarning(
                    category="temporal_consistency",
                    message=f"Low temporal consistency: {temporal_score:.3f} < {threshold:.3f}",
                    expected_value=threshold,
                    actual_value=temporal_score,
                    threshold=threshold,
                    recommendation="Consider using different compression parameters to preserve temporal consistency",
                )
            )

    def _validate_multi_metric_combinations(
        self,
        result: ValidationResult,
        original_metadata: GifMetadata,
        compression_metrics: dict[str, Any],
    ) -> None:
        """Validate complex multi-metric combinations for advanced issue detection."""

        # Combination 1: Low quality + High compression ratio = Suspicious efficiency
        composite_quality = compression_metrics.get("composite_quality")
        compression_ratio = compression_metrics.get("compression_ratio")

        if composite_quality and compression_ratio:
            quality_compression_ratio = composite_quality / min(
                compression_ratio / 10, 1
            )
            if quality_compression_ratio < 0.6:
                result.warnings.append(
                    ValidationWarning(
                        category="quality_compression_mismatch",
                        message=f"Suspicious quality/compression ratio: Quality={composite_quality:.3f}, "
                        f"Compression={compression_ratio:.1f}x",
                        recommendation="Review compression parameters - quality may be unnecessarily low",
                    )
                )

        # Combination 2: Significant FPS change + No frame reduction = Timing issue
        original_fps = original_metadata.orig_fps
        compressed_fps = compression_metrics.get("orig_fps", original_fps)
        original_frames = original_metadata.orig_frames
        compressed_frames = compression_metrics.get(
            "compressed_frame_count", original_frames
        )

        if (
            original_fps
            and compressed_fps
            and original_frames
            and compressed_frames
            and original_frames == compressed_frames
        ):  # No frame reduction
            fps_change = (
                abs(compressed_fps - original_fps) / original_fps
                if original_fps > 0
                else 0
            )
            if fps_change > 0.1:  # 10% FPS change without frame reduction
                result.issues.append(
                    ValidationIssue(
                        category="fps_timing_issue",
                        message=f"FPS changed without frame reduction: {original_fps:.1f}fps → {compressed_fps:.1f}fps",
                        expected_value=original_fps,
                        actual_value=compressed_fps,
                    )
                )

        # Combination 3: High disposal artifacts + Low temporal consistency = Animation corruption
        disposal_pre = compression_metrics.get("disposal_artifacts_pre")
        temporal_consistency = compression_metrics.get("temporal_consistency_post")

        if disposal_pre and temporal_consistency:
            if disposal_pre < 0.8 and temporal_consistency < 0.7:
                result.issues.append(
                    ValidationIssue(
                        category="animation_corruption",
                        message=f"Animation corruption detected: Poor disposal artifacts ({disposal_pre:.3f}) + "
                        f"Low temporal consistency ({temporal_consistency:.3f})",
                        severity="CRITICAL",
                    )
                )

        # Combination 4: Extreme frame reduction + Poor quality = Over-compression
        if original_frames and compressed_frames and composite_quality:
            frame_reduction = compressed_frames / original_frames
            if frame_reduction < 0.3 and composite_quality < 0.6:
                result.warnings.append(
                    ValidationWarning(
                        category="over_compression",
                        message=f"Possible over-compression: {frame_reduction*100:.0f}% frames retained, "
                        f"{composite_quality*100:.0f}% quality",
                        recommendation="Consider reducing compression aggressiveness",
                    )
                )

    def _validate_temporal_artifacts(
        self,
        result: ValidationResult,
        compression_metrics: dict[str, Any],
        thresholds: dict[str, float],
    ) -> None:
        """Validate enhanced temporal artifact detection scores from Task 1.2.

        This validates the new temporal artifact metrics including flicker excess,
        flat-region flicker, and temporal pumping detection.
        """
        # Check if temporal artifact metrics are available
        flicker_excess = compression_metrics.get("flicker_excess")
        flat_flicker_ratio = compression_metrics.get("flat_flicker_ratio")
        temporal_pumping = compression_metrics.get("temporal_pumping_score")
        lpips_t_mean = compression_metrics.get("lpips_t_mean")

        if all(
            _is_missing(v) for v in (flicker_excess, flat_flicker_ratio, temporal_pumping, lpips_t_mean)
        ):
            result.warnings.append(
                ValidationWarning(
                    category="temporal_artifacts",
                    message="Enhanced temporal artifact data unavailable for validation",
                )
            )
            return

        # Validate flicker excess
        flicker_threshold = thresholds.get("flicker_excess_threshold", 0.02)
        if not _is_missing(flicker_excess) and flicker_excess > flicker_threshold:
            result.issues.append(
                ValidationIssue(
                    category="flicker_excess",
                    message=f"Excessive flicker detected: {flicker_excess:.4f} > {flicker_threshold:.4f}",
                    expected_value=flicker_threshold,
                    actual_value=flicker_excess,
                    threshold=flicker_threshold,
                    severity="WARNING",
                )
            )

        # Validate flat region flicker
        flat_flicker_threshold = thresholds.get("flat_flicker_ratio_threshold", 0.1)
        if (
            not _is_missing(flat_flicker_ratio)
            and flat_flicker_ratio > flat_flicker_threshold
        ):
            result.issues.append(
                ValidationIssue(
                    category="background_flicker",
                    message=f"Background flicker detected in {flat_flicker_ratio*100:.1f}% of stable regions - check disposal methods",
                    expected_value=flat_flicker_threshold,
                    actual_value=flat_flicker_ratio,
                    threshold=flat_flicker_threshold,
                )
            )

        # Validate temporal pumping
        pumping_threshold = thresholds.get("temporal_pumping_threshold", 0.15)
        if not _is_missing(temporal_pumping) and temporal_pumping > pumping_threshold:
            result.issues.append(
                ValidationIssue(
                    category="temporal_pumping",
                    message=f"Temporal pumping detected: {temporal_pumping:.3f} > {pumping_threshold:.3f} - quality oscillation creates 'breathing' effect",
                    expected_value=pumping_threshold,
                    actual_value=temporal_pumping,
                    threshold=pumping_threshold,
                )
            )

        # Validate LPIPS-T for perceptual temporal consistency
        lpips_threshold = thresholds.get("lpips_t_threshold", 0.05)
        if not _is_missing(lpips_t_mean) and lpips_t_mean > lpips_threshold:
            result.issues.append(
                ValidationIssue(
                    category="perceptual_temporal_degradation",
                    message=f"Poor temporal consistency detected: {lpips_t_mean:.4f} > {lpips_threshold:.4f} - check for compression artifacts",
                    expected_value=lpips_threshold,
                    actual_value=lpips_t_mean,
                    threshold=lpips_threshold,
                    severity="WARNING",
                )
            )

    def _validate_deep_perceptual_metrics(
        self,
        result: ValidationResult,
        compression_metrics: dict[str, Any],
        thresholds: dict[str, float],
    ) -> None:
        """Validate deep perceptual quality metrics from Task 2.2.

        This validates the new LPIPS-based spatial quality metrics that catch
        perceptual issues traditional metrics miss.
        """
        # Check if deep perceptual metrics are available
        lpips_quality_mean = compression_metrics.get("lpips_quality_mean")
        lpips_quality_p95 = compression_metrics.get("lpips_quality_p95")
        lpips_quality_max = compression_metrics.get("lpips_quality_max")
        deep_perceptual_used = (
            compression_metrics.get("deep_perceptual_device", "fallback") != "fallback"
        )

        all_lpips_missing = all(
            _is_missing(v) for v in (lpips_quality_mean, lpips_quality_p95, lpips_quality_max)
        )
        if all_lpips_missing or not deep_perceptual_used:
            if not deep_perceptual_used:
                # Deep perceptual not attempted — only warn for borderline quality.
                composite_quality = compression_metrics.get("composite_quality")
                if (
                    not _is_missing(composite_quality) and 0.3 <= composite_quality <= 0.7
                ):  # Borderline quality
                    result.warnings.append(
                        ValidationWarning(
                            category="deep_perceptual_unavailable",
                            message="Deep perceptual metrics unavailable for borderline quality case",
                            recommendation="Consider investigating why LPIPS calculation failed",
                        )
                    )
            else:
                # deep_perceptual_used=True but all scores missing: measurement failed.
                # Always warn regardless of composite_quality.
                result.warnings.append(
                    ValidationWarning(
                        category="deep_perceptual_unavailable",
                        message="Deep perceptual (LPIPS) metrics unavailable despite device being active — measurement failed end-to-end",
                        recommendation="Consider investigating why LPIPS calculation failed",
                    )
                )
            return

        # Validate LPIPS quality - higher scores indicate more perceptual difference (worse quality)
        lpips_threshold = thresholds.get("lpips_quality_threshold", 0.3)

        if not _is_missing(lpips_quality_mean) and lpips_quality_mean > lpips_threshold:
            result.issues.append(
                ValidationIssue(
                    category="perceptual_quality_degradation",
                    message=f"Poor perceptual quality detected: LPIPS {lpips_quality_mean:.4f} > {lpips_threshold:.4f} - significant visual differences detected",
                    expected_value=lpips_threshold,
                    actual_value=lpips_quality_mean,
                    threshold=lpips_threshold,
                    severity="WARNING",
                )
            )

        # Validate extreme cases with p95 metric
        extreme_threshold = thresholds.get("lpips_quality_extreme_threshold", 0.5)
        if not _is_missing(lpips_quality_p95) and lpips_quality_p95 > extreme_threshold:
            result.issues.append(
                ValidationIssue(
                    category="extreme_perceptual_degradation",
                    message=f"Extreme perceptual degradation in worst frames: LPIPS P95 {lpips_quality_p95:.4f} > {extreme_threshold:.4f}",
                    expected_value=extreme_threshold,
                    actual_value=lpips_quality_p95,
                    threshold=extreme_threshold,
                    severity="CRITICAL",
                )
            )

        # Check for very high maximum values that indicate severe artifacts
        max_threshold = thresholds.get("lpips_quality_max_threshold", 0.7)
        if not _is_missing(lpips_quality_max) and lpips_quality_max > max_threshold:
            result.warnings.append(
                ValidationWarning(
                    category="severe_perceptual_artifacts",
                    message=f"Severe artifacts in worst frame: LPIPS max {lpips_quality_max:.4f} > {max_threshold:.4f}",
                    expected_value=max_threshold,
                    actual_value=lpips_quality_max,
                    threshold=max_threshold,
                    recommendation="Check for compression artifacts or frame corruption",
                )
            )

        # Combination check: Low composite quality + High LPIPS = Comprehensive quality degradation
        composite_quality = compression_metrics.get("composite_quality")
        if (
            not _is_missing(composite_quality)
            and not _is_missing(lpips_quality_mean)
            and composite_quality < 0.6
            and lpips_quality_mean > 0.2
        ):
            result.issues.append(
                ValidationIssue(
                    category="comprehensive_quality_failure",
                    message=f"Both traditional and perceptual quality poor: Composite {composite_quality:.3f} + LPIPS {lpips_quality_mean:.3f}",
                    severity="CRITICAL",
                )
            )

    def _validate_ssimulacra2_metrics(
        self,
        result: ValidationResult,
        compression_metrics: dict[str, Any],
        thresholds: dict[str, float],
    ) -> None:
        """Validate SSIMULACRA2 perceptual quality metrics from Task 3.2.

        This validates the modern SSIMULACRA2 perceptual quality metric that provides
        good human visual alignment for borderline quality cases.
        """
        # Check if SSIMULACRA2 metrics are available
        ssimulacra2_mean = compression_metrics.get("ssimulacra2_mean")
        ssimulacra2_p95 = compression_metrics.get("ssimulacra2_p95")
        ssimulacra2_min = compression_metrics.get("ssimulacra2_min")
        ssimulacra2_triggered = compression_metrics.get("ssimulacra2_triggered", 0.0)

        all_scores_missing = all(
            _is_missing(v) for v in (ssimulacra2_mean, ssimulacra2_p95, ssimulacra2_min)
        )
        if all_scores_missing or ssimulacra2_triggered == 0.0:
            if ssimulacra2_triggered == 0.0:
                # Explicitly disabled — skip silently.
                return
            # triggered=1.0 but all scores missing: measurement failed end-to-end.
            # Always warn; the original borderline-only guard was too narrow and left
            # higher-quality GIFs silently unchecked when ssimulacra2 crashed.
            result.warnings.append(
                ValidationWarning(
                    category="ssimulacra2_unavailable",
                    message="SSIMULACRA2 metrics unavailable despite being triggered — measurement failed end-to-end",
                    recommendation="Check if ssimulacra2 binary is installed and accessible",
                )
            )
            return

        # Validate SSIMULACRA2 quality - scores are normalized (0-1, higher = better quality)
        low_threshold = thresholds.get("ssimulacra2_low_threshold", 0.3)  # Low quality
        medium_threshold = thresholds.get(
            "ssimulacra2_threshold", 0.5
        )  # Medium quality
        thresholds.get("ssimulacra2_high_threshold", 0.7)  # High quality

        # Check mean quality
        if not _is_missing(ssimulacra2_mean) and ssimulacra2_mean < low_threshold:
            result.issues.append(
                ValidationIssue(
                    category="ssimulacra2_poor_quality",
                    message=f"Poor SSIMULACRA2 quality: {ssimulacra2_mean:.3f} < {low_threshold:.3f} - significant perceptual degradation",
                    expected_value=low_threshold,
                    actual_value=ssimulacra2_mean,
                    threshold=low_threshold,
                    severity="CRITICAL",
                )
            )
        elif not _is_missing(ssimulacra2_mean) and ssimulacra2_mean < medium_threshold:
            result.warnings.append(
                ValidationWarning(
                    category="ssimulacra2_medium_quality",
                    message=f"Medium SSIMULACRA2 quality: {ssimulacra2_mean:.3f} < {medium_threshold:.3f} - noticeable perceptual differences",
                    expected_value=medium_threshold,
                    actual_value=ssimulacra2_mean,
                    threshold=medium_threshold,
                    recommendation="Consider less aggressive compression settings",
                )
            )

        # Check worst case (minimum) quality
        if not _is_missing(ssimulacra2_min) and ssimulacra2_min < low_threshold:
            result.issues.append(
                ValidationIssue(
                    category="ssimulacra2_worst_frame",
                    message=f"Worst frame SSIMULACRA2 quality very poor: {ssimulacra2_min:.3f} < {low_threshold:.3f}",
                    expected_value=low_threshold,
                    actual_value=ssimulacra2_min,
                    threshold=low_threshold,
                    severity="WARNING",
                )
            )

        # Check 95th percentile quality
        if not _is_missing(ssimulacra2_p95) and ssimulacra2_p95 < medium_threshold:
            result.warnings.append(
                ValidationWarning(
                    category="ssimulacra2_poor_consistency",
                    message=f"Inconsistent SSIMULACRA2 quality: P95 {ssimulacra2_p95:.3f} < {medium_threshold:.3f}",
                    expected_value=medium_threshold,
                    actual_value=ssimulacra2_p95,
                    threshold=medium_threshold,
                    recommendation="Check for frame-specific compression artifacts",
                )
            )

        # Combination check: SSIMULACRA2 disagrees with composite quality
        composite_quality = compression_metrics.get("composite_quality")
        if (
            not _is_missing(composite_quality)
            and not _is_missing(ssimulacra2_mean)
            and composite_quality > 0.7  # High composite quality
            and ssimulacra2_mean < medium_threshold  # But poor perceptual quality
        ):
            result.warnings.append(
                ValidationWarning(
                    category="quality_metric_disagreement",
                    message=f"Quality metrics disagree: Composite {composite_quality:.3f} high but SSIMULACRA2 {ssimulacra2_mean:.3f} poor",
                    recommendation="Review compression with focus on perceptual quality",
                )
            )

    def get_validation_result(
        self, gif_name: str, pipeline_id: str
    ) -> ValidationResult | None:
        """Get validation result for a specific GIF and pipeline combination."""
        key = f"{gif_name}_{pipeline_id}"
        return self.validation_results.get(key)

    def get_validation_summary(self) -> dict[str, Any]:
        """Get summary statistics of all validation results."""
        if not self.validation_results:
            return {"total_validations": 0}

        status_counts = {status.value: 0 for status in ValidationStatus}
        total_issues = 0
        total_warnings = 0

        for result in self.validation_results.values():
            status_counts[result.status.value] += 1
            total_issues += len(result.issues)
            total_warnings += len(result.warnings)

        return {
            "total_validations": len(self.validation_results),
            "status_breakdown": status_counts,
            "total_issues": total_issues,
            "total_warnings": total_warnings,
            "pass_rate": status_counts[ValidationStatus.PASS.value]
            / len(self.validation_results)
            * 100,
        }

    def clear_results(self) -> None:
        """Clear all stored validation results."""
        self.validation_results.clear()
        logger.info("Validation results cleared")
