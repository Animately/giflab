"""
Data structures for GifLab validation system.

Defines the core data types used for validation results, statuses, and configuration.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class ValidationStatus(Enum):
    """Validation status levels for compression results."""

    PASS = "PASS"  # All validation checks passed
    WARNING = "WARNING"  # Minor issues detected, compression acceptable
    ERROR = "ERROR"  # Significant issues, compression problematic
    ARTIFACT = "ARTIFACT"  # Disposal artifacts or corruption detected
    UNKNOWN = "UNKNOWN"  # Validation could not be performed


@dataclass
class ValidationIssue:
    """Represents a validation error or critical issue."""

    category: str  # e.g., "fps_consistency", "quality_threshold", "frame_count"
    message: str  # Human-readable description
    expected_value: Any | None = None  # What was expected
    actual_value: Any | None = None  # What was measured
    threshold: float | None = None  # Threshold that was violated
    severity: str = "ERROR"  # ERROR, CRITICAL


@dataclass
class ValidationWarning:
    """Represents a validation warning or minor issue."""

    category: str  # e.g., "efficiency_suboptimal", "quality_below_ideal"
    message: str  # Human-readable description
    expected_value: Any | None = None
    actual_value: Any | None = None
    threshold: float | None = None
    recommendation: str | None = None  # Suggested improvement


@dataclass
class ValidationMetrics:
    """Summary of key metrics used in validation.

    Includes metrics from all validation phases:
    - Phase 1: Frame/timing, quality, temporal consistency
    - Phase 2: Deep perceptual quality (LPIPS)
    - Phase 3: Text/UI content validation and sharpness analysis
    """

    # Frame and timing metrics
    original_frame_count: int
    compressed_frame_count: int
    original_fps: float
    compressed_fps: float
    frame_reduction_ratio: float | None = None
    fps_deviation_percent: float | None = None

    # Quality metrics
    composite_quality: float | None = None
    efficiency: float | None = None
    compression_ratio: float | None = None

    # Temporal and artifact metrics
    temporal_consistency_pre: float | None = None
    temporal_consistency_post: float | None = None
    disposal_artifacts_pre: float | None = None
    disposal_artifacts_post: float | None = None
    disposal_artifacts_delta: float | None = None

    # Enhanced temporal artifact metrics (Task 1.2)
    flicker_excess: float | None = None
    flicker_frame_ratio: float | None = None
    flat_flicker_ratio: float | None = None
    flat_region_count: int | None = None
    temporal_pumping_score: float | None = None
    quality_oscillation_frequency: float | None = None
    lpips_t_mean: float | None = None
    lpips_t_p95: float | None = None

    # Deep perceptual metrics (Task 2.2)
    lpips_quality_mean: float | None = None
    lpips_quality_p95: float | None = None
    lpips_quality_max: float | None = None
    deep_perceptual_frame_count: int | None = None
    deep_perceptual_downscaled: float | None = None

    # Size metrics
    original_size_kb: float | None = None
    compressed_size_kb: float | None = None

    # Phase 3: Text and UI content validation metrics
    text_ui_edge_density: float | None = (
        None  # Edge density measurement for UI detection
    )
    text_ui_component_count: int | None = None  # Number of UI components detected
    ocr_conf_delta_mean: float | None = None  # Mean OCR confidence change
    ocr_conf_delta_min: float | None = None  # Minimum OCR confidence change
    ocr_regions_analyzed: int | None = None  # Number of OCR regions processed
    mtf50_ratio_mean: float | None = None  # Mean MTF50 sharpness ratio
    mtf50_ratio_min: float | None = None  # Minimum MTF50 sharpness ratio
    edge_sharpness_score: float | None = None  # Overall edge sharpness score
    has_text_ui_content: bool | None = None  # Boolean flag for UI content presence

    # Phase 3: SSIMULACRA2 perceptual quality metrics
    ssimulacra2_mean: float | None = None  # Mean SSIMULACRA2 quality score
    ssimulacra2_p95: float | None = None  # 95th percentile SSIMULACRA2 score
    ssimulacra2_min: float | None = None  # Minimum SSIMULACRA2 score
    ssimulacra2_frame_count: float | None = None  # Number of frames processed
    ssimulacra2_triggered: float | None = None  # Boolean flag for SSIMULACRA2 usage


@dataclass
class ValidationResult:
    """Complete validation result for a compression operation."""

    # Identification
    pipeline_id: str
    gif_name: str
    content_type: str

    # Overall validation status
    status: ValidationStatus

    # Issues and warnings
    issues: list[ValidationIssue] = field(default_factory=list)
    warnings: list[ValidationWarning] = field(default_factory=list)

    # Metrics summary
    metrics: ValidationMetrics | None = None

    # Validation configuration used
    effective_thresholds: dict[str, Any] = field(default_factory=dict)

    # Execution metadata
    validation_time_ms: int | None = None
    config_version: str | None = None

    def has_errors(self) -> bool:
        """Check if validation found any errors."""
        return self.status in (ValidationStatus.ERROR, ValidationStatus.ARTIFACT)

    def has_warnings(self) -> bool:
        """Check if validation found any warnings."""
        return len(self.warnings) > 0 or self.status == ValidationStatus.WARNING

    def is_acceptable(self) -> bool:
        """Check if compression result is acceptable despite any warnings."""
        return self.status in (ValidationStatus.PASS, ValidationStatus.WARNING)

    def get_summary(self) -> str:
        """Get a brief summary of validation results."""
        status_emoji = {
            ValidationStatus.PASS: "🟢",
            ValidationStatus.WARNING: "🟡",
            ValidationStatus.ERROR: "🔴",
            ValidationStatus.ARTIFACT: "⚠️",
            ValidationStatus.UNKNOWN: "❓",
        }

        emoji = status_emoji.get(self.status, "❓")
        error_count = len(self.issues)
        warning_count = len(self.warnings)

        summary = f"{emoji} {self.status.value}"
        if error_count > 0:
            summary += f" ({error_count} errors"
            if warning_count > 0:
                summary += f", {warning_count} warnings"
            summary += ")"
        elif warning_count > 0:
            summary += f" ({warning_count} warnings)"

        return summary

    def get_detailed_report(self) -> str:
        """Get a detailed validation report for terminal output."""
        lines = []
        lines.append(f"Validation Report: {self.gif_name} ({self.content_type})")
        lines.append(f"Pipeline: {self.pipeline_id}")
        lines.append(f"Status: {self.get_summary()}")
        lines.append("")

        if self.metrics:
            lines.append("Key Metrics:")
            lines.append(
                f"  Frames: {self.metrics.original_frame_count} → {self.metrics.compressed_frame_count}"
            )
            lines.append(
                f"  FPS: {self.metrics.original_fps:.1f} → {self.metrics.compressed_fps:.1f}"
            )
            if self.metrics.composite_quality is not None:
                lines.append(f"  Quality: {self.metrics.composite_quality:.3f}")
            if self.metrics.efficiency is not None:
                lines.append(f"  Efficiency: {self.metrics.efficiency:.3f}")
            lines.append("")

        if self.issues:
            lines.append("Issues:")
            for issue in self.issues:
                lines.append(f"  ❌ {issue.category}: {issue.message}")
            lines.append("")

        if self.warnings:
            lines.append("Warnings:")
            for warning in self.warnings:
                lines.append(f"  ⚠️  {warning.category}: {warning.message}")
                if warning.recommendation:
                    lines.append(f"     Suggestion: {warning.recommendation}")
            lines.append("")

        return "\n".join(lines)


@dataclass
class ValidationConfig:
    """Configuration for validation thresholds and rules."""

    # Frame validation
    frame_reduction_tolerance: float = 0.1  # 10% tolerance for frame reduction accuracy

    # FPS validation
    fps_deviation_tolerance: float = 0.1  # 10% tolerance for FPS changes
    fps_warning_threshold: float = 0.05  # 5% warning threshold for FPS changes

    # Quality thresholds
    minimum_quality_floor: float = 0.7  # Minimum acceptable composite quality
    quality_warning_threshold: float = 0.8  # Warning threshold for quality

    # Note: No efficiency thresholds - efficiency is purely informational

    # Disposal artifact thresholds
    disposal_artifact_threshold: float = 0.85  # Minimum acceptable disposal score
    disposal_artifact_delta_threshold: float = 0.1  # Max change in disposal score

    # Temporal consistency thresholds
    temporal_consistency_threshold: float = 0.75  # Minimum temporal consistency

    # Frame-drop alignment threshold ([[giflab-alignment-warning-threshold]]).
    # Drives the validation-checker ``alignment_uncertain`` warning rule.
    # Deliberately separate from MetricsConfig.ALIGNMENT_WARNING_THRESHOLD
    # (which drives the metrics-pipeline ``alignment_warning`` key); both
    # default 0.98 and live in their natural homes.
    alignment_warning_threshold: float = 0.98  # Minimum frame-drop alignment

    # Enhanced temporal artifact thresholds (Task 1.2)
    flicker_excess_threshold: float = 0.02  # Maximum acceptable flicker excess
    flat_flicker_ratio_threshold: float = 0.1  # Maximum flicker in stable regions
    temporal_pumping_threshold: float = 0.15  # Maximum temporal quality oscillation
    lpips_t_threshold: float = 0.05  # LPIPS temporal threshold

    # Deep perceptual quality thresholds (Task 2.2)
    lpips_quality_threshold: float = 0.3  # LPIPS spatial quality threshold
    lpips_quality_extreme_threshold: float = 0.5  # Extreme LPIPS P95 threshold
    lpips_quality_max_threshold: float = 0.7  # Maximum LPIPS threshold

    # SSIMULACRA2 perceptual quality thresholds (Task 3.2)
    ssimulacra2_threshold: float = 0.5  # Medium quality threshold (normalized score)
    ssimulacra2_high_threshold: float = 0.7  # High quality threshold
    ssimulacra2_low_threshold: float = 0.3  # Low quality threshold

    # Content-type specific adjustments
    content_type_adjustments: dict[str, dict[str, float]] = field(default_factory=dict)

    # Pipeline-specific rules
    pipeline_specific_rules: dict[str, dict[str, float]] = field(default_factory=dict)

    @classmethod
    def load_from_file(cls, config_path: Path) -> "ValidationConfig":
        """Load validation configuration from YAML file."""
        import yaml

        with open(config_path) as f:
            config_data = yaml.safe_load(f)

        return cls(**config_data)

    def get_effective_thresholds(
        self, content_type: str | None = None, pipeline_type: str | None = None
    ) -> dict[str, float]:
        """Get effective thresholds with content-type and pipeline adjustments."""
        # Start with base thresholds
        thresholds = {
            "frame_reduction_tolerance": self.frame_reduction_tolerance,
            "fps_deviation_tolerance": self.fps_deviation_tolerance,
            "fps_warning_threshold": self.fps_warning_threshold,
            "minimum_quality_floor": self.minimum_quality_floor,
            "quality_warning_threshold": self.quality_warning_threshold,
            "disposal_artifact_threshold": self.disposal_artifact_threshold,
            "disposal_artifact_delta_threshold": self.disposal_artifact_delta_threshold,
            "temporal_consistency_threshold": self.temporal_consistency_threshold,
            "alignment_warning_threshold": self.alignment_warning_threshold,
            # Enhanced temporal artifact thresholds (Task 1.2)
            "flicker_excess_threshold": self.flicker_excess_threshold,
            "flat_flicker_ratio_threshold": self.flat_flicker_ratio_threshold,
            "temporal_pumping_threshold": self.temporal_pumping_threshold,
            "lpips_t_threshold": self.lpips_t_threshold,
            # Deep perceptual quality thresholds (Task 2.2)
            "lpips_quality_threshold": self.lpips_quality_threshold,
            "lpips_quality_extreme_threshold": self.lpips_quality_extreme_threshold,
            "lpips_quality_max_threshold": self.lpips_quality_max_threshold,
            # SSIMULACRA2 perceptual quality thresholds (Task 3.2)
            "ssimulacra2_threshold": self.ssimulacra2_threshold,
            "ssimulacra2_high_threshold": self.ssimulacra2_high_threshold,
            "ssimulacra2_low_threshold": self.ssimulacra2_low_threshold,
        }

        # Apply content-type specific adjustments
        if content_type and content_type in self.content_type_adjustments:
            adjustments = self.content_type_adjustments[content_type]
            thresholds.update(adjustments)

        # Apply pipeline-specific rules
        if pipeline_type and pipeline_type in self.pipeline_specific_rules:
            rules = self.pipeline_specific_rules[pipeline_type]
            thresholds.update(rules)

        return thresholds
