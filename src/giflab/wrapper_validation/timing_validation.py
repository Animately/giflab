"""Timing validation for GIF compression operations.

This module implements frame timing validation to detect timing corruption
during compression operations, specifically focusing on frame delay preservation
and timing drift detection using a timing grid approach.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from PIL import Image

from .types import ValidationResult

if TYPE_CHECKING:
    from ..config import ValidationConfig

logger = logging.getLogger(__name__)


@dataclass
class TimingMetrics:
    """Container for timing validation metrics."""

    original_durations: list[int]
    compressed_durations: list[int]
    grid_ms: int
    total_duration_diff_ms: int
    max_timing_drift_ms: int
    timing_drift_score: float
    grid_length_original: int
    grid_length_compressed: int
    alignment_accuracy: float


class TimingGridValidator:
    """Validates frame timing integrity using a timing grid approach.

    This validator ensures that frame delays are preserved correctly during
    compression operations and detects timing drift that could break animation
    smoothness.
    """

    def __init__(self, config: "ValidationConfig | None" = None):
        """Initialize timing grid validator.

        Args:
            config: ValidationConfig instance, or None to use default config
        """
        from ..config import DEFAULT_VALIDATION_CONFIG

        # Use provided config or default
        cfg = config or DEFAULT_VALIDATION_CONFIG

        self.grid_ms = cfg.TIMING_GRID_MS
        self.timing_thresholds = {
            "max_drift_ms": cfg.TIMING_MAX_DRIFT_MS,
            "duration_diff_threshold": cfg.TIMING_DURATION_DIFF_THRESHOLD,
            "alignment_accuracy_min": cfg.TIMING_ALIGNMENT_ACCURACY_MIN,
        }

    def extract_frame_durations(self, gif_path: Path) -> list[int]:
        """Extract individual frame durations from a GIF file.

        Args:
            gif_path: Path to the GIF file

        Returns:
            List of frame durations in milliseconds

        Raises:
            ValueError: If GIF cannot be processed or has invalid frame data
        """
        try:
            durations = []

            with Image.open(gif_path) as img:
                if not hasattr(img, "n_frames") or img.n_frames <= 1:
                    # Single frame GIF - return default duration
                    return [100]  # 100ms default

                total_frames = img.n_frames

                # Extract duration for each frame
                for i in range(total_frames):
                    img.seek(i)
                    # ``info.get("duration", 100)`` only falls back to 100ms when
                    # PIL reports *no* duration at all (key absent). An explicit
                    # per-frame delay of 0 is authored intent ("render as fast as
                    # possible") and is returned verbatim — it must be preserved,
                    # not fabricated up to 100ms. Substituting a synthetic 100ms
                    # for genuine zeros produced phantom total-duration
                    # differences whenever the original and compressed frame
                    # counts diverged (see
                    # [[giflab-timing-validation-zero-delay-fallback]]).
                    duration = img.info.get("duration", 100)  # missing key -> 100ms

                    # Clamp to a sane range. Negative durations are nonsensical
                    # (not authored), so floor at 0; cap the absurd upper bound
                    # at 10s. Genuine 0ms delays pass through untouched.
                    if duration < 0:
                        duration = 0
                    elif duration > 10000:
                        duration = 10000  # Cap at 10 seconds

                    durations.append(duration)

            if not durations:
                durations = [100]  # Fallback for edge cases

            return durations

        except Exception as e:
            logger.error(f"Failed to extract frame durations from {gif_path}: {e}")
            raise ValueError(f"Cannot extract frame durations: {e}") from e

    def align_to_grid(self, duration: int) -> int:
        """Align frame duration to the nearest timing grid point.

        Args:
            duration: Original duration in milliseconds

        Returns:
            Duration aligned to grid in milliseconds
        """
        if self.grid_ms <= 0:
            return duration
        return round(duration / self.grid_ms) * self.grid_ms

    def calculate_grid_length(self, durations: list[int]) -> int:
        """Calculate total animation length in grid units.

        Args:
            durations: List of frame durations in milliseconds

        Returns:
            Total animation length in grid units
        """
        aligned_total = sum(self.align_to_grid(d) for d in durations)
        return aligned_total // self.grid_ms if self.grid_ms > 0 else 0

    def calculate_timing_drift(
        self, original_durations: list[int], compressed_durations: list[int]
    ) -> dict[str, Any]:
        """Calculate timing drift between original and compressed animations.

        Args:
            original_durations: Original frame durations in milliseconds
            compressed_durations: Compressed frame durations in milliseconds

        Returns:
            Dictionary containing timing drift metrics
        """
        # Handle frame count mismatches
        min_frames = min(len(original_durations), len(compressed_durations))
        if min_frames == 0:
            return {
                "max_drift_ms": 0,
                "total_duration_diff_ms": 0,
                "drift_points": [],
                "cumulative_drift": [],
                "frames_analyzed": 0,
            }

        original = original_durations[:min_frames]
        compressed = compressed_durations[:min_frames]

        # Calculate cumulative timing drift
        cumulative_original = 0
        cumulative_compressed = 0
        cumulative_drift = []
        drift_points = []
        max_drift = 0

        for i, (orig_dur, comp_dur) in enumerate(
            zip(original, compressed, strict=True)
        ):
            cumulative_original += orig_dur
            cumulative_compressed += comp_dur

            drift = abs(cumulative_original - cumulative_compressed)
            cumulative_drift.append(drift)

            if drift > max_drift:
                max_drift = drift
                drift_points.append(
                    {
                        "frame_index": i,
                        "drift_ms": drift,
                        "cumulative_original": cumulative_original,
                        "cumulative_compressed": cumulative_compressed,
                    }
                )

        # Calculate total duration difference
        total_original = sum(original_durations)
        total_compressed = sum(compressed_durations)
        total_duration_diff = abs(total_original - total_compressed)

        return {
            "max_drift_ms": max_drift,
            "total_duration_diff_ms": total_duration_diff,
            "drift_points": drift_points[-5:],  # Keep last 5 worst drift points
            "cumulative_drift": cumulative_drift,
            "frames_analyzed": min_frames,
        }

    def calculate_alignment_accuracy(
        self, original_durations: list[int], compressed_durations: list[int]
    ) -> float:
        """Calculate how accurately frame delays are preserved after grid alignment.

        Args:
            original_durations: Original frame durations
            compressed_durations: Compressed frame durations

        Returns:
            Alignment accuracy as a float between 0.0 and 1.0
        """
        if not original_durations or not compressed_durations:
            return 0.0

        min_frames = min(len(original_durations), len(compressed_durations))
        if min_frames == 0:
            return 0.0

        matches = 0
        for i in range(min_frames):
            orig_aligned = self.align_to_grid(original_durations[i])
            comp_aligned = self.align_to_grid(compressed_durations[i])
            if orig_aligned == comp_aligned:
                matches += 1

        return matches / min_frames

    def validate_timing_integrity(
        self, original_gif: Path, compressed_gif: Path
    ) -> ValidationResult:
        """Validate frame timing integrity between original and compressed GIFs.

        Args:
            original_gif: Path to original GIF file
            compressed_gif: Path to compressed GIF file

        Returns:
            ValidationResult containing timing validation results
        """
        try:
            # Extract frame durations
            original_durations = self.extract_frame_durations(original_gif)
            compressed_durations = self.extract_frame_durations(compressed_gif)

            # Calculate timing drift metrics
            drift_metrics = self.calculate_timing_drift(
                original_durations, compressed_durations
            )

            # Calculate grid lengths
            grid_length_original = self.calculate_grid_length(original_durations)
            grid_length_compressed = self.calculate_grid_length(compressed_durations)

            # Calculate alignment accuracy
            alignment_accuracy = self.calculate_alignment_accuracy(
                original_durations, compressed_durations
            )

            # Calculate timing drift score (normalized 0-1, lower is better)
            max_drift = drift_metrics["max_drift_ms"]
            duration_diff = drift_metrics["total_duration_diff_ms"]
            timing_drift_score = min(1.0, (max_drift + duration_diff) / 500.0)

            # Determine if validation passes
            is_valid = (
                max_drift <= self.timing_thresholds["max_drift_ms"]
                and duration_diff <= self.timing_thresholds["duration_diff_threshold"]
                and alignment_accuracy
                >= self.timing_thresholds["alignment_accuracy_min"]
            )

            # Create timing metrics object
            timing_metrics = TimingMetrics(
                original_durations=original_durations,
                compressed_durations=compressed_durations,
                grid_ms=self.grid_ms,
                total_duration_diff_ms=duration_diff,
                max_timing_drift_ms=max_drift,
                timing_drift_score=timing_drift_score,
                grid_length_original=grid_length_original,
                grid_length_compressed=grid_length_compressed,
                alignment_accuracy=alignment_accuracy,
            )

            # Build error message if validation fails
            error_message = None
            if not is_valid:
                error_parts = []
                if max_drift > self.timing_thresholds["max_drift_ms"]:
                    error_parts.append(
                        f"timing drift {max_drift}ms exceeds threshold {self.timing_thresholds['max_drift_ms']}ms"
                    )
                if duration_diff > self.timing_thresholds["duration_diff_threshold"]:
                    error_parts.append(
                        f"duration difference {duration_diff}ms exceeds threshold {self.timing_thresholds['duration_diff_threshold']}ms"
                    )
                if (
                    alignment_accuracy
                    < self.timing_thresholds["alignment_accuracy_min"]
                ):
                    error_parts.append(
                        f"alignment accuracy {alignment_accuracy:.2%} below minimum {self.timing_thresholds['alignment_accuracy_min']:.2%}"
                    )
                error_message = f"Timing validation failed: {', '.join(error_parts)}"

            return ValidationResult(
                is_valid=is_valid,
                validation_type="timing_grid_validation",
                expected={
                    "max_drift_ms": self.timing_thresholds["max_drift_ms"],
                    "duration_diff_threshold": self.timing_thresholds[
                        "duration_diff_threshold"
                    ],
                    "alignment_accuracy_min": self.timing_thresholds[
                        "alignment_accuracy_min"
                    ],
                    "grid_ms": self.grid_ms,
                },
                actual={
                    "max_timing_drift_ms": max_drift,
                    "total_duration_diff_ms": duration_diff,
                    "alignment_accuracy": alignment_accuracy,
                    "timing_drift_score": timing_drift_score,
                },
                error_message=error_message,
                details={
                    "timing_metrics": timing_metrics,
                    "drift_analysis": drift_metrics,
                    "original_frame_count": len(original_durations),
                    "compressed_frame_count": len(compressed_durations),
                    "grid_length_original": grid_length_original,
                    "grid_length_compressed": grid_length_compressed,
                    "thresholds_used": self.timing_thresholds.copy(),
                },
            )

        except OSError as e:
            logger.error(f"Timing validation file access error: {e}")
            return ValidationResult(
                is_valid=False,
                validation_type="timing_file_error",
                expected="successful_file_access",
                actual="file_access_failed",
                error_message=f"Cannot access GIF files for timing validation: {str(e)}",
                details={
                    "exception": str(e),
                    "original_gif": str(original_gif),
                    "compressed_gif": str(compressed_gif),
                    "error_type": "file_access",
                },
            )
        except (ValueError, TypeError, AttributeError) as e:
            logger.error(f"Timing validation calculation error: {e}")
            return ValidationResult(
                is_valid=False,
                validation_type="timing_calculation_error",
                expected="successful_timing_calculation",
                actual="calculation_failed",
                error_message=f"Timing calculation failed: {str(e)}",
                details={
                    "exception": str(e),
                    "original_gif": str(original_gif),
                    "compressed_gif": str(compressed_gif),
                    "error_type": "calculation",
                },
            )
        except Exception as e:
            # Use logging.exception for better tracebacks of unexpected errors
            logger.exception(f"Unexpected timing validation error: {e}")
            return ValidationResult(
                is_valid=False,
                validation_type="timing_validation_error",
                expected="successful_timing_validation",
                actual="validation_exception",
                error_message=f"Timing validation failed with unexpected error: {str(e)}",
                details={
                    "exception": str(e),
                    "original_gif": str(original_gif),
                    "compressed_gif": str(compressed_gif),
                    "error_type": "unexpected",
                },
            )


def validate_frame_timing_for_operation(
    original_gif: Path,
    compressed_gif: Path,
    operation_type: str = "compression",
    config: "ValidationConfig | None" = None,
) -> ValidationResult:
    """Convenience function for timing validation in compression pipelines.

    Args:
        original_gif: Path to original GIF
        compressed_gif: Path to compressed GIF
        operation_type: Type of operation performed (for context)
        config: ValidationConfig instance, or None to use default config

    Returns:
        ValidationResult for timing validation
    """
    validator = TimingGridValidator(config=config)
    result = validator.validate_timing_integrity(original_gif, compressed_gif)

    # Add operation context to the result
    if result.details:
        result.details["operation_type"] = operation_type

    return result


def extract_timing_metrics_for_csv(
    validation_result: ValidationResult,
) -> dict[str, Any]:
    """Extract timing metrics for CSV output from validation result.

    Args:
        validation_result: ValidationResult from timing validation

    Returns:
        Dictionary of timing metrics suitable for CSV output
    """
    if (
        not validation_result.details
        or "timing_metrics" not in validation_result.details
    ):
        # Return default/empty metrics if validation failed or no metrics available.
        #
        # alignment_accuracy is NaN here (not 0.0): on this default branch the
        # alignment was NOT measured (validation failed / no timing metrics), so
        # "could not measure" must be distinguishable from a genuine 0% alignment.
        # A 0.0 sentinel would always read as "fully misaligned" and false-trip
        # the downstream alignment warning (0.0 < 0.98). The genuine-computation
        # path below (and ``calculate_alignment_accuracy``'s empty-list 0.0) are a
        # different, real measurement and are left untouched.
        # ([[giflab-alignment-warning-threshold]])
        return {
            "timing_grid_ms": 0,
            "grid_length": 0,
            "duration_diff_ms": 0,
            "timing_drift_score": 1.0,
            "max_timing_drift_ms": 0,
            "alignment_accuracy": float("nan"),
        }

    timing_metrics = validation_result.details["timing_metrics"]

    return {
        "timing_grid_ms": timing_metrics.grid_ms,
        "grid_length": timing_metrics.grid_length_original,
        "duration_diff_ms": timing_metrics.total_duration_diff_ms,
        "timing_drift_score": timing_metrics.timing_drift_score,
        "max_timing_drift_ms": timing_metrics.max_timing_drift_ms,
        "alignment_accuracy": timing_metrics.alignment_accuracy,
    }
