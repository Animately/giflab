"""SSIMULACRA2 perceptual quality metrics for GIF compression validation.

This module provides integration with the SSIMULACRA2 CLI tool for modern
perceptual quality assessment. SSIMULACRA2 is used conditionally for borderline
quality cases to provide additional validation beyond traditional metrics.

SSIMULACRA2 returns scores in the range -inf..100:
- 90+: Very high quality (indistinguishable from original)
- 70: High quality
- 50: Medium quality
- 30: Low quality
- <30: Poor quality
"""

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .config import MetricsConfig
from .metrics import extract_gif_frames

logger = logging.getLogger(__name__)

# Default SSIMULACRA2 binary path
DEFAULT_SSIMULACRA2_PATH = "/opt/homebrew/bin/ssimulacra2"

# Score normalization constants
SSIMULACRA2_EXCELLENT_SCORE = 90.0  # Maps to 1.0
SSIMULACRA2_POOR_SCORE = 10.0  # Maps to 0.0


class Ssimulacra2Validator:
    """SSIMULACRA2 perceptual quality validator for GIF compression."""

    def __init__(self, binary_path: str = DEFAULT_SSIMULACRA2_PATH):
        """Initialize SSIMULACRA2 validator.

        Args:
            binary_path: Path to ssimulacra2 binary
        """
        self.binary_path = Path(binary_path)

    def is_available(self) -> bool:
        """Check if SSIMULACRA2 binary is available."""
        try:
            subprocess.run(
                [str(self.binary_path), "--help"], capture_output=True, timeout=5
            )
            # SSIMULACRA2 returns non-zero even for --help, check if binary exists
            return self.binary_path.exists() and self.binary_path.is_file()
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            return False

    def should_use_ssimulacra2(self, composite_quality: float | None) -> bool:
        """Determine if SSIMULACRA2 should be calculated based on quality.

        Args:
            composite_quality: Current composite quality score (0-1)

        Returns:
            True if SSIMULACRA2 should be calculated
        """
        if composite_quality is None:
            return True  # Calculate for first pass

        # Use for borderline and poor quality cases
        return composite_quality < 0.7

    def normalize_score(self, raw_score: float) -> float:
        """Normalize SSIMULACRA2 score from -inf..100 to 0..1 range.

        Args:
            raw_score: Raw SSIMULACRA2 score

        Returns:
            Normalized score between 0 and 1
        """
        import math

        if not math.isfinite(raw_score):
            return 0.0
        if raw_score >= SSIMULACRA2_EXCELLENT_SCORE:
            return 1.0
        elif raw_score <= SSIMULACRA2_POOR_SCORE:
            return 0.0
        else:
            # Linear interpolation between poor and excellent scores
            return (raw_score - SSIMULACRA2_POOR_SCORE) / (
                SSIMULACRA2_EXCELLENT_SCORE - SSIMULACRA2_POOR_SCORE
            )

    def _export_frame_to_png(self, frame: np.ndarray, output_path: Path) -> None:
        """Export a single frame to PNG format.

        Args:
            frame: Frame as numpy array (H, W, C)
            output_path: Path to save PNG file
        """
        # Convert numpy array to PIL Image
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8)

        image = Image.fromarray(frame)
        image.save(output_path, "PNG")

    def _run_ssimulacra2_on_pair(self, orig_png: Path, comp_png: Path) -> float:
        """Run SSIMULACRA2 on a pair of PNG images.

        Args:
            orig_png: Path to original PNG
            comp_png: Path to compressed PNG

        Returns:
            Raw SSIMULACRA2 score

        Raises:
            subprocess.CalledProcessError: If SSIMULACRA2 execution fails
        """
        try:
            result = subprocess.run(
                [str(self.binary_path), str(orig_png), str(comp_png)],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode != 0:
                raise subprocess.CalledProcessError(
                    result.returncode,
                    [str(self.binary_path), str(orig_png), str(comp_png)],
                    result.stderr,
                )

            # Parse the numeric score from output
            # Handle both plain number format and "SSIMULACRA2 score: X.X" format
            import re
            score_str = result.stdout.strip()
            matches = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", score_str)
            if matches:
                return float(matches[-1])
            return float(score_str)

        except (subprocess.TimeoutExpired, ValueError) as e:
            logger.error(f"SSIMULACRA2 execution failed: {e}")
            raise

    def calculate_ssimulacra2_metrics(
        self,
        original_frames: list[np.ndarray],
        compressed_frames: list[np.ndarray],
        config: MetricsConfig,
    ) -> dict[str, float]:
        """Calculate SSIMULACRA2 metrics for frame pairs.

        Args:
            original_frames: List of original frames
            compressed_frames: List of compressed frames
            config: Metrics configuration

        Returns:
            Dictionary of SSIMULACRA2 metrics
        """
        if not self.is_available():
            logger.warning("SSIMULACRA2 binary not available, returning defaults")
            return {
                "ssimulacra2_mean": 50.0,
                "ssimulacra2_p95": 50.0,
                "ssimulacra2_min": 50.0,
                "ssimulacra2_frame_count": 0.0,
                "ssimulacra2_triggered": 0.0,
            }

        if len(original_frames) != len(compressed_frames):
            raise ValueError("Frame count mismatch between original and compressed")

        max_frames = min(
            len(original_frames), getattr(config, "SSIMULACRA2_MAX_FRAMES", 30)
        )

        # Sample frames uniformly if we have too many
        frame_indices = self._sample_frame_indices(len(original_frames), max_frames)

        scores = []

        with tempfile.TemporaryDirectory(prefix="ssimulacra2_") as temp_dir:
            temp_path = Path(temp_dir)

            for i, frame_idx in enumerate(frame_indices):
                orig_png = temp_path / f"orig_{i:04d}.png"
                comp_png = temp_path / f"comp_{i:04d}.png"

                try:
                    # Export frames to PNG
                    self._export_frame_to_png(original_frames[frame_idx], orig_png)
                    self._export_frame_to_png(compressed_frames[frame_idx], comp_png)

                    # Calculate SSIMULACRA2 score
                    raw_score = self._run_ssimulacra2_on_pair(orig_png, comp_png)
                    normalized_score = self.normalize_score(raw_score)
                    scores.append(normalized_score)

                    logger.debug(
                        f"Frame {frame_idx}: SSIMULACRA2 raw={raw_score:.2f}, "
                        f"normalized={normalized_score:.3f}"
                    )

                except Exception as e:
                    logger.error(f"SSIMULACRA2 failed for frame {frame_idx}: {e}")
                    # Use fallback score for failed frames
                    scores.append(0.5)

        if not scores:
            logger.error("No SSIMULACRA2 scores calculated")
            scores = [0.5]  # Fallback

        return {
            "ssimulacra2_mean": float(np.mean(scores)),
            "ssimulacra2_p95": float(np.percentile(scores, 95)),
            "ssimulacra2_min": float(np.min(scores)),
            "ssimulacra2_frame_count": float(len(scores)),
            "ssimulacra2_triggered": 1.0,
        }

    def _sample_frame_indices(self, total_frames: int, max_frames: int) -> list[int]:
        """Sample frame indices for analysis.

        Uses uniform sampling to distribute frames evenly across the GIF.

        Args:
            total_frames: Total number of available frames
            max_frames: Maximum number of frames to sample

        Returns:
            List of frame indices to analyze
        """
        if total_frames <= max_frames:
            return list(range(total_frames))

        # Uniform sampling
        indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
        return list(indices)


def calculate_ssimulacra2_quality_metrics(
    original_frames: list[np.ndarray],
    compressed_frames: list[np.ndarray],
    config: MetricsConfig,
    binary_path: str = DEFAULT_SSIMULACRA2_PATH,
) -> dict[str, float]:
    """Calculate SSIMULACRA2 quality metrics (main entry point).

    Args:
        original_frames: List of original frames as numpy arrays
        compressed_frames: List of compressed frames as numpy arrays
        config: Metrics configuration
        binary_path: Path to SSIMULACRA2 binary

    Returns:
        Dictionary of SSIMULACRA2 metrics
    """
    validator = Ssimulacra2Validator(binary_path)
    return validator.calculate_ssimulacra2_metrics(
        original_frames, compressed_frames, config
    )


def should_use_ssimulacra2(composite_quality: float | None) -> bool:
    """Determine if SSIMULACRA2 should be calculated (convenience function).

    Args:
        composite_quality: Current composite quality score (0-1)

    Returns:
        True if SSIMULACRA2 should be calculated
    """
    validator = Ssimulacra2Validator()
    return validator.should_use_ssimulacra2(composite_quality)
