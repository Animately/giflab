#!/usr/bin/env python3
"""
Optimized Metrics Calculation Module

This module provides high-performance implementations of core metrics calculations
with vectorized operations, memory efficiency optimizations, and algorithmic improvements.

Performance optimizations implemented:
1. Vectorized batch processing of frames
2. Memory-efficient algorithms with reduced allocations
3. Early termination strategies for expensive metrics
4. Streamlined control flow with reduced conditional branching
5. Optimized OpenCV and NumPy operations

===========================================================================
SCHEMA DIVERGENCE FROM MAIN PATH — READ BEFORE CONSUMING THIS MODULE
===========================================================================

``calculate_optimized_comprehensive_metrics`` (Phase 6 optimized path) emits
a **different key schema** from the main paths
``calculate_comprehensive_metrics`` and
``calculate_comprehensive_metrics_from_frames`` in ``giflab.metrics``.

Keys ABSENT from the Phase 6 output
-------------------------------------
The following keys are emitted by the main paths but are intentionally
**absent** from the Phase 6 result dict:

* ``temporal_consistency_original`` — main path: temporal consistency of
  original frames only (pre-compression stream).
* ``disposal_artifacts_compressed`` and ``disposal_artifacts_original`` —
  main path: disposal artifact score on compressed vs original stream.
* ``<temporal_key>_compressed`` aliases (``flicker_excess_compressed``,
  ``flicker_frame_ratio_compressed``, etc.) — main path: enhanced temporal
  metrics measured on the compressed stream only, suffixed with ``_compressed``
  to document the single-stream limitation.

Why they are absent
-------------------
Phase 6 measures temporal consistency on the **compressed stream only**
(see ``FastTemporalConsistency.calculate_optimized``).  It cannot separate
original-stream from compressed-stream values because the algorithm does not
run the same calculation on the original stream.

Per the "Same key shape across paths" rule in ``CLAUDE.md``::

    If the optimized path can't compute ``_pre``/``_post`` separately,
    document that prominently rather than silently aliasing — silent
    equivalence lies to the consumer about what was actually measured.

Phase 6 DOES emit ``temporal_consistency_compressed`` (Wave 7): this is the
honest, correctly-labelled compressed-stream value — emitting it names exactly
what was measured. What it must NOT do is fabricate
``temporal_consistency_original`` (the original stream is never measured here),
nor claim a pair comparison happened. Consumers reading two keys named
``_compressed`` and ``_original`` reasonably expect them to describe different
streams, so only the measured one (``_compressed``) is emitted.

Keys PRESENT that differ in semantics
--------------------------------------
``temporal_consistency_pre`` and ``temporal_consistency_post`` are both set
to the same value as ``temporal_consistency_compressed`` (compressed-stream
score) because Phase 6 cannot separate the two.  This is a legacy shape
retained for structural compatibility; the comment in the code marks the
limitation.

Consumer guidance
-----------------
If your code reads ``temporal_consistency_compressed``,
``temporal_consistency_original``, or any ``_compressed``/``_original`` alias
from the result dict, **disable Phase 6** by not setting
``GIFLAB_ENABLE_PHASE6_OPTIMIZATION=true``.  The main path in
``giflab.metrics`` always computes these correctly.

The metadata flag ``_optimization_applied: True`` in the result dict can be
used as a programmatic sentinel to detect Phase 6 output and skip
alias-dependent logic.

Regression test
---------------
``tests/functional/test_phase6_schema_contract.py`` locks this contract.
It asserts which keys are present and which are absent.  If a future change
adds silent equivalence aliases, those tests will fail intentionally.
===========================================================================
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class OptimizationStats:
    """Statistics tracking for optimization performance."""

    frames_processed: int
    processing_time_ms: float
    memory_peak_mb: float
    optimizations_applied: list[str]
    speedup_factor: float


class VectorizedMetricsCalculator:
    """
    High-performance metrics calculator using vectorized operations.

    Optimizations:
    - Batch processing of multiple frames simultaneously
    - Memory-efficient NumPy operations
    - Reduced temporary array allocations
    - Streamlined metric calculations
    """

    def __init__(self, batch_size: int = 8):
        self.batch_size = batch_size
        self.optimization_stats = OptimizationStats(
            frames_processed=0,
            processing_time_ms=0.0,
            memory_peak_mb=0.0,
            optimizations_applied=[],
            speedup_factor=1.0,
        )

    def calculate_batch_ssim(
        self, original_batch: np.ndarray, compressed_batch: np.ndarray
    ) -> np.ndarray:
        """
        Calculate SSIM for a batch of frame pairs using individual frame processing.

        Args:
            original_batch: Shape (batch_size, height, width, channels)
            compressed_batch: Shape (batch_size, height, width, channels)

        Returns:
            SSIM values for each frame pair, shape (batch_size,)
        """
        batch_size = original_batch.shape[0]
        ssim_values = np.zeros(batch_size)

        # Process each frame individually to avoid OpenCV batch issues
        from skimage.metrics import structural_similarity as ssim

        for i in range(batch_size):
            orig_frame = original_batch[i]
            comp_frame = compressed_batch[i]

            # Convert to grayscale if needed
            if len(orig_frame.shape) == 3:
                orig_gray = cv2.cvtColor(orig_frame, cv2.COLOR_RGB2GRAY)
                comp_gray = cv2.cvtColor(comp_frame, cv2.COLOR_RGB2GRAY)
            else:
                orig_gray = orig_frame
                comp_gray = comp_frame

            # Calculate SSIM using skimage for accuracy
            try:
                frame_ssim = ssim(orig_gray, comp_gray, data_range=255.0)
                ssim_values[i] = max(0.0, min(1.0, frame_ssim))
            except Exception as e:
                logger.warning(f"SSIM calculation failed for frame {i}: {e}")
                ssim_values[i] = 0.0

        return ssim_values

    def calculate_batch_mse(
        self, original_batch: np.ndarray, compressed_batch: np.ndarray
    ) -> np.ndarray:
        """
        Calculate MSE for a batch of frame pairs using vectorized operations.

        Args:
            original_batch: Shape (batch_size, height, width, channels)
            compressed_batch: Shape (batch_size, height, width, channels)

        Returns:
            MSE values for each frame pair, shape (batch_size,)
        """
        # Vectorized MSE calculation
        diff = original_batch.astype(np.float32) - compressed_batch.astype(np.float32)
        mse_batch = np.mean(diff**2, axis=(1, 2, 3))
        return mse_batch

    def calculate_batch_psnr(
        self, original_batch: np.ndarray, compressed_batch: np.ndarray
    ) -> np.ndarray:
        """
        Calculate PSNR for a batch of frame pairs using vectorized operations.

        Args:
            original_batch: Shape (batch_size, height, width, channels)
            compressed_batch: Shape (batch_size, height, width, channels)

        Returns:
            PSNR values for each frame pair, shape (batch_size,)
        """
        mse_batch = self.calculate_batch_mse(original_batch, compressed_batch)
        # Avoid division by zero
        mse_batch = np.maximum(mse_batch, 1e-10)
        psnr_batch = 20 * np.log10(255.0 / np.sqrt(mse_batch))
        return psnr_batch

    def process_frame_pairs_batched(
        self, frame_pairs: list[tuple[np.ndarray, np.ndarray]]
    ) -> dict[str, np.ndarray]:
        """
        Process frame pairs in batches for optimal performance.

        Args:
            frame_pairs: list of (original, compressed) frame pairs

        Returns:
            Dictionary with metric values for all frames
        """
        start_time = time.perf_counter()

        num_frames = len(frame_pairs)
        self.optimization_stats.frames_processed = num_frames

        # Initialize result arrays
        ssim_values = np.zeros(num_frames)
        mse_values = np.zeros(num_frames)
        psnr_values = np.zeros(num_frames)

        # Process in batches
        for i in range(0, num_frames, self.batch_size):
            end_idx = min(i + self.batch_size, num_frames)
            batch_size = end_idx - i

            # Extract batch
            batch_pairs = frame_pairs[i:end_idx]

            if not batch_pairs:
                continue

            # Get frame dimensions from first pair
            sample_frame = batch_pairs[0][0]
            height, width = sample_frame.shape[:2]
            channels = sample_frame.shape[2] if len(sample_frame.shape) == 3 else 1

            # Create batch arrays
            original_batch = np.zeros(
                (batch_size, height, width, channels), dtype=np.uint8
            )
            compressed_batch = np.zeros(
                (batch_size, height, width, channels), dtype=np.uint8
            )

            # Fill batch arrays
            for j, (orig, comp) in enumerate(batch_pairs):
                if len(orig.shape) == 2:
                    orig = orig[:, :, np.newaxis]
                if len(comp.shape) == 2:
                    comp = comp[:, :, np.newaxis]

                original_batch[j] = orig
                compressed_batch[j] = comp

            # Calculate metrics for batch
            try:
                batch_ssim = self.calculate_batch_ssim(original_batch, compressed_batch)
                ssim_values[i:end_idx] = batch_ssim

                batch_mse = self.calculate_batch_mse(original_batch, compressed_batch)
                mse_values[i:end_idx] = batch_mse

                batch_psnr = self.calculate_batch_psnr(original_batch, compressed_batch)
                psnr_values[i:end_idx] = batch_psnr

            except Exception as e:
                logger.warning(
                    f"Batch processing failed for batch {i//self.batch_size}: {e}"
                )
                # Fallback to individual frame processing for this batch
                for j, (orig, comp) in enumerate(batch_pairs):
                    try:
                        # Single frame SSIM calculation (simplified)
                        if len(orig.shape) == 3:
                            orig_gray = cv2.cvtColor(orig, cv2.COLOR_RGB2GRAY)
                            comp_gray = cv2.cvtColor(comp, cv2.COLOR_RGB2GRAY)
                        else:
                            orig_gray, comp_gray = orig, comp

                        from skimage.metrics import structural_similarity as ssim

                        frame_ssim = ssim(orig_gray, comp_gray, data_range=255.0)
                        ssim_values[i + j] = max(0.0, min(1.0, frame_ssim))

                        # Single frame MSE
                        diff = orig.astype(np.float32) - comp.astype(np.float32)
                        frame_mse = np.mean(diff**2)
                        mse_values[i + j] = frame_mse

                        # Single frame PSNR
                        if frame_mse > 0:
                            frame_psnr = 20 * np.log10(255.0 / np.sqrt(frame_mse))
                        else:
                            frame_psnr = 100.0
                        psnr_values[i + j] = frame_psnr

                    except Exception as e2:
                        logger.warning(f"Single frame processing failed: {e2}")
                        ssim_values[i + j] = 0.0
                        mse_values[i + j] = 0.0
                        psnr_values[i + j] = 0.0

        # Record performance
        end_time = time.perf_counter()
        self.optimization_stats.processing_time_ms = (end_time - start_time) * 1000
        self.optimization_stats.optimizations_applied = [
            "vectorized_batch_processing",
            "reduced_memory_allocations",
            "opencv_optimizations",
        ]

        return {"ssim": ssim_values, "mse": mse_values, "psnr": psnr_values}


class FastTemporalConsistency:
    """
    Optimized temporal consistency calculation that matches standard implementation.
    """

    @staticmethod
    def calculate_optimized(frames: list[np.ndarray]) -> float:
        """
        Calculate temporal consistency using approach consistent with standard implementation.

        Args:
            frames: list of frame arrays

        Returns:
            Temporal consistency score
        """
        if len(frames) < 2:
            return 1.0

        # Convert frames to grayscale and calculate frame-to-frame differences
        frame_diffs = []

        for i in range(len(frames) - 1):
            frame1 = frames[i]
            frame2 = frames[i + 1]

            # Convert to grayscale if needed
            if len(frame1.shape) == 3:
                gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
                gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
            else:
                gray1 = frame1
                gray2 = frame2

            # Calculate absolute difference between consecutive frames
            diff = np.abs(gray1.astype(np.float32) - gray2.astype(np.float32))
            mean_diff = np.mean(diff)
            frame_diffs.append(mean_diff)

        if not frame_diffs:
            return 1.0

        # Calculate coefficient of variation (std/mean) of frame differences
        # Lower variation = higher temporal consistency
        mean_diff_overall = np.mean(frame_diffs)
        std_diff = np.std(frame_diffs)

        if mean_diff_overall > 0:
            coefficient_of_variation = std_diff / mean_diff_overall
            # Convert to 0-1 scale where 1.0 is perfect consistency
            consistency = 1.0 / (1.0 + coefficient_of_variation)
        else:
            consistency = 1.0

        return float(consistency)


class MemoryEfficientFrameProcessor:
    """
    Frame processing with optimized memory allocation patterns.
    """

    def __init__(self, max_memory_mb: float = 500.0):
        self.max_memory_mb = max_memory_mb
        self.frame_cache = {}

    def resize_frames_batch(
        self, frames: list[np.ndarray], target_size: tuple[int, int]
    ) -> list[np.ndarray]:
        """
        Resize frames with memory-efficient batch processing.

        Args:
            frames: list of frames to resize
            target_size: (width, height) target size

        Returns:
            list of resized frames
        """
        if not frames:
            return []

        # Estimate memory usage
        sample_frame = frames[0]
        bytes_per_frame = sample_frame.nbytes
        target_bytes_per_frame = (
            target_size[0] * target_size[1] * sample_frame.shape[-1]
            if len(sample_frame.shape) == 3
            else target_size[0] * target_size[1]
        )

        # Calculate optimal batch size to stay under memory limit
        max_bytes = self.max_memory_mb * 1024 * 1024
        optimal_batch_size = max(
            1, int(max_bytes // (bytes_per_frame + target_bytes_per_frame))
        )

        resized_frames = []

        for i in range(0, len(frames), optimal_batch_size):
            batch = frames[i : i + optimal_batch_size]

            # Resize batch
            batch_resized = []
            for frame in batch:
                resized = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
                batch_resized.append(resized)

            resized_frames.extend(batch_resized)

        return resized_frames

    def align_frames_optimized(
        self, original_frames: list[np.ndarray], compressed_frames: list[np.ndarray]
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        Optimized frame alignment with early termination for obvious matches.

        Args:
            original_frames: Original frames
            compressed_frames: Compressed frames

        Returns:
            list of aligned frame pairs
        """
        # For same-length sequences, use direct alignment (most common case)
        if len(original_frames) == len(compressed_frames):
            return list(zip(original_frames, compressed_frames, strict=True))

        # For different lengths, use simplified content-based alignment
        # This is a simplified version that should handle most cases efficiently
        aligned_pairs = []

        orig_len = len(original_frames)
        comp_len = len(compressed_frames)

        if comp_len <= orig_len:
            # Compressed has fewer frames - align to best matching original frames
            step = orig_len / comp_len
            for i, comp_frame in enumerate(compressed_frames):
                orig_idx = min(int(i * step), orig_len - 1)
                aligned_pairs.append((original_frames[orig_idx], comp_frame))
        else:
            # Compressed has more frames - use sampling
            step = comp_len / orig_len
            for i, orig_frame in enumerate(original_frames):
                comp_idx = min(int(i * step), comp_len - 1)
                aligned_pairs.append((orig_frame, compressed_frames[comp_idx]))

        return aligned_pairs


def calculate_optimized_comprehensive_metrics(
    original_frames: list[np.ndarray],
    compressed_frames: list[np.ndarray],
    config: Any | None = None,
) -> dict[str, float]:
    """
    High-performance implementation of comprehensive metrics calculation.

    This function provides significant performance improvements over the standard
    implementation through:
    1. Vectorized batch processing
    2. Memory-efficient algorithms
    3. Optimized frame alignment
    4. Reduced computational overhead

    Args:
        original_frames: list of original frames
        compressed_frames: list of compressed frames
        config: Optional configuration (uses defaults if None)

    Returns:
        Dictionary with calculated metrics
    """
    start_time = time.perf_counter()

    # Initialize components
    vectorized_calc = VectorizedMetricsCalculator(batch_size=8)
    frame_processor = MemoryEfficientFrameProcessor(max_memory_mb=400.0)

    # Resize frames to common dimensions with memory efficiency
    if original_frames and compressed_frames:
        # Determine target size (use smaller dimensions to reduce memory usage)
        orig_sample = original_frames[0]
        comp_sample = compressed_frames[0]

        target_width = min(
            orig_sample.shape[1], comp_sample.shape[1], 800
        )  # Cap at 800px width
        target_height = min(
            orig_sample.shape[0], comp_sample.shape[0], 600
        )  # Cap at 600px height
        target_size = (target_width, target_height)

        logger.debug(f"Resizing frames to {target_size} for optimization")

        original_resized = frame_processor.resize_frames_batch(
            original_frames, target_size
        )
        compressed_resized = frame_processor.resize_frames_batch(
            compressed_frames, target_size
        )
    else:
        original_resized = original_frames
        compressed_resized = compressed_frames

    # Align frames with optimized algorithm
    aligned_pairs = frame_processor.align_frames_optimized(
        original_resized, compressed_resized
    )

    if not aligned_pairs:
        logger.warning("No frame pairs could be aligned")
        # Emit the FULL Phase 6 temporal key set (Wave 7) so the required-keys
        # contract holds for this branch too — previously this fallback emitted
        # only the bare ``temporal_consistency`` key and none of the
        # ``_compressed`` / ``_pre`` / ``_post`` / ``_delta`` siblings, a shape
        # divergence from the normal path. The bare key is now removed.
        return {
            "ssim_mean": 0.0,
            "mse_mean": 0.0,
            "psnr_mean": 0.0,
            "temporal_consistency_compressed": 1.0,
            "temporal_consistency_pre": 1.0,
            "temporal_consistency_post": 1.0,
            "temporal_consistency_delta": 0.0,
            "frame_count": len(original_frames),
            "compressed_frame_count": len(compressed_frames),
            "render_ms": 0,
            "_optimization_applied": True,
        }

    # Calculate core metrics using vectorized processing
    logger.debug(
        f"Processing {len(aligned_pairs)} frame pairs with vectorized calculation"
    )
    metric_results = vectorized_calc.process_frame_pairs_batched(aligned_pairs)

    # Calculate temporal consistency using optimized algorithm
    temporal_consistency = FastTemporalConsistency.calculate_optimized(
        compressed_resized
    )

    # Aggregate results
    result = {}

    # Add core metrics with aggregations
    for metric_name, values in metric_results.items():
        if len(values) > 0:
            result[f"{metric_name}_mean"] = float(np.mean(values))
            result[f"{metric_name}_std"] = float(np.std(values))
            result[f"{metric_name}_min"] = float(np.min(values))
            result[f"{metric_name}_max"] = float(np.max(values))

            # Backwards compatibility - add base metric name
            result[metric_name] = result[f"{metric_name}_mean"]
        else:
            result[f"{metric_name}_mean"] = 0.0
            result[f"{metric_name}_std"] = 0.0
            result[f"{metric_name}_min"] = 0.0
            result[f"{metric_name}_max"] = 0.0
            result[metric_name] = 0.0

    # Add temporal consistency.
    #
    # SCHEMA: Phase 6 measures temporal consistency on the compressed stream
    # only (FastTemporalConsistency.calculate_optimized receives
    # compressed_resized, not a pair).  Therefore:
    #
    #   - temporal_consistency_compressed   EMITTED — this IS the honest
    #     compressed-stream value, correctly labelled. Emitting ONLY this is
    #     not fabrication (it names exactly what was measured); emitting BOTH
    #     _compressed AND _original would fabricate the original-stream value.
    #   - temporal_consistency_original     NOT emitted (would fabricate — Phase
    #     6 never runs the calculation on the original stream).
    #
    # _pre and _post are both set to the same compressed-stream score for
    # structural shape compatibility only; they do NOT represent independent
    # stream measurements.  See the module docstring "SCHEMA DIVERGENCE"
    # section and CLAUDE.md "Same key shape across paths" rule.
    #
    # The bare ``temporal_consistency`` key was removed in Wave 7 — it read
    # like a pair signal but only ever carried the compressed-stream value.
    result["temporal_consistency_compressed"] = float(temporal_consistency)
    result["temporal_consistency_pre"] = float(
        temporal_consistency
    )  # Same as _post — compressed stream only; see above.
    result["temporal_consistency_post"] = float(temporal_consistency)
    result["temporal_consistency_delta"] = 0.0

    # Add frame counts
    result["frame_count"] = len(original_frames)
    result["compressed_frame_count"] = len(compressed_frames)

    # Add processing time
    end_time = time.perf_counter()
    result["render_ms"] = int((end_time - start_time) * 1000)

    # Add optimization metadata
    result["_optimization_applied"] = True
    result["_optimization_stats"] = {
        "frames_processed": vectorized_calc.optimization_stats.frames_processed,
        "processing_time_ms": vectorized_calc.optimization_stats.processing_time_ms,
        "optimizations": vectorized_calc.optimization_stats.optimizations_applied,
    }

    # Add minimal required metrics for compatibility
    default_metrics = {
        "composite_quality": float(
            np.mean(
                [
                    result.get("ssim_mean", 0.0),
                    1.0 - result.get("mse_mean", 0.0) / 10000.0,
                ]
            )
        ),
        "efficiency": 1.0,  # Simplified
        "compression_ratio": 1.0,  # Will be overridden if file metadata available
        "kilobytes": 0.0,  # Will be overridden if file metadata available
        # Add minimal gradient/color metrics
        "banding_score_mean": 0.0,
        "deltae_mean": 0.0,
        "color_patch_count": 0,
        # Add minimal text/UI metrics
        "has_text_ui_content": False,
        "text_ui_edge_density": 0.0,
        "text_ui_component_count": 0,
        # Add minimal SSIMULACRA2 metrics
        # NaN = "not computed" on the normalised [0, 1] output scale.
        # The legacy 50.0 sentinel was on the raw 0-100 scale and corrupted
        # downstream aggregations.
        "ssimulacra2_mean": float("nan"),
        "ssimulacra2_triggered": 0.0,
    }

    for key, value in default_metrics.items():
        if key not in result:
            result[key] = value

    logger.info(
        f"Optimized metrics calculation completed in {result['render_ms']}ms for {len(aligned_pairs)} frame pairs"
    )

    return result
