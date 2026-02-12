"""Integration of frame sampling with metrics calculation."""

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ..config import DEFAULT_METRICS_CONFIG, FRAME_SAMPLING, MetricsConfig
from ..metrics import (
    align_frames,
    calculate_comprehensive_metrics_from_frames,
    resize_to_common_dimensions,
)
from .frame_sampler import (
    SamplingResult,
    SamplingStrategy,
    create_sampler,
)

logger = logging.getLogger(__name__)


def calculate_metrics_with_sampling(
    original_frames: list[np.ndarray],
    compressed_frames: list[np.ndarray],
    config: MetricsConfig | None = None,
    sampling_enabled: bool | None = None,
    sampling_strategy: str | None = None,
    frame_reduction_context: bool = False,
    file_metadata: dict[str, Any] | None = None,
    return_sampling_info: bool = False,
) -> dict[str, Any]:
    """
    Calculate comprehensive metrics with optional frame sampling.

    This wrapper function adds sampling support to the standard metrics calculation.
    When sampling is enabled, it intelligently selects representative frames and
    calculates metrics on the subset, providing confidence intervals for accuracy.

    Args:
        original_frames: list of original frames as numpy arrays
        compressed_frames: list of compressed frames as numpy arrays
        config: Optional metrics configuration (uses default if None)
        sampling_enabled: Override sampling enable/disable (uses config if None)
        sampling_strategy: Override sampling strategy (uses config if None)
        frame_reduction_context: If True, adjusts disposal artifact detection
        file_metadata: Optional dict with file-specific metadata
        return_sampling_info: If True, includes sampling metadata in results

    Returns:
        Dictionary with comprehensive metrics, optionally including sampling info
    """
    if config is None:
        config = DEFAULT_METRICS_CONFIG

    # Determine if sampling should be used
    if sampling_enabled is None:
        sampling_enabled = FRAME_SAMPLING.get("enabled", True)

    # Get sampling strategy
    if sampling_strategy is None:
        sampling_strategy = FRAME_SAMPLING.get("default_strategy", "adaptive")

    # Check if we should sample
    num_original = len(original_frames)
    num_compressed = len(compressed_frames)
    min_frames = FRAME_SAMPLING.get("min_frames_threshold", 30)

    sampling_result = None
    sampled_original = original_frames
    sampled_compressed = compressed_frames

    if sampling_enabled and num_original >= min_frames:
        try:
            # Map strategy string to enum
            strategy_map = {
                "uniform": SamplingStrategy.UNIFORM,
                "adaptive": SamplingStrategy.ADAPTIVE,
                "progressive": SamplingStrategy.PROGRESSIVE,
                "scene_aware": SamplingStrategy.SCENE_AWARE,
                "full": SamplingStrategy.FULL,
            }

            strategy_enum = strategy_map.get(
                sampling_strategy.lower(), SamplingStrategy.ADAPTIVE
            )

            # Get strategy-specific config
            strategy_config = FRAME_SAMPLING.get(sampling_strategy.lower(), {})

            # Create sampler with appropriate parameters
            sampler = create_sampler(
                strategy=strategy_enum,
                min_frames_threshold=min_frames,
                confidence_level=FRAME_SAMPLING.get("confidence_level", 0.95),
                verbose=FRAME_SAMPLING.get("verbose", False),
                **strategy_config,
            )

            # Sample frames
            logger.info(
                f"Applying {sampling_strategy} sampling to {num_original} frames"
            )

            # For progressive sampling, we need a metric function
            if strategy_enum == SamplingStrategy.PROGRESSIVE:
                # Define a simple metric function for progressive sampling
                def frame_diff_metric(idx):
                    if idx == 0 or idx >= len(original_frames):
                        return 0.0
                    # Use MSE between consecutive frames as metric
                    prev = original_frames[idx - 1].astype(np.float32)
                    curr = original_frames[idx].astype(np.float32)
                    return np.mean((curr - prev) ** 2)

                sampling_result = sampler.sample(
                    original_frames, metric_func=frame_diff_metric
                )
            else:
                sampling_result = sampler.sample(original_frames)

            # Extract sampled frames
            sampled_indices = sampling_result.sampled_indices
            sampled_original = [original_frames[i] for i in sampled_indices]

            # For compressed frames, we need to handle alignment
            # Sample the same relative positions from compressed frames
            if num_compressed == num_original:
                # Same frame count, use same indices
                sampled_compressed = [compressed_frames[i] for i in sampled_indices]
            else:
                # Different frame counts, need proportional sampling
                compressed_indices = [
                    min(int(i * num_compressed / num_original), num_compressed - 1)
                    for i in sampled_indices
                ]
                # Remove duplicates while preserving order
                seen = set()
                compressed_indices = [
                    x for x in compressed_indices if not (x in seen or seen.add(x))
                ]
                sampled_compressed = [compressed_frames[i] for i in compressed_indices]

            logger.info(
                f"Sampled {len(sampled_original)} original and "
                f"{len(sampled_compressed)} compressed frames "
                f"({sampling_result.sampling_rate:.1%} sampling rate)"
            )

        except Exception as e:
            logger.warning(f"Frame sampling failed: {e}, using all frames")
            sampling_result = None
            sampled_original = original_frames
            sampled_compressed = compressed_frames

    # Calculate metrics on sampled frames
    metrics = calculate_comprehensive_metrics_from_frames(
        sampled_original,
        sampled_compressed,
        config=config,
        frame_reduction_context=frame_reduction_context,
        file_metadata=file_metadata,
    )

    # Add sampling information if requested
    if return_sampling_info and sampling_result is not None:
        metrics["_sampling_info"] = {
            "sampling_applied": True,
            "strategy_used": sampling_result.strategy_used,
            "frames_sampled": sampling_result.num_sampled,
            "total_frames": sampling_result.total_frames,
            "sampling_rate": sampling_result.sampling_rate,
            "confidence_level": sampling_result.confidence_level,
        }

        # Add confidence interval if available
        if sampling_result.confidence_interval:
            metrics["_sampling_info"]["confidence_interval"] = {
                "lower": sampling_result.confidence_interval[0],
                "upper": sampling_result.confidence_interval[1],
            }

        # Add strategy-specific metadata
        if sampling_result.metadata:
            metrics["_sampling_info"]["strategy_metadata"] = sampling_result.metadata
    elif return_sampling_info:
        metrics["_sampling_info"] = {
            "sampling_applied": False,
            "reason": "insufficient_frames"
            if num_original < min_frames
            else "disabled",
        }

    return metrics


def apply_sampling_to_frames(
    frames: list[np.ndarray],
    sampling_strategy: str = "adaptive",
    confidence_level: float = 0.95,
    verbose: bool = False,
) -> tuple[list[np.ndarray], SamplingResult]:
    """
    Apply sampling to a list of frames.

    This is a utility function for applying sampling independently of metrics.

    Args:
        frames: list of frames to sample
        sampling_strategy: Strategy to use (uniform, adaptive, progressive, scene_aware)
        confidence_level: Target confidence level for sampling
        verbose: Enable verbose logging

    Returns:
        tuple of (sampled_frames, sampling_result)
    """
    # Map strategy string to enum
    strategy_map = {
        "uniform": SamplingStrategy.UNIFORM,
        "adaptive": SamplingStrategy.ADAPTIVE,
        "progressive": SamplingStrategy.PROGRESSIVE,
        "scene_aware": SamplingStrategy.SCENE_AWARE,
        "full": SamplingStrategy.FULL,
    }

    strategy_enum = strategy_map.get(
        sampling_strategy.lower(), SamplingStrategy.ADAPTIVE
    )

    # Get strategy-specific config from global config
    strategy_config = FRAME_SAMPLING.get(sampling_strategy.lower(), {})

    # Create sampler
    sampler = create_sampler(
        strategy=strategy_enum,
        min_frames_threshold=FRAME_SAMPLING.get("min_frames_threshold", 30),
        confidence_level=confidence_level,
        verbose=verbose,
        **strategy_config,
    )

    # Sample frames
    sampling_result = sampler.sample(frames)

    # Extract sampled frames
    sampled_frames = [frames[i] for i in sampling_result.sampled_indices]

    return sampled_frames, sampling_result


def estimate_sampling_speedup(
    num_frames: int,
    sampling_strategy: str = "adaptive",
) -> float:
    """
    Estimate the speedup factor from using sampling.

    Args:
        num_frames: Number of frames in the GIF
        sampling_strategy: Sampling strategy to use

    Returns:
        Estimated speedup factor (e.g., 2.0 means 2x faster)
    """
    # Get min frames threshold
    min_frames = FRAME_SAMPLING.get("min_frames_threshold", 30)

    if num_frames < min_frames:
        return 1.0  # No speedup if not sampling

    # Get expected sampling rate for strategy
    strategy_rates = {
        "uniform": FRAME_SAMPLING.get("uniform", {}).get("sampling_rate", 0.3),
        "adaptive": 0.4,  # Adaptive varies, use average estimate
        "progressive": 0.25,  # Progressive starts low
        "scene_aware": 0.35,  # Scene-aware varies by content
        "full": 1.0,
    }

    sampling_rate = strategy_rates.get(sampling_strategy.lower(), 0.3)

    # Speedup is roughly inverse of sampling rate
    # But there's overhead, so cap maximum speedup
    speedup = 1.0 / max(sampling_rate, 0.1)
    return min(speedup, 10.0)  # Cap at 10x speedup
