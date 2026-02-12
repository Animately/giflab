"""Base frame sampler interface and data structures."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


class SamplingStrategy(Enum):
    """Available sampling strategies."""

    UNIFORM = "uniform"
    ADAPTIVE = "adaptive"
    PROGRESSIVE = "progressive"
    SCENE_AWARE = "scene_aware"
    FULL = "full"  # No sampling, process all frames


@dataclass
class SamplingResult:
    """Result of frame sampling operation."""

    sampled_indices: list[int]  # Indices of sampled frames
    total_frames: int  # Total number of frames
    sampling_rate: float  # Percentage of frames sampled
    confidence_level: float | None = None  # Statistical confidence
    confidence_interval: tuple[float, float] | None = None  # CI bounds
    strategy_used: str = ""  # Name of strategy used
    metadata: dict[str, Any] = None  # Additional strategy-specific data

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

        # Calculate sampling rate if not set
        if self.total_frames > 0:
            self.sampling_rate = len(self.sampled_indices) / self.total_frames

    @property
    def num_sampled(self) -> int:
        """Number of frames sampled."""
        return len(self.sampled_indices)

    def is_full_sampling(self) -> bool:
        """Check if all frames were sampled."""
        return self.num_sampled == self.total_frames


class FrameSampler(ABC):
    """Abstract base class for frame sampling strategies."""

    def __init__(
        self,
        min_frames_threshold: int = 30,
        confidence_level: float = 0.95,
        verbose: bool = False,
    ):
        """
        Initialize frame sampler.

        Args:
            min_frames_threshold: Minimum frames required for sampling
            confidence_level: Target confidence level for statistical sampling
            verbose: Enable verbose logging
        """
        self.min_frames_threshold = min_frames_threshold
        self.confidence_level = confidence_level
        self.verbose = verbose

    @abstractmethod
    def sample(self, frames: list[np.ndarray], **kwargs) -> SamplingResult:
        """
        Sample frames based on strategy.

        Args:
            frames: list of frames to sample from
            **kwargs: Strategy-specific parameters

        Returns:
            SamplingResult with sampled indices and metadata
        """
        pass

    def should_sample(self, num_frames: int) -> bool:
        """
        Determine if sampling should be applied.

        Args:
            num_frames: Number of frames in the GIF

        Returns:
            True if sampling should be applied
        """
        return num_frames >= self.min_frames_threshold

    def calculate_confidence_interval(
        self, samples: list[float], confidence_level: float = None
    ) -> tuple[float, float]:
        """
        Calculate confidence interval for sampled metrics.

        Args:
            samples: Sampled metric values
            confidence_level: Confidence level (uses default if None)

        Returns:
            tuple of (lower_bound, upper_bound)
        """
        if confidence_level is None:
            confidence_level = self.confidence_level

        if len(samples) < 2:
            # Can't calculate CI with < 2 samples
            mean = np.mean(samples) if samples else 0
            return (mean, mean)

        # Calculate mean and standard error
        mean = np.mean(samples)
        std_err = np.std(samples, ddof=1) / np.sqrt(len(samples))

        # Calculate confidence interval using t-distribution
        # For simplicity, using z-score approximation
        z_scores = {
            0.90: 1.645,
            0.95: 1.96,
            0.99: 2.576,
        }
        z = z_scores.get(confidence_level, 1.96)

        margin = z * std_err
        return (mean - margin, mean + margin)

    def log_sampling_info(self, result: SamplingResult):
        """Log sampling information if verbose."""
        if self.verbose:
            logger.info(
                f"Sampled {result.num_sampled}/{result.total_frames} frames "
                f"({result.sampling_rate:.1%}) using {result.strategy_used}"
            )
            if result.confidence_interval:
                logger.info(
                    f"Confidence interval ({result.confidence_level:.0%}): "
                    f"[{result.confidence_interval[0]:.4f}, "
                    f"{result.confidence_interval[1]:.4f}]"
                )


class FrameDifferenceCalculator:
    """Helper class for calculating frame differences."""

    @staticmethod
    def calculate_mse(frame1: np.ndarray, frame2: np.ndarray) -> float:
        """Calculate MSE between two frames."""
        if frame1.shape != frame2.shape:
            raise ValueError("Frames must have the same shape")

        # Convert to float to avoid overflow
        diff = frame1.astype(np.float32) - frame2.astype(np.float32)
        mse = np.mean(diff**2)
        return float(mse)

    @staticmethod
    def calculate_histogram_difference(
        frame1: np.ndarray, frame2: np.ndarray, bins: int = 256
    ) -> float:
        """
        Calculate histogram difference between frames.

        Returns value between 0 (identical) and 1 (completely different).
        """
        # Calculate histograms for each channel
        hist_diff = 0
        num_channels = frame1.shape[2] if len(frame1.shape) == 3 else 1

        for channel in range(num_channels):
            if num_channels > 1:
                data1 = frame1[:, :, channel].flatten()
                data2 = frame2[:, :, channel].flatten()
            else:
                data1 = frame1.flatten()
                data2 = frame2.flatten()

            hist1, _ = np.histogram(data1, bins=bins, range=(0, 256))
            hist2, _ = np.histogram(data2, bins=bins, range=(0, 256))

            # Normalize histograms
            hist1 = hist1.astype(np.float32) / hist1.sum()
            hist2 = hist2.astype(np.float32) / hist2.sum()

            # Calculate histogram intersection
            intersection = np.minimum(hist1, hist2).sum()
            hist_diff += 1 - intersection

        return hist_diff / num_channels

    @staticmethod
    def detect_scene_change(
        frame1: np.ndarray, frame2: np.ndarray, threshold: float = 0.3
    ) -> bool:
        """
        Detect if there's a scene change between frames.

        Args:
            frame1: First frame
            frame2: Second frame
            threshold: Threshold for scene change detection

        Returns:
            True if scene change detected
        """
        # Use histogram difference for scene detection
        diff = FrameDifferenceCalculator.calculate_histogram_difference(frame1, frame2)
        return diff > threshold


def create_sampler(
    strategy: SamplingStrategy,
    min_frames_threshold: int = 30,
    confidence_level: float = 0.95,
    verbose: bool = False,
    **kwargs,
) -> FrameSampler:
    """
    Factory function to create a sampler.

    Args:
        strategy: Sampling strategy to use
        min_frames_threshold: Minimum frames for sampling
        confidence_level: Target confidence level
        verbose: Enable verbose logging
        **kwargs: Strategy-specific parameters

    Returns:
        FrameSampler instance
    """
    from .strategies import (
        AdaptiveSampler,
        ProgressiveSampler,
        SceneAwareSampler,
        UniformSampler,
    )

    samplers = {
        SamplingStrategy.UNIFORM: UniformSampler,
        SamplingStrategy.ADAPTIVE: AdaptiveSampler,
        SamplingStrategy.PROGRESSIVE: ProgressiveSampler,
        SamplingStrategy.SCENE_AWARE: SceneAwareSampler,
    }

    if strategy == SamplingStrategy.FULL:
        # Return a pass-through sampler for full processing
        return UniformSampler(
            sampling_rate=1.0,
            min_frames_threshold=float("inf"),
            confidence_level=confidence_level,
            verbose=verbose,
        )

    sampler_class = samplers.get(strategy)
    if not sampler_class:
        raise ValueError(f"Unknown sampling strategy: {strategy}")

    return sampler_class(
        min_frames_threshold=min_frames_threshold,
        confidence_level=confidence_level,
        verbose=verbose,
        **kwargs,
    )
