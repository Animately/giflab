"""Concrete implementations of frame sampling strategies."""

import logging
from collections.abc import Callable
from typing import Any, Optional

import numpy as np

from .frame_sampler import (
    FrameDifferenceCalculator,
    FrameSampler,
    SamplingResult,
)

logger = logging.getLogger(__name__)


class UniformSampler(FrameSampler):
    """
    Uniform sampling strategy that selects frames at regular intervals.
    """

    def __init__(
        self,
        sampling_rate: float = 0.3,
        min_frames_threshold: int = 30,
        confidence_level: float = 0.95,
        verbose: bool = False,
    ):
        """
        Initialize uniform sampler.

        Args:
            sampling_rate: Fraction of frames to sample (0.0 to 1.0)
            min_frames_threshold: Minimum frames for sampling
            confidence_level: Target confidence level
            verbose: Enable verbose logging
        """
        super().__init__(min_frames_threshold, confidence_level, verbose)
        self.sampling_rate = max(0.0, min(1.0, sampling_rate))

    def sample(self, frames: list[np.ndarray], **kwargs) -> SamplingResult:
        """
        Sample frames uniformly at regular intervals.

        Args:
            frames: list of frames to sample
            **kwargs: Additional parameters (ignored)

        Returns:
            SamplingResult with uniformly sampled indices
        """
        num_frames = len(frames)

        # Check if sampling should be applied
        if not self.should_sample(num_frames):
            # Return all frames
            return SamplingResult(
                sampled_indices=list(range(num_frames)),
                total_frames=num_frames,
                sampling_rate=1.0,
                strategy_used="uniform (full)",
            )

        # Calculate number of frames to sample
        num_samples = max(2, int(num_frames * self.sampling_rate))

        # Always include first and last frame
        if num_samples >= 2:
            # Calculate step size for uniform distribution
            step = (num_frames - 1) / (num_samples - 1)
            indices = [int(i * step) for i in range(num_samples)]

            # Ensure unique indices
            indices = sorted(set(indices))
        else:
            indices = [0, num_frames - 1]

        result = SamplingResult(
            sampled_indices=indices,
            total_frames=num_frames,
            sampling_rate=len(indices) / num_frames,
            strategy_used="uniform",
            metadata={
                "target_rate": self.sampling_rate,
                "actual_samples": len(indices),
            },
        )

        self.log_sampling_info(result)
        return result


class AdaptiveSampler(FrameSampler):
    """
    Adaptive sampling based on frame-to-frame differences.
    Samples more densely in areas of high motion/change.
    """

    def __init__(
        self,
        base_rate: float = 0.2,
        motion_threshold: float = 0.1,
        max_rate: float = 0.8,
        min_frames_threshold: int = 30,
        confidence_level: float = 0.95,
        verbose: bool = False,
    ):
        """
        Initialize adaptive sampler.

        Args:
            base_rate: Base sampling rate for low-motion areas
            motion_threshold: Threshold for detecting significant motion
            max_rate: Maximum sampling rate for high-motion areas
            min_frames_threshold: Minimum frames for sampling
            confidence_level: Target confidence level
            verbose: Enable verbose logging
        """
        super().__init__(min_frames_threshold, confidence_level, verbose)
        self.base_rate = max(0.1, min(1.0, base_rate))
        self.motion_threshold = motion_threshold
        self.max_rate = max(self.base_rate, min(1.0, max_rate))

    def sample(self, frames: list[np.ndarray], **kwargs) -> SamplingResult:
        """
        Sample frames adaptively based on motion/change detection.

        Args:
            frames: list of frames to sample
            **kwargs: Additional parameters

        Returns:
            SamplingResult with adaptively sampled indices
        """
        num_frames = len(frames)

        # Check if sampling should be applied
        if not self.should_sample(num_frames):
            return SamplingResult(
                sampled_indices=list(range(num_frames)),
                total_frames=num_frames,
                sampling_rate=1.0,
                strategy_used="adaptive (full)",
            )

        # Calculate frame differences
        differences = []
        for i in range(1, num_frames):
            diff = FrameDifferenceCalculator.calculate_histogram_difference(
                frames[i - 1], frames[i]
            )
            differences.append(diff)

        # Determine sampling density based on differences
        indices = [0]  # Always include first frame

        # Calculate adaptive sampling rate for each segment
        mean_diff = np.mean(differences)
        std_diff = np.std(differences)

        for i, diff in enumerate(differences):
            # Calculate local sampling rate based on motion
            if diff > mean_diff + std_diff:
                # High motion area
                local_rate = self.max_rate
            elif diff > mean_diff:
                # Medium motion
                local_rate = (self.base_rate + self.max_rate) / 2
            else:
                # Low motion
                local_rate = self.base_rate

            # Decide whether to sample this frame
            if np.random.random() < local_rate:
                indices.append(i + 1)

        # Always include last frame
        if (num_frames - 1) not in indices:
            indices.append(num_frames - 1)

        indices = sorted(set(indices))

        # Calculate motion statistics
        high_motion_frames = sum(1 for d in differences if d > mean_diff + std_diff)
        motion_intensity = np.mean(differences)

        result = SamplingResult(
            sampled_indices=indices,
            total_frames=num_frames,
            sampling_rate=len(indices) / num_frames,
            strategy_used="adaptive",
            metadata={
                "motion_intensity": float(motion_intensity),
                "high_motion_frames": high_motion_frames,
                "mean_difference": float(mean_diff),
                "std_difference": float(std_diff),
            },
        )

        self.log_sampling_info(result)
        return result


class ProgressiveSampler(FrameSampler):
    """
    Progressive sampling that iteratively refines the sample set
    until confidence interval requirements are met.
    """

    def __init__(
        self,
        initial_rate: float = 0.1,
        increment_rate: float = 0.1,
        max_iterations: int = 5,
        target_ci_width: float = 0.1,
        min_frames_threshold: int = 30,
        confidence_level: float = 0.95,
        verbose: bool = False,
    ):
        """
        Initialize progressive sampler.

        Args:
            initial_rate: Initial sampling rate
            increment_rate: Rate increase per iteration
            max_iterations: Maximum sampling iterations
            target_ci_width: Target confidence interval width
            min_frames_threshold: Minimum frames for sampling
            confidence_level: Target confidence level
            verbose: Enable verbose logging
        """
        super().__init__(min_frames_threshold, confidence_level, verbose)
        self.initial_rate = max(0.05, min(1.0, initial_rate))
        self.increment_rate = max(0.05, min(0.5, increment_rate))
        self.max_iterations = max(1, max_iterations)
        self.target_ci_width = target_ci_width

    def sample(
        self, frames: list[np.ndarray], metric_func: Callable | None = None, **kwargs
    ) -> SamplingResult:
        """
        Sample frames progressively, refining based on confidence intervals.

        Args:
            frames: list of frames to sample
            metric_func: Optional function to calculate metric for CI
            **kwargs: Additional parameters

        Returns:
            SamplingResult with progressively sampled indices
        """
        num_frames = len(frames)

        # Check if sampling should be applied
        if not self.should_sample(num_frames):
            return SamplingResult(
                sampled_indices=list(range(num_frames)),
                total_frames=num_frames,
                sampling_rate=1.0,
                strategy_used="progressive (full)",
            )

        # If no metric function provided, use frame differences as proxy
        if metric_func is None:

            def default_metric(frame_idx):
                if frame_idx == 0:
                    return 0.0
                return FrameDifferenceCalculator.calculate_histogram_difference(
                    frames[frame_idx - 1], frames[frame_idx]
                )

            metric_func = default_metric

        # Progressive sampling
        sampled_indices = set()
        current_rate = self.initial_rate
        iteration = 0
        ci_width = float("inf")

        while iteration < self.max_iterations and ci_width > self.target_ci_width:
            # Calculate number of additional samples needed
            target_samples = int(num_frames * current_rate)
            additional_needed = target_samples - len(sampled_indices)

            if additional_needed > 0:
                # Sample additional frames uniformly from unsampled indices
                unsampled = [i for i in range(num_frames) if i not in sampled_indices]
                if unsampled:
                    step = max(1, len(unsampled) // additional_needed)
                    new_samples = unsampled[::step][:additional_needed]
                    sampled_indices.update(new_samples)

            # Calculate metrics for sampled frames
            if len(sampled_indices) >= 2:
                metrics = [metric_func(i) for i in sorted(sampled_indices)]
                ci_lower, ci_upper = self.calculate_confidence_interval(
                    metrics, self.confidence_level
                )
                ci_width = ci_upper - ci_lower

                if self.verbose:
                    logger.info(
                        f"Iteration {iteration + 1}: "
                        f"Sampled {len(sampled_indices)}/{num_frames} frames, "
                        f"CI width: {ci_width:.4f}"
                    )

            # Increase sampling rate for next iteration
            current_rate = min(1.0, current_rate + self.increment_rate)
            iteration += 1

        # Always include first and last frame
        sampled_indices.add(0)
        sampled_indices.add(num_frames - 1)

        indices = sorted(sampled_indices)

        # Calculate final CI if we have metrics
        if len(indices) >= 2:
            final_metrics = [metric_func(i) for i in indices]
            ci_lower, ci_upper = self.calculate_confidence_interval(
                final_metrics, self.confidence_level
            )
        else:
            ci_lower, ci_upper = None, None

        result = SamplingResult(
            sampled_indices=indices,
            total_frames=num_frames,
            sampling_rate=len(indices) / num_frames,
            confidence_level=self.confidence_level,
            confidence_interval=(ci_lower, ci_upper) if ci_lower is not None else None,
            strategy_used="progressive",
            metadata={
                "iterations": iteration,
                "final_ci_width": ci_width if ci_width != float("inf") else None,
                "target_ci_width": self.target_ci_width,
            },
        )

        self.log_sampling_info(result)
        return result


class SceneAwareSampler(FrameSampler):
    """
    Scene-aware sampling that ensures frames from different scenes
    are included, with additional sampling at scene boundaries.
    """

    def __init__(
        self,
        scene_threshold: float = 0.3,
        boundary_window: int = 2,
        min_scene_samples: int = 3,
        base_rate: float = 0.3,
        min_frames_threshold: int = 30,
        confidence_level: float = 0.95,
        verbose: bool = False,
    ):
        """
        Initialize scene-aware sampler.

        Args:
            scene_threshold: Threshold for scene change detection
            boundary_window: Frames to sample around scene boundaries
            min_scene_samples: Minimum samples per detected scene
            base_rate: Base sampling rate within scenes
            min_frames_threshold: Minimum frames for sampling
            confidence_level: Target confidence level
            verbose: Enable verbose logging
        """
        super().__init__(min_frames_threshold, confidence_level, verbose)
        self.scene_threshold = scene_threshold
        self.boundary_window = max(1, boundary_window)
        self.min_scene_samples = max(1, min_scene_samples)
        self.base_rate = max(0.1, min(1.0, base_rate))

    def detect_scenes(self, frames: list[np.ndarray]) -> list[tuple[int, int]]:
        """
        Detect scene boundaries in the frame sequence.

        Args:
            frames: list of frames

        Returns:
            list of (start_idx, end_idx) tuples for each scene
        """
        if len(frames) < 2:
            return [(0, len(frames) - 1)]

        scenes = []
        current_start = 0

        for i in range(1, len(frames)):
            # Check for scene change
            is_scene_change = FrameDifferenceCalculator.detect_scene_change(
                frames[i - 1], frames[i], threshold=self.scene_threshold
            )

            if is_scene_change:
                # End current scene
                scenes.append((current_start, i - 1))
                current_start = i

        # Add final scene
        scenes.append((current_start, len(frames) - 1))

        return scenes

    def sample(self, frames: list[np.ndarray], **kwargs) -> SamplingResult:
        """
        Sample frames with scene awareness.

        Args:
            frames: list of frames to sample
            **kwargs: Additional parameters

        Returns:
            SamplingResult with scene-aware sampled indices
        """
        num_frames = len(frames)

        # Check if sampling should be applied
        if not self.should_sample(num_frames):
            return SamplingResult(
                sampled_indices=list(range(num_frames)),
                total_frames=num_frames,
                sampling_rate=1.0,
                strategy_used="scene_aware (full)",
            )

        # Detect scenes
        scenes = self.detect_scenes(frames)
        sampled_indices = set()

        if self.verbose:
            logger.info(f"Detected {len(scenes)} scenes in {num_frames} frames")

        for scene_start, scene_end in scenes:
            scene_length = scene_end - scene_start + 1

            # Always sample scene boundaries
            sampled_indices.add(scene_start)
            sampled_indices.add(scene_end)

            # Sample frames around boundaries
            for offset in range(1, min(self.boundary_window, scene_length // 2)):
                if scene_start + offset <= scene_end:
                    sampled_indices.add(scene_start + offset)
                if scene_end - offset >= scene_start:
                    sampled_indices.add(scene_end - offset)

            # Sample additional frames within the scene
            if scene_length > self.min_scene_samples:
                # Calculate how many more samples we need
                current_scene_samples = len(
                    [i for i in sampled_indices if scene_start <= i <= scene_end]
                )

                additional_needed = max(
                    0,
                    min(
                        int(scene_length * self.base_rate),
                        self.min_scene_samples - current_scene_samples,
                    ),
                )

                if additional_needed > 0:
                    # Sample uniformly within the scene
                    unsampled_in_scene = [
                        i
                        for i in range(scene_start, scene_end + 1)
                        if i not in sampled_indices
                    ]

                    if unsampled_in_scene:
                        step = max(1, len(unsampled_in_scene) // additional_needed)
                        new_samples = unsampled_in_scene[::step][:additional_needed]
                        sampled_indices.update(new_samples)

        indices = sorted(sampled_indices)

        result = SamplingResult(
            sampled_indices=indices,
            total_frames=num_frames,
            sampling_rate=len(indices) / num_frames,
            strategy_used="scene_aware",
            metadata={
                "num_scenes": len(scenes),
                "scene_boundaries": [(s, e) for s, e in scenes],
                "scene_threshold": self.scene_threshold,
            },
        )

        self.log_sampling_info(result)
        return result
