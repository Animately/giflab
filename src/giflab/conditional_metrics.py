"""
Conditional Metrics Calculator for GifLab.

This module implements intelligent, conditional metric calculation with early exit
strategies to optimize performance. It uses quick quality assessment to determine
which metrics are necessary, avoiding expensive calculations when possible.
"""

import hashlib
import logging
import os
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

import numpy as np
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio

logger = logging.getLogger(__name__)

class QualityTier(Enum):
    """Quality tier classifications for progressive metric calculation."""

    HIGH = "high"  # Quality > 0.9, minimal artifacts
    MEDIUM = "medium"  # Quality 0.5-0.9, some artifacts
    LOW = "low"  # Quality < 0.5, significant artifacts
    UNKNOWN = "unknown"  # Unable to determine

@dataclass
class ContentProfile:
    """Profile of GIF content characteristics."""

    has_text: bool = False
    has_ui_elements: bool = False
    has_temporal_changes: bool = False
    has_color_gradients: bool = False
    is_monochrome: bool = False
    frame_similarity: float = 0.0  # Average similarity between frames
    complexity_score: float = 0.0  # Overall complexity rating

@dataclass
class QualityAssessment:
    """Initial quality assessment results."""

    tier: QualityTier
    confidence: float  # 0.0 to 1.0
    base_psnr: float
    base_mse: float
    frame_consistency: float
    details: dict[str, Any]

class FrameHashCache:
    """Cache for frame hashes and similarity detection."""

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache: dict[str, np.ndarray] = {}
        self.hash_cache: dict[str, bool] = {}
        self.similarity_cache: dict[tuple[str, str], float] = {}
        self.access_count = 0
        self.hit_count = 0

    def get_frame_hash(self, frame: np.ndarray) -> str:
        """Generate or retrieve hash for a frame."""
        frame_bytes = frame.tobytes()

        # Generate hash
        hasher = hashlib.md5()
        hasher.update(frame_bytes)
        frame_hash = hasher.hexdigest()

        # Check if we've seen this hash before
        if frame_hash in self.hash_cache:
            self.hit_count += 1
        else:
            # Cache with LRU eviction
            if len(self.hash_cache) >= self.max_size:
                # Simple eviction - remove first item
                self.hash_cache.pop(next(iter(self.hash_cache)))

            self.hash_cache[frame_hash] = True

        self.access_count += 1

        return frame_hash

    def get_similarity(self, hash1: str, hash2: str) -> float | None:
        """Retrieve cached similarity score between two frames."""
        key = tuple(sorted([hash1, hash2]))
        return self.similarity_cache.get(key)

    def cache_similarity(self, hash1: str, hash2: str, similarity: float):
        """Cache similarity score between two frames."""
        key = tuple(sorted([hash1, hash2]))

        if len(self.similarity_cache) >= self.max_size:
            # Simple eviction
            self.similarity_cache.pop(next(iter(self.similarity_cache)))

        self.similarity_cache[key] = similarity

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        return {
            "hash_cache_size": len(self.hash_cache),
            "similarity_cache_size": len(self.similarity_cache),
            "total_accesses": self.access_count,
            "cache_hits": self.hit_count,
            "hit_rate": self.hit_count / max(1, self.access_count),
        }

class ConditionalMetricsCalculator:
    """
    Implements conditional metric calculation with early exit strategies.

    This calculator performs quick quality assessment first, then determines
    which metrics are necessary based on quality tier and content profile.
    """

    def __init__(
        self,
        enable_caching: bool = True,
        quality_thresholds: dict[str, float] | None = None,
        max_sample_frames: int = 5,
    ):
        """
        Initialize the conditional metrics calculator.

        Args:
            enable_caching: Whether to enable frame hash caching
            quality_thresholds: Custom quality tier thresholds
            max_sample_frames: Maximum frames to sample for initial assessment
        """
        self.enable_caching = enable_caching
        self.frame_cache = FrameHashCache() if enable_caching else None
        self.max_sample_frames = max_sample_frames

        # Default quality thresholds (can be overridden)
        self.quality_thresholds = quality_thresholds or {
            "high": 0.9,  # PSNR > 35 dB typically
            "medium": 0.5,  # PSNR 25-35 dB
            "low": 0.0,  # PSNR < 25 dB
        }

        # Load configuration from environment
        self._load_env_config()

        # Track metrics for performance analysis
        self.metrics_skipped = 0
        self.metrics_calculated = 0
        self.time_saved = 0.0

    def _load_env_config(self):
        """Load configuration from environment variables."""
        # Enable/disable conditional processing
        self.enabled = (
            os.environ.get("GIFLAB_ENABLE_CONDITIONAL_METRICS", "true").lower()
            == "true"
        )

        # Quality thresholds
        if "GIFLAB_QUALITY_HIGH_THRESHOLD" in os.environ:
            self.quality_thresholds["high"] = float(
                os.environ["GIFLAB_QUALITY_HIGH_THRESHOLD"]
            )
        if "GIFLAB_QUALITY_MEDIUM_THRESHOLD" in os.environ:
            self.quality_thresholds["medium"] = float(
                os.environ["GIFLAB_QUALITY_MEDIUM_THRESHOLD"]
            )

        # Sample frame count
        if "GIFLAB_QUALITY_SAMPLE_FRAMES" in os.environ:
            self.max_sample_frames = int(os.environ["GIFLAB_QUALITY_SAMPLE_FRAMES"])

        # Feature flags for specific optimizations
        self.skip_expensive_on_high_quality = (
            os.environ.get("GIFLAB_SKIP_EXPENSIVE_ON_HIGH_QUALITY", "true").lower()
            == "true"
        )

        self.use_progressive_calculation = (
            os.environ.get("GIFLAB_USE_PROGRESSIVE_CALCULATION", "true").lower()
            == "true"
        )

        self.cache_frame_hashes = (
            os.environ.get("GIFLAB_CACHE_FRAME_HASHES", "true").lower() == "true"
        )

        logger.debug(
            f"Conditional metrics config: enabled={self.enabled}, "
            f"thresholds={self.quality_thresholds}, "
            f"sample_frames={self.max_sample_frames}"
        )

    def assess_quality(
        self, frames_original: list[np.ndarray], frames_compressed: list[np.ndarray]
    ) -> QualityAssessment:
        """
        Perform quick quality assessment using fast metrics.

        Args:
            frames_original: Original frames
            frames_compressed: Compressed frames

        Returns:
            Quality assessment with tier classification
        """
        start_time = time.time()

        # Sample frames for quick assessment
        num_frames = min(len(frames_original), self.max_sample_frames)
        if num_frames < len(frames_original):
            # Sample evenly across the GIF
            indices = np.linspace(0, len(frames_original) - 1, num_frames, dtype=int)
        else:
            indices = range(num_frames)

        # Calculate quick metrics on sampled frames
        mse_values = []
        psnr_values = []

        for idx in indices:
            frame_orig = frames_original[idx]
            frame_comp = frames_compressed[idx]

            # Ensure frames are in the same format
            if frame_orig.shape != frame_comp.shape:
                logger.warning(f"Frame shape mismatch at index {idx}")
                continue

            # Calculate MSE and PSNR
            mse = mean_squared_error(frame_orig, frame_comp)
            mse_values.append(mse)

            if mse > 0:
                # Calculate PSNR with proper scaling
                if frame_orig.dtype == np.uint8:
                    max_val = 255.0
                else:
                    max_val = 1.0
                psnr = 10 * np.log10((max_val**2) / mse)
                psnr_values.append(psnr)
            else:
                psnr_values.append(float("inf"))  # Perfect match

        # Aggregate metrics
        avg_mse = np.mean(mse_values) if mse_values else 0.0

        # Handle PSNR averaging (filter out infinity values)
        finite_psnr = [p for p in psnr_values if p != float("inf")]
        if finite_psnr:
            avg_psnr = np.mean(finite_psnr)
        elif any(p == float("inf") for p in psnr_values):
            # All or some values are perfect (infinite PSNR)
            avg_psnr = 40.0  # Use high value for perfect quality
        else:
            avg_psnr = 0.0

        # Calculate frame consistency (variance in quality)
        if finite_psnr:
            frame_consistency = 1.0 - (
                np.std(finite_psnr) / max(1.0, np.mean(finite_psnr))
            )
        elif all(p == float("inf") for p in psnr_values):
            frame_consistency = 1.0  # Perfect consistency for identical frames
        else:
            frame_consistency = 0.5
        frame_consistency = max(0.0, min(1.0, frame_consistency))

        # Determine quality tier based on PSNR
        if avg_psnr >= 35:  # Excellent quality
            tier = QualityTier.HIGH
            confidence = min(1.0, avg_psnr / 40.0)
        elif avg_psnr >= 25:  # Good quality
            tier = QualityTier.MEDIUM
            confidence = 0.5 + (avg_psnr - 25) / 20.0
        else:  # Poor quality
            tier = QualityTier.LOW
            confidence = max(0.1, avg_psnr / 25.0)

        # Adjust confidence based on consistency
        confidence *= 0.7 + 0.3 * frame_consistency

        assessment_time = time.time() - start_time

        return QualityAssessment(
            tier=tier,
            confidence=confidence,
            base_psnr=avg_psnr,
            base_mse=avg_mse,
            frame_consistency=frame_consistency,
            details={
                "num_samples": len(indices),
                "assessment_time": assessment_time,
                "mse_values": mse_values,
                "psnr_values": psnr_values,
            },
        )

    def detect_content_profile(
        self, frames: list[np.ndarray], quick_mode: bool = True
    ) -> ContentProfile:
        """
        Detect content characteristics to guide metric selection.

        Args:
            frames: Frames to analyze
            quick_mode: Use quick heuristics vs full analysis

        Returns:
            Content profile with detected characteristics
        """
        profile = ContentProfile()

        if not frames:
            return profile

        # Sample frames for analysis
        sample_size = min(3 if quick_mode else 5, len(frames))
        sample_indices = np.linspace(0, len(frames) - 1, sample_size, dtype=int)
        sampled_frames = [frames[i] for i in sample_indices]

        # Check for monochrome content
        profile.is_monochrome = self._is_monochrome(sampled_frames[0])

        # Detect temporal changes
        if len(frames) > 1:
            profile.has_temporal_changes = self._detect_temporal_changes(sampled_frames)
            profile.frame_similarity = self._calculate_frame_similarity(sampled_frames)

        # Quick content detection using edge detection
        for frame in sampled_frames[:2]:  # Check first 2 samples
            edges = self._detect_edges_quick(frame)
            edge_density = np.mean(edges)

            # Heuristics for content detection
            if edge_density > 0.1:  # High edge density suggests UI/text
                profile.has_ui_elements = True

            if edge_density > 0.15 and self._has_rectangular_regions(edges):
                profile.has_text = True

            # Check for gradients (smooth transitions)
            if not profile.is_monochrome:
                profile.has_color_gradients = self._detect_gradients(frame)

        # Calculate overall complexity
        profile.complexity_score = self._calculate_complexity_score(profile)

        return profile

    def _is_monochrome(self, frame: np.ndarray) -> bool:
        """Check if frame is monochrome."""
        if len(frame.shape) == 2:
            return True

        if len(frame.shape) == 3 and frame.shape[2] >= 3:
            # Check if R, G, B channels are similar
            r, g, b = frame[:, :, 0], frame[:, :, 1], frame[:, :, 2]
            rg_diff = np.mean(np.abs(r - g))
            rb_diff = np.mean(np.abs(r - b))

            # Threshold for considering as monochrome
            threshold = 5 if frame.dtype == np.uint8 else 0.02
            return bool(rg_diff < threshold and rb_diff < threshold)

        return False

    def _detect_temporal_changes(self, frames: list[np.ndarray]) -> bool:
        """Detect if there are significant temporal changes between frames."""
        if len(frames) < 2:
            return False

        # Calculate frame differences
        differences = []
        for i in range(1, len(frames)):
            diff = np.mean(np.abs(frames[i] - frames[i - 1]))
            differences.append(diff)

        # Threshold for significant change
        threshold = 10 if frames[0].dtype == np.uint8 else 0.04
        return bool(any(d > threshold for d in differences))

    def _calculate_frame_similarity(self, frames: list[np.ndarray]) -> float:
        """Calculate average similarity between consecutive frames."""
        if len(frames) < 2:
            return 1.0

        similarities = []
        for i in range(1, len(frames)):
            # Use cached similarity if available
            if self.frame_cache:
                hash1 = self.frame_cache.get_frame_hash(frames[i - 1])
                hash2 = self.frame_cache.get_frame_hash(frames[i])

                cached_sim = self.frame_cache.get_similarity(hash1, hash2)
                if cached_sim is not None:
                    similarities.append(cached_sim)
                    continue

            # Calculate similarity
            mse = mean_squared_error(frames[i - 1], frames[i])
            max_val = 255.0 if frames[0].dtype == np.uint8 else 1.0
            similarity = 1.0 - (mse / (max_val**2))
            similarities.append(similarity)

            # Cache the result
            if self.frame_cache:
                self.frame_cache.cache_similarity(hash1, hash2, similarity)

        return np.mean(similarities) if similarities else 1.0

    def _detect_edges_quick(self, frame: np.ndarray) -> np.ndarray:
        """Quick edge detection using Sobel-like filter."""
        if len(frame.shape) == 3:
            # Convert to grayscale
            gray = np.mean(frame, axis=2)
        else:
            gray = frame

        # Simple edge detection using differences
        edges_h = np.abs(np.diff(gray, axis=0))
        edges_v = np.abs(np.diff(gray, axis=1))

        # Combine and threshold
        edges = np.zeros_like(gray)
        edges[:-1, :] += edges_h
        edges[:, :-1] += edges_v

        threshold = 20 if frame.dtype == np.uint8 else 0.08
        return edges > threshold

    def _has_rectangular_regions(self, edges: np.ndarray) -> bool:
        """Check if edge map contains rectangular regions (suggesting UI/text)."""
        # Simple heuristic: look for horizontal and vertical lines
        h_lines = np.sum(edges, axis=1)
        v_lines = np.sum(edges, axis=0)

        # Count significant lines
        h_threshold = edges.shape[1] * 0.3
        v_threshold = edges.shape[0] * 0.3

        num_h_lines = np.sum(h_lines > h_threshold)
        num_v_lines = np.sum(v_lines > v_threshold)

        # Multiple lines suggest rectangular regions
        return bool(num_h_lines > 2 and num_v_lines > 2)

    def _detect_gradients(self, frame: np.ndarray) -> bool:
        """Detect if frame contains smooth gradients."""
        if len(frame.shape) == 3:
            # Check each channel
            for channel in range(min(3, frame.shape[2])):
                if self._channel_has_gradient(frame[:, :, channel]):
                    return True
        else:
            return self._channel_has_gradient(frame)

        return False

    def _channel_has_gradient(self, channel: np.ndarray) -> bool:
        """Check if a single channel has smooth gradients."""
        # Calculate local variance
        kernel_size = 5
        h, w = channel.shape

        if h < kernel_size or w < kernel_size:
            return False

        # Sample patches
        num_samples = 10
        variances = []

        for _ in range(num_samples):
            y = np.random.randint(0, h - kernel_size)
            x = np.random.randint(0, w - kernel_size)
            patch = channel[y : y + kernel_size, x : x + kernel_size]
            variances.append(np.var(patch))

        # Low variance suggests smooth gradients
        avg_variance = np.mean(variances)
        threshold = 100 if channel.dtype == np.uint8 else 0.01

        return bool(avg_variance < threshold)

    def _calculate_complexity_score(self, profile: ContentProfile) -> float:
        """Calculate overall complexity score from content profile."""
        score = 0.0

        if profile.has_text:
            score += 0.3
        if profile.has_ui_elements:
            score += 0.2
        if profile.has_temporal_changes:
            score += 0.3
        if profile.has_color_gradients:
            score += 0.2

        # Adjust for frame similarity
        score *= 2.0 - profile.frame_similarity

        return min(1.0, score)

    def select_metrics(
        self, quality: QualityAssessment, profile: ContentProfile
    ) -> dict[str, bool]:
        """
        Select which metrics to calculate based on quality and content.

        Args:
            quality: Quality assessment results
            profile: Content profile

        Returns:
            Dictionary of metric names and whether to calculate them
        """
        metrics = {
            # Basic metrics (always calculated)
            "mse": True,
            "psnr": True,
            "ssim": True,
            # Advanced metrics (conditional)
            "lpips": False,
            "fsim": False,
            "vif": False,
            "ssimulacra2": False,
            # Specialized metrics (conditional)
            "temporal_artifacts": False,
            "text_ui_validation": False,
            "color_gradients": False,
            "edge_similarity": False,
            "texture_similarity": False,
        }

        # High quality: skip expensive metrics
        if quality.tier == QualityTier.HIGH and self.skip_expensive_on_high_quality:
            logger.info(
                f"High quality detected (PSNR={quality.base_psnr:.1f}), "
                "skipping expensive metrics"
            )
            self.metrics_calculated += 3  # Basic metrics always calculated
            self.metrics_skipped += (
                9  # Count skipped expensive metrics (12 total - 3 basic)
            )
            return metrics

        # Low quality: enable all relevant metrics
        if quality.tier == QualityTier.LOW:
            metrics["lpips"] = True
            metrics["fsim"] = True
            metrics["vif"] = True
            metrics["ssimulacra2"] = True
            metrics["edge_similarity"] = True
            metrics["texture_similarity"] = True

            # Enable specialized metrics based on content
            if profile.has_temporal_changes:
                metrics["temporal_artifacts"] = True
            if profile.has_text or profile.has_ui_elements:
                metrics["text_ui_validation"] = True
            if profile.has_color_gradients:
                metrics["color_gradients"] = True

            self.metrics_calculated += sum(1 for v in metrics.values() if v)
            return metrics

        # Medium quality: selective metrics based on content and confidence
        if quality.tier == QualityTier.MEDIUM:
            # Always include some perceptual metrics for medium quality
            metrics["lpips"] = True
            metrics["fsim"] = quality.confidence < 0.7  # Include if less confident

            # Include SSIMULACRA2 for lower confidence
            if quality.confidence < 0.6:
                metrics["ssimulacra2"] = True

            # Content-based selection
            if profile.complexity_score > 0.5:
                metrics["vif"] = True
                metrics["texture_similarity"] = True

            if profile.has_temporal_changes and profile.frame_similarity < 0.8:
                metrics["temporal_artifacts"] = True

            if profile.has_text or profile.has_ui_elements:
                metrics["text_ui_validation"] = True
                metrics["edge_similarity"] = True

            if profile.has_color_gradients:
                metrics["color_gradients"] = True

            self.metrics_calculated += sum(1 for v in metrics.values() if v)
            return metrics

        # Unknown quality: be conservative, calculate most metrics
        metrics.update(
            {"lpips": True, "fsim": True, "ssimulacra2": True, "edge_similarity": True}
        )

        self.metrics_calculated += sum(1 for v in metrics.values() if v)
        return metrics

    def calculate_progressive(
        self,
        frames_original: list[np.ndarray],
        frames_compressed: list[np.ndarray],
        metrics_calculator: Any,
        force_all: bool = False,
    ) -> dict[str, Any]:
        """
        Calculate metrics progressively with early exit.

        Args:
            frames_original: Original frames
            frames_compressed: Compressed frames
            metrics_calculator: The main metrics calculator instance
            force_all: Force calculation of all metrics (bypass optimization)

        Returns:
            Dictionary of calculated metrics
        """
        if force_all or not self.enabled:
            # Bypass optimization, calculate all metrics
            logger.debug("Conditional metrics bypassed, calculating all metrics")
            return metrics_calculator.calculate_all_metrics(
                frames_original, frames_compressed
            )

        start_time = time.time()

        # Step 1: Quick quality assessment
        quality = self.assess_quality(frames_original, frames_compressed)
        logger.info(
            f"Quality assessment: {quality.tier.value} "
            f"(confidence={quality.confidence:.2f}, PSNR={quality.base_psnr:.1f})"
        )

        # Step 2: Content profiling
        profile = self.detect_content_profile(frames_compressed, quick_mode=True)
        logger.debug(
            f"Content profile: text={profile.has_text}, "
            f"ui={profile.has_ui_elements}, "
            f"temporal={profile.has_temporal_changes}, "
            f"complexity={profile.complexity_score:.2f}"
        )

        # Step 3: Select metrics to calculate
        selected_metrics = self.select_metrics(quality, profile)

        num_selected = sum(1 for v in selected_metrics.values() if v)
        num_skipped = sum(1 for v in selected_metrics.values() if not v)
        logger.info(f"Calculating {num_selected} metrics, skipping {num_skipped}")

        # Step 4: Calculate selected metrics
        results = metrics_calculator.calculate_selected_metrics(
            frames_original, frames_compressed, selected_metrics
        )

        # Add metadata about the optimization
        results["_optimization_metadata"] = {
            "quality_tier": quality.tier.value,
            "quality_confidence": quality.confidence,
            "base_psnr": quality.base_psnr,
            "content_complexity": profile.complexity_score,
            "metrics_calculated": num_selected,
            "metrics_skipped": num_skipped,
            "optimization_time": time.time() - start_time,
        }

        # Track time saved (estimate)
        self.time_saved += num_skipped * 0.1  # Rough estimate: 0.1s per skipped metric

        return results

    def get_optimization_stats(self) -> dict[str, Any]:
        """Get statistics about optimization performance."""
        stats = {
            "metrics_calculated": self.metrics_calculated,
            "metrics_skipped": self.metrics_skipped,
            "estimated_time_saved": self.time_saved,
            "optimization_ratio": self.metrics_skipped
            / max(1, self.metrics_calculated + self.metrics_skipped),
        }

        if self.frame_cache:
            stats["cache_stats"] = self.frame_cache.get_cache_stats()

        return stats

    def reset_stats(self):
        """Reset optimization statistics."""
        self.metrics_calculated = 0
        self.metrics_skipped = 0
        self.time_saved = 0.0

        if self.frame_cache:
            self.frame_cache = FrameHashCache()
