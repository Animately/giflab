"""
Enhanced temporal artifact detection for GIF compression validation.

This module implements advanced temporal artifact detection metrics designed specifically
for debugging compression failures, including:
- Flicker excess detection using LPIPS-T between consecutive frames
- Flat-region flicker detection for background stability validation
- Enhanced disposal artifact detection with better background tracking
- Temporal pumping detection for quality oscillation
"""

import logging
from typing import Any, Literal, Optional, Protocol, TypedDict, Union

import cv2
import numpy as np
import torch

# Import model caching to prevent memory leaks
from .model_cache import LPIPSModelCache

logger = logging.getLogger(__name__)

try:
    import lpips

    LPIPS_AVAILABLE = True
except ImportError:
    logger.warning(
        "LPIPS not available. Temporal artifact detection will use fallback methods."
    )
    LPIPS_AVAILABLE = False


# Type definitions
class LPIPSModel(Protocol):
    """Protocol for LPIPS model interface."""

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        ...

    def eval(self) -> None:
        ...

    def to(self, device: str) -> "LPIPSModel":
        ...


class TemporalMetrics(TypedDict):
    """Type definition for temporal artifact metrics."""

    lpips_t_mean: float
    lpips_t_p95: float
    lpips_t_max: float
    lpips_frame_count: int


class FlickerMetrics(TypedDict):
    """Type definition for flicker metrics."""

    flicker_excess: float
    flicker_frame_count: int
    flicker_frame_ratio: float
    lpips_t_mean: float
    lpips_t_p95: float


class MemoryMonitor:
    """Monitor and manage GPU memory usage for batch processing."""

    def __init__(self, device: str, memory_threshold: float = 0.8):
        """Initialize memory monitor.

        Args:
            device: PyTorch device string
            memory_threshold: Memory usage threshold (0.0-1.0) above which to trigger cleanup
        """
        self.device = device
        self.memory_threshold = memory_threshold
        self.is_cuda = device.startswith("cuda")
        self._baseline_memory = self._get_memory_usage() if self.is_cuda else 0

    def _get_memory_usage(self) -> float:
        """Get current memory usage as fraction of total."""
        if not self.is_cuda:
            return 0.0

        try:
            allocated = torch.cuda.memory_allocated(self.device)
            total = torch.cuda.get_device_properties(self.device).total_memory
            return allocated / total
        except Exception:
            return 0.0

    def get_safe_batch_size(
        self,
        frame_shape: tuple[int, int, int],
        max_batch_size: int = 32,
        min_batch_size: int = 1,
    ) -> int:
        """Calculate safe batch size based on memory and frame dimensions.

        Args:
            frame_shape: (height, width, channels) of frames
            max_batch_size: Maximum batch size to consider
            min_batch_size: Minimum batch size to return

        Returns:
            Safe batch size for processing
        """
        if not self.is_cuda:
            return max_batch_size  # No memory concerns on CPU

        try:
            # Estimate memory per frame pair (input + preprocessed + gradients)
            h, w, c = frame_shape
            bytes_per_frame = (
                h * w * c * 4 * 6
            )  # float32 * 6 (rough estimate for all tensors)

            available = torch.cuda.get_device_properties(self.device).total_memory
            current_usage = torch.cuda.memory_allocated(self.device)
            free_memory = (available * self.memory_threshold) - current_usage

            if free_memory <= 0:
                return min_batch_size

            # Calculate batch size with safety margin
            safe_batch = int(free_memory // bytes_per_frame * 0.7)  # 70% safety margin
            return max(min_batch_size, min(safe_batch, max_batch_size))

        except (RuntimeError, AttributeError, TypeError) as e:
            logger.debug(f"Error calculating batch size: {e}, using default")
            return max(4, max_batch_size // 4)  # Conservative fallback

    def should_cleanup_memory(self) -> bool:
        """Check if memory cleanup is needed."""
        if not self.is_cuda:
            return False
        return self._get_memory_usage() > self.memory_threshold

    def cleanup_memory(self) -> None:
        """Perform memory cleanup if needed."""
        if self.should_cleanup_memory():
            logger.debug(f"Memory usage above {self.memory_threshold:.1%}, cleaning up")
            torch.cuda.empty_cache()

    def log_memory_usage(self, context: str = "") -> None:
        """Log current memory usage for debugging."""
        if self.is_cuda:
            usage = self._get_memory_usage()
            logger.debug(
                f"GPU memory usage{' (' + context + ')' if context else ''}: {usage:.1%}"
            )


class TemporalArtifactDetector:
    """Enhanced temporal artifact detector using perceptual metrics."""

    def __init__(
        self,
        device: str | None = None,
        force_mse_fallback: bool = False,
        memory_threshold: float = 0.8,
    ):
        """Initialize temporal artifact detector.

        Args:
            device: PyTorch device ('cpu', 'cuda', etc.). Auto-detects if None.
            force_mse_fallback: If True, skip LPIPS and use enhanced MSE fallback.
                               Useful for consistent behavior or when LPIPS is problematic.
            memory_threshold: Memory usage threshold for cleanup (0.0-1.0).
        """
        # Auto-detect device if not specified
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            # Validate explicitly requested device
            if device.startswith("cuda") and not torch.cuda.is_available():
                logger.warning(
                    f"CUDA device '{device}' requested but CUDA is not available, falling back to CPU"
                )
                self.device = "cpu"
            else:
                self.device = device
        self._lpips_model: LPIPSModel | Literal[False] | None = None
        self.force_mse_fallback = force_mse_fallback
        self.memory_monitor = MemoryMonitor(self.device, memory_threshold)

        if force_mse_fallback:
            logger.info("LPIPS disabled by configuration, using enhanced MSE fallback")
            self._lpips_model = False

    def _get_lpips_model(self) -> LPIPSModel | None:
        """Lazy initialization of LPIPS model with enhanced fallback handling."""
        if (
            self._lpips_model is None
            and LPIPS_AVAILABLE
            and not self.force_mse_fallback
        ):
            try:
                # Check device compatibility
                if self.device == "cuda" and not torch.cuda.is_available():
                    logger.warning(
                        "CUDA requested but not available, falling back to CPU for LPIPS"
                    )
                    self.device = "cpu"
                    # Update memory monitor with new device
                    self.memory_monitor = MemoryMonitor(
                        self.device, self.memory_monitor.memory_threshold
                    )

                # Initialize LPIPS model
                logger.debug(f"Initializing LPIPS model on device: {self.device}")
                # Use cached model to prevent memory leaks
                model = LPIPSModelCache.get_model(
                    net="alex", version="0.1", spatial=False, device=self.device
                )
                if model is not None:
                    self._lpips_model = model
                logger.info("LPIPS model initialized successfully")
                self.memory_monitor.log_memory_usage("after LPIPS init")

            except RuntimeError as e:
                if (
                    "out of memory" in str(e).lower()
                    or "cuda out of memory" in str(e).lower()
                ):
                    logger.warning(
                        f"LPIPS model failed due to memory constraints: {e}. "
                        "Falling back to CPU or MSE-based processing."
                    )
                    # Try CPU fallback if we were using CUDA
                    if self.device.startswith("cuda"):
                        try:
                            logger.info("Attempting CPU fallback for LPIPS model")
                            self.device = "cpu"
                            self.memory_monitor = MemoryMonitor(self.device)
                            # Use cached model for CPU fallback too
                            cpu_model = LPIPSModelCache.get_model(
                                net="alex", version="0.1", spatial=False, device="cpu"
                            )
                            if cpu_model is not None:
                                self._lpips_model = cpu_model
                            logger.info("LPIPS model initialized successfully on CPU")
                            return cpu_model  # type: ignore[no-any-return]
                        except (RuntimeError, ImportError, AttributeError) as cpu_e:
                            logger.warning(f"CPU fallback also failed: {cpu_e}")
                else:
                    logger.warning(f"LPIPS model initialization failed: {e}")
                self._lpips_model = False

            except ImportError as e:
                logger.warning(f"LPIPS import failed: {e}. Using MSE-based fallback.")
                self._lpips_model = False

            except (AttributeError, TypeError, OSError) as e:
                logger.warning(
                    f"Unexpected error initializing LPIPS model: {e}. "
                    "Using MSE-based fallback for temporal artifact detection."
                )
                self._lpips_model = False

        return (
            self._lpips_model
            if isinstance(self._lpips_model, type(self._lpips_model))
            and self._lpips_model is not False
            else None
        )

    def __del__(self) -> None:
        """Clean up resources when the detector is destroyed."""
        # Release LPIPS model reference
        if hasattr(self, "_lpips_model"):
            if self._lpips_model is not None and self._lpips_model is not False:
                # Release the model reference from cache
                if hasattr(self, "device"):
                    LPIPSModelCache.release_model(
                        net="alex", version="0.1", spatial=False, device=self.device
                    )
                self._lpips_model = None

    def _prepare_frame_pairs(
        self, frames: list[np.ndarray], batch_start: int, batch_end: int
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Prepare frame pairs for LPIPS processing.

        Args:
            frames: List of RGB frames
            batch_start: Starting index for batch
            batch_end: Ending index for batch

        Returns:
            Tuple of (frame1_tensors, frame2_tensors)
        """
        frame1_tensors = []
        frame2_tensors = []

        for i in range(batch_start, batch_end):
            frame1_tensor = self.preprocess_for_lpips(frames[i])
            frame2_tensor = self.preprocess_for_lpips(frames[i + 1])
            frame1_tensors.append(frame1_tensor)
            frame2_tensors.append(frame2_tensor)

        return frame1_tensors, frame2_tensors

    def _cleanup_tensors(self, *tensor_args: torch.Tensor | list[torch.Tensor]) -> None:
        """Clean up tensors and tensor lists, triggering memory cleanup if needed.

        Args:
            *tensor_args: Variable number of tensors or tensor lists to delete
        """
        for tensor_arg in tensor_args:
            if isinstance(tensor_arg, list):
                # Handle list of tensors
                for tensor in tensor_arg:
                    del tensor
                del tensor_arg
            else:
                # Handle individual tensor
                del tensor_arg

        # Only cleanup memory when needed, not after every batch
        self.memory_monitor.cleanup_memory()

    def _process_lpips_batch(
        self, frames: list[np.ndarray], batch_size: int = 8
    ) -> list[float]:
        """Process frames through LPIPS model in batches with adaptive sizing.

        Args:
            frames: List of RGB frames
            batch_size: Initial batch size (will be adapted based on memory)

        Returns:
            List of LPIPS scores between consecutive frames
        """
        lpips_model = self._get_lpips_model()
        if lpips_model is None:
            return []  # Fallback will be handled by caller

        lpips_scores = []
        num_comparisons = len(frames) - 1

        # Adapt batch size based on memory and frame dimensions if we have frames
        if frames:
            frame_shape = frames[0].shape
            # Ensure we have a 3D shape (H, W, C) for batch size calculation
            if len(frame_shape) == 2:
                # Grayscale image - add channel dimension
                normalized_shape = (frame_shape[0], frame_shape[1], 1)
            elif len(frame_shape) == 3:
                # RGB image - use as is
                normalized_shape = frame_shape
            else:
                # Unexpected shape - use a reasonable default
                normalized_shape = (256, 256, 3)
            batch_size = self.memory_monitor.get_safe_batch_size(
                normalized_shape, batch_size
            )

        self.memory_monitor.log_memory_usage("before LPIPS processing")

        try:
            with torch.no_grad():
                for batch_start in range(0, num_comparisons, batch_size):
                    batch_end = min(batch_start + batch_size, num_comparisons)

                    # Prepare batch tensors
                    frame1_tensors, frame2_tensors = self._prepare_frame_pairs(
                        frames, batch_start, batch_end
                    )

                    if frame1_tensors:  # Only process if we have frames
                        try:
                            # Stack into batch tensors
                            batch_frame1 = torch.cat(frame1_tensors, dim=0)
                            batch_frame2 = torch.cat(frame2_tensors, dim=0)

                            # Calculate perceptual distances for the batch
                            batch_distances = lpips_model(batch_frame1, batch_frame2)

                            # Extract individual scores
                            for distance in batch_distances:
                                lpips_scores.append(float(distance.cpu().item()))

                            # Clean up batch tensors
                            self._cleanup_tensors(
                                batch_frame1,
                                batch_frame2,
                                batch_distances,
                                frame1_tensors,
                                frame2_tensors,
                            )

                        except RuntimeError as e:
                            if "out of memory" in str(e).lower():
                                logger.warning(
                                    f"OOM in batch processing, reducing batch size: {e}"
                                )
                                # Cleanup and try with smaller batch
                                self._cleanup_tensors(frame1_tensors, frame2_tensors)
                                if hasattr(locals(), "batch_frame1"):
                                    self._cleanup_tensors(batch_frame1, batch_frame2)
                                if hasattr(locals(), "batch_distances"):
                                    del batch_distances

                                # Force cleanup and retry with smaller batch
                                torch.cuda.empty_cache()
                                smaller_batch_size = max(1, batch_size // 2)
                                logger.info(
                                    f"Retrying with reduced batch size: {smaller_batch_size}"
                                )

                                # Recursive call with smaller batch size for remaining frames
                                remaining_frames = frames[batch_start:]
                                if len(remaining_frames) > 1:
                                    remaining_scores = self._process_lpips_batch(
                                        remaining_frames, smaller_batch_size
                                    )
                                    lpips_scores.extend(remaining_scores)
                                break
                            else:
                                raise  # Re-raise non-OOM errors

        except (RuntimeError, AttributeError, TypeError) as e:
            logger.error(f"Error in LPIPS batch processing: {e}")
            raise  # Let caller handle fallback

        self.memory_monitor.log_memory_usage("after LPIPS processing")
        return lpips_scores

    def preprocess_for_lpips(self, frame: np.ndarray) -> torch.Tensor:
        """Preprocess frame for LPIPS calculation.

        Args:
            frame: RGB frame as numpy array [H, W, 3]

        Returns:
            Preprocessed tensor ready for LPIPS [1, 3, H, W]
        """
        # Ensure frame is RGB and float32
        if frame.dtype != np.float32:
            frame = frame.astype(np.float32) / 255.0

        # Convert to torch tensor and normalize to [-1, 1]
        frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0)
        frame_tensor = frame_tensor * 2.0 - 1.0

        return frame_tensor.to(self.device)

    def calculate_lpips_temporal(
        self, frames: list[np.ndarray], batch_size: int = 8
    ) -> TemporalMetrics:
        """Calculate LPIPS between consecutive frames for temporal consistency.

        Args:
            frames: List of RGB frames as numpy arrays
            batch_size: Number of frame pairs to process simultaneously for memory efficiency

        Returns:
            Dictionary with LPIPS temporal metrics
        """
        if len(frames) < 2:
            return {
                "lpips_t_mean": 0.0,
                "lpips_t_p95": 0.0,
                "lpips_t_max": 0.0,
                "lpips_frame_count": len(frames),
            }

        # Try LPIPS processing first
        try:
            lpips_scores = self._process_lpips_batch(frames, batch_size)
        except (RuntimeError, AttributeError, TypeError, ValueError) as e:
            logger.error(f"Error calculating LPIPS temporal: {e}")
            logger.debug("Falling back to MSE-based temporal calculation")
            return self._calculate_mse_temporal_fallback(frames)

        # If no scores obtained, fall back to MSE
        if lpips_scores is None or len(lpips_scores) == 0:
            logger.debug("No LPIPS scores obtained, using MSE fallback")
            return self._calculate_mse_temporal_fallback(frames)

        return {
            "lpips_t_mean": float(np.mean(lpips_scores)),
            "lpips_t_p95": float(np.percentile(lpips_scores, 95)),
            "lpips_t_max": float(np.max(lpips_scores)),
            "lpips_frame_count": len(frames),
        }

    def _calculate_mse_temporal_fallback(
        self, frames: list[np.ndarray]
    ) -> TemporalMetrics:
        """Enhanced MSE-based temporal calculation when LPIPS is unavailable.

        Uses perceptually-weighted MSE that better approximates LPIPS behavior
        by emphasizing luminance differences and edge regions.
        """
        mse_scores = []

        for i in range(len(frames) - 1):
            frame1 = frames[i].astype(np.float32) / 255.0
            frame2 = frames[i + 1].astype(np.float32) / 255.0

            # Convert to LAB color space for perceptually uniform differences
            try:
                # Convert RGB to LAB for perceptual weighting
                lab1 = (
                    cv2.cvtColor(
                        (frame1 * 255).astype(np.uint8), cv2.COLOR_RGB2LAB
                    ).astype(np.float32)
                    / 255.0
                )
                lab2 = (
                    cv2.cvtColor(
                        (frame2 * 255).astype(np.uint8), cv2.COLOR_RGB2LAB
                    ).astype(np.float32)
                    / 255.0
                )

                # Weight luminance (L) channel more heavily (matches perceptual importance)
                l_diff = (lab1[:, :, 0] - lab2[:, :, 0]) ** 2
                a_diff = (lab1[:, :, 1] - lab2[:, :, 1]) ** 2
                b_diff = (lab1[:, :, 2] - lab2[:, :, 2]) ** 2

                # Perceptual weighting: L=70%, a=15%, b=15%
                perceptual_mse = 0.7 * l_diff + 0.15 * a_diff + 0.15 * b_diff

                # Add edge-sensitive weighting
                gray1 = (
                    cv2.cvtColor(
                        (frame1 * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY
                    ).astype(np.float32)
                    / 255.0
                )
                edges = (
                    cv2.Canny((gray1 * 255).astype(np.uint8), 50, 150).astype(
                        np.float32
                    )
                    / 255.0
                )

                # Boost differences near edges (where LPIPS is most sensitive)
                edge_weight = 1.0 + edges * 0.5
                weighted_mse = perceptual_mse * edge_weight

                mse = float(np.mean(weighted_mse))

            except (cv2.error, ValueError, TypeError) as e:
                logger.debug(f"LAB conversion failed, using RGB MSE: {e}")
                # Fallback to RGB MSE if LAB conversion fails
                mse = float(np.mean((frame1 - frame2) ** 2))

            # Normalize to approximate LPIPS range [0, 1] with better scaling
            # LPIPS typically ranges 0-0.8 for natural images, so adjust scaling
            normalized_mse = min(mse * 8.0, 1.0)  # More aggressive scaling than before
            mse_scores.append(normalized_mse)

        if not mse_scores:
            return {
                "lpips_t_mean": 0.0,
                "lpips_t_p95": 0.0,
                "lpips_t_max": 0.0,
                "lpips_frame_count": len(frames),
            }

        return {
            "lpips_t_mean": float(np.mean(mse_scores)),
            "lpips_t_p95": float(np.percentile(mse_scores, 95)),
            "lpips_t_max": float(np.max(mse_scores)),
            "lpips_frame_count": len(frames),
        }

    def detect_flicker_excess(
        self, frames: list[np.ndarray], threshold: float = 0.02, batch_size: int = 8
    ) -> FlickerMetrics:
        """Detect flicker excess using LPIPS-T between consecutive frames.

        Flicker manifests as high perceptual differences between consecutive frames
        that don't correlate with intended animation patterns.

        Args:
            frames: List of RGB frames
            threshold: LPIPS-T threshold above which flicker is detected
            batch_size: Number of frame pairs to process simultaneously

        Returns:
            Dictionary with flicker excess metrics
        """
        # Use the optimized temporal calculation that already handles batch processing
        lpips_metrics = self.calculate_lpips_temporal(frames, batch_size=batch_size)

        # Get individual scores for flicker analysis
        # We need the raw scores, so we'll get them from our batch processor
        lpips_scores = []
        if len(frames) >= 2:
            try:
                lpips_scores = self._process_lpips_batch(frames, batch_size)
            except (RuntimeError, AttributeError, TypeError, ValueError) as e:
                logger.error(f"Error getting LPIPS scores for flicker analysis: {e}")
                # Use empty scores, will result in zero flicker metrics

        # Analyze flicker patterns
        flicker_excess = 0.0
        flicker_frames = 0

        if lpips_scores:
            # Identify frames with excessive perceptual difference
            excessive_frames = [score > threshold for score in lpips_scores]
            flicker_frames = sum(excessive_frames)

            # Calculate excess beyond threshold
            excess_values = [max(0, score - threshold) for score in lpips_scores]
            flicker_excess = float(np.mean(excess_values))

        return {
            "flicker_excess": flicker_excess,
            "flicker_frame_count": flicker_frames,
            "flicker_frame_ratio": flicker_frames / max(1, len(frames) - 1),
            "lpips_t_mean": lpips_metrics["lpips_t_mean"],
            "lpips_t_p95": lpips_metrics["lpips_t_p95"],
        }

    def identify_flat_regions(
        self, frame: np.ndarray, variance_threshold: float = 10.0
    ) -> list[tuple[int, int, int, int]]:
        """Identify flat/stable regions in a frame that should not flicker.

        Args:
            frame: RGB frame as numpy array
            variance_threshold: Pixel variance threshold for flat regions

        Returns:
            List of (x, y, width, height) tuples for flat regions
        """
        # Convert to grayscale for variance calculation
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32)

        # Use sliding window to find low-variance regions
        kernel_size = 16
        flat_regions = []

        h, w = gray.shape
        step = kernel_size // 2

        for y in range(0, h - kernel_size, step):
            for x in range(0, w - kernel_size, step):
                patch = gray[y : y + kernel_size, x : x + kernel_size]
                variance = float(np.var(patch))

                if variance < variance_threshold:
                    flat_regions.append((x, y, kernel_size, kernel_size))

        # Merge overlapping regions (simplified approach)
        if len(flat_regions) > 1:
            flat_regions = self._merge_overlapping_regions(flat_regions)

        return flat_regions

    def _merge_overlapping_regions(
        self, regions: list[tuple[int, int, int, int]]
    ) -> list[tuple[int, int, int, int]]:
        """Merge overlapping rectangular regions."""
        if not regions:
            return []

        # Simple merge: combine regions that overlap by >50%
        merged = [regions[0]]

        for region in regions[1:]:
            x, y, w, h = region
            merged_with_existing = False

            for i, existing in enumerate(merged):
                ex, ey, ew, eh = existing

                # Check for overlap
                overlap_x = max(0, min(x + w, ex + ew) - max(x, ex))
                overlap_y = max(0, min(y + h, ey + eh) - max(y, ey))
                overlap_area = overlap_x * overlap_y

                region_area = w * h
                existing_area = ew * eh

                if overlap_area > 0.5 * min(region_area, existing_area):
                    # Merge regions
                    new_x = min(x, ex)
                    new_y = min(y, ey)
                    new_w = max(x + w, ex + ew) - new_x
                    new_h = max(y + h, ey + eh) - new_y
                    merged[i] = (new_x, new_y, new_w, new_h)
                    merged_with_existing = True
                    break

            if not merged_with_existing:
                merged.append(region)

        return merged

    def detect_flat_region_flicker(
        self, frames: list[np.ndarray], variance_threshold: float = 10.0
    ) -> dict[str, float]:
        """Detect flicker in regions that should be stable (background areas).

        Args:
            frames: List of RGB frames
            variance_threshold: Threshold for identifying flat regions

        Returns:
            Dictionary with flat region flicker metrics
        """
        if len(frames) < 2:
            return {
                "flat_flicker_ratio": 0.0,
                "flat_region_count": 0,
                "flat_region_variance_mean": 0.0,
            }

        # Identify flat regions in first frame
        flat_regions = self.identify_flat_regions(frames[0], variance_threshold)

        if not flat_regions:
            return {
                "flat_flicker_ratio": 0.0,
                "flat_region_count": 0,
                "flat_region_variance_mean": 0.0,
            }

        # Track variance in these regions across frames
        region_variances = []

        for region in flat_regions:
            x, y, w, h = region

            # Extract region pixels from all frames
            region_pixels = []
            for frame in frames:
                # Ensure bounds are within frame
                actual_h, actual_w = frame.shape[:2]
                x_end = min(x + w, actual_w)
                y_end = min(y + h, actual_h)

                if x < actual_w and y < actual_h:
                    patch = frame[y:y_end, x:x_end]
                    region_pixels.append(patch.astype(np.float32))

            if len(region_pixels) >= 2:
                # Calculate temporal variance for this region
                temporal_variance = self._calculate_temporal_variance(region_pixels)
                region_variances.append(temporal_variance)

        if not region_variances:
            return {
                "flat_flicker_ratio": 0.0,
                "flat_region_count": len(flat_regions),
                "flat_region_variance_mean": 0.0,
            }

        mean_variance = float(np.mean(region_variances))

        # High variance in flat regions indicates flicker
        # Use adaptive threshold based on overall image characteristics
        flicker_threshold = variance_threshold * 2.0
        flickering_regions = sum(
            1 for var in region_variances if var > flicker_threshold
        )

        return {
            "flat_flicker_ratio": flickering_regions / len(flat_regions),
            "flat_region_count": len(flat_regions),
            "flat_region_variance_mean": mean_variance,
        }

    def _calculate_temporal_variance(self, region_pixels: list[np.ndarray]) -> float:
        """Calculate temporal variance for a region across frames."""
        if len(region_pixels) < 2:
            return 0.0

        # Stack patches and calculate variance across time
        try:
            # Ensure all patches have the same shape
            min_h = min(patch.shape[0] for patch in region_pixels)
            min_w = min(patch.shape[1] for patch in region_pixels)

            normalized_patches = []
            for patch in region_pixels:
                resized_patch = patch[:min_h, :min_w]
                normalized_patches.append(resized_patch)

            # Stack along time axis
            temporal_stack = np.stack(
                normalized_patches, axis=0
            )  # [time, h, w, channels]

            # Calculate variance across time axis for each pixel
            temporal_variance = np.var(temporal_stack, axis=0)

            # Return mean variance across all pixels in region
            return float(np.mean(temporal_variance))

        except (ValueError, IndexError, np.AxisError) as e:
            logger.error(f"Error calculating temporal variance: {e}")
            return 0.0

    def detect_temporal_pumping(self, frames: list[np.ndarray]) -> dict[str, float]:
        """Detect temporal pumping - quality oscillation between frames.

        Temporal pumping manifests as cyclic quality variations that create
        a "breathing" or "pumping" visual effect.

        Args:
            frames: List of RGB frames

        Returns:
            Dictionary with temporal pumping metrics
        """
        if len(frames) < 3:
            return {
                "temporal_pumping_score": 0.0,
                "quality_oscillation_frequency": 0.0,
                "quality_variance": 0.0,
            }

        # Calculate frame-wise quality metrics
        frame_qualities = []

        for frame in frames:
            # Use multiple quality indicators
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32)

            # Edge density as quality indicator
            edges = cv2.Canny(gray.astype(np.uint8), 50, 150)
            edge_density = float(np.sum(edges > 0) / edges.size)

            # Local variance as quality indicator
            local_variance = float(np.var(gray))

            # Combine metrics
            quality = edge_density * 0.7 + (local_variance / 10000.0) * 0.3
            frame_qualities.append(quality)

        # Analyze quality oscillation
        quality_variance = float(np.var(frame_qualities))

        # Detect periodic patterns in quality
        quality_diffs = np.diff(frame_qualities)
        oscillation_score = 0.0

        if len(quality_diffs) >= 2:
            # Look for alternating pattern in quality differences
            sign_changes = 0
            for i in range(len(quality_diffs) - 1):
                if np.sign(quality_diffs[i]) != np.sign(quality_diffs[i + 1]):
                    sign_changes += 1

            # High sign changes indicate oscillation
            oscillation_frequency = sign_changes / len(quality_diffs)

            # Combine frequency with magnitude
            oscillation_magnitude = float(np.std(quality_diffs))
            oscillation_score = oscillation_frequency * oscillation_magnitude

        return {
            "temporal_pumping_score": oscillation_score,
            "quality_oscillation_frequency": oscillation_frequency
            if "oscillation_frequency" in locals()
            else 0.0,
            "quality_variance": quality_variance,
        }


def calculate_enhanced_temporal_metrics(
    original_frames: list[np.ndarray],
    compressed_frames: list[np.ndarray],
    device: str | None = None,
    force_mse_fallback: bool = False,
    batch_size: int = 8,
) -> dict[str, float]:
    """Calculate comprehensive temporal artifact metrics for validation.

    This is the main entry point for temporal artifact detection that integrates
    all the enhanced detection methods with memory-efficient batch processing.

    Args:
        original_frames: List of original RGB frames
        compressed_frames: List of compressed RGB frames
        device: PyTorch device for LPIPS computation
        force_mse_fallback: If True, skip LPIPS and use enhanced MSE fallback
        batch_size: Number of frame pairs to process simultaneously for memory efficiency

    Returns:
        Dictionary with all temporal artifact metrics
    """
    detector = TemporalArtifactDetector(
        device=device, force_mse_fallback=force_mse_fallback
    )

    # Ensure frames are aligned
    min_frame_count = min(len(original_frames), len(compressed_frames))
    if min_frame_count == 0:
        return {
            "flicker_excess": 0.0,
            "flat_flicker_ratio": 0.0,
            "temporal_pumping_score": 0.0,
            "lpips_t_mean": 0.0,
            "lpips_t_p95": 0.0,
        }

    original_frames = original_frames[:min_frame_count]
    compressed_frames = compressed_frames[:min_frame_count]

    # For very large frame counts, automatically adjust batch size to manage memory
    if min_frame_count > 100:
        # Reduce batch size for very large sequences
        batch_size = max(4, batch_size // 2)
        logger.info(
            f"Large frame sequence ({min_frame_count} frames), using batch_size={batch_size}"
        )

    # Calculate temporal metrics on compressed frames with batch processing
    flicker_metrics = detector.detect_flicker_excess(
        compressed_frames, batch_size=batch_size
    )
    flat_flicker_metrics = detector.detect_flat_region_flicker(compressed_frames)
    pumping_metrics = detector.detect_temporal_pumping(compressed_frames)

    # Combine all metrics
    combined_metrics = {
        "flicker_excess": flicker_metrics["flicker_excess"],
        "flicker_frame_ratio": flicker_metrics["flicker_frame_ratio"],
        "flat_flicker_ratio": flat_flicker_metrics["flat_flicker_ratio"],
        "flat_region_count": flat_flicker_metrics["flat_region_count"],
        "temporal_pumping_score": pumping_metrics["temporal_pumping_score"],
        "quality_oscillation_frequency": pumping_metrics[
            "quality_oscillation_frequency"
        ],
        "lpips_t_mean": flicker_metrics["lpips_t_mean"],
        "lpips_t_p95": flicker_metrics["lpips_t_p95"],
        "frame_count": min_frame_count,
    }

    return combined_metrics


def calculate_enhanced_temporal_metrics_from_paths(
    original_path: str,
    compressed_path: str,
    device: str | None = None,
    force_mse_fallback: bool = False,
    batch_size: int = 8,
) -> dict[str, float]:
    """Calculate enhanced temporal metrics from GIF file paths.

    This is a convenience wrapper around calculate_enhanced_temporal_metrics
    that handles frame extraction from file paths with memory-efficient processing.

    Args:
        original_path: Path to original GIF file
        compressed_path: Path to compressed GIF file
        device: PyTorch device for LPIPS computation
        force_mse_fallback: If True, skip LPIPS and use enhanced MSE fallback
        batch_size: Number of frame pairs to process simultaneously

    Returns:
        Dictionary with temporal artifact metrics
    """
    try:
        # Import frame extraction utility
        from pathlib import Path

        from .metrics import extract_gif_frames

        # Extract frames from both files
        original_result = extract_gif_frames(Path(original_path))
        compressed_result = extract_gif_frames(Path(compressed_path))

        original_frames = original_result.frames
        compressed_frames = compressed_result.frames

        return calculate_enhanced_temporal_metrics(
            original_frames,
            compressed_frames,
            device=device,
            force_mse_fallback=force_mse_fallback,
            batch_size=batch_size,
        )

    except (FileNotFoundError, PermissionError) as e:
        logger.error(f"File access error calculating temporal metrics from paths: {e}")
        return {
            "flicker_excess": 0.0,
            "flat_flicker_ratio": 0.0,
            "temporal_pumping_score": 0.0,
            "lpips_t_mean": 0.0,
            "lpips_t_p95": 0.0,
        }
    except (ImportError, AttributeError) as e:
        logger.error(f"Module import error calculating temporal metrics: {e}")
        return {
            "flicker_excess": 0.0,
            "flat_flicker_ratio": 0.0,
            "temporal_pumping_score": 0.0,
            "lpips_t_mean": 0.0,
            "lpips_t_p95": 0.0,
        }
    except (ValueError, TypeError) as e:
        logger.error(f"Data processing error calculating temporal metrics: {e}")
        return {
            "flicker_excess": 0.0,
            "flat_flicker_ratio": 0.0,
            "temporal_pumping_score": 0.0,
            "lpips_t_mean": 0.0,
            "lpips_t_p95": 0.0,
        }
    except Exception as e:
        logger.error(f"Unexpected error calculating temporal metrics from paths: {e}")
        return {
            "flicker_excess": 0.0,
            "flat_flicker_ratio": 0.0,
            "temporal_pumping_score": 0.0,
            "lpips_t_mean": 0.0,
            "lpips_t_p95": 0.0,
        }


# =============================================================================
# TEMPORAL ARTIFACTS DETECTION SYSTEM DOCUMENTATION
# =============================================================================

"""
## Temporal Artifacts Detection System

### Overview

The temporal artifacts detection system provides comprehensive analysis of GIF
compression artifacts that manifest across time. This is essential for debugging
compression failures where quality degrades due to temporal inconsistencies.

### Key Metrics

1. **Flicker Excess Detection**
   - Uses LPIPS (Learned Perceptual Image Patch Similarity) for perceptual analysis
   - Detects frame-to-frame inconsistencies that cause visible flicker
   - Falls back to MSE-based metrics if LPIPS is unavailable

2. **Flat-Region Flicker Detection**
   - Identifies stable background regions that should remain consistent
   - Detects unwanted variations in supposedly flat areas
   - Critical for validating background preservation in compression

3. **Temporal Pumping Detection**
   - Identifies quality oscillations across the animation
   - Detects periodic variations in compression quality
   - Important for smooth animation playback

### Usage Examples

#### Basic Usage
```python
from giflab.temporal_artifacts import calculate_enhanced_temporal_metrics

# Load your frames (as RGB numpy arrays)
original_frames = load_frames("original.gif")
compressed_frames = load_frames("compressed.gif")

# Calculate comprehensive metrics
metrics = calculate_enhanced_temporal_metrics(
    original_frames, compressed_frames, device="cpu"
)

# Key metrics to examine
print(f"Flicker Excess: {metrics['flicker_excess']:.4f}")
print(f"Flat Region Flicker: {metrics['flat_flicker_ratio']:.4f}")
print(f"Temporal Pumping: {metrics['temporal_pumping_score']:.4f}")
```

#### Advanced Usage with Custom Detector
```python
from giflab.temporal_artifacts import TemporalArtifactDetector

# Initialize detector with GPU if available
detector = TemporalArtifactDetector(device="cuda")

# Individual metric calculations
flicker_metrics = detector.detect_flicker_excess(compressed_frames)
flat_metrics = detector.detect_flat_region_flicker(compressed_frames)
pumping_metrics = detector.detect_temporal_pumping(compressed_frames)

# Access detailed breakdown
print(f"Flicker Frame Ratio: {flicker_metrics['flicker_frame_ratio']:.3f}")
print(f"Quality Oscillation Freq: {pumping_metrics['quality_oscillation_frequency']:.3f}")
```

#### Integration with GIF Analysis
```python
from giflab.temporal_artifacts import calculate_enhanced_temporal_metrics_from_paths

# Calculate temporal metrics from original and compressed GIF paths
metrics = calculate_enhanced_temporal_metrics_from_paths(
    original_path="original.gif",
    compressed_path="compressed.gif",
)

# Access temporal artifact scores
print(f"Flicker excess: {metrics.get('flicker_excess', 0):.3f}")
print(f"Temporal pumping: {metrics.get('temporal_pumping_score', 0):.3f}")
```

### Interpretation Guide

#### Flicker Excess
- **Range**: 0.0 - 1.0+ (higher is worse)
- **Good**: < 0.05 (imperceptible flicker)
- **Concerning**: > 0.1 (visible flicker artifacts)
- **Critical**: > 0.2 (severe flicker issues)

#### Flat Region Flicker Ratio
- **Range**: 0.0 - 1.0 (proportion of flat regions with flicker)
- **Good**: < 0.1 (most backgrounds stable)
- **Concerning**: > 0.3 (many backgrounds unstable)
- **Critical**: > 0.5 (majority of backgrounds flickering)

#### Temporal Pumping Score
- **Range**: 0.0 - 1.0+ (higher indicates more pumping)
- **Good**: < 0.1 (consistent quality)
- **Concerning**: > 0.2 (noticeable quality variation)
- **Critical**: > 0.4 (severe quality oscillation)

### Technical Details

#### LPIPS Integration
- Uses AlexNet-based perceptual similarity measurement
- Automatically falls back to MSE if LPIPS unavailable
- GPU acceleration supported for large frame sequences
- Preprocessing handles different input formats automatically

#### Memory Optimization
- Lazy loading of LPIPS model to save memory
- Batch processing for large frame sequences
- Automatic cleanup of temporary tensors
- Supports both CPU and GPU processing

#### Error Handling
- Graceful fallback when LPIPS is unavailable
- Handles mismatched frame counts automatically
- Validates input data types and shapes
- Provides informative logging for debugging

### Dependencies

#### Required
- numpy: Array operations and frame handling
- opencv-python (cv2): Image processing and flat region detection
- torch: Tensor operations and LPIPS preprocessing

#### Optional
- lpips: Perceptual similarity measurement (falls back to MSE if unavailable)
- CUDA: GPU acceleration for large datasets

### Performance Considerations

- LPIPS model initialization has ~2-3 second overhead
- GPU processing 5-10x faster for large frame sequences
- Memory usage scales with frame count and resolution
- Consider processing in batches for very large GIFs (>100 frames)

### Integration Points

This system integrates with:
- `giflab.prediction_runner.PredictionRunner` for pipeline execution
- `giflab.validation` systems for automated quality checks
- `giflab.metrics` for comprehensive quality assessment
- `giflab.cli` commands for batch processing

For troubleshooting compression issues, focus on:
1. High flicker_excess values indicating frame instability
2. High flat_flicker_ratio suggesting background corruption
3. High temporal_pumping_score indicating quality inconsistency
"""


# Global singleton instance management
_global_temporal_detector: TemporalArtifactDetector | None = None


def cleanup_global_temporal_detector() -> None:
    """Clean up the global temporal detector instance and release model references."""
    global _global_temporal_detector
    if _global_temporal_detector is not None:
        # Release the LPIPS model reference
        if hasattr(_global_temporal_detector, "_lpips_model"):
            if (
                _global_temporal_detector._lpips_model is not None
                and _global_temporal_detector._lpips_model is not False
            ):
                # Release the model reference from cache
                LPIPSModelCache.release_model(
                    net="alex",
                    version="0.1",
                    spatial=False,
                    device=_global_temporal_detector.device,
                )
            _global_temporal_detector._lpips_model = None
        _global_temporal_detector = None
        logger.debug("Global temporal detector cleaned up")


def get_temporal_detector(
    device: str = "cpu", memory_threshold: float = 0.8, force_mse_fallback: bool = False
) -> TemporalArtifactDetector:
    """Get or create a global temporal detector instance.

    Args:
        device: Device for computation ('cpu' or 'cuda')
        memory_threshold: Memory threshold for automatic fallback
        force_mse_fallback: Force MSE-based processing instead of LPIPS

    Returns:
        Global TemporalArtifactDetector instance
    """
    global _global_temporal_detector

    # Check if we need to create a new instance
    if (
        _global_temporal_detector is None
        or _global_temporal_detector.device != device
        or _global_temporal_detector.force_mse_fallback != force_mse_fallback
    ):
        # Clean up existing instance if present
        if _global_temporal_detector is not None:
            cleanup_global_temporal_detector()

        # Create new instance
        _global_temporal_detector = TemporalArtifactDetector(
            device=device,
            memory_threshold=memory_threshold,
            force_mse_fallback=force_mse_fallback,
        )
        logger.debug(f"Created new global temporal detector on device: {device}")

    return _global_temporal_detector
