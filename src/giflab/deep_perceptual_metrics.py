"""
Deep Perceptual Metrics for GIF Quality Assessment

This module provides deep perceptual similarity metrics using learned models
like LPIPS to catch perceptual quality issues that traditional metrics miss.

Key Features:
- LPIPS-based spatial perceptual similarity measurement
- Intelligent frame downscaling for performance
- Batch processing with GPU acceleration when available
- Conditional triggering for borderline quality cases
- Memory-efficient processing with adaptive batch sizing

Dependencies:
- torch: For tensor operations and LPIPS model inference
- torchvision: For image transformations and preprocessing
- lpips: Learned perceptual similarity model (falls back gracefully if unavailable)
- cv2: For image resizing and preprocessing
"""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Optional, Protocol, Union

import numpy as np

# Import resized frame caching for optimization
from .caching.resized_frame_cache import resize_frame_cached

# Import lazy loading system
from .lazy_imports import (
    is_lpips_available,
    is_torch_available,
    lazy_import,
)

# Import model caching to prevent memory leaks
from .model_cache import LPIPSModelCache, cleanup_model_cache

# Import memory monitoring from temporal artifacts
from .temporal_artifacts import MemoryMonitor

# Lazy load heavy dependencies. The TYPE_CHECKING import gives mypy the real
# module (so ``torch.Tensor`` annotations resolve); at runtime the name stays
# a LazyModule proxy.
if TYPE_CHECKING:
    import torch
else:
    torch = lazy_import("torch")
lpips = lazy_import("lpips")

# Check availability without importing
TORCH_AVAILABLE = is_torch_available()
LPIPS_AVAILABLE = is_lpips_available()


# Lazy load torch submodules
def _get_torch_functional() -> Any:
    """Get torch.nn.functional lazily."""
    if TORCH_AVAILABLE:
        import torch.nn.functional as F

        return F
    return None


def _get_torchvision_transforms() -> Any:
    """Get torchvision.transforms lazily."""
    if TORCH_AVAILABLE:
        import torchvision.transforms as transforms

        return transforms
    return None


# For backward compatibility, we'll assign these on first use
F = None
transforms = None

# Log if LPIPS is not available
if not LPIPS_AVAILABLE:
    logger = logging.getLogger(__name__)
    logger.info(
        "LPIPS not available. Deep perceptual metrics will use fallback methods."
    )

try:
    import cv2

    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

logger = logging.getLogger(__name__)


class LPIPSModel(Protocol):
    """Protocol for LPIPS model interface (for type hints)."""

    def __call__(self, x: "torch.Tensor", y: "torch.Tensor") -> "torch.Tensor":
        """Calculate LPIPS distance between tensors."""

    def to(self, device: str) -> "LPIPSModel":
        """Move model to device."""

    def eval(self) -> "LPIPSModel":
        """Set model to evaluation mode."""


@dataclass
class DeepPerceptualMetrics:
    """Container for deep perceptual quality metrics."""

    lpips_quality_mean: float
    lpips_quality_p95: float
    lpips_quality_max: float
    frame_count: int
    downscaled: bool
    device_used: str


@dataclass
class PerceptualValidationResult:
    """Container for perceptual validation results."""

    lpips_quality_mean: float
    lpips_quality_p95: float
    lpips_quality_max: float
    quality_acceptable: bool
    frames_processed: int
    downscaled: bool


class DeepPerceptualValidator:
    """Deep perceptual quality validator using LPIPS and other learned metrics."""

    def __init__(
        self,
        device: str = "auto",
        downscale_size: int = 512,
        force_fallback: bool = False,
        memory_threshold: float = 0.8,
        use_resize_cache: bool = True,
    ):
        """Initialize deep perceptual validator.

        Args:
            device: PyTorch device for computation ('auto', 'cpu', 'cuda', or specific GPU)
            downscale_size: Maximum dimension to downscale frames to (default: 512)
            force_fallback: If True, skip LPIPS and use traditional metrics
            memory_threshold: Memory threshold for adaptive batch sizing (0-1)
            use_resize_cache: If True, use resized frame cache for efficiency
        """
        self.downscale_size = downscale_size
        self.force_fallback = force_fallback
        self.use_resize_cache = use_resize_cache
        self.device = self._determine_device(device)
        self._lpips_model: LPIPSModel | Literal[False] | None = None
        self.memory_monitor = MemoryMonitor(self.device, memory_threshold)

        if force_fallback:
            logger.info("Deep perceptual metrics disabled by configuration")
            self._lpips_model = False

    def _determine_device(self, device: str) -> str:
        """Determine optimal device for computation."""
        if not TORCH_AVAILABLE:
            return "cpu"

        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        elif device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            return "cpu"
        else:
            return device

    def _get_lpips_model(self) -> LPIPSModel | None:
        """Lazy initialization of LPIPS model with enhanced fallback handling."""
        if self._lpips_model is None and LPIPS_AVAILABLE and not self.force_fallback:
            try:
                logger.debug(
                    f"Getting LPIPS model from cache for device: {self.device}"
                )
                # Use cached model to prevent memory leaks
                model = LPIPSModelCache.get_model(
                    net="alex", version="0.1", spatial=False, device=self.device
                )
                if model is not None:
                    self._lpips_model = model
                    logger.info(
                        "LPIPS model initialized successfully for spatial quality assessment"
                    )
                self.memory_monitor.log_memory_usage("after LPIPS init")

            except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                if "out of memory" in str(e).lower():
                    logger.warning(f"LPIPS model failed due to memory constraints: {e}")
                    # Try CPU fallback if we were using CUDA
                    if self.device.startswith("cuda"):
                        try:
                            logger.info("Attempting CPU fallback for LPIPS model")
                            self.device = "cpu"
                            self.memory_monitor = MemoryMonitor(self.device)
                            # Use cached model for CPU fallback too
                            model = LPIPSModelCache.get_model(
                                net="alex", version="0.1", spatial=False, device="cpu"
                            )
                            if model is not None:
                                self._lpips_model = model
                                logger.info(
                                    "LPIPS model initialized successfully on CPU"
                                )
                                return model  # type: ignore
                        except Exception as cpu_e:
                            logger.warning(f"CPU fallback also failed: {cpu_e}")
                else:
                    logger.warning(f"LPIPS model initialization failed: {e}")
                self._lpips_model = False

            except Exception as e:
                logger.warning(f"Unexpected error initializing LPIPS model: {e}")
                self._lpips_model = False

        if (
            isinstance(self._lpips_model, type(self._lpips_model))
            and self._lpips_model is not False
        ):
            return self._lpips_model
        return None

    def __del__(self) -> None:
        """Clean up resources when the validator is destroyed."""
        # Release LPIPS model reference
        if hasattr(self, "_lpips_model"):
            if self._lpips_model is not None and self._lpips_model is not False:
                # Release the model reference from cache
                if hasattr(self, "device"):
                    LPIPSModelCache.release_model(
                        net="alex", version="0.1", spatial=False, device=self.device
                    )
                self._lpips_model = None

    def _downscale_frame_if_needed(self, frame: np.ndarray) -> tuple[np.ndarray, bool]:
        """Downscale frame if larger than target size.

        Args:
            frame: Input frame (H, W, C)

        Returns:
            Tuple of (processed_frame, was_downscaled)
        """
        if not CV2_AVAILABLE:
            return frame, False

        h, w = frame.shape[:2]
        max_dim = max(h, w)

        if max_dim <= self.downscale_size:
            return frame, False

        # Calculate new dimensions maintaining aspect ratio
        scale = self.downscale_size / max_dim
        new_w = int(w * scale)
        new_h = int(h * scale)

        # Use cached resizing for efficient processing
        # Enable cache by default unless explicitly disabled
        use_cache = getattr(self, "use_resize_cache", True)
        downscaled = resize_frame_cached(
            frame, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4, use_cache=use_cache
        )
        return downscaled, True

    def _preprocess_for_lpips(self, frame: np.ndarray) -> torch.Tensor:
        """Preprocess frame for LPIPS calculation.

        Args:
            frame: Input frame as numpy array (H, W, C) in range [0, 255]

        Returns:
            Preprocessed tensor ready for LPIPS [1, 3, H, W] in range [-1, 1]
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available for LPIPS preprocessing")

        # Convert to float and normalize to [0, 1]
        frame_float = frame.astype(np.float32) / 255.0

        # Convert to torch tensor and rearrange to CHW
        tensor = torch.from_numpy(frame_float).permute(2, 0, 1)

        # Normalize to [-1, 1] range for LPIPS
        tensor = tensor * 2.0 - 1.0

        # Add batch dimension and move to device
        tensor = tensor.unsqueeze(0).to(self.device)

        return tensor

    def _calculate_spatial_lpips(
        self,
        original_frames: list[np.ndarray],
        compressed_frames: list[np.ndarray],
        batch_size: int = 8,
    ) -> list[float]:
        """Calculate spatial LPIPS between corresponding original and compressed frames.

        Args:
            original_frames: List of original frames
            compressed_frames: List of compressed frames
            batch_size: Batch size for processing

        Returns:
            List of LPIPS scores between corresponding frames
        """
        lpips_model = self._get_lpips_model()
        if lpips_model is None:
            return []

        if len(original_frames) != len(compressed_frames):
            logger.warning(
                f"Frame count mismatch: {len(original_frames)} vs {len(compressed_frames)}"
            )
            min_frames = min(len(original_frames), len(compressed_frames))
            original_frames = original_frames[:min_frames]
            compressed_frames = compressed_frames[:min_frames]

        lpips_scores = []
        num_frames = len(original_frames)

        # Adapt batch size based on memory and frame dimensions
        if original_frames:
            frame_shape = original_frames[0].shape
            if len(frame_shape) == 3:  # RGB frame (H, W, C)
                safe_shape = (frame_shape[0], frame_shape[1], frame_shape[2])
                batch_size = self.memory_monitor.get_safe_batch_size(
                    safe_shape, batch_size
                )
            else:
                batch_size = min(batch_size, 4)  # Conservative fallback

        self.memory_monitor.log_memory_usage("before spatial LPIPS processing")

        try:
            with torch.no_grad():
                for batch_start in range(0, num_frames, batch_size):
                    batch_end = min(batch_start + batch_size, num_frames)

                    # Prepare batch tensors
                    orig_tensors = []
                    comp_tensors = []

                    for i in range(batch_start, batch_end):
                        # Downscale if needed
                        orig_frame, _ = self._downscale_frame_if_needed(
                            original_frames[i]
                        )
                        comp_frame, _ = self._downscale_frame_if_needed(
                            compressed_frames[i]
                        )

                        # Preprocess for LPIPS
                        orig_tensor = self._preprocess_for_lpips(orig_frame)
                        comp_tensor = self._preprocess_for_lpips(comp_frame)

                        orig_tensors.append(orig_tensor)
                        comp_tensors.append(comp_tensor)

                    if orig_tensors:  # Only process if we have frames
                        try:
                            # Stack into batch tensors
                            batch_orig = torch.cat(orig_tensors, dim=0)
                            batch_comp = torch.cat(comp_tensors, dim=0)

                            # Calculate perceptual distances for the batch
                            batch_distances = lpips_model(batch_orig, batch_comp)

                            # Extract individual scores
                            for distance in batch_distances:
                                lpips_scores.append(float(distance.cpu().item()))

                            # Cleanup batch tensors
                            del batch_orig, batch_comp, batch_distances
                            for tensor in orig_tensors + comp_tensors:
                                del tensor

                        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                            if "out of memory" in str(e).lower():
                                logger.warning(
                                    f"OOM in spatial LPIPS, reducing batch size: {e}"
                                )
                                # Cleanup and retry with smaller batch
                                for tensor in orig_tensors + comp_tensors:
                                    del tensor
                                torch.cuda.empty_cache()

                                smaller_batch_size = max(1, batch_size // 2)
                                logger.info(
                                    f"Retrying spatial LPIPS with batch size: {smaller_batch_size}"
                                )

                                # Recursive call with smaller batch size
                                remaining_orig = original_frames[batch_start:]
                                remaining_comp = compressed_frames[batch_start:]
                                remaining_scores = self._calculate_spatial_lpips(
                                    remaining_orig, remaining_comp, smaller_batch_size
                                )
                                lpips_scores.extend(remaining_scores)
                                break
                            else:
                                raise

        except Exception as e:
            logger.error(f"Error in spatial LPIPS processing: {e}")
            raise

        self.memory_monitor.log_memory_usage("after spatial LPIPS processing")
        return lpips_scores

    def calculate_deep_perceptual_metrics(
        self,
        original_frames: list[np.ndarray],
        compressed_frames: list[np.ndarray],
        max_frames: int | None = None,
    ) -> DeepPerceptualMetrics:
        """Calculate deep perceptual quality metrics between frame sets.

        Args:
            original_frames: List of original frames
            compressed_frames: List of compressed frames
            max_frames: Maximum frames to process (for sampling)

        Returns:
            DeepPerceptualMetrics with calculated scores
        """
        # Sample frames if too many
        if max_frames and len(original_frames) > max_frames:
            step = len(original_frames) // max_frames
            indices = list(range(0, len(original_frames), step))[:max_frames]
            original_frames = [original_frames[i] for i in indices]
            compressed_frames = [compressed_frames[i] for i in indices]
            logger.info(
                f"Sampled {len(original_frames)} frames from {len(indices)} for deep perceptual analysis"
            )

        # Check if downscaling will be used
        will_downscale = False
        if original_frames:
            h, w = original_frames[0].shape[:2]
            will_downscale = max(h, w) > self.downscale_size

        try:
            # Calculate spatial LPIPS scores
            lpips_scores = self._calculate_spatial_lpips(
                original_frames, compressed_frames
            )

            if not lpips_scores:
                # Fallback case: no scores could be computed. NaN ("not
                # measured"), not a 0.5 midpoint sentinel — a fabricated score
                # silently inflates composite_quality (1.0 - 0.5 = 0.5 of full
                # LPIPS weight) and corpus aggregates.
                logger.warning("No LPIPS scores obtained, using fallback values")
                return DeepPerceptualMetrics(
                    lpips_quality_mean=float("nan"),
                    lpips_quality_p95=float("nan"),
                    lpips_quality_max=float("nan"),
                    frame_count=len(original_frames),
                    downscaled=will_downscale,
                    device_used="fallback",
                )

            # Calculate statistics
            lpips_mean = float(np.mean(lpips_scores))
            lpips_p95 = float(np.percentile(lpips_scores, 95))
            lpips_max = float(np.max(lpips_scores))

            return DeepPerceptualMetrics(
                lpips_quality_mean=lpips_mean,
                lpips_quality_p95=lpips_p95,
                lpips_quality_max=lpips_max,
                frame_count=len(original_frames),
                downscaled=will_downscale,
                device_used=self.device,
            )

        except Exception as e:
            logger.error(f"Error calculating deep perceptual metrics: {e}")
            # Return fallback metrics with NaN score keys ("not measured"),
            # not a 0.5 midpoint sentinel. See the no-scores branch above.
            return DeepPerceptualMetrics(
                lpips_quality_mean=float("nan"),
                lpips_quality_p95=float("nan"),
                lpips_quality_max=float("nan"),
                frame_count=len(original_frames),
                downscaled=will_downscale,
                device_used="fallback" if self.force_fallback else self.device,
            )

    def validate_perceptual_quality(
        self,
        original_frames: list[np.ndarray],
        compressed_frames: list[np.ndarray],
        quality_threshold: float = 0.3,
        max_frames: int | None = None,
    ) -> PerceptualValidationResult:
        """Validate perceptual quality using deep metrics.

        Args:
            original_frames: List of original frames
            compressed_frames: List of compressed frames
            quality_threshold: LPIPS threshold above which quality is unacceptable
            max_frames: Maximum frames to process

        Returns:
            PerceptualValidationResult with validation decision
        """
        metrics = self.calculate_deep_perceptual_metrics(
            original_frames, compressed_frames, max_frames
        )

        # Quality is acceptable if mean LPIPS is below threshold
        # (lower LPIPS = more similar = better quality).
        #
        # NaN handling: when LPIPS couldn't be measured, lpips_quality_mean is
        # NaN. ``nan <= threshold`` is False in Python, so quality_acceptable
        # is False — i.e. an unmeasurable LPIPS is treated as "not acceptable"
        # (cannot confirm acceptable), which is the conservative, honest
        # outcome. This coincidentally matches the previous 0.5-sentinel
        # behaviour (0.5 > 0.3 threshold -> also False) but now reflects
        # "unassessable" rather than a fabricated mid-quality score.
        quality_acceptable = metrics.lpips_quality_mean <= quality_threshold

        return PerceptualValidationResult(
            lpips_quality_mean=metrics.lpips_quality_mean,
            lpips_quality_p95=metrics.lpips_quality_p95,
            lpips_quality_max=metrics.lpips_quality_max,
            quality_acceptable=quality_acceptable,
            frames_processed=metrics.frame_count,
            downscaled=metrics.downscaled,
        )


# Module-level validator instance to reuse across calls (prevents repeated model loading)
_global_validator: DeepPerceptualValidator | None = None


def cleanup_global_validator() -> None:
    """Clean up the global validator instance to free memory."""
    global _global_validator
    if _global_validator is not None:
        # Release the LPIPS model reference
        if hasattr(_global_validator, "_lpips_model"):
            if (
                _global_validator._lpips_model is not None
                and _global_validator._lpips_model is not False
            ):
                # Release the model reference from cache
                LPIPSModelCache.release_model(
                    net="alex",
                    version="0.1",
                    spatial=False,
                    device=_global_validator.device,
                )
            _global_validator._lpips_model = None
        _global_validator = None
        logger.debug("Cleared global validator instance")


def _get_or_create_validator(
    device: str = "auto",
    downscale_size: int = 512,
    force_fallback: bool = False,
    use_resize_cache: bool = True,
) -> DeepPerceptualValidator:
    """Get or create a reusable validator instance.

    This prevents creating new validators (and loading models) on every call.
    """
    global _global_validator

    # Create validator if it doesn't exist or configuration changed significantly
    if (
        _global_validator is None
        or _global_validator.force_fallback != force_fallback
        or _global_validator.downscale_size != downscale_size
        or getattr(_global_validator, "use_resize_cache", True) != use_resize_cache
    ):
        _global_validator = DeepPerceptualValidator(
            device=device,
            downscale_size=downscale_size,
            force_fallback=force_fallback,
            use_resize_cache=use_resize_cache,
        )

    return _global_validator


def should_use_deep_perceptual(composite_quality: float | None) -> bool:
    """Determine if deep perceptual metrics should be calculated based on composite quality.

    Args:
        composite_quality: Current composite quality score (0-1)

    Returns:
        True if deep perceptual metrics should be calculated
    """
    if composite_quality is None:
        return True  # Calculate if we don't have composite quality yet

    # Use for borderline cases where traditional metrics may not be reliable
    # Quality between 0.3 and 0.7 represents the "uncertainty zone"
    if 0.3 <= composite_quality <= 0.7:
        return True

    # Also use for very poor quality to better understand failure modes
    if composite_quality < 0.3:
        return True

    return False


def calculate_deep_perceptual_quality_metrics(
    original_frames: list[np.ndarray],
    compressed_frames: list[np.ndarray],
    config: dict | None = None,
) -> dict[str, float | str]:
    """Main entry point for deep perceptual quality metrics calculation.

    Args:
        original_frames: List of original frames
        compressed_frames: List of compressed frames
        config: Configuration dictionary

    Returns:
        Dictionary with deep perceptual quality metrics
    """
    if config is None:
        config = {}

    device = config.get("device", "auto")
    downscale_size = config.get("lpips_downscale_size", 512)
    max_frames = config.get("lpips_max_frames", 100)
    force_fallback = config.get("disable_deep_perceptual", False)
    use_resize_cache = config.get("use_resize_cache", True)

    try:
        # Use reusable validator to prevent repeated model loading
        validator = _get_or_create_validator(
            device=device,
            downscale_size=downscale_size,
            force_fallback=force_fallback,
            use_resize_cache=use_resize_cache,
        )

        metrics = validator.calculate_deep_perceptual_metrics(
            original_frames, compressed_frames, max_frames
        )

        return {
            "lpips_quality_mean": metrics.lpips_quality_mean,
            "lpips_quality_p95": metrics.lpips_quality_p95,
            "lpips_quality_max": metrics.lpips_quality_max,
            "deep_perceptual_frame_count": float(metrics.frame_count),
            "deep_perceptual_downscaled": float(metrics.downscaled),
            "deep_perceptual_device": metrics.device_used,
        }

    except Exception as e:
        logger.error(f"Deep perceptual metrics calculation failed: {e}")
        # Score keys NaN ("not measured"), not a 0.5 midpoint sentinel; the
        # midpoint silently inflated composite_quality and corpus aggregates.
        return {
            "lpips_quality_mean": float("nan"),
            "lpips_quality_p95": float("nan"),
            "lpips_quality_max": float("nan"),
            "deep_perceptual_frame_count": float(len(original_frames)),
            "deep_perceptual_downscaled": 0.0,
            "deep_perceptual_device": "fallback",
        }


# Module documentation
__doc__ += """

## Usage Examples

### Basic Usage
```python
from giflab.deep_perceptual_metrics import DeepPerceptualValidator

validator = DeepPerceptualValidator()
metrics = validator.calculate_deep_perceptual_metrics(original_frames, compressed_frames)
print(f"LPIPS quality: {metrics.lpips_quality_mean:.3f}")
```

### Conditional Usage
```python
from giflab.deep_perceptual_metrics import should_use_deep_perceptual, calculate_deep_perceptual_quality_metrics

if should_use_deep_perceptual(composite_quality):
    deep_metrics = calculate_deep_perceptual_quality_metrics(original_frames, compressed_frames)
```

## Performance Notes

- Frame downscaling to 512px reduces computation time by ~4x
- GPU acceleration provides 2-3x speedup over CPU
- Batch processing reduces memory overhead
- Adaptive batch sizing prevents OOM errors

## Integration Points

This module integrates with:
- `metrics.py`: Main metrics calculation pipeline
- `enhanced_metrics.py`: Composite quality calculation
- `config.py`: Configuration management
- `temporal_artifacts.py`: Shared LPIPS infrastructure
"""
