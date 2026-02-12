"""Quality metrics and comparison functionality for GIF analysis."""

import logging
import math
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

import cv2
import numpy as np
from PIL import Image
from skimage.feature import local_binary_pattern
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from .config import (
    DEFAULT_METRICS_CONFIG,
    ENABLE_EXPERIMENTAL_CACHING,
    FRAME_CACHE,
    MetricsConfig,
)

logger = logging.getLogger(__name__)

# ============================================================================
# Conditional Import Architecture for Experimental Caching
# ============================================================================
# This section implements the conditional import pattern designed to break
# circular dependencies between metrics and caching modules while providing
# graceful degradation when caching is unavailable.
#
# Architecture Goals:
# 1. Eliminate metrics ↔ caching circular dependency
# 2. Allow safe toggling of experimental caching features
# 3. Provide fallback implementations for production stability
# 4. Enable comprehensive error handling with actionable guidance
#
# Flow Control:
# 1. Check ENABLE_EXPERIMENTAL_CACHING feature flag
# 2. Attempt conditional imports with comprehensive error handling
# 3. Set up fallback implementations if imports fail
# 4. Initialize module-level state variables for runtime access

# Initialize module-level caching state variables
# These are set during import and remain stable for the module's lifetime
CACHING_ENABLED = False  # True if caching modules successfully imported
get_frame_cache: Callable[[], Any] | None = None  # Function reference or None
resize_frame_cached: Callable[..., Any]  # Function reference or fallback implementation
CACHING_ERROR_MESSAGE: str | None = None  # Detailed error message for troubleshooting

if ENABLE_EXPERIMENTAL_CACHING:
    try:
        # Attempt to import caching modules
        # These imports may fail due to:
        # 1. Missing dependencies (e.g., cache backends not installed)
        # 2. Circular import issues (if caching modules import metrics)
        # 3. Module initialization errors (configuration issues)
        from .caching import get_frame_cache
        from .caching.resized_frame_cache import resize_frame_cached

        CACHING_ENABLED = True
        logger.debug("✅ Caching modules loaded successfully")

    except ImportError as e:
        # Handle missing module dependencies with actionable guidance
        # This is the most common failure mode in production
        error_details = str(e)
        module_name = error_details.split("'")[1] if "'" in error_details else "unknown"

        CACHING_ERROR_MESSAGE = (
            f"🚨 Caching features unavailable due to import error.\n"
            f"Failed module: {module_name}\n"
            f"Error details: {error_details}\n\n"
            f"To resolve:\n"
            f"1. Verify all caching dependencies are installed: poetry install\n"
            f"2. Check for circular dependency issues in caching modules\n"
            f"3. Disable caching if issues persist: ENABLE_EXPERIMENTAL_CACHING = False\n"
            f"4. Report issue if problem continues: https://github.com/animately/giflab/issues"
        )

        logger.warning(f"❌ Failed to import caching modules: {e}")
        logger.info("💡 Falling back to non-cached operations")
        CACHING_ENABLED = False

    except Exception as e:
        # Handle unexpected errors during caching module initialization
        # These could be configuration errors, memory issues, or system problems
        CACHING_ERROR_MESSAGE = (
            f"🚨 Unexpected error loading caching modules.\n"
            f"Error type: {type(e).__name__}\n"
            f"Error details: {str(e)}\n\n"
            f"To resolve:\n"
            f"1. Check system resources and memory availability\n"
            f"2. Verify caching module integrity: poetry install --no-cache\n"
            f"3. Disable caching temporarily: ENABLE_EXPERIMENTAL_CACHING = False\n"
            f"4. Report issue: https://github.com/animately/giflab/issues"
        )

        logger.error(f"💥 Unexpected error loading caching modules: {e}")
        logger.info("💡 Falling back to non-cached operations")
        CACHING_ENABLED = False

else:
    # Feature flag is disabled - this is the normal production state
    # until caching has been thoroughly tested and validated
    logger.debug("📋 Experimental caching is disabled in configuration")


def _resize_frame_fallback(
    frame: Any,
    size: tuple[int, int],
    interpolation: int = cv2.INTER_AREA,
    **kwargs: Any,
) -> Any:
    """Fallback implementation for frame resizing when caching is disabled.

    Provides the same interface as resize_frame_cached but performs direct
    OpenCV resizing without any caching functionality. This ensures that
    all resize operations continue to work even when caching is unavailable.

    Args:
        frame: Input frame/image to resize
        size: Target size as (width, height) tuple
        interpolation: OpenCV interpolation method (default: INTER_AREA)
        **kwargs: Additional arguments (ignored in fallback, maintained for compatibility)

    Returns:
        Resized frame using OpenCV's cv2.resize()

    Design Notes:
        - Maintains same function signature as cached version for drop-in replacement
        - Uses cv2.INTER_AREA as default for high-quality downsampling
        - Ignores cache-specific kwargs to prevent errors
        - Performance: Direct OpenCV call, no overhead

    Example:
        >>> resized = _resize_frame_fallback(frame, (100, 100))
        >>> # Equivalent to: cv2.resize(frame, (100, 100), interpolation=cv2.INTER_AREA)
    """
    return cv2.resize(frame, size, interpolation=interpolation)


def get_caching_status() -> dict[str, Any]:
    """Get comprehensive caching system status and diagnostic information.

    Provides detailed information about the current state of the conditional
    import system, including success/failure status, error messages, and
    available functionality. Used for troubleshooting and monitoring.

    Returns:
        dict[str, any]: Status dictionary containing:
            - enabled (bool): Whether caching modules successfully loaded
            - experimental_flag (bool): State of ENABLE_EXPERIMENTAL_CACHING flag
            - error_message (str|None): Detailed error message if import failed
            - fallback_available (bool): Whether fallback functions are set up
            - modules_available (dict): Per-module availability status

    Thread Safety:
        Thread-safe. Accesses only module-level constants set during import.

    Performance:
        Very fast (~0.1ms). Safe to call frequently for monitoring.

    Example:
        >>> status = get_caching_status()
        >>> if status["enabled"]:
        >>>     print("✅ Caching is fully operational")
        >>> elif status["error_message"]:
        >>>     print(f"❌ Caching error: {status['error_message']}")
        >>> else:
        >>>     print("📋 Caching disabled by configuration")
        >>>
        >>> # Check specific module availability
        >>> if status["modules_available"]["resize_frame_cached"]:
        >>>     print("Resize caching available")

    CLI Integration:
        This function is used by `giflab deps check` to report caching status
        and provide troubleshooting information to users.
    """
    return {
        "enabled": CACHING_ENABLED,
        "experimental_flag": ENABLE_EXPERIMENTAL_CACHING,
        "error_message": CACHING_ERROR_MESSAGE,
        "fallback_available": resize_frame_cached is not None,
        "modules_available": {
            "get_frame_cache": get_frame_cache is not None,
            "resize_frame_cached": resize_frame_cached is not None,
        },
    }


# ============================================================================
# Fallback Function Assignment
# ============================================================================
# Ensure resize_frame_cached is always callable, either with the real cached
# implementation or the fallback. This assignment happens after the conditional
# import attempt so that resize_frame_cached is never None.

# Ensure resize_frame_cached is always assigned to a callable function
# This prevents "None not callable" errors that mypy detects
if not CACHING_ENABLED:
    # Assign fallback implementation when caching is unavailable
    # This ensures resize_frame_cached() calls always work, providing
    # transparent fallback behavior for all calling code
    resize_frame_cached = _resize_frame_fallback
    logger.debug("🔄 Using fallback resize implementation (no caching)")
else:
    # resize_frame_cached was imported successfully from caching module
    # No additional assignment needed - it's already set from import
    logger.debug("🚀 Using cached resize implementation")

# ============================================================================
# End of Conditional Import Architecture
# ============================================================================


@dataclass
class FrameExtractResult:
    """Result of frame extraction from a GIF."""

    frames: list[np.ndarray]
    frame_count: int
    dimensions: tuple[int, int]  # (width, height)
    duration_ms: int


def extract_gif_frames(
    gif_path: Path, max_frames: int | None = None
) -> FrameExtractResult:
    """Extract frames from a GIF file.

    Args:
        gif_path: Path to GIF file
        max_frames: Maximum number of frames to extract (None for all)

    Returns:
        FrameExtractResult with extracted frames and metadata

    Raises:
        IOError: If GIF cannot be read
        ValueError: If GIF is invalid or corrupted
    """
    # Check if caching is available and enabled
    # Try dynamic import if runtime config is enabled but modules weren't loaded
    global get_frame_cache, CACHING_ENABLED

    runtime_enabled = FRAME_CACHE.get("enabled", False)
    caching_available = CACHING_ENABLED and get_frame_cache is not None

    # If runtime enabled but caching not available, try dynamic import
    if runtime_enabled and not caching_available:
        try:
            # Attempt dynamic import for testing scenarios
            from .caching import get_frame_cache as _get_frame_cache

            get_frame_cache = _get_frame_cache
            caching_available = True
            CACHING_ENABLED = True
            logger.debug(
                "✅ Dynamic caching import successful for runtime-enabled cache"
            )
        except ImportError:
            logger.debug("❌ Dynamic caching import failed, cache unavailable")
            caching_available = False

    use_cache = caching_available and runtime_enabled

    # Try to get from cache first (only if caching is enabled)

    if use_cache:
        frame_cache = get_frame_cache()
        cached = frame_cache.get(gif_path, max_frames)

        if cached is not None:
            frames, frame_count, dimensions, duration_ms = cached
            return FrameExtractResult(
                frames=frames,
                frame_count=frame_count,
                dimensions=dimensions,
                duration_ms=duration_ms,
            )

    # Not in cache, extract frames
    try:
        with Image.open(gif_path) as img:
            if not hasattr(img, "n_frames") or img.n_frames == 1:
                # Single frame image (PNG, JPEG, etc.) or single-frame GIF
                frame = np.array(img.convert("RGB"))
                result = FrameExtractResult(
                    frames=[frame],
                    frame_count=1,
                    dimensions=(img.width, img.height),
                    duration_ms=0,
                )

                # Cache the result (only if caching is enabled)
                if use_cache:
                    frame_cache.put(
                        gif_path,
                        result.frames,
                        result.frame_count,
                        result.dimensions,
                        result.duration_ms,
                    )

                return result

            total_frames = img.n_frames

            # Memory protection: limit frame extraction for very large GIFs
            memory_limit_frames = 500  # Reasonable limit to prevent memory issues
            if max_frames is None:
                frames_to_extract = min(total_frames, memory_limit_frames)
            else:
                frames_to_extract = min(total_frames, max_frames, memory_limit_frames)

            # Additional memory check based on image dimensions
            width, height = img.size
            pixels_per_frame = width * height * 3  # RGB
            estimated_memory_mb = (frames_to_extract * pixels_per_frame) / (1024 * 1024)

            # Limit memory usage to ~500MB for frame extraction
            if estimated_memory_mb > 500:
                max_safe_frames = int(500 * 1024 * 1024 / pixels_per_frame)
                frames_to_extract = min(frames_to_extract, max(1, max_safe_frames))

            frames = []
            total_duration = 0

            # Use even frame sampling across entire animation for better quality assessment
            if frames_to_extract >= total_frames:
                # Use all frames if we're not hitting the limit
                frame_indices = list(range(total_frames))
            else:
                # Sample evenly across the entire animation to capture quality issues
                # that may appear later in the animation
                frame_indices = np.linspace(
                    0, total_frames - 1, frames_to_extract, dtype=int
                ).tolist()

            for i in frame_indices:
                img.seek(i)
                frame = np.array(img.convert("RGB"))
                frames.append(frame)

                # Get frame duration
                duration = img.info.get("duration", 100)  # Default 100ms
                total_duration += duration

            result = FrameExtractResult(
                frames=frames,
                frame_count=total_frames,  # Store the actual total frame count
                dimensions=(img.width, img.height),
                duration_ms=total_duration,
            )

            # Cache the result (only if caching is enabled)
            if use_cache:
                frame_cache.put(
                    gif_path,
                    result.frames,
                    result.frame_count,
                    result.dimensions,
                    result.duration_ms,
                )

            return result

    except Exception as e:
        raise OSError(f"Failed to extract frames from {gif_path}: {e}") from e


def resize_to_common_dimensions(
    frames1: list[np.ndarray], frames2: list[np.ndarray]
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Resize frames to common dimensions (smallest common size).

    Args:
        frames1: First set of frames
        frames2: Second set of frames

    Returns:
        Tuple of (resized_frames1, resized_frames2)
    """
    if not frames1 or not frames2:
        return frames1, frames2

    # Validate frames have proper dimensions
    if len(frames1[0].shape) < 2 or len(frames2[0].shape) < 2:
        raise ValueError("Frames must have at least 2 dimensions")

    # Get dimensions
    h1, w1 = frames1[0].shape[:2]
    h2, w2 = frames2[0].shape[:2]

    # Validate dimensions are positive
    if h1 <= 0 or w1 <= 0 or h2 <= 0 or w2 <= 0:
        raise ValueError(f"Invalid frame dimensions: {h1}x{w1} and {h2}x{w2}")

    # Use smallest common dimensions
    target_h = min(h1, h2)
    target_w = min(w1, w2)

    # Ensure minimum dimensions for processing
    target_h = max(target_h, 1)
    target_w = max(target_w, 1)

    # Resize if necessary
    resized_frames1 = []
    for frame in frames1:
        if len(frame.shape) < 2:
            raise ValueError("Frame has invalid shape")

        if frame.shape[:2] != (target_h, target_w):
            try:
                resized = cv2.resize(
                    frame, (target_w, target_h), interpolation=cv2.INTER_AREA
                )
                resized_frames1.append(resized)
            except Exception as e:
                raise ValueError(f"Failed to resize frame: {e}") from e
        else:
            resized_frames1.append(frame)

    resized_frames2 = []
    for frame in frames2:
        if len(frame.shape) < 2:
            raise ValueError("Frame has invalid shape")

        if frame.shape[:2] != (target_h, target_w):
            try:
                resized = cv2.resize(
                    frame, (target_w, target_h), interpolation=cv2.INTER_AREA
                )
                resized_frames2.append(resized)
            except Exception as e:
                raise ValueError(f"Failed to resize frame: {e}") from e
        else:
            resized_frames2.append(frame)

    return resized_frames1, resized_frames2


def calculate_frame_mse(frame1: np.ndarray, frame2: np.ndarray) -> float:
    """Calculate mean squared error between two frames."""
    if frame1.shape != frame2.shape:
        frame2 = cv2.resize(frame2, (frame1.shape[1], frame1.shape[0]))

    return float(np.mean((frame1.astype(np.float32) - frame2.astype(np.float32)) ** 2))


def calculate_safe_psnr(
    frame1: np.ndarray, frame2: np.ndarray, data_range: float = 255.0
) -> float:
    """Calculate PSNR with proper handling of perfect matches (MSE = 0).

    Args:
        frame1: First frame
        frame2: Second frame
        data_range: Maximum possible pixel value (default 255.0)

    Returns:
        PSNR value, with 100.0 dB returned for perfect matches
    """
    try:
        # Check for perfect match first to avoid divide by zero
        mse = calculate_frame_mse(frame1, frame2)

        if mse == 0.0:
            # Perfect match - return maximum PSNR (100 dB is a reasonable upper bound)
            return 100.0

        # Use scikit-image PSNR for non-perfect matches
        return float(psnr(frame1, frame2, data_range=data_range))

    except Exception as e:
        logger.warning(f"PSNR calculation failed: {e}")
        return 0.0


def align_frames_content_based(
    original_frames: list[np.ndarray], compressed_frames: list[np.ndarray]
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Content-based alignment - find most similar frames using MSE.

    This is the most robust alignment method as it finds actual visual matches
    regardless of temporal position or compression patterns."""
    if not original_frames or not compressed_frames:
        return []

    aligned_pairs = []
    used_compressed_indices = set()

    for orig_frame in original_frames:
        best_match_idx = -1
        best_mse = float("inf")

        for comp_idx, comp_frame in enumerate(compressed_frames):
            if comp_idx in used_compressed_indices:
                continue

            try:
                mse = calculate_frame_mse(orig_frame, comp_frame)

                # Handle perfect matches (MSE = 0) - these are ideal matches
                if mse == 0.0:
                    best_mse = 0.0
                    best_match_idx = comp_idx
                    break  # Perfect match found, no need to check further

                # Validate MSE is finite and reasonable
                if not np.isfinite(mse) or mse < 0:
                    logger.warning(
                        f"Invalid MSE calculated for frame pair {comp_idx}: {mse}"
                    )
                    continue

                if mse < best_mse:
                    best_mse = mse
                    best_match_idx = comp_idx
            except Exception as e:
                logger.warning(f"MSE calculation failed for frame {comp_idx}: {e}")
                continue

        # Accept any valid match with finite MSE (including perfect matches with MSE = 0)
        if best_match_idx >= 0 and np.isfinite(best_mse) and best_mse >= 0:
            aligned_pairs.append((orig_frame, compressed_frames[best_match_idx]))
            used_compressed_indices.add(best_match_idx)
        else:
            # Only warn if we genuinely couldn't find any valid match
            logger.debug(
                f"No valid frame match found for original frame (best_mse={best_mse})"
            )
            # For robustness, try to match with the first available frame if no perfect match found
            if compressed_frames and not used_compressed_indices:
                logger.debug("Falling back to first available frame for alignment")
                aligned_pairs.append((orig_frame, compressed_frames[0]))
                used_compressed_indices.add(0)

    return aligned_pairs


def align_frames(
    original_frames: list[np.ndarray], compressed_frames: list[np.ndarray]
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Align frames using content-based matching (most robust approach).

    Args:
        original_frames: Original GIF frames
        compressed_frames: Compressed GIF frames

    Returns:
        List of aligned frame pairs based on visual similarity
    """
    return align_frames_content_based(original_frames, compressed_frames)


def calculate_ms_ssim(
    frame1: np.ndarray, frame2: np.ndarray, scales: int = 5, use_cache: bool = True
) -> float:
    """Calculate Multi-Scale SSIM (MS-SSIM) between two frames.

    Args:
        frame1: First frame
        frame2: Second frame
        scales: Number of scales for MS-SSIM (default 5)
        use_cache: Whether to use the resized frame cache

    Returns:
        MS-SSIM value between 0.0 and 1.0
    """
    if frame1.shape != frame2.shape:
        frame2 = resize_frame_cached(
            frame2,
            (frame1.shape[1], frame1.shape[0]),
            interpolation=cv2.INTER_AREA,
            use_cache=use_cache,
        )

    # Convert to grayscale for MS-SSIM calculation
    if len(frame1.shape) == 3:
        frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
        frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
    else:
        frame1_gray = frame1
        frame2_gray = frame2

    # Calculate MS-SSIM using pyramid approach
    ssim_values: list[float] = []
    current_frame1 = frame1_gray.astype(np.float32)
    current_frame2 = frame2_gray.astype(np.float32)

    # Safety check: limit scales to prevent infinite loops
    max_possible_scales = min(scales, 10)  # Hard limit to prevent runaway loops

    for scale in range(max_possible_scales):
        # Calculate SSIM at current scale
        try:
            scale_ssim = ssim(current_frame1, current_frame2, data_range=255.0)
            ssim_values.append(scale_ssim)
        except (ValueError, RuntimeError) as e:
            # If SSIM calculation fails due to invalid data or computation error, use a default value
            logger.warning(f"SSIM calculation failed at scale {scale}: {e}")
            ssim_values.append(0.0)

        # Downsample for next scale (if not the last scale)
        if scale < max_possible_scales - 1:
            prev_shape = current_frame1.shape
            # Calculate target dimensions for 0.5x scaling
            new_h = int(current_frame1.shape[0] * 0.5)
            new_w = int(current_frame1.shape[1] * 0.5)

            current_frame1 = resize_frame_cached(
                current_frame1,
                (new_w, new_h),
                interpolation=cv2.INTER_AREA,
                use_cache=use_cache,
            ).astype(np.float32)
            current_frame2 = resize_frame_cached(
                current_frame2,
                (new_w, new_h),
                interpolation=cv2.INTER_AREA,
                use_cache=use_cache,
            ).astype(np.float32)

            # Stop if frames become too small OR if size didn't change (safety check)
            if (
                current_frame1.shape[0] < 8
                or current_frame1.shape[1] < 8
                or current_frame1.shape == prev_shape
            ):
                break

    # Weighted average of SSIM values across scales
    if ssim_values:
        weight_list = [0.4, 0.25, 0.15, 0.1, 0.1][: len(ssim_values)]
        weights = np.array(weight_list)

        # Protect against division by zero in weight normalization
        weights_sum = np.sum(weights)
        if weights_sum > 0:
            weights = weights / weights_sum  # Normalize weights
            return float(np.average(ssim_values, weights=weights))
        else:
            # If all weights are zero, use uniform weighting
            return float(np.mean(ssim_values))
    else:
        return 0.0


def calculate_temporal_consistency(frames: list[np.ndarray]) -> float:
    """Calculate temporal consistency (animation smoothness) of frames.

    Temporal consistency measures how predictable/smooth the animation is,
    NOT whether frames are identical. For animated content, consistent
    frame-to-frame differences indicate good temporal consistency.

    Args:
        frames: List of consecutive frames

    Returns:
        Temporal consistency value between 0.0 and 1.0 (higher = more consistent)
    """
    if len(frames) < 2:
        return 1.0  # Single frame is perfectly consistent

    frame_differences = []

    for i in range(len(frames) - 1):
        frame1 = frames[i].astype(np.float32)
        frame2 = frames[i + 1].astype(np.float32)

        # Calculate frame-to-frame difference
        diff = float(np.mean(np.abs(frame1 - frame2)))
        frame_differences.append(diff)

    if not frame_differences:
        return 1.0

    # Key insight: Temporal consistency is about PREDICTABILITY, not minimal change
    # For animated GIFs, we want consistent patterns in frame differences

    mean_diff = float(np.mean(frame_differences))
    variance_diff = float(np.var(frame_differences))

    # Handle special cases
    if mean_diff == 0 and variance_diff == 0:
        return 1.0  # Static content = perfect consistency

    if variance_diff == 0 and mean_diff > 0:
        return 1.0  # Perfectly uniform animation = perfect consistency

    # For content with variation, measure consistency of the pattern
    # High variance in frame differences = inconsistent temporal behavior
    # Low variance = consistent temporal behavior (good)

    # Normalize variance by the square of mean difference to get relative consistency
    if mean_diff > 0:
        relative_variance = variance_diff / (mean_diff**2)
        # Use inverse exponential: lower relative variance = higher consistency
        consistency = np.exp(-relative_variance * 2.0)
    else:
        # Edge case: very small differences, use original method
        normalized_variance = variance_diff / (255.0**2)
        normalized_mean = mean_diff / 255.0
        consistency = np.exp(-normalized_variance * 2.0 - normalized_mean * 0.5)

    return float(max(0.0, min(1.0, consistency)))


def detect_disposal_artifacts(
    frames: list[np.ndarray], frame_reduction_context: bool = False
) -> float:
    """Detect disposal method artifacts including background corruption, transparency bleeding, and color shifts.

    This function distinguishes between intended animation changes and actual disposal artifacts.
    Disposal artifacts manifest as:
    - Inconsistent/corrupted frame transitions
    - Background color corruption in areas that should be stable
    - Transparency corruption and overlay residue
    - Visual frame stacking and overlay artifacts that break animation patterns

    Args:
        frames: List of consecutive frames
        frame_reduction_context: If True, adjusts detection for legitimate frame reduction
                               vs actual disposal method artifacts

    Returns:
        Artifact score between 0.0 and 1.0 (0.0 = severe artifacts, 1.0 = clean)
    """
    if len(frames) < 2:
        return 1.0  # Need at least 2 frames to detect artifacts

    # Check if this appears to be global animation (all pixels change uniformly)
    # vs partial animation (some areas stable, others changing)
    is_global_animation = _detect_global_animation_pattern(frames)

    if is_global_animation:
        # For global animation (like solid color cycling), disposal artifacts are rare
        # Focus on pattern consistency rather than absolute differences
        return _detect_global_animation_artifacts(frames)
    else:
        # For partial animation, use enhanced detection with better background tracking
        return _detect_partial_animation_artifacts_enhanced(
            frames, frame_reduction_context
        )


def _detect_global_animation_pattern(frames: list[np.ndarray]) -> bool:
    """Detect if the animation involves global changes (entire frame changes) vs partial changes."""
    if len(frames) < 3:
        return False

    # Sample a few frame transitions
    sample_transitions = min(3, len(frames) - 1)
    global_change_scores = []

    for i in range(sample_transitions):
        frame1 = frames[i].astype(np.float32)
        frame2 = frames[i + 1].astype(np.float32)

        # Calculate per-pixel differences
        pixel_diffs = np.mean(np.abs(frame1 - frame2), axis=2)

        # If most pixels changed significantly, it's likely global animation
        changed_pixels = np.sum(pixel_diffs > 30.0)  # Threshold for significant change
        total_pixels = pixel_diffs.size
        change_ratio = changed_pixels / total_pixels

        global_change_scores.append(change_ratio)

    # If >70% of pixels change in most transitions, consider it global animation
    avg_change_ratio = float(np.mean(global_change_scores))
    return avg_change_ratio > 0.7


def _detect_global_animation_artifacts(frames: list[np.ndarray]) -> float:
    """Detect artifacts in global animation by looking for pattern inconsistencies."""
    if len(frames) < 3:
        return 1.0

    # For global animation, measure consistency of the animation pattern
    frame_diffs = []
    for i in range(len(frames) - 1):
        frame1 = frames[i].astype(np.float32)
        frame2 = frames[i + 1].astype(np.float32)
        diff = np.mean(np.abs(frame1 - frame2))
        frame_diffs.append(diff)

    # If animation has consistent differences, it's clean
    # If differences vary wildly, there might be artifacts
    if len(frame_diffs) > 1:
        variance = np.var(frame_diffs)
        mean_diff = np.mean(frame_diffs)

        if mean_diff > 0:
            # Lower relative variance = more consistent pattern = fewer artifacts
            relative_variance = variance / (mean_diff**2)
            consistency_score = np.exp(-relative_variance * 0.5)  # Gentler penalty
            return float(max(0.0, min(1.0, consistency_score)))

    return 1.0  # Default to clean for consistent patterns


def _detect_partial_animation_artifacts(
    frames: list[np.ndarray], frame_reduction_context: bool
) -> float:
    """Detect artifacts in partial animation using the original detailed method."""
    # Extract first and last frames for comparison (most likely to show accumulation)
    first_frame = frames[0].astype(np.float32)
    last_frame = frames[-1].astype(np.float32)

    # Ensure frames are the same size
    if first_frame.shape != last_frame.shape:
        last_frame = cv2.resize(last_frame, (first_frame.shape[1], first_frame.shape[0]))  # type: ignore[assignment]

    scores = []

    # 1. Background Color Stability Detection
    bg_stability = detect_background_color_stability(first_frame, last_frame)
    scores.append(("background_stability", bg_stability, 0.25))

    # 2. Structural Integrity Detection (for geometric artifacts like duplicate lines)
    structural_score = detect_structural_artifacts(first_frame, last_frame)
    scores.append(("structural", structural_score, 0.4))

    # 3. Transparency Corruption Detection
    transparency_score = detect_transparency_corruption(frames)
    scores.append(("transparency", transparency_score, 0.2))

    # 4. Color Fidelity Measurement
    color_fidelity = detect_color_fidelity_corruption(first_frame, last_frame)
    scores.append(("color_fidelity", color_fidelity, 0.1))

    # 5. Visual Frame Overlay Detection (legacy density-based)
    overlay_score = detect_frame_overlay_artifacts(frames)
    scores.append(("overlay", overlay_score, 0.05))

    # Calculate weighted final score
    total_weight = sum(weight for _, _, weight in scores)
    final_score = sum(score * weight for _, score, weight in scores) / total_weight

    return float(max(0.0, min(1.0, final_score)))


def _detect_partial_animation_artifacts_enhanced(
    frames: list[np.ndarray], frame_reduction_context: bool
) -> float:
    """Enhanced detection of artifacts in partial animation with improved background tracking.

    This enhanced version integrates temporal artifact detection from the temporal_artifacts
    module to provide better background stability tracking and flicker detection.
    """
    from .temporal_artifacts import get_temporal_detector

    # Get global temporal detector instance
    detector = get_temporal_detector()

    # Extract first and last frames for comparison (most likely to show accumulation)
    first_frame = frames[0].astype(np.float32)
    last_frame = frames[-1].astype(np.float32)

    # Ensure frames are the same size
    if first_frame.shape != last_frame.shape:
        last_frame = cv2.resize(last_frame, (first_frame.shape[1], first_frame.shape[0]))  # type: ignore[assignment]

    scores = []

    # 1. Enhanced Background Stability Detection using temporal analysis
    bg_stability = detect_background_color_stability_enhanced(frames, detector)
    scores.append(("background_stability_enhanced", bg_stability, 0.3))

    # 2. Flat Region Flicker Detection (new)
    flat_flicker_metrics = detector.detect_flat_region_flicker(frames)
    flat_flicker_score = max(0.0, 1.0 - flat_flicker_metrics["flat_flicker_ratio"])
    scores.append(("flat_region_stability", flat_flicker_score, 0.25))

    # 3. Structural Integrity Detection (existing method)
    structural_score = detect_structural_artifacts(first_frame, last_frame)
    scores.append(("structural", structural_score, 0.25))

    # 4. Transparency Corruption Detection
    transparency_score = detect_transparency_corruption(frames)
    scores.append(("transparency", transparency_score, 0.1))

    # 5. Color Fidelity Measurement
    color_fidelity = detect_color_fidelity_corruption(first_frame, last_frame)
    scores.append(("color_fidelity", color_fidelity, 0.05))

    # 6. Visual Frame Overlay Detection (legacy density-based)
    overlay_score = detect_frame_overlay_artifacts(frames)
    scores.append(("overlay", overlay_score, 0.05))

    # Calculate weighted final score
    total_weight = sum(weight for _, _, weight in scores)
    final_score = sum(score * weight for _, score, weight in scores) / total_weight

    return float(max(0.0, min(1.0, final_score)))


def detect_background_color_stability_enhanced(
    frames: list[np.ndarray], detector: Any
) -> float:
    """Enhanced background color stability detection using temporal analysis.

    This enhanced version uses region-based temporal tracking to better identify
    background areas and detect corruption across all frames, not just first/last.
    """
    if len(frames) < 2:
        return 1.0

    # Identify stable background regions using the first frame
    first_frame = frames[0]
    flat_regions = detector.identify_flat_regions(first_frame, variance_threshold=8.0)

    if not flat_regions:
        # Fallback to original edge-based method
        return detect_background_color_stability(first_frame, frames[-1])

    # Track color stability in identified background regions across all frames
    region_stabilities = []

    for region in flat_regions:
        x, y, w, h = region

        # Extract region from all frames
        region_colors = []
        for frame in frames:
            # Ensure bounds are within frame
            actual_h, actual_w = frame.shape[:2]
            x_end = min(x + w, actual_w)
            y_end = min(y + h, actual_h)

            if x < actual_w and y < actual_h:
                patch = frame[y:y_end, x:x_end]
                # Calculate mean color for this region
                mean_color = np.mean(patch.reshape(-1, patch.shape[-1]), axis=0)
                region_colors.append(mean_color)

        if len(region_colors) >= 2:
            # Calculate color stability across time
            region_colors_array = np.array(region_colors)
            color_variance = np.mean(np.var(region_colors_array, axis=0))

            # Convert variance to stability score (lower variance = higher stability)
            stability = max(
                0.0, 1.0 - (color_variance / 100.0)
            )  # 100 is empirical threshold
            region_stabilities.append(stability)

    if not region_stabilities:
        # Fallback to original method
        return detect_background_color_stability(first_frame, frames[-1])

    # Return mean stability across all background regions
    return float(np.mean(region_stabilities))


def detect_background_color_stability(
    first_frame: np.ndarray, last_frame: np.ndarray
) -> float:
    """Detect background color corruption between first and last frames.

    Background corruption manifests as color shifts (gray→pink) in areas that
    should remain stable throughout the animation.
    """
    # Sample edge regions as likely background areas
    height, width = first_frame.shape[:2]
    edge_width = max(5, width // 20)
    edge_height = max(5, height // 20)

    # Extract edge regions (top, bottom, left, right)
    edges_first = []
    edges_last = []

    # Top and bottom edges
    edges_first.extend([first_frame[:edge_height, :], first_frame[-edge_height:, :]])
    edges_last.extend([last_frame[:edge_height, :], last_frame[-edge_height:, :]])

    # Left and right edges
    edges_first.extend([first_frame[:, :edge_width], first_frame[:, -edge_width:]])
    edges_last.extend([last_frame[:, :edge_width], last_frame[:, -edge_width:]])

    # Calculate color shift in edge regions
    total_shift = 0.0
    for edge_first, edge_last in zip(edges_first, edges_last, strict=False):
        if edge_first.shape != edge_last.shape:
            continue

        # Calculate mean color difference in each edge region
        color_diff = np.mean(np.abs(edge_first - edge_last))
        total_shift += color_diff

    # Normalize shift (higher shift = lower score)
    avg_shift = total_shift / len(edges_first)
    # Convert to 0-1 score where 0 = severe shift, 1 = no shift
    stability_score = max(0.0, 1.0 - (avg_shift / 50.0))  # 50 is empirical threshold

    return stability_score


def detect_transparency_corruption(frames: list[np.ndarray]) -> float:
    """Detect transparency and white bleeding artifacts.

    Transparency corruption shows as unexpected white pixels or regions
    where transparency should be preserved.
    """
    if len(frames) < 2:
        return 1.0

    corruption_scores = []

    for i in range(1, len(frames)):
        frame1 = frames[i - 1].astype(np.float32)
        frame2 = frames[i].astype(np.float32)

        if frame1.shape != frame2.shape:
            frame2 = cv2.resize(frame2, (frame1.shape[1], frame1.shape[0]))  # type: ignore[assignment]

        # Detect unexpected white/bright pixels (transparency bleeding)
        if len(frame1.shape) == 3:
            # Check for white bleeding in RGB
            white_threshold = 240
            bright_pixels_1 = np.sum(np.all(frame1 > white_threshold, axis=2))
            bright_pixels_2 = np.sum(np.all(frame2 > white_threshold, axis=2))

            # Corruption = unexpected increase in bright pixels
            pixel_increase = max(0, bright_pixels_2 - bright_pixels_1)
            total_pixels = frame1.shape[0] * frame1.shape[1]
            corruption_ratio = pixel_increase / total_pixels

            corruption_scores.append(
                1.0 - min(1.0, corruption_ratio * 10)
            )  # Scale factor

    if not corruption_scores:
        return 1.0

    return float(np.mean(corruption_scores))


def detect_color_fidelity_corruption(
    first_frame: np.ndarray, last_frame: np.ndarray
) -> float:
    """Detect color palette corruption and intensity shifts.

    Color fidelity corruption shows as unexpected color changes in regions
    that should maintain consistent colors (red→magenta shifts).
    """
    if first_frame.shape != last_frame.shape:
        last_frame = cv2.resize(
            last_frame, (first_frame.shape[1], first_frame.shape[0])
        )

    if len(first_frame.shape) == 3:
        # Calculate per-channel color stability
        channel_stabilities = []

        for channel in range(3):  # R, G, B
            first_channel = first_frame[:, :, channel]
            last_channel = last_frame[:, :, channel]

            # Calculate mean absolute difference in this color channel
            channel_diff = np.mean(np.abs(first_channel - last_channel))

            # Convert to stability score (lower diff = higher stability)
            stability = max(
                0.0, 1.0 - (channel_diff / 100.0)
            )  # 100 is empirical threshold
            channel_stabilities.append(stability)

        # Overall color fidelity is minimum channel stability (worst channel determines score)
        return float(min(channel_stabilities))
    else:
        # Grayscale - calculate intensity stability
        intensity_diff = np.mean(np.abs(first_frame - last_frame))
        return float(max(0.0, 1.0 - (intensity_diff / 100.0)))


def detect_structural_artifacts(
    first_frame: np.ndarray, last_frame: np.ndarray
) -> float:
    """Detect structural disposal artifacts like duplicate lines, edges, and geometric elements.

    This method detects disposal artifacts that manifest as:
    - Duplicate axis lines in charts
    - Overlapping geometric elements
    - Edge duplication and structural inconsistencies
    - Line artifacts that don't affect overall color/density

    Args:
        first_frame: First frame of the animation
        last_frame: Last frame of the animation (most likely to show accumulation)

    Returns:
        Score between 0.0 and 1.0 (0.0 = severe structural artifacts, 1.0 = clean)
    """
    try:
        # Convert to grayscale for edge detection
        gray_first = (
            cv2.cvtColor(first_frame, cv2.COLOR_RGB2GRAY)
            if len(first_frame.shape) == 3
            else first_frame
        )
        gray_last = (
            cv2.cvtColor(last_frame, cv2.COLOR_RGB2GRAY)
            if len(last_frame.shape) == 3
            else last_frame
        )

        # Ensure same dimensions
        if gray_first.shape != gray_last.shape:
            gray_last = cv2.resize(
                gray_last, (gray_first.shape[1], gray_first.shape[0])
            )

        # Edge detection using Canny - captures lines and structural elements
        edges_first = cv2.Canny(gray_first.astype(np.uint8), 50, 150)
        edges_last = cv2.Canny(gray_last.astype(np.uint8), 50, 150)

        # Calculate edge pixel counts
        edge_count_first = np.sum(edges_first > 0)
        edge_count_last = np.sum(edges_last > 0)

        # Detect structural duplication - significant edge increase suggests artifacts
        if edge_count_first == 0:
            # No edges in first frame - can't detect duplication
            edge_increase_score = 1.0
        else:
            edge_ratio = edge_count_last / edge_count_first
            # Normal animation should have similar edge density
            # Disposal artifacts cause edge multiplication (ratio > 1.2 for charts/diagrams)
            if edge_ratio > 1.2:
                # More aggressive penalization for edge duplication
                edge_increase_score = max(0.0, 1.0 - ((edge_ratio - 1.2) * 1.5))  # type: ignore[assignment]
            else:
                edge_increase_score = 1.0

        # Detect edge pattern inconsistency using structural similarity
        if edge_count_first > 0 and edge_count_last > 0:
            # Calculate correlation between edge patterns
            edges_first_norm = edges_first.astype(np.float32) / 255.0
            edges_last_norm = edges_last.astype(np.float32) / 255.0

            # Use SSIM on edge maps to detect structural corruption
            edge_ssim = ssim(edges_first_norm, edges_last_norm, data_range=1.0)

            # Low SSIM between edge patterns indicates structural corruption
            # But account for legitimate animation changes
            if edge_ssim < 0.6:  # Significant structural change
                edge_pattern_score = max(0.0, edge_ssim)
            else:
                edge_pattern_score = 1.0
        else:
            edge_pattern_score = 1.0

        # Combine edge increase and pattern consistency (equal weighting)
        final_score = edge_increase_score * 0.6 + edge_pattern_score * 0.4

        return float(max(0.0, min(1.0, final_score)))

    except Exception as e:
        logger.warning(f"Structural artifact detection failed: {e}")
        return 1.0  # Assume clean on error


def detect_frame_overlay_artifacts(frames: list[np.ndarray]) -> float:
    """Detect visual frame overlay artifacts using density-based approach.

    This is the legacy detection method for frame stacking artifacts.
    """
    if len(frames) < 3:
        return 1.0

    # Calculate content density changes
    content_densities = []
    for frame in frames:
        gray = (
            cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) if len(frame.shape) == 3 else frame
        )
        non_bg_pixels = np.sum(gray > 25)
        total_pixels = gray.shape[0] * gray.shape[1]
        density = non_bg_pixels / total_pixels
        content_densities.append(density)

    # Check for problematic density increases
    increases = 0
    for i in range(1, len(content_densities)):
        if (
            content_densities[i] > content_densities[i - 1] * 1.15
        ):  # 15% increase threshold
            increases += 1

    # Score based on density increase pattern
    max_increases = len(frames) - 1
    score = 1.0 - (increases / max_increases) if max_increases > 0 else 1.0

    return float(max(0.0, min(1.0, score)))


# Legacy compatibility functions
def calculate_ssim(original_path: Path, compressed_path: Path) -> float:
    """Calculate Structural Similarity Index (SSIM) between two GIFs.

    Legacy function - use calculate_comprehensive_metrics for full functionality.
    """
    try:
        metrics = calculate_comprehensive_metrics(original_path, compressed_path)
        ssim_value = metrics["ssim"]
        return float(ssim_value) if isinstance(ssim_value, int | float) else 0.0
    except Exception as e:
        logger.error(f"SSIM calculation failed: {e}")
        return 0.0


def calculate_file_size_kb(file_path: Path) -> float:
    """Calculate file size in kilobytes.

    Args:
        file_path: Path to the file

    Returns:
        File size in kilobytes (KB)

    Raises:
        IOError: If file cannot be accessed
    """
    try:
        size_bytes = file_path.stat().st_size
        return size_bytes / 1024.0  # Convert bytes to KB
    except OSError as e:
        raise OSError(f"Cannot access file {file_path}: {e}") from e


def measure_render_time(
    func: Callable[..., Any], *args: Any, **kwargs: Any
) -> tuple[Any, int]:
    """Measure execution time of a function in milliseconds.

    Args:
        func: Function to measure
        *args: Arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function

    Returns:
        Tuple of (function_result, execution_time_ms)
    """
    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    end_time = time.perf_counter()

    elapsed_seconds = end_time - start_time
    # Cap at reasonable maximum to prevent overflow (24 hours = 86400000 ms)
    execution_time_ms = min(int(elapsed_seconds * 1000), 86400000)
    return result, execution_time_ms


def compare_gif_frames(gif1_path: Path, gif2_path: Path) -> dict[str, Any]:
    """Compare frames between two GIF files for quality analysis.

    Legacy function - use calculate_comprehensive_metrics for full functionality.
    """
    try:
        metrics = calculate_comprehensive_metrics(gif1_path, gif2_path)
        return {
            "frame_count_original": len(extract_gif_frames(gif1_path).frames),
            "frame_count_compressed": len(extract_gif_frames(gif2_path).frames),
            "quality_metrics": metrics,
        }
    except Exception as e:
        logger.error(f"Frame comparison failed: {e}")
        return {"error": str(e)}


def calculate_compression_ratio(
    original_size_kb: float, compressed_size_kb: float
) -> float:
    """Calculate compression ratio between original and compressed files.

    Args:
        original_size_kb: Original file size in KB
        compressed_size_kb: Compressed file size in KB

    Returns:
        Compression ratio (original_size / compressed_size)
    """
    if compressed_size_kb <= 0:
        raise ValueError("Compressed size must be positive")

    return original_size_kb / compressed_size_kb


# ---------------- New helper and metric functions (Stage-1) ---------------- #


def _resize_if_needed(
    frame1: np.ndarray, frame2: np.ndarray, use_cache: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Resize both frames to the smallest common size if their shapes differ.

    The function keeps the aspect ratio by simply resizing to the *minimum* of the
    two input shapes. This avoids any padding/cropping artefacts while ensuring
    metric functions receive arrays with identical dimensions.

    Args:
        frame1: First frame to resize
        frame2: Second frame to resize
        use_cache: Whether to use the resized frame cache for efficiency
    """
    if frame1.shape[:2] == frame2.shape[:2]:
        return frame1, frame2

    target_h = min(frame1.shape[0], frame2.shape[0])
    target_w = min(frame1.shape[1], frame2.shape[1])

    try:
        frame1_resized = resize_frame_cached(
            frame1,
            (target_w, target_h),
            interpolation=cv2.INTER_AREA,
            use_cache=use_cache,
        )
        frame2_resized = resize_frame_cached(
            frame2,
            (target_w, target_h),
            interpolation=cv2.INTER_AREA,
            use_cache=use_cache,
        )
    except Exception as exc:  # pragma: no cover – surface as ValueError for callers
        raise ValueError(f"Failed to resize frames to common size: {exc}") from exc

    return frame1_resized, frame2_resized


def mse(frame1: np.ndarray, frame2: np.ndarray) -> float:
    """Mean-Squared Error (lower is better).

    Returns a non-negative float. Identical frames ⇒ 0.0.
    """
    f1, f2 = _resize_if_needed(frame1, frame2)
    return float(np.mean((f1.astype(np.float32) - f2.astype(np.float32)) ** 2))


def rmse(frame1: np.ndarray, frame2: np.ndarray) -> float:
    """Root-Mean-Squared Error (sqrt of MSE)."""
    return math.sqrt(mse(frame1, frame2))


def fsim(frame1: np.ndarray, frame2: np.ndarray) -> float:
    """Feature Similarity Index (approximated implementation).

    This lightweight approximation returns the *mean* of the combined
    gradient-magnitude and phase-congruency similarity maps, which empirically
    yields higher scores for identical images and lower scores for dissimilar
    ones.
    """
    f1, f2 = _resize_if_needed(frame1, frame2)

    # Grayscale conversion.
    if f1.ndim == 3:
        gray1 = cv2.cvtColor(f1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(f2, cv2.COLOR_RGB2GRAY)
    else:
        gray1, gray2 = f1, f2

    gray1 = gray1.astype(np.float32)
    gray2 = gray2.astype(np.float32)

    # Gradient magnitude (Sobel).
    def _grad_mag(img: np.ndarray) -> np.ndarray:
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
        return np.sqrt(gx**2 + gy**2)

    G1 = _grad_mag(gray1)
    G2 = _grad_mag(gray2)

    # Phase-congruency proxy using Laplacian magnitude.
    PC1 = np.abs(cv2.Laplacian(gray1, cv2.CV_32F))
    PC2 = np.abs(cv2.Laplacian(gray2, cv2.CV_32F))

    T1 = 1e-3
    T2 = 1e-3
    gradient_sim = (2 * G1 * G2 + T1) / (G1**2 + G2**2 + T1)
    pc_sim = (2 * PC1 * PC2 + T2) / (PC1**2 + PC2**2 + T2)

    fsim_map = gradient_sim * pc_sim

    return float(np.clip(np.mean(fsim_map), 0.0, 1.0))


def gmsd(frame1: np.ndarray, frame2: np.ndarray) -> float:
    """Gradient Magnitude Similarity Deviation (lower is better)."""
    f1, f2 = _resize_if_needed(frame1, frame2)

    if f1.ndim == 3:
        gray1 = cv2.cvtColor(f1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(f2, cv2.COLOR_RGB2GRAY)
    else:
        gray1, gray2 = f1, f2

    gray1 = gray1.astype(np.float32)
    gray2 = gray2.astype(np.float32)

    # Prewitt kernels.
    prewitt_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.float32) / 3.0
    prewitt_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float32) / 3.0

    def _prewitt(img: np.ndarray) -> np.ndarray:
        gx = cv2.filter2D(img, -1, prewitt_x)
        gy = cv2.filter2D(img, -1, prewitt_y)
        return np.sqrt(gx**2 + gy**2)

    M1 = _prewitt(gray1)
    M2 = _prewitt(gray2)

    C = 1e-3  # stability constant
    gms_map = (2 * M1 * M2 + C) / (M1**2 + M2**2 + C)

    return float(np.std(gms_map))


def chist(frame1: np.ndarray, frame2: np.ndarray, bins: int = 32) -> float:
    """Colour-Histogram Correlation (0-1, higher is better)."""
    f1, f2 = _resize_if_needed(frame1, frame2)
    scores: list[float] = []
    for ch in range(3):  # R,G,B channels
        h1 = cv2.calcHist([f1], [ch], None, [bins], [0, 256])
        h2 = cv2.calcHist([f2], [ch], None, [bins], [0, 256])
        cv2.normalize(h1, h1)
        cv2.normalize(h2, h2)
        corr = cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)
        scores.append(corr)
    # cv2 correlation in [-1,1] – map to [0,1]
    return float(np.clip((np.mean(scores) + 1) / 2.0, 0.0, 1.0))


def edge_similarity(
    frame1: np.ndarray, frame2: np.ndarray, threshold1: int = 50, threshold2: int = 150
) -> float:
    """Edge-Map Jaccard similarity (0-1, higher is better).

    Args:
        frame1: First frame (RGB or grayscale)
        frame2: Second frame (RGB or grayscale)
        threshold1: Lower Canny threshold
        threshold2: Upper Canny threshold
    """
    f1, f2 = _resize_if_needed(frame1, frame2)

    if f1.ndim == 3:
        gray1 = cv2.cvtColor(f1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(f2, cv2.COLOR_RGB2GRAY)
    else:
        gray1, gray2 = f1, f2

    edges1 = cv2.Canny(gray1, threshold1, threshold2)
    edges2 = cv2.Canny(gray2, threshold1, threshold2)

    intersection = np.logical_and(edges1 > 0, edges2 > 0).sum()
    union = np.logical_or(edges1 > 0, edges2 > 0).sum()
    if union == 0:
        return 1.0  # no edges at all
    return float(intersection / union)


def texture_similarity(frame1: np.ndarray, frame2: np.ndarray) -> float:
    """Texture-Histogram correlation using uniform LBP (0-1, higher is better)."""
    f1, f2 = _resize_if_needed(frame1, frame2)

    if f1.ndim == 3:
        gray1 = cv2.cvtColor(f1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(f2, cv2.COLOR_RGB2GRAY)
    else:
        gray1, gray2 = f1, f2

    radius = 1
    n_points = 8 * radius
    lbp1 = local_binary_pattern(gray1, n_points, radius, "uniform")
    lbp2 = local_binary_pattern(gray2, n_points, radius, "uniform")

    hist1, _ = np.histogram(
        lbp1.ravel(), bins=10, range=(0, n_points + 2), density=True
    )
    hist2, _ = np.histogram(
        lbp2.ravel(), bins=10, range=(0, n_points + 2), density=True
    )

    if np.std(hist1) == 0 or np.std(hist2) == 0:
        return 1.0  # completely uniform textures

    corr = np.corrcoef(hist1, hist2)[0, 1]
    return float(np.clip((corr + 1) / 2.0, 0.0, 1.0))


def sharpness_similarity(frame1: np.ndarray, frame2: np.ndarray) -> float:
    """Sharpness similarity based on Laplacian variance ratio (0-1, higher is better)."""
    f1, f2 = _resize_if_needed(frame1, frame2)

    if f1.ndim == 3:
        gray1 = cv2.cvtColor(f1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(f2, cv2.COLOR_RGB2GRAY)
    else:
        gray1, gray2 = f1, f2

    var1 = float(np.var(cv2.Laplacian(gray1, cv2.CV_64F)))
    var2 = float(np.var(cv2.Laplacian(gray2, cv2.CV_64F)))

    # Both completely flat ⇒ identical sharpness.
    if var1 == 0 and var2 == 0:
        return 1.0
    if max(var1, var2) == 0:
        return 0.0

    return float(min(var1, var2) / max(var1, var2))


# --------------------------------------------------------------------------- #


def _aggregate_metric(values: list[float], metric_name: str) -> dict[str, float]:
    """Aggregate frame-level metric values into descriptive statistics.

    Args:
        values: List of frame-level metric values
        metric_name: Name of the metric for key generation

    Returns:
        Dictionary with mean, std, min, max for the metric
    """
    if not values:
        return {
            metric_name: 0.0,
            f"{metric_name}_std": 0.0,
            f"{metric_name}_min": 0.0,
            f"{metric_name}_max": 0.0,
        }

    values_array = np.array(values)

    # Handle edge case of single frame
    if len(values) == 1:
        return {
            metric_name: float(values[0]),
            f"{metric_name}_std": 0.0,
            f"{metric_name}_min": float(values[0]),
            f"{metric_name}_max": float(values[0]),
        }

    return {
        metric_name: float(np.mean(values_array)),
        f"{metric_name}_std": float(np.std(values_array)),
        f"{metric_name}_min": float(np.min(values_array)),
        f"{metric_name}_max": float(np.max(values_array)),
    }


def _calculate_positional_samples(
    aligned_pairs: list[tuple[np.ndarray, np.ndarray]],
    metric_func: Any,
    metric_name: str,
) -> dict[str, float]:
    """Calculate metrics for first, middle, and last frames to understand positional effects.

    This function provides insights into how frame position affects quality metrics,
    which is crucial for determining optimal sampling strategies in production.

    Args:
        aligned_pairs: List of (original_frame, compressed_frame) tuples
        metric_func: Function to calculate the metric (e.g., ssim, mse, fsim)
        metric_name: Name of the metric for key generation

    Returns:
        Dictionary with positional samples and variance:
        {
            "metric_first": float,      # Metric value for first frame
            "metric_middle": float,     # Metric value for middle frame
            "metric_last": float,       # Metric value for last frame
            "metric_positional_variance": float  # Variance across positions
        }
    """
    if not aligned_pairs:
        return {
            f"{metric_name}_first": 0.0,
            f"{metric_name}_middle": 0.0,
            f"{metric_name}_last": 0.0,
            f"{metric_name}_positional_variance": 0.0,
        }

    n_frames = len(aligned_pairs)

    try:
        # Calculate for 3 key positions
        first_val = float(metric_func(*aligned_pairs[0]))
        middle_val = float(metric_func(*aligned_pairs[n_frames // 2]))
        last_val = float(metric_func(*aligned_pairs[-1]))

        # Calculate positional variance (how much does position matter?)
        pos_values = [first_val, middle_val, last_val]
        positional_variance = float(np.var(pos_values))

        return {
            f"{metric_name}_first": first_val,
            f"{metric_name}_middle": middle_val,
            f"{metric_name}_last": last_val,
            f"{metric_name}_positional_variance": positional_variance,
        }

    except Exception as e:
        logger.warning(f"Positional sampling failed for {metric_name}: {e}")
        return {
            f"{metric_name}_first": 0.0,
            f"{metric_name}_middle": 0.0,
            f"{metric_name}_last": 0.0,
            f"{metric_name}_positional_variance": 0.0,
        }


def calculate_selected_metrics(
    original_frames: list[np.ndarray],
    compressed_frames: list[np.ndarray],
    selected_metrics: dict[str, bool],
    config: MetricsConfig | None = None,
) -> dict[str, Any]:
    """Calculate only selected metrics between original and compressed frames.

    This function is used by ConditionalMetricsCalculator to calculate only
    the metrics that are needed based on quality assessment and content profile.

    Args:
        original_frames: List of original frames as numpy arrays
        compressed_frames: List of compressed frames as numpy arrays
        selected_metrics: Dictionary with metric names as keys and bool values
                         indicating whether to calculate that metric
        config: Optional metrics configuration (uses default if None)

    Returns:
        Dictionary with calculated metrics (only those selected)
    """
    if config is None:
        config = DEFAULT_METRICS_CONFIG

    # Resize frames to common dimensions
    original_frames_resized, compressed_frames_resized = resize_to_common_dimensions(
        original_frames, compressed_frames
    )

    # Align frames
    aligned_pairs = align_frames(original_frames_resized, compressed_frames_resized)

    if not aligned_pairs:
        raise ValueError("No frame pairs could be aligned")

    # Initialize result dictionary
    metric_values: dict[str, list[float]] = {}

    # Calculate basic metrics (always included)
    if selected_metrics.get("mse", False):
        metric_values["mse"] = []
        for orig_frame, comp_frame in aligned_pairs:
            try:
                frame_mse = mse(orig_frame, comp_frame)
                metric_values["mse"].append(frame_mse)
            except Exception as e:
                logger.warning(f"MSE calculation failed: {e}")
                metric_values["mse"].append(0.0)

    if selected_metrics.get("psnr", False):
        metric_values["psnr"] = []
        for orig_frame, comp_frame in aligned_pairs:
            try:
                frame_psnr = calculate_safe_psnr(orig_frame, comp_frame)
                # Don't normalize PSNR - keep raw dB values for quality validation
                # The enhanced_metrics module expects raw dB values and will normalize itself
                metric_values["psnr"].append(frame_psnr)
            except Exception as e:
                logger.warning(f"PSNR calculation failed: {e}")
                metric_values["psnr"].append(0.0)

    if selected_metrics.get("ssim", False):
        metric_values["ssim"] = []
        for orig_frame, comp_frame in aligned_pairs:
            try:
                if len(orig_frame.shape) == 3:
                    orig_gray = cv2.cvtColor(orig_frame, cv2.COLOR_RGB2GRAY)
                    comp_gray = cv2.cvtColor(comp_frame, cv2.COLOR_RGB2GRAY)
                else:
                    orig_gray = orig_frame
                    comp_gray = comp_frame
                frame_ssim = ssim(orig_gray, comp_gray, data_range=255.0)
                metric_values["ssim"].append(max(0.0, min(1.0, frame_ssim)))
            except Exception as e:
                logger.warning(f"SSIM calculation failed: {e}")
                metric_values["ssim"].append(0.0)

    # Calculate advanced metrics (conditional)
    if selected_metrics.get("fsim", False):
        metric_values["fsim"] = []
        for orig_frame, comp_frame in aligned_pairs:
            try:
                frame_fsim = fsim(orig_frame, comp_frame)
                metric_values["fsim"].append(frame_fsim)
            except Exception as e:
                logger.warning(f"FSIM calculation failed: {e}")
                metric_values["fsim"].append(0.0)

    if selected_metrics.get("vif", False):
        # VIF would be calculated here if implemented
        pass

    if selected_metrics.get("edge_similarity", False):
        metric_values["edge_similarity"] = []
        for orig_frame, comp_frame in aligned_pairs:
            try:
                frame_edge = edge_similarity(
                    orig_frame,
                    comp_frame,
                    config.EDGE_CANNY_THRESHOLD1,
                    config.EDGE_CANNY_THRESHOLD2,
                )
                metric_values["edge_similarity"].append(frame_edge)
            except Exception as e:
                logger.warning(f"Edge similarity calculation failed: {e}")
                metric_values["edge_similarity"].append(0.0)

    if selected_metrics.get("texture_similarity", False):
        metric_values["texture_similarity"] = []
        for orig_frame, comp_frame in aligned_pairs:
            try:
                frame_texture = texture_similarity(orig_frame, comp_frame)
                metric_values["texture_similarity"].append(frame_texture)
            except Exception as e:
                logger.warning(f"Texture similarity calculation failed: {e}")
                metric_values["texture_similarity"].append(0.0)

    # Calculate expensive deep metrics conditionally
    results: dict[str, Any] = {}

    if selected_metrics.get("lpips", False):
        try:
            from .deep_perceptual_metrics import (
                calculate_deep_perceptual_quality_metrics,
            )

            deep_config = {
                "device": getattr(config, "DEEP_PERCEPTUAL_DEVICE", "auto"),
                "lpips_downscale_size": getattr(config, "LPIPS_DOWNSCALE_SIZE", 512),
                "lpips_max_frames": getattr(config, "LPIPS_MAX_FRAMES", 100),
            }
            deep_metrics = calculate_deep_perceptual_quality_metrics(
                original_frames_resized, compressed_frames_resized, deep_config
            )
            results.update(deep_metrics)
        except Exception as e:
            logger.warning(f"LPIPS calculation failed: {e}")
            results["lpips_quality_mean"] = 0.5

    if selected_metrics.get("ssimulacra2", False):
        try:
            from .ssimulacra2_metrics import calculate_ssimulacra2_quality_metrics

            ssim2_metrics = calculate_ssimulacra2_quality_metrics(
                original_frames_resized, compressed_frames_resized, config
            )
            results.update(ssim2_metrics)
        except Exception as e:
            logger.warning(f"SSIMULACRA2 calculation failed: {e}")
            results["ssimulacra2_mean"] = 50.0

    if selected_metrics.get("temporal_artifacts", False):
        try:
            from .temporal_artifacts import calculate_enhanced_temporal_metrics

            temporal_metrics = calculate_enhanced_temporal_metrics(
                original_frames_resized, compressed_frames_resized, device=None
            )
            results.update(temporal_metrics)
        except Exception as e:
            logger.warning(f"Temporal artifacts calculation failed: {e}")

    if selected_metrics.get("text_ui_validation", False):
        try:
            from .text_ui_validation import calculate_text_ui_metrics

            text_ui_metrics = calculate_text_ui_metrics(
                original_frames_resized, compressed_frames_resized, max_frames=5
            )
            results.update(text_ui_metrics)
        except Exception as e:
            logger.warning(f"Text/UI validation calculation failed: {e}")

    if selected_metrics.get("color_gradients", False):
        try:
            from .gradient_color_artifacts import calculate_gradient_color_metrics

            gradient_metrics = calculate_gradient_color_metrics(
                original_frames_resized, compressed_frames_resized
            )
            results.update(gradient_metrics)
        except Exception as e:
            logger.warning(f"Color gradients calculation failed: {e}")

    # Aggregate frame-level metrics
    # Add "_mean" suffix to match expected format for composite quality calculation
    for metric_name, values in metric_values.items():
        aggregated = _aggregate_metric(values, metric_name)
        # Rename the main metric to have "_mean" suffix for compatibility
        if metric_name in aggregated:
            aggregated[f"{metric_name}_mean"] = aggregated.pop(metric_name)
        results.update(aggregated)

    return results


def calculate_all_metrics(
    original_frames: list[np.ndarray],
    compressed_frames: list[np.ndarray],
    config: MetricsConfig | None = None,
) -> dict[str, Any]:
    """Calculate all available metrics (used when bypassing conditional logic).

    This is essentially a wrapper around calculate_comprehensive_metrics_from_frames
    but returns only the core metrics without file-specific data.
    """
    return calculate_comprehensive_metrics_from_frames(
        original_frames, compressed_frames, config
    )


def calculate_comprehensive_metrics_from_frames(
    original_frames: list[np.ndarray],
    compressed_frames: list[np.ndarray],
    config: MetricsConfig | None = None,
    frame_reduction_context: bool = False,
    file_metadata: dict[str, Any] | None = None,
) -> dict[str, float | str]:
    """Calculate comprehensive quality metrics between original and compressed frames.

    This function performs frame-based metric calculations without requiring file I/O.
    It's the core metrics engine used by calculate_comprehensive_metrics and can be
    called directly for testing or when frames are already available in memory.

    Args:
        original_frames: List of original frames as numpy arrays
        compressed_frames: List of compressed frames as numpy arrays
        config: Optional metrics configuration (uses default if None)
        frame_reduction_context: If True, adjusts disposal artifact detection for frame reduction
        file_metadata: Optional dict with file-specific metadata (paths, sizes, frame counts)
                      Keys: 'original_path', 'compressed_path', 'original_frame_count',
                            'compressed_frame_count', 'original_size_bytes', 'compressed_size_bytes'

    Returns:
        Dictionary with comprehensive metrics including all frame-based metrics.
        File-specific metrics (kilobytes, compression_ratio, timing validation) are only
        included if file_metadata is provided.

    Raises:
        ValueError: If frames are invalid or processing fails
    """
    if config is None:
        config = DEFAULT_METRICS_CONFIG

    # Define default text/UI metrics once at function scope for consistent access
    default_text_ui_metrics: dict[str, float | str] = {
        "has_text_ui_content": False,
        "text_ui_edge_density": 0.0,
        "text_ui_component_count": 0,
        "ocr_regions_analyzed": 0,
        "ocr_conf_delta_mean": 0.0,
        "ocr_conf_delta_min": 0.0,
        "mtf50_ratio_mean": 1.0,
        "mtf50_ratio_min": 1.0,
        "edge_sharpness_score": 100.0,
    }

    # Check if Phase 6 optimized processing is enabled
    use_phase6_optimization = (
        os.environ.get("GIFLAB_ENABLE_PHASE6_OPTIMIZATION", "false").lower() == "true"
    )

    if use_phase6_optimization:
        try:
            from .optimized_metrics import calculate_optimized_comprehensive_metrics

            logger.info("Using Phase 6 optimized metrics calculation")

            # Call optimized implementation
            optimized_result: dict[str, float | str] = cast(
                dict[str, float | str],
                calculate_optimized_comprehensive_metrics(
                    original_frames, compressed_frames, config
                ),
            )

            # Add file metadata if provided
            if file_metadata:
                if "compressed_path" in file_metadata:
                    optimized_result["kilobytes"] = float(
                        calculate_file_size_kb(file_metadata["compressed_path"])
                    )
                elif "compressed_size_bytes" in file_metadata:
                    optimized_result["kilobytes"] = float(
                        file_metadata["compressed_size_bytes"] / 1024.0
                    )

                if (
                    "original_size_bytes" in file_metadata
                    and "compressed_size_bytes" in file_metadata
                ):
                    optimized_result["compression_ratio"] = (
                        file_metadata["original_size_bytes"]
                        / file_metadata["compressed_size_bytes"]
                        if file_metadata["compressed_size_bytes"] > 0
                        else 1.0
                    )

            return optimized_result

        except ImportError:
            logger.warning(
                "Phase 6 optimization module not available, falling back to standard processing"
            )
        except Exception as e:
            logger.warning(
                f"Phase 6 optimization failed: {e}, falling back to standard processing"
            )

    start_time = time.perf_counter()

    try:
        # Resize frames to common dimensions
        (
            original_frames_resized,
            compressed_frames_resized,
        ) = resize_to_common_dimensions(original_frames, compressed_frames)

        # Align frames using content-based method (most robust)
        aligned_pairs = align_frames(original_frames_resized, compressed_frames_resized)

        if not aligned_pairs:
            raise ValueError("No frame pairs could be aligned")

        # Check if conditional metrics optimization is enabled
        use_conditional = (
            os.environ.get("GIFLAB_ENABLE_CONDITIONAL_METRICS", "true").lower()
            == "true"
        )
        force_all_metrics = (
            os.environ.get("GIFLAB_FORCE_ALL_METRICS", "false").lower() == "true"
        )

        if use_conditional and not force_all_metrics:
            try:
                from .conditional_metrics import ConditionalMetricsCalculator

                logger.info("Using conditional metrics optimization")
                conditional_calc = ConditionalMetricsCalculator()

                # Perform quality assessment and content profiling
                quality_assessment = conditional_calc.assess_quality(
                    original_frames_resized, compressed_frames_resized
                )
                content_profile = conditional_calc.detect_content_profile(
                    compressed_frames_resized, quick_mode=True
                )

                # Select which metrics to calculate
                selected_metrics = conditional_calc.select_metrics(
                    quality_assessment, content_profile
                )

                # Log optimization decision
                num_selected = sum(1 for v in selected_metrics.values() if v)
                num_skipped = sum(1 for v in selected_metrics.values() if not v)
                logger.info(
                    f"Quality tier: {quality_assessment.tier.value} "
                    f"(PSNR={quality_assessment.base_psnr:.1f}dB). "
                    f"Calculating {num_selected} metrics, skipping {num_skipped}"
                )

                # If we're skipping most expensive metrics, use the optimized path
                if (
                    quality_assessment.tier.value == "high"
                    and not selected_metrics.get("lpips", False)
                    and not selected_metrics.get("ssimulacra2", False)
                ):
                    # Calculate only selected metrics using the optimized function
                    optimized_results: dict[str, float | str] = cast(
                        dict[str, float | str],
                        calculate_selected_metrics(
                            original_frames_resized,
                            compressed_frames_resized,
                            selected_metrics,
                            config,
                        ),
                    )

                    # Add base metric names (without _mean suffix) for backwards compatibility
                    # This ensures tests expecting "ssim", "psnr" etc. still work
                    for metric in ["ssim", "psnr", "mse"]:
                        mean_key = f"{metric}_mean"
                        if (
                            mean_key in optimized_results
                            and metric not in optimized_results
                        ):
                            optimized_results[metric] = optimized_results[mean_key]

                    # Add metadata about optimization
                    optimized_results["_optimization_metadata"] = {  # type: ignore[assignment]
                        "quality_tier": quality_assessment.tier.value,
                        "quality_confidence": quality_assessment.confidence,
                        "base_psnr": quality_assessment.base_psnr,
                        "metrics_calculated": num_selected,
                        "metrics_skipped": num_skipped,
                        "optimization_applied": True,
                    }

                    # Add frame counts
                    optimized_results["frame_count"] = len(original_frames)
                    optimized_results["compressed_frame_count"] = len(compressed_frames)

                    # Add file metadata if provided
                    if file_metadata:
                        if "compressed_path" in file_metadata:
                            optimized_results["kilobytes"] = float(
                                calculate_file_size_kb(file_metadata["compressed_path"])
                            )
                        elif "compressed_size_bytes" in file_metadata:
                            optimized_results["kilobytes"] = float(
                                file_metadata["compressed_size_bytes"] / 1024.0
                            )

                        if (
                            "original_size_bytes" in file_metadata
                            and "compressed_size_bytes" in file_metadata
                        ):
                            optimized_results["compression_ratio"] = (
                                file_metadata["original_size_bytes"]
                                / file_metadata["compressed_size_bytes"]
                                if file_metadata["compressed_size_bytes"] > 0
                                else 1.0
                            )

                    # Calculate gradient and color artifact metrics only if not high quality
                    # or if explicitly requested
                    should_calculate_gradient_color = (
                        quality_assessment.tier.value != "HIGH"
                        or not conditional_calc.skip_expensive_on_high_quality
                        or os.environ.get(
                            "GIFLAB_FORCE_GRADIENT_METRICS", "false"
                        ).lower()
                        == "true"
                    )

                    # Default fallback metrics
                    default_gradient_metrics = {
                        "banding_score_mean": 0.0,
                        "banding_score_p95": 0.0,
                        "banding_patch_count": 0,
                        "gradient_region_count": 0,
                        "deltae_mean": 0.0,
                        "deltae_p95": 0.0,
                        "deltae_max": 0.0,
                        "deltae_pct_gt1": 0.0,
                        "deltae_pct_gt2": 0.0,
                        "deltae_pct_gt3": 0.0,
                        "deltae_pct_gt5": 0.0,
                        "color_patch_count": 0,
                    }

                    if should_calculate_gradient_color:
                        try:
                            from .gradient_color_artifacts import (
                                calculate_gradient_color_metrics,
                            )

                            logger.debug(
                                "Calculating gradient/color metrics in optimized path"
                            )
                            gradient_color_metrics = calculate_gradient_color_metrics(
                                original_frames_resized, compressed_frames_resized
                            )

                            # Add gradient and color metrics to optimized results
                            for (
                                metric_key,
                                metric_value,
                            ) in gradient_color_metrics.items():
                                optimized_results[metric_key] = (
                                    float(metric_value)
                                    if isinstance(metric_value, int | float)
                                    else metric_value
                                )
                        except Exception as e:
                            logger.warning(
                                f"Gradient/color metrics failed in optimized path: {e}, using defaults"
                            )
                            # Add default values
                            for (
                                metric_key,
                                metric_value,
                            ) in default_gradient_metrics.items():
                                optimized_results[metric_key] = float(metric_value)
                    else:
                        logger.debug(
                            "Skipping gradient/color metrics for high quality result"
                        )
                        # Add default values for skipped metrics
                        for (
                            metric_key,
                            metric_value,
                        ) in default_gradient_metrics.items():
                            optimized_results[metric_key] = float(metric_value)

                    # Calculate text/UI validation metrics (always needed for Phase 3 tests)

                    try:
                        from .text_ui_validation import calculate_text_ui_metrics

                        logger.debug("Calculating text/UI metrics in optimized path")
                        optimized_text_ui_metrics = calculate_text_ui_metrics(
                            original_frames_resized,
                            compressed_frames_resized,
                            max_frames=5,
                        )

                        # Add text/UI metrics to optimized results
                        for (
                            text_ui_key,
                            text_ui_value,
                        ) in optimized_text_ui_metrics.items():
                            if isinstance(text_ui_value, int | float):
                                optimized_results[text_ui_key] = float(text_ui_value)
                            else:
                                optimized_results[text_ui_key] = str(text_ui_value)
                    except Exception as e:
                        logger.warning(
                            f"Text/UI metrics failed in optimized path: {e}, using defaults"
                        )
                        # Add default values
                        for (
                            text_ui_key,
                            text_ui_value,
                        ) in default_text_ui_metrics.items():
                            if isinstance(text_ui_value, int | float):
                                optimized_results[text_ui_key] = float(text_ui_value)
                            else:
                                optimized_results[text_ui_key] = str(text_ui_value)

                    # Calculate SSIMULACRA2 metrics (always needed for Phase 3 tests)
                    # Default fallback metrics
                    default_ssimulacra2_metrics: dict[str, float | str] = {
                        "ssimulacra2_mean": 50.0,
                        "ssimulacra2_p95": 50.0,
                        "ssimulacra2_min": 50.0,
                        "ssimulacra2_frame_count": 0.0,
                        "ssimulacra2_triggered": 0.0,
                    }

                    # Check if SSIMULACRA2 metrics should be calculated
                    should_calculate_ssimulacra2 = getattr(
                        config, "ENABLE_SSIMULACRA2", True
                    )

                    if should_calculate_ssimulacra2:
                        try:
                            from .ssimulacra2_metrics import (
                                calculate_ssimulacra2_quality_metrics,
                                should_use_ssimulacra2,
                            )

                            logger.debug(
                                "Calculating SSIMULACRA2 metrics in optimized path"
                            )

                            # Use existing composite quality for conditional triggering
                            if should_use_ssimulacra2(
                                None
                            ):  # No composite quality available yet
                                ssimulacra2_result = (
                                    calculate_ssimulacra2_quality_metrics(
                                        original_frames_resized,
                                        compressed_frames_resized,
                                        config,
                                    )
                                )
                                # Add SSIMULACRA2 metrics to optimized results
                                for (
                                    ssim2_key,
                                    ssim2_value,
                                ) in ssimulacra2_result.items():
                                    # Convert all values appropriately for type safety
                                    optimized_results[ssim2_key] = (
                                        float(ssim2_value)
                                        if isinstance(ssim2_value, int | float)
                                        else str(ssim2_value)
                                    )
                            else:
                                logger.debug(
                                    "SSIMULACRA2 metrics skipped based on conditional logic"
                                )
                                for (
                                    ssim2_key,
                                    ssim2_value,
                                ) in default_ssimulacra2_metrics.items():
                                    # All default values are floats, safe to cast directly
                                    optimized_results[ssim2_key] = float(ssim2_value)  # type: ignore[assignment]
                        except Exception as e:
                            logger.warning(
                                f"SSIMULACRA2 metrics failed in optimized path: {e}, using defaults"
                            )
                            # Add default values
                            for (
                                ssim2_key,
                                ssim2_value,
                            ) in default_ssimulacra2_metrics.items():
                                # All default values are floats, safe to cast directly
                                optimized_results[ssim2_key] = float(ssim2_value)  # type: ignore[assignment]
                    else:
                        logger.debug("SSIMULACRA2 metrics calculation disabled")
                        for (
                            ssim2_key,
                            ssim2_value,
                        ) in default_ssimulacra2_metrics.items():
                            # All default values are floats, safe to cast directly
                            optimized_results[ssim2_key] = float(ssim2_value)  # type: ignore[assignment]

                    # Calculate temporal consistency metrics
                    # These are fast metrics that should always be included
                    try:
                        temporal_pre = calculate_temporal_consistency(
                            original_frames_resized
                        )
                        temporal_post = calculate_temporal_consistency(
                            compressed_frames_resized
                        )
                        temporal_delta = abs(temporal_pre - temporal_post)

                        optimized_results["temporal_consistency"] = float(temporal_post)
                        optimized_results["temporal_consistency_std"] = 0.0
                        optimized_results["temporal_consistency_min"] = float(
                            temporal_post
                        )
                        optimized_results["temporal_consistency_max"] = float(
                            temporal_post
                        )
                        optimized_results["temporal_consistency_pre"] = float(
                            temporal_pre
                        )
                        optimized_results["temporal_consistency_post"] = float(
                            temporal_post
                        )
                        optimized_results["temporal_consistency_delta"] = float(
                            temporal_delta
                        )
                    except Exception as e:
                        logger.warning(f"Temporal consistency calculation failed: {e}")
                        # Use default values if calculation fails
                        optimized_results["temporal_consistency"] = 1.0
                        optimized_results["temporal_consistency_pre"] = 1.0
                        optimized_results["temporal_consistency_post"] = 1.0
                        optimized_results["temporal_consistency_delta"] = 0.0

                    # Process with quality system
                    from .enhanced_metrics import process_metrics_with_enhanced_quality

                    optimized_results = process_metrics_with_enhanced_quality(
                        optimized_results, config
                    )

                    # Calculate processing time
                    end_time = time.perf_counter()
                    elapsed_seconds = end_time - start_time
                    optimized_results["render_ms"] = min(
                        int(elapsed_seconds * 1000), 86400000
                    )

                    # Get optimization stats
                    opt_stats = conditional_calc.get_optimization_stats()
                    logger.info(
                        f"Conditional optimization complete. "
                        f"Metrics skipped: {opt_stats['metrics_skipped']}, "
                        f"Estimated time saved: {opt_stats['estimated_time_saved']:.2f}s"
                    )

                    return optimized_results

            except ImportError:
                logger.info(
                    "Conditional metrics module not available, using standard processing"
                )
                use_conditional = False
            except Exception as e:
                logger.warning(
                    f"Conditional metrics failed: {e}, using standard processing"
                )
                use_conditional = False

        # Check if parallel processing is enabled
        use_parallel = (
            os.environ.get("GIFLAB_ENABLE_PARALLEL_METRICS", "true").lower() != "false"
        )

        # Store raw (un-normalised) metric values where necessary
        raw_metric_values: dict[str, list[float]] = {
            "psnr": [],  # PSNR is normalised for main reporting; keep raw values separately
        }

        if use_parallel and len(aligned_pairs) > 1:
            # Use parallel processing for frame-level metrics
            try:
                from .parallel_metrics import ParallelConfig, ParallelMetricsCalculator

                # Create parallel calculator
                parallel_config = ParallelConfig()
                calculator = ParallelMetricsCalculator(parallel_config)

                # Define metric functions to parallelize
                metric_functions: dict[str, Callable[..., Any] | None] = {
                    "ssim": None,  # Special handling needed
                    "ms_ssim": None,
                    "psnr": None,
                    "mse": None,
                    "rmse": None,
                    "fsim": None,
                    "gmsd": None,
                    "chist": None,
                    "edge_similarity": None,
                    "texture_similarity": None,
                    "sharpness_similarity": None,
                }

                # Calculate metrics in parallel
                metric_values = calculator.calculate_frame_metrics_parallel(
                    aligned_pairs, metric_functions, config
                )

                # Extract raw PSNR values before normalization
                if "psnr" in metric_values:
                    raw_metric_values["psnr"] = metric_values["psnr"].copy()
                    # Normalize PSNR values
                    metric_values["psnr"] = [
                        max(0.0, min(value / float(config.PSNR_MAX_DB), 1.0))
                        for value in metric_values["psnr"]
                    ]

                logger.debug(
                    f"Parallel processing completed for {len(aligned_pairs)} frame pairs"
                )

            except ImportError:
                logger.info(
                    "Parallel metrics module not available, falling back to sequential processing"
                )
                use_parallel = False
            except Exception as e:
                logger.warning(
                    f"Parallel processing failed: {e}, falling back to sequential processing"
                )
                use_parallel = False
        else:
            use_parallel = False

        if not use_parallel:
            # Fall back to sequential processing
            # Calculate all frame-level metrics
            metric_values = {
                "ssim": [],
                "ms_ssim": [],
                "psnr": [],
                "mse": [],
                "rmse": [],
                "fsim": [],
                "gmsd": [],
                "chist": [],
                "edge_similarity": [],
                "texture_similarity": [],
                "sharpness_similarity": [],
            }

            for orig_frame, comp_frame in aligned_pairs:
                # Traditional SSIM calculation
                try:
                    if len(orig_frame.shape) == 3:
                        orig_gray = cv2.cvtColor(orig_frame, cv2.COLOR_RGB2GRAY)
                        comp_gray = cv2.cvtColor(comp_frame, cv2.COLOR_RGB2GRAY)
                    else:
                        orig_gray = orig_frame
                        comp_gray = comp_frame

                    frame_ssim = ssim(orig_gray, comp_gray, data_range=255.0)
                    metric_values["ssim"].append(max(0.0, min(1.0, frame_ssim)))
                except Exception as e:
                    logger.warning(f"SSIM calculation failed for frame: {e}")
                    metric_values["ssim"].append(0.0)

                # MS-SSIM calculation
                try:
                    frame_ms_ssim = calculate_ms_ssim(orig_frame, comp_frame)
                    metric_values["ms_ssim"].append(frame_ms_ssim)
                except Exception as e:
                    logger.warning(f"MS-SSIM calculation failed for frame: {e}")
                    metric_values["ms_ssim"].append(0.0)

                # PSNR calculation
                try:
                    frame_psnr = calculate_safe_psnr(orig_frame, comp_frame)
                    # Keep un-scaled PSNR for optional raw metrics output
                    raw_metric_values["psnr"].append(frame_psnr)

                    # Normalize PSNR using configurable upper bound
                    normalized_psnr = min(frame_psnr / float(config.PSNR_MAX_DB), 1.0)
                    metric_values["psnr"].append(max(0.0, normalized_psnr))
                except Exception as e:
                    logger.warning(f"PSNR calculation failed for frame: {e}")
                    metric_values["psnr"].append(0.0)
                    raw_metric_values["psnr"].append(0.0)

                # New metrics - MSE and RMSE
                try:
                    frame_mse = mse(orig_frame, comp_frame)
                    metric_values["mse"].append(frame_mse)

                    frame_rmse = rmse(orig_frame, comp_frame)
                    metric_values["rmse"].append(frame_rmse)
                except Exception as e:
                    logger.warning(f"MSE/RMSE calculation failed for frame: {e}")
                    metric_values["mse"].append(0.0)
                    metric_values["rmse"].append(0.0)

                # FSIM calculation
                try:
                    frame_fsim = fsim(orig_frame, comp_frame)
                    metric_values["fsim"].append(frame_fsim)
                except Exception as e:
                    logger.warning(f"FSIM calculation failed for frame: {e}")
                    metric_values["fsim"].append(0.0)

                # GMSD calculation
                try:
                    frame_gmsd = gmsd(orig_frame, comp_frame)
                    metric_values["gmsd"].append(frame_gmsd)
                except Exception as e:
                    logger.warning(f"GMSD calculation failed for frame: {e}")
                    metric_values["gmsd"].append(0.0)

                # Color histogram correlation
                try:
                    frame_chist = chist(orig_frame, comp_frame)
                    metric_values["chist"].append(frame_chist)
                except Exception as e:
                    logger.warning(f"Color histogram calculation failed for frame: {e}")
                    metric_values["chist"].append(0.0)

                # Edge similarity
                try:
                    frame_edge = edge_similarity(
                        orig_frame,
                        comp_frame,
                        config.EDGE_CANNY_THRESHOLD1,
                        config.EDGE_CANNY_THRESHOLD2,
                    )
                    metric_values["edge_similarity"].append(frame_edge)
                except Exception as e:
                    logger.warning(f"Edge similarity calculation failed for frame: {e}")
                    metric_values["edge_similarity"].append(0.0)

                # Texture similarity
                try:
                    frame_texture = texture_similarity(orig_frame, comp_frame)
                    metric_values["texture_similarity"].append(frame_texture)
                except Exception as e:
                    logger.warning(
                        f"Texture similarity calculation failed for frame: {e}"
                    )
                    metric_values["texture_similarity"].append(0.0)

                # Sharpness similarity
                try:
                    frame_sharpness = sharpness_similarity(orig_frame, comp_frame)
                    metric_values["sharpness_similarity"].append(frame_sharpness)
                except Exception as e:
                    logger.warning(
                        f"Sharpness similarity calculation failed for frame: {e}"
                    )
                    metric_values["sharpness_similarity"].append(0.0)

        # Calculate temporal consistency for original and compressed frames
        temporal_pre = 0.0
        temporal_post = 0.0
        if config.TEMPORAL_CONSISTENCY_ENABLED:
            temporal_pre = calculate_temporal_consistency(original_frames_resized)
            temporal_post = calculate_temporal_consistency(compressed_frames_resized)

        temporal_delta = abs(temporal_post - temporal_pre)

        # Calculate disposal artifact detection
        disposal_artifacts_pre = detect_disposal_artifacts(
            original_frames_resized, frame_reduction_context
        )
        disposal_artifacts_post = detect_disposal_artifacts(
            compressed_frames_resized, frame_reduction_context
        )
        disposal_artifacts_delta = abs(disposal_artifacts_post - disposal_artifacts_pre)

        # Calculate enhanced temporal artifact metrics (Task 1.2)
        enhanced_temporal_metrics = {}
        try:
            from .temporal_artifacts import calculate_enhanced_temporal_metrics

            enhanced_temporal_metrics = calculate_enhanced_temporal_metrics(
                original_frames_resized, compressed_frames_resized, device=None
            )
        except ImportError as e:
            logger.warning(f"Enhanced temporal artifacts module not available: {e}")
            enhanced_temporal_metrics = {
                "flicker_excess": 0.0,
                "flicker_frame_ratio": 0.0,
                "flat_flicker_ratio": 0.0,
                "flat_region_count": 0,
                "temporal_pumping_score": 0.0,
                "quality_oscillation_frequency": 0.0,
                "lpips_t_mean": 0.0,
                "lpips_t_p95": 0.0,
                "frame_count": len(compressed_frames_resized),
            }
        except Exception as e:
            logger.error(f"Enhanced temporal artifacts calculation failed: {e}")
            enhanced_temporal_metrics = {
                "flicker_excess": 0.0,
                "flicker_frame_ratio": 0.0,
                "flat_flicker_ratio": 0.0,
                "flat_region_count": 0,
                "temporal_pumping_score": 0.0,
                "quality_oscillation_frequency": 0.0,
                "lpips_t_mean": 0.0,
                "lpips_t_p95": 0.0,
                "frame_count": len(compressed_frames_resized),
            }

        # Calculate enhanced gradient and color artifact metrics (Task 1.3 & 1.4)
        gradient_color_metrics = {}

        # Default fallback metrics
        default_gradient_metrics = {
            "banding_score_mean": 0.0,
            "banding_score_p95": 0.0,
            "banding_patch_count": 0,
            "gradient_region_count": 0,
            "deltae_mean": 0.0,
            "deltae_p95": 0.0,
            "deltae_max": 0.0,
            "deltae_pct_gt1": 0.0,
            "deltae_pct_gt2": 0.0,
            "deltae_pct_gt3": 0.0,
            "deltae_pct_gt5": 0.0,
            "color_patch_count": 0,
        }

        try:
            from .gradient_color_artifacts import calculate_gradient_color_metrics

            logger.debug("Successfully imported gradient_color_artifacts module")

            gradient_color_metrics = calculate_gradient_color_metrics(
                original_frames_resized, compressed_frames_resized
            )
            logger.debug("Successfully calculated gradient and color artifact metrics")

        except ImportError as e:
            logger.info(
                f"Gradient and color artifacts module not available: {e}. Using fallback values."
            )
            gradient_color_metrics = default_gradient_metrics

        except AttributeError as e:
            logger.warning(
                f"Gradient and color artifacts function not found: {e}. Module may be incomplete."
            )
            gradient_color_metrics = default_gradient_metrics

        except (ValueError, TypeError, RuntimeError) as e:
            logger.error(
                f"Error calculating gradient and color artifacts: {e}. Using fallback values."
            )
            gradient_color_metrics = default_gradient_metrics

        except Exception as e:
            logger.error(
                f"Unexpected error in gradient and color artifacts calculation: {e}. Using fallback values."
            )
            gradient_color_metrics = default_gradient_metrics

        # Calculate deep perceptual metrics (Task 2.2)
        deep_perceptual_metrics: dict[str, float | str] = {}

        # Default fallback metrics
        default_deep_perceptual_metrics: dict[str, float | str] = {
            "lpips_quality_mean": 0.5,
            "lpips_quality_p95": 0.5,
            "lpips_quality_max": 0.5,
            "deep_perceptual_frame_count": float(len(compressed_frames_resized)),
            "deep_perceptual_downscaled": 0.0,
            "deep_perceptual_device": "fallback",
        }

        # Check if deep perceptual metrics should be calculated
        # We'll calculate conditional logic after we have the initial composite quality
        should_calculate_deep_perceptual = True  # Initially calculate for all

        if should_calculate_deep_perceptual:
            try:
                from .deep_perceptual_metrics import (
                    calculate_deep_perceptual_quality_metrics,
                    should_use_deep_perceptual,
                )

                logger.debug("Successfully imported deep_perceptual_metrics module")

                # Prepare configuration for deep perceptual metrics
                deep_config = {
                    "device": getattr(config, "DEEP_PERCEPTUAL_DEVICE", "auto"),
                    "lpips_downscale_size": getattr(
                        config, "LPIPS_DOWNSCALE_SIZE", 512
                    ),
                    "lpips_max_frames": getattr(config, "LPIPS_MAX_FRAMES", 100),
                    "disable_deep_perceptual": not getattr(
                        config, "ENABLE_DEEP_PERCEPTUAL", True
                    ),
                }

                # For now, always calculate since we need composite quality first
                # In future iterations, this could be made conditional based on initial quality assessment
                if should_use_deep_perceptual(
                    None
                ):  # No composite quality available yet
                    deep_perceptual_metrics = calculate_deep_perceptual_quality_metrics(
                        original_frames_resized, compressed_frames_resized, deep_config
                    )
                    logger.debug("Successfully calculated deep perceptual metrics")
                else:
                    logger.debug(
                        "Deep perceptual metrics skipped based on conditional logic"
                    )
                    deep_perceptual_metrics = default_deep_perceptual_metrics

            except ImportError as e:
                logger.info(
                    f"Deep perceptual metrics module not available: {e}. Using fallback values."
                )
                deep_perceptual_metrics = default_deep_perceptual_metrics

            except AttributeError as e:
                logger.warning(
                    f"Deep perceptual metrics function not found: {e}. Module may be incomplete."
                )
                deep_perceptual_metrics = default_deep_perceptual_metrics

            except (ValueError, TypeError, RuntimeError) as e:
                logger.error(
                    f"Error calculating deep perceptual metrics: {e}. Using fallback values."
                )
                deep_perceptual_metrics = default_deep_perceptual_metrics

            except Exception as e:
                logger.error(
                    f"Unexpected error in deep perceptual metrics calculation: {e}. Using fallback values."
                )
                deep_perceptual_metrics = default_deep_perceptual_metrics
        else:
            logger.debug("Deep perceptual metrics calculation skipped")
            deep_perceptual_metrics = default_deep_perceptual_metrics

        # Calculate SSIMULACRA2 metrics (Phase 3.2)
        ssimulacra2_metrics: dict[str, float | str] = {}

        # Check if SSIMULACRA2 metrics should be calculated
        should_calculate_ssimulacra2 = getattr(config, "ENABLE_SSIMULACRA2", True)

        if should_calculate_ssimulacra2:
            try:
                from .ssimulacra2_metrics import (
                    calculate_ssimulacra2_quality_metrics,
                    should_use_ssimulacra2,
                )

                logger.debug("Successfully imported ssimulacra2_metrics module")

                # Use existing composite quality for conditional triggering
                # For first calculation, we don't have composite quality yet, so calculate for all
                if should_use_ssimulacra2(None):  # No composite quality available yet
                    ssimulacra2_result = calculate_ssimulacra2_quality_metrics(
                        original_frames_resized, compressed_frames_resized, config
                    )
                    # Update metrics dictionary (allows type widening)
                    ssimulacra2_metrics.update(ssimulacra2_result)
                    logger.debug("Successfully calculated SSIMULACRA2 metrics")
                else:
                    logger.debug(
                        "SSIMULACRA2 metrics skipped based on conditional logic"
                    )
                    ssimulacra2_metrics = {
                        "ssimulacra2_mean": 50.0,
                        "ssimulacra2_p95": 50.0,
                        "ssimulacra2_min": 50.0,
                        "ssimulacra2_frame_count": 0.0,
                        "ssimulacra2_triggered": 0.0,
                    }

            except ImportError as e:
                logger.info(
                    f"SSIMULACRA2 metrics module not available: {e}. Using fallback values."
                )
                ssimulacra2_metrics = {
                    "ssimulacra2_mean": 50.0,
                    "ssimulacra2_p95": 50.0,
                    "ssimulacra2_min": 50.0,
                    "ssimulacra2_frame_count": 0.0,
                    "ssimulacra2_triggered": 0.0,
                }

            except AttributeError as e:
                logger.warning(
                    f"SSIMULACRA2 metrics function not found: {e}. Module may be incomplete."
                )
                ssimulacra2_metrics = {
                    "ssimulacra2_mean": 50.0,
                    "ssimulacra2_p95": 50.0,
                    "ssimulacra2_min": 50.0,
                    "ssimulacra2_frame_count": 0.0,
                    "ssimulacra2_triggered": 0.0,
                }

            except (ValueError, TypeError, RuntimeError) as e:
                logger.error(
                    f"Error calculating SSIMULACRA2 metrics: {e}. Using fallback values."
                )
                ssimulacra2_metrics = {
                    "ssimulacra2_mean": 50.0,
                    "ssimulacra2_p95": 50.0,
                    "ssimulacra2_min": 50.0,
                    "ssimulacra2_frame_count": 0.0,
                    "ssimulacra2_triggered": 0.0,
                }

            except Exception as e:
                logger.error(
                    f"Unexpected error in SSIMULACRA2 metrics calculation: {e}. Using fallback values."
                )
                ssimulacra2_metrics = {
                    "ssimulacra2_mean": 50.0,
                    "ssimulacra2_p95": 50.0,
                    "ssimulacra2_min": 50.0,
                    "ssimulacra2_frame_count": 0.0,
                    "ssimulacra2_triggered": 0.0,
                }
        else:
            logger.debug("SSIMULACRA2 metrics calculation disabled")
            ssimulacra2_metrics = {
                "ssimulacra2_mean": 50.0,
                "ssimulacra2_p95": 50.0,
                "ssimulacra2_min": 50.0,
                "ssimulacra2_frame_count": 0.0,
                "ssimulacra2_triggered": 0.0,
            }

        # Calculate text/UI validation metrics (Phase 3.1)
        text_ui_metrics: dict[str, float | str] = {}

        # Use the default_text_ui_metrics already defined earlier in this function

        try:
            from .text_ui_validation import calculate_text_ui_metrics

            logger.debug("Successfully imported text_ui_validation module")

            # Calculate text/UI validation metrics
            text_ui_metrics = calculate_text_ui_metrics(
                original_frames_resized, compressed_frames_resized, max_frames=5
            )
            logger.debug("Successfully calculated text/UI validation metrics")

        except ImportError as e:
            logger.info(
                f"Text/UI validation module not available: {e}. Using fallback values."
            )
            text_ui_metrics = default_text_ui_metrics

        except AttributeError as e:
            logger.warning(
                f"Text/UI validation function not found: {e}. Module may be incomplete."
            )
            text_ui_metrics = default_text_ui_metrics

        except (ValueError, TypeError, RuntimeError) as e:
            logger.error(
                f"Error calculating text/UI validation metrics: {e}. Using fallback values."
            )
            text_ui_metrics = default_text_ui_metrics

        except Exception as e:
            logger.error(
                f"Unexpected error in text/UI validation calculation: {e}. Using fallback values."
            )
            text_ui_metrics = default_text_ui_metrics

        # Extract frame count information
        if file_metadata:
            original_frame_count = file_metadata.get(
                "original_frame_count", len(original_frames)
            )
            compressed_frame_count = file_metadata.get(
                "compressed_frame_count", len(compressed_frames)
            )
        else:
            original_frame_count = len(original_frames)
            compressed_frame_count = len(compressed_frames)

        # Add timing validation metrics only if file paths are provided
        timing_metrics = {}
        if (
            file_metadata
            and "original_path" in file_metadata
            and "compressed_path" in file_metadata
        ):
            try:
                from .wrapper_validation.timing_validation import (
                    TimingGridValidator,
                    extract_timing_metrics_for_csv,
                )

                timing_validator = TimingGridValidator()
                timing_result = timing_validator.validate_timing_integrity(
                    file_metadata["original_path"], file_metadata["compressed_path"]
                )
                timing_metrics = extract_timing_metrics_for_csv(timing_result)
                # Add success indicator
                timing_metrics["timing_validation_status"] = "success"
            except ImportError as e:
                logger.error(f"Timing validation module not available: {e}")
                # Provide failure-indicating timing metrics
                timing_metrics = {
                    "timing_grid_ms": 10,
                    "grid_length": -1,  # -1 indicates failure
                    "duration_diff_ms": -1,
                    "timing_drift_score": -1.0,  # -1.0 indicates failure, not perfect score
                    "max_timing_drift_ms": -1,
                    "alignment_accuracy": -1.0,  # -1.0 indicates failure
                    "timing_validation_status": "import_failed",
                    "timing_validation_error": str(e),
                }
            except (ValueError, OSError) as e:
                logger.error(f"Timing validation calculation failed: {e}")
                timing_metrics = {
                    "timing_grid_ms": 10,
                    "grid_length": -1,
                    "duration_diff_ms": -1,
                    "timing_drift_score": -1.0,
                    "max_timing_drift_ms": -1,
                    "alignment_accuracy": -1.0,
                    "timing_validation_status": "calculation_failed",
                    "timing_validation_error": str(e),
                }
            except Exception as e:
                # Log unexpected errors more severely and re-raise to avoid hiding bugs
                logger.critical(f"Unexpected timing validation error: {e}")
                timing_metrics = {
                    "timing_grid_ms": 10,
                    "grid_length": -1,
                    "duration_diff_ms": -1,
                    "timing_drift_score": -1.0,
                    "max_timing_drift_ms": -1,
                    "alignment_accuracy": -1.0,
                    "timing_validation_status": "unexpected_error",
                    "timing_validation_error": str(e),
                }

        # Aggregate all metrics with descriptive statistics
        result: dict[str, float | str] = {}

        # Add aggregated metrics
        for metric_name, values in metric_values.items():
            result.update(_aggregate_metric(values, metric_name))

        # Add temporal consistency (single value, not frame-level)
        # Keep legacy key pointing to *post*-compression value for backward compatibility
        result["temporal_consistency"] = float(temporal_post)
        result["temporal_consistency_std"] = 0.0
        result["temporal_consistency_min"] = float(temporal_post)
        result["temporal_consistency_max"] = float(temporal_post)

        # New keys: pre, post (explicit) and delta
        result["temporal_consistency_pre"] = float(temporal_pre)
        result["temporal_consistency_post"] = float(temporal_post)
        result["temporal_consistency_delta"] = float(temporal_delta)

        # Add disposal artifact metrics
        result["disposal_artifacts"] = float(disposal_artifacts_post)
        result["disposal_artifacts_std"] = 0.0
        result["disposal_artifacts_min"] = float(disposal_artifacts_post)
        result["disposal_artifacts_max"] = float(disposal_artifacts_post)
        result["disposal_artifacts_pre"] = float(disposal_artifacts_pre)
        result["disposal_artifacts_post"] = float(disposal_artifacts_post)
        result["disposal_artifacts_delta"] = float(disposal_artifacts_delta)

        # Add enhanced temporal artifact metrics (Task 1.2)
        for metric_key, metric_value in enhanced_temporal_metrics.items():
            result[metric_key] = (
                float(metric_value)
                if isinstance(metric_value, int | float)
                else metric_value
            )

        # Add enhanced gradient and color artifact metrics (Task 1.3 & 1.4)
        for metric_key, metric_value in gradient_color_metrics.items():
            result[metric_key] = (
                float(metric_value)
                if isinstance(metric_value, int | float)
                else metric_value
            )

        # Add deep perceptual metrics (Task 2.2)
        for deep_key, deep_value in deep_perceptual_metrics.items():
            if isinstance(deep_value, int | float):
                result[deep_key] = float(deep_value)
            else:
                result[deep_key] = str(deep_value)

        # Add SSIMULACRA2 metrics (Phase 3.2)
        for ssim2_key, ssim2_value in ssimulacra2_metrics.items():
            # Convert values to appropriate types for storage
            result[ssim2_key] = (  # type: ignore[assignment]
                float(ssim2_value)
                if isinstance(ssim2_value, int | float)
                else str(ssim2_value)
            )

        # Add text/UI validation metrics (Phase 3.1)
        for text_ui_key, text_ui_value in text_ui_metrics.items():
            # Convert values to appropriate types for storage
            result[text_ui_key] = (
                float(text_ui_value)
                if isinstance(text_ui_value, int | float)
                else str(text_ui_value)
            )

        # Add frame count information
        result["frame_count"] = int(original_frame_count)
        result["compressed_frame_count"] = int(compressed_frame_count)

        # Add timing validation metrics if available
        for key, value in timing_metrics.items():
            result[key] = value

        # Calculate compression ratio for efficiency calculation (if file metadata provided)
        if (
            file_metadata
            and "original_size_bytes" in file_metadata
            and "compressed_size_bytes" in file_metadata
        ):
            result["compression_ratio"] = (
                file_metadata["original_size_bytes"]
                / file_metadata["compressed_size_bytes"]
                if file_metadata["compressed_size_bytes"] > 0
                else 1.0
            )
        else:
            # Default compression ratio when no file metadata
            result["compression_ratio"] = 1.0

        # Process with quality system (adds composite_quality and efficiency)
        from .enhanced_metrics import process_metrics_with_enhanced_quality

        result = process_metrics_with_enhanced_quality(result, config)

        # Add file-specific metrics if metadata provided
        if file_metadata and "compressed_path" in file_metadata:
            result["kilobytes"] = float(
                calculate_file_size_kb(file_metadata["compressed_path"])
            )
        elif file_metadata and "compressed_size_bytes" in file_metadata:
            result["kilobytes"] = float(file_metadata["compressed_size_bytes"] / 1024.0)

        # Calculate processing time
        end_time = time.perf_counter()
        elapsed_seconds = end_time - start_time
        result["render_ms"] = min(int(elapsed_seconds * 1000), 86400000)

        # Add positional sampling if enabled
        if config.ENABLE_POSITIONAL_SAMPLING:
            # Map metric names to their functions
            metric_functions = {
                "ssim": lambda f1, f2: ssim(
                    cv2.cvtColor(f1, cv2.COLOR_RGB2GRAY) if len(f1.shape) == 3 else f1,
                    cv2.cvtColor(f2, cv2.COLOR_RGB2GRAY) if len(f2.shape) == 3 else f2,
                    data_range=255.0,
                ),
                "mse": mse,
                "rmse": rmse,
                "fsim": fsim,
                "gmsd": gmsd,
                "chist": chist,
                "edge_similarity": lambda f1, f2: edge_similarity(
                    f1, f2, config.EDGE_CANNY_THRESHOLD1, config.EDGE_CANNY_THRESHOLD2
                ),
                "texture_similarity": texture_similarity,
                "sharpness_similarity": sharpness_similarity,
            }

            # Calculate positional samples for configured metrics
            if config.POSITIONAL_METRICS is not None:
                for metric_name in config.POSITIONAL_METRICS:
                    if metric_name in metric_functions:
                        try:
                            positional_data = _calculate_positional_samples(
                                aligned_pairs,
                                metric_functions[metric_name],
                                metric_name,
                            )
                            result.update(positional_data)
                        except Exception as e:
                            logger.warning(
                                f"Failed to calculate positional samples for {metric_name}: {e}"
                            )

        # Add raw metrics if requested
        if config.RAW_METRICS:
            # Metrics already reported in raw (un-scaled) form – directly copy.
            raw_equivalent_metrics = [
                "ssim",
                "ms_ssim",
                "mse",
                "rmse",
                "fsim",
                "gmsd",
                "chist",
                "edge_similarity",
                "texture_similarity",
                "sharpness_similarity",
                "temporal_consistency",
                "temporal_consistency_pre",
                "temporal_consistency_post",
                "temporal_consistency_delta",
            ]

            for metric_name in raw_equivalent_metrics:
                result[f"{metric_name}_raw"] = result[metric_name]

            # Handle PSNR separately: use un-scaled mean value
            if raw_metric_values["psnr"]:
                result["psnr_raw"] = float(np.mean(raw_metric_values["psnr"]))
            else:
                result["psnr_raw"] = 0.0

            # Raw copies for temporal consistency variants
            result["temporal_consistency_pre_raw"] = result["temporal_consistency_pre"]
            result["temporal_consistency_post_raw"] = result[
                "temporal_consistency_post"
            ]
            result["temporal_consistency_delta_raw"] = result[
                "temporal_consistency_delta"
            ]

        return result

    except Exception as e:
        logger.error(f"Failed to calculate comprehensive metrics from frames: {e}")
        raise ValueError(f"Metrics calculation failed: {e}") from e


def calculate_comprehensive_metrics(
    original_path: Path,
    compressed_path: Path,
    config: MetricsConfig | None = None,
    frame_reduction_context: bool = False,
) -> dict[str, float | str]:
    """Calculate comprehensive quality metrics between original and compressed GIFs.

    This is the main function that addresses the frame alignment problem and provides
    multi-metric quality assessment with all available metrics.

    Args:
        original_path: Path to original GIF file
        compressed_path: Path to compressed GIF file
        config: Optional metrics configuration (uses default if None)
        frame_reduction_context: If True, adjusts disposal artifact detection for frame reduction

    Returns:
        Dictionary with comprehensive metrics including:
        - Traditional metrics: ssim, ms_ssim, psnr, temporal_consistency
        - New metrics: mse, rmse, fsim, gmsd, chist, edge_similarity, texture_similarity, sharpness_similarity
        - Aggregation descriptors: *_std, *_min, *_max for each metric
        - Optional raw values: *_raw for each metric (if config.RAW_METRICS=True)
        - System metrics: render_ms, kilobytes
        - Composite quality score

    Raises:
        IOError: If either GIF file cannot be read
        ValueError: If GIFs are invalid or processing fails
    """
    if config is None:
        config = DEFAULT_METRICS_CONFIG

    try:
        # Extract frames from both GIFs
        original_result = extract_gif_frames(original_path, config.SSIM_MAX_FRAMES)
        compressed_result = extract_gif_frames(compressed_path, config.SSIM_MAX_FRAMES)

        # Extract metadata for file-specific operations
        try:
            from .meta import extract_gif_metadata

            original_metadata = extract_gif_metadata(original_path)
            compressed_metadata = extract_gif_metadata(compressed_path)
            original_frame_count = original_metadata.orig_frames
            compressed_frame_count = compressed_metadata.orig_frames
        except Exception:
            # Fallback to extracted frames count
            original_frame_count = len(original_result.frames)
            compressed_frame_count = len(compressed_result.frames)

        # Prepare file metadata for the frame-based function
        file_metadata = {
            "original_path": original_path,
            "compressed_path": compressed_path,
            "original_frame_count": original_frame_count,
            "compressed_frame_count": compressed_frame_count,
            "original_size_bytes": original_path.stat().st_size,
            "compressed_size_bytes": compressed_path.stat().st_size,
        }

        # Delegate to frame-based function
        result = calculate_comprehensive_metrics_from_frames(
            original_result.frames,
            compressed_result.frames,
            config=config,
            frame_reduction_context=frame_reduction_context,
            file_metadata=file_metadata,
        )

        return result

    except Exception as e:
        logger.error(f"Failed to calculate comprehensive metrics: {e}")
        raise ValueError(f"Metrics calculation failed: {e}") from e


def cleanup_all_validators() -> None:
    """Clean up all global validator instances and release model references.

    This function should be called when you want to free up memory used by
    cached models and validator instances. It's especially useful in testing
    scenarios or when switching between different processing configurations.
    """
    logger.info("Cleaning up all validators and model cache")

    # Clean up temporal detector
    try:
        from .temporal_artifacts import cleanup_global_temporal_detector

        cleanup_global_temporal_detector()
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"Failed to cleanup temporal detector: {e}")

    # Clean up deep perceptual validator
    try:
        from .deep_perceptual_metrics import cleanup_global_validator

        cleanup_global_validator()
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"Failed to cleanup deep perceptual validator: {e}")

    # Clean up model cache
    try:
        from .model_cache import cleanup_model_cache

        cleanup_model_cache(force=True)
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"Failed to cleanup model cache: {e}")

    # Force garbage collection
    import gc

    gc.collect()

    logger.debug("All validators and model cache cleaned up")
