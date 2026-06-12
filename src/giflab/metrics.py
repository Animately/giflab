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
    # Whether the source GIF carried per-pixel transparency. Defaults False so
    # the 7+ non-metrics callers of ``extract_gif_frames`` simply ignore it.
    # The dual-composite path in ``calculate_comprehensive_metrics`` reads it to
    # decide whether a black-background second pass is warranted (opaque GIFs
    # short-circuit to a single white pass at zero added cost).
    has_alpha: bool = False


def _frame_to_rgb_composited(
    img: Image.Image, background: tuple[int, int, int]
) -> np.ndarray:
    """Convert a PIL frame to RGB by compositing transparent pixels onto a fixed background.

    Audit-fix: ``Image.convert('RGB')`` on a palette+transparency GIF resolves
    transparent pixels through the file's declared background palette index.
    That palette entry's COLOUR is unstable across re-encoding — animately
    (and most compressors) rearrange the palette during recompression, so the
    same transparent region resolves to a different RGB value in the original
    vs. the compressed file. Every pixel-comparison metric (ssim, ms_ssim,
    chist, lpips, …) then treats that as real content disagreement and
    collapses, even on essentially lossless compressions.

    Fix: always go via RGBA, then alpha-blend onto a fixed RGB background.
    The result is invariant to palette ordering and to PIL's background-index
    interpretation. White is the conventional choice for email/marketing
    content (`MetricsConfig.ALPHA_BACKGROUND`).

    See: `audit-fix/extract-gif-frames-alpha-compositing-bug` and the
    `TestExtractGifFramesAlphaCompositing` regression test.
    """
    rgba = img.convert("RGBA")
    bg = Image.new("RGBA", rgba.size, (*background, 255))
    composited = Image.alpha_composite(bg, rgba).convert("RGB")
    return np.array(composited)


def _frame_has_transparency(img: Image.Image) -> bool:
    """Return True if *img* carries per-pixel transparency.

    Detection mirrors what ``_frame_to_rgb_composited`` would composite away:
    a palette GIF with a ``transparency`` index, or any mode with a real alpha
    channel whose minimum alpha is below fully-opaque. This must be computed on
    the SOURCE frame BEFORE compositing — once composited to RGB the alpha is
    gone, which is exactly why the warm-cache path has to persist the flag (it
    can never be recomputed from the cached RGB frames).
    """
    if "transparency" in img.info:
        return True
    if img.mode in ("RGBA", "LA", "PA"):
        rgba = img.convert("RGBA")
        alpha = rgba.getchannel("A")
        # getextrema returns (min, max); any pixel below 255 means transparency.
        return bool(alpha.getextrema()[0] < 255)
    if img.mode == "P":
        # Palette image without a declared transparency index has no alpha.
        return False
    return False


def extract_gif_frames(
    gif_path: Path,
    max_frames: int | None = None,
    alpha_background: tuple[int, int, int] | None = None,
) -> FrameExtractResult:
    """Extract frames from a GIF file.

    Args:
        gif_path: Path to GIF file
        max_frames: Maximum number of frames to extract (None for all)
        alpha_background: RGB background used to composite transparent pixels.
            ``None`` (the default) uses ``DEFAULT_METRICS_CONFIG.ALPHA_BACKGROUND``
            (white) — preserving the single-white behaviour every existing
            caller relies on. The dual-composite path passes an explicit
            ``(0, 0, 0)`` for the black second pass; a diagnostic override hatch
            (e.g. ``(128, 128, 128)``) is available for one-off debugging.

    Returns:
        FrameExtractResult with extracted frames and metadata. ``has_alpha`` is
        set True when the source GIF carried transparency (so the file-level
        dual-composite path knows a black second pass is warranted).

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

    # Audit-fix: composite RGBA onto a fixed background before returning RGB
    # frames. PIL's `.convert('RGB')` on palette+transparency GIFs resolves
    # transparent pixels via the file's background palette colour, which is
    # unstable across re-encoding (palette gets reordered). White is the
    # conventional choice for email/marketing content. See
    # _frame_to_rgb_composited for the full rationale. ``alpha_background``
    # lets the file-level dual-composite path request the black second pass.
    if alpha_background is None:
        background = DEFAULT_METRICS_CONFIG.ALPHA_BACKGROUND
    else:
        background = alpha_background

    # Try to get from cache first (only if caching is enabled). The background
    # is folded into the cache key for non-white passes, so the white and black
    # passes never collide.
    if use_cache and get_frame_cache is not None:
        frame_cache = get_frame_cache()
        cached = frame_cache.get(gif_path, max_frames, alpha_background=background)

        if cached is not None:
            frames, frame_count, dimensions, duration_ms, has_alpha = cached
            return FrameExtractResult(
                frames=frames,
                frame_count=frame_count,
                dimensions=dimensions,
                duration_ms=duration_ms,
                has_alpha=has_alpha,
            )

    # Not in cache, extract frames
    try:
        with Image.open(gif_path) as img:
            if not hasattr(img, "n_frames") or img.n_frames == 1:
                # Single frame image (PNG, JPEG, etc.) or single-frame GIF.
                # Detect transparency BEFORE compositing (alpha is gone after).
                has_alpha = _frame_has_transparency(img)
                frame = _frame_to_rgb_composited(img, background)
                result = FrameExtractResult(
                    frames=[frame],
                    frame_count=1,
                    dimensions=(img.width, img.height),
                    duration_ms=0,
                    has_alpha=has_alpha,
                )

                # Cache the result (only if caching is enabled)
                if use_cache:
                    frame_cache.put(
                        gif_path,
                        result.frames,
                        result.frame_count,
                        result.dimensions,
                        result.duration_ms,
                        alpha_background=background,
                        has_alpha=result.has_alpha,
                    )

                return result

            total_frames = img.n_frames

            # Additional memory check based on image dimensions
            width, height = img.size
            frame_indices = _compute_frame_indices(
                total_frames, width, height, max_frames
            )

            frames = []
            total_duration = 0
            has_alpha = False

            for i in frame_indices:
                img.seek(i)
                # Track transparency across the sampled frames. A GIF whose
                # transparency appears only on a later frame still warrants the
                # black dual-composite pass, so OR across the walk. Free: the
                # frame is already decoded here for compositing.
                if not has_alpha:
                    has_alpha = _frame_has_transparency(img)
                frame = _frame_to_rgb_composited(img, background)
                frames.append(frame)

                # Get frame duration
                duration = img.info.get("duration", 100)  # Default 100ms
                total_duration += duration

            result = FrameExtractResult(
                frames=frames,
                frame_count=total_frames,  # Store the actual total frame count
                dimensions=(img.width, img.height),
                duration_ms=total_duration,
                has_alpha=has_alpha,
            )

            # Cache the result (only if caching is enabled)
            if use_cache:
                frame_cache.put(
                    gif_path,
                    result.frames,
                    result.frame_count,
                    result.dimensions,
                    result.duration_ms,
                    alpha_background=background,
                    has_alpha=result.has_alpha,
                )

            return result

    except Exception as e:
        raise OSError(f"Failed to extract frames from {gif_path}: {e}") from e


def _compute_frame_indices(
    total_frames: int,
    width: int,
    height: int,
    max_frames: int | None,
) -> list[int]:
    """Compute the set of frame indices to sample from an animation.

    Single-sources the sampling logic so the standard and any future fast
    path stay in lockstep. Applies the same two memory guards as before:
    a hard 500-frame cap and a ~500MB RGB-frame-buffer cap, then samples
    evenly across the whole animation (so quality issues that appear only
    late are still captured).
    """
    # Memory protection: limit frame extraction for very large GIFs
    memory_limit_frames = 500  # Reasonable limit to prevent memory issues
    if max_frames is None:
        frames_to_extract = min(total_frames, memory_limit_frames)
    else:
        frames_to_extract = min(total_frames, max_frames, memory_limit_frames)

    pixels_per_frame = width * height * 3  # RGB
    estimated_memory_mb = (frames_to_extract * pixels_per_frame) / (1024 * 1024)

    # Limit memory usage to ~500MB for frame extraction
    if estimated_memory_mb > 500:
        max_safe_frames = int(500 * 1024 * 1024 / pixels_per_frame)
        frames_to_extract = min(frames_to_extract, max(1, max_safe_frames))

    # Use even frame sampling across entire animation for better quality assessment
    if frames_to_extract >= total_frames:
        # Use all frames if we're not hitting the limit
        return list(range(total_frames))
    # Sample evenly across the entire animation to capture quality issues
    # that may appear later in the animation
    return [int(i) for i in np.linspace(0, total_frames - 1, frames_to_extract, dtype=int)]


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

    Note:
        Investigated in the 2026-05-22 metrics audit as a possible source of
        the smooth_gradient "bump-up" at strong animately lossy levels. Ruled
        out: the only clamp is the MSE==0 perfect-match cap at 100 dB, which
        does not activate for lossy compression. The bump-up is an
        animately-side saturation artefact — see comment on the SSIM clamp
        site in this file and scripts/audit/sanity.py:LOSSY_LEVELS.
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

    Note:
        Investigated in the 2026-05-22 metrics audit as a possible source of
        the smooth_gradient "bump-up" at strong animately lossy levels. Ruled
        out: this function has no clamping or NaN-replacement that could mask
        a monotonic ordering. The bump-up is an animately-side saturation
        artefact (its --lossy parameter caps around level ~125 on
        low-complexity gradient content, producing distinct but bounded
        outputs whose local-window similarity differs by ~0.005). See
        comment on the SSIM clamp site in this file and
        scripts/audit/sanity.py:LOSSY_LEVELS.
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
            # If SSIM at this scale can't be computed, record NaN (not 0.0).
            # np.average below is NaN-aware so a single failed scale doesn't
            # drag the whole MS-SSIM toward zero; a 0.0 here would be a
            # fabricated "perfectly dissimilar" scale weight.
            logger.warning(f"SSIM calculation failed at scale {scale}: {e}")
            ssim_values.append(float("nan"))

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

    # Weighted average of SSIM values across scales.
    #
    # NaN-aware: a scale whose SSIM couldn't be computed is NaN; drop those
    # scales (and their weights) before averaging rather than letting NaN
    # propagate through the whole MS-SSIM. If *every* scale is NaN, the
    # metric genuinely couldn't be computed -> NaN, not 0.0.
    if ssim_values:
        scores = np.array(ssim_values, dtype=float)
        weight_list = [0.4, 0.25, 0.15, 0.1, 0.1][: len(ssim_values)]
        weights = np.array(weight_list)

        valid = ~np.isnan(scores)
        if not bool(np.any(valid)):
            return float("nan")
        scores = scores[valid]
        weights = weights[valid]

        # Protect against division by zero in weight normalization
        weights_sum = np.sum(weights)
        if weights_sum > 0:
            weights = weights / weights_sum  # Normalize weights
            return float(np.average(scores, weights=weights))
        else:
            # If all weights are zero, use uniform weighting
            return float(np.mean(scores))
    else:
        # No scales computed at all — not measured.
        return float("nan")


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

# Flat-content fallback constants — control smooth degradation for structure-based
# metrics (fsim, gmsd, edge_similarity, sharpness_similarity) on solid-colour frames.
#
# Structure-based metrics return meaningless values on flat content (no edges, no
# gradients, no phase congruency) because every derivative is zero and stability
# constants in the metric formulas make zero/zero resolve to 1.0 or 0.0 regardless
# of whether the two frames are the same colour.
#
# Fix strategy: detect flat content and replace the metric with an honest
# smooth function of the L2 colour distance between the two frames' means:
#
#   score = identity_value * (1 - blend) + worst_value * blend
#   blend = clamp((L2 - FLAT_IDENTITY_FLOOR) / FLAT_DEGRADATION_SCALE, 0, 1)
#
# Constants:
# - FLAT_STD_THRESHOLD: per-channel std (uint8 units) below which a frame is
#   considered flat. 1.0 = "less than one DN of variation", below the noise
#   floor of any real-world image.
# - FLAT_IDENTITY_FLOOR: L2 below which the two flat frames are "the same colour"
#   and score exactly identity_value. 1.0 tolerates sub-DN floating-point
#   round-trips through PIL/cv2 (e.g. RGB→YUV→RGB introduces <1 DN drift).
# - FLAT_DEGRADATION_SCALE: blend saturates to worst_value at
#   L2 = FLAT_IDENTITY_FLOOR + FLAT_DEGRADATION_SCALE. 50.0 corresponds to
#   approximately 29 DN per channel (well past JND for solid-colour patches;
#   natural compression drift on flat regions is <5 DN).
#
# See docs/metrics-audit/2026-05-22/report.md and task notes
# giflab-fsim-flat-content-returns-1 + giflab-flat-mean-tol-recalibration.
FLAT_STD_THRESHOLD: float = 1.0
FLAT_IDENTITY_FLOOR: float = 1.0
FLAT_DEGRADATION_SCALE: float = 50.0

# Threshold (in uint8 DN) for the ptp() fast-reject in ``_is_flat_frame``.
# Derived from ``FLAT_STD_THRESHOLD`` so the two stay in lock-step: changing
# the std threshold automatically widens or narrows the ptp gate.
#
# Derivation: for a two-value uint8 distribution over N pixels (k pixels at the
# high value and N-k at the low value, ptp = h-l), the variance is
# ``(k/N)*(1 - k/N)*ptp²``. The maximum std for a given ptp is ``ptp/2`` (at
# k = N/2). So fast-rejecting at ``ptp > 2 * FLAT_STD_THRESHOLD`` would only
# be tight at the 50/50 split — for sparser clusters the divergence widens.
# Empirically, ``4 * FLAT_STD_THRESHOLD`` is the smallest gate that catches
# all real-content frames (any visible edge has ptp ≫ 4) while keeping the
# false-reject zone confined to extreme sparse-outlier pathologies (see
# docstring for the precise divergence math).
_PTP_FAST_REJECT_THRESHOLD = int(round(FLAT_STD_THRESHOLD * 4))

# Grayscale peak-to-peak ceiling below which a frame is "near flat" for the
# texture_similarity near-flat guard. Derived from _PTP_FAST_REJECT_THRESHOLD
# (= 4 DN at FLAT_STD_THRESHOLD = 1.0) so it stays in lock-step with the flat
# constants — no independent free tunable.
#
# Why texture_similarity needs this and the four sibling structure metrics do
# not: LBP encodes LOCAL ORDER relations, so it is intensity-INVERSION
# invariant. A frame with a tiny gradient and its 255-complement (a
# catastrophically different colour pair) produce nearly identical uniform-LBP
# histograms and the raw corrcoef path scores ~0.9995 — even though
# ``_is_flat_frame`` correctly reports the pair as non-flat (the gradient
# carries real per-channel std > FLAT_STD_THRESHOLD). The fsim/gmsd/edge/
# sharpness metrics degrade naturally on such a gradient, so they need no
# near-flat guard; only the inversion-invariant LBP metric does.
#
# Derivation: 8 × _PTP_FAST_REJECT_THRESHOLD = 32 DN. This sits well above the
# residual near-flat escape zone (gray_ptp 12–24 for sub-DN-band gradients)
# and far below any frame with visible structure (a single 30-DN palette step
# already exceeds it; real content measures gray_ptp ≥ ~175). The near-flat
# blend weight is a CONTINUOUS function of gray_ptp (no cliff): it reaches
# full strength only as both frames approach perfect flatness and is exactly
# zero for any textured frame.
_TEXTURE_NEAR_FLAT_PTP_CEILING = _PTP_FAST_REJECT_THRESHOLD * 8


def _flat_colour_degradation(
    frame1: np.ndarray,
    frame2: np.ndarray,
    *,
    identity_value: float,
    worst_value: float,
) -> float:
    """Smooth degradation from ``identity_value`` to ``worst_value`` by colour L2.

    Shared by ``_flat_content_fallback`` (both-flat branch) and the
    ``texture_similarity`` near-flat guard so the two regimes degrade on the
    exact same curve. No hard threshold cliff:

        blend = clamp((L2 - FLAT_IDENTITY_FLOOR) / FLAT_DEGRADATION_SCALE, 0, 1)
        score = identity_value * (1 - blend) + worst_value * blend

    L2 below ``FLAT_IDENTITY_FLOOR`` returns exactly ``identity_value`` (sub-DN
    round-trip drift is perceptually invisible and must not be penalised).
    """
    L2 = _flat_mean_distance(frame1, frame2)
    if L2 < FLAT_IDENTITY_FLOOR:
        return identity_value
    blend = min(1.0, (L2 - FLAT_IDENTITY_FLOOR) / FLAT_DEGRADATION_SCALE)
    return identity_value * (1.0 - blend) + worst_value * blend


def _is_flat_frame(frame: np.ndarray, threshold: float = FLAT_STD_THRESHOLD) -> bool:
    """Return True if every channel of ``frame`` has std below ``threshold``.

    Used by structure-based metrics to detect inputs they cannot meaningfully
    analyse (no edges, no gradients, no phase congruency).

    Performance note:
        The function uses a two-phase check to minimise per-call cost on the
        common case (non-flat frames with real content):

        1. **ptp() fast-reject** — ``np.ptp()`` (peak-to-peak = max − min) on
           uint8 data is a single O(n) scan without a dtype cast.  If any
           channel's ptp exceeds ``_PTP_FAST_REJECT_THRESHOLD`` (= 4 DN at
           ``FLAT_STD_THRESHOLD = 1.0``) the frame is rejected as non-flat
           without allocating the float32 copy or computing std.

        2. **Accurate std fallback** — only reached for frames within the
           ptp threshold on all channels.  These are either genuinely flat or
           a hair off — the full float-cast + std check resolves them
           correctly.

        On a 100-frame 480×480 GIF this avoids ~3.8 s of overhead:
        ``astype(float32).std`` costs ~9.7 ms/frame × 2 calls per metric ×
        4 affected metrics.  The ptp() path costs ~0.2 ms/frame instead.

        ``np.ptp(arr)`` (standalone function) is used instead of
        ``arr.ptp()`` (instance method) because the instance method was
        removed in NumPy 2.0 — the standalone form works in both 1.x and 2.x.

    Behavioural divergence from the pre-optimisation path:
        For a 480×480 (N = 230,400 pixels) frame at ``ptp = 5`` (the smallest
        value that triggers the fast-reject), the unoptimised std-based check
        returns ``True`` (flat) for any high-value cluster size up to
        k ≈ 9,600 pixels (≈ 4.17% of the frame), because the two-value
        variance ``(k/N)*(1 - k/N)*25`` only crosses ``threshold² = 1.0`` at
        k ≈ 9,600.  The fast-reject path rejects all such inputs as non-flat.

        This 0–4% sparse-cluster zone is the entire behavioural divergence.
        Beyond k ≈ 9,600 both paths agree (std ≥ 1.0 → non-flat); at
        ``ptp ≤ 4`` both paths fall through to the accurate check.

        Real GIF flat frames have ``ptp = 0`` (solid colour), so the divergence
        only affects synthetic pathologies — never real content.  See task
        note ``giflab-is-flat-frame-perf-optimisation`` for the full rationale.
    """
    if frame.ndim == 3:
        # Fast-reject: per-channel ptp() on uint8 slices — dtype-native O(n) scan.
        # Slicing each channel (frame[..., c]) avoids the reshape + axis reduce and
        # is ~20× faster than flat.ptp(axis=0) on a 480×480 frame.
        # ``np.ptp(...)`` (not ``arr.ptp()``) for NumPy 2.x forward-compat.
        n_ch = frame.shape[-1]
        for c in range(n_ch):
            if np.ptp(frame[..., c]) > _PTP_FAST_REJECT_THRESHOLD:
                return False
    else:
        if np.ptp(frame) > _PTP_FAST_REJECT_THRESHOLD:
            return False
    # Accurate fallback for frames within the ptp threshold on all channels.
    arr = frame.astype(np.float32)
    if arr.ndim == 2:
        return float(np.std(arr)) < threshold
    # Per-channel std for RGB; treat as flat only if every channel is flat.
    return bool(np.all(np.std(arr.reshape(-1, arr.shape[-1]), axis=0) < threshold))


def _flat_mean_distance(frame1: np.ndarray, frame2: np.ndarray) -> float:
    """L2 distance between per-channel mean colours of two frames (uint8 units).

    Used to decide how different two flat frames are — feeds the smooth
    degradation formula in ``_flat_content_fallback``.
    """
    m1 = np.mean(
        frame1.astype(np.float32).reshape(
            -1, frame1.shape[-1] if frame1.ndim == 3 else 1
        ),
        axis=0,
    )
    m2 = np.mean(
        frame2.astype(np.float32).reshape(
            -1, frame2.shape[-1] if frame2.ndim == 3 else 1
        ),
        axis=0,
    )
    return float(np.linalg.norm(m1 - m2))


def _flat_content_fallback(
    frame1: np.ndarray,
    frame2: np.ndarray,
    *,
    identity_value: float,
    worst_value: float,
) -> float | None:
    """Return an honest fallback value for structure-based metrics on flat content.

    Uses smooth degradation from ``identity_value`` to ``worst_value`` as the
    L2 colour distance between the two frames grows — no hard threshold cliff.

    Returns:
        - ``identity_value`` if both frames are flat and L2 < FLAT_IDENTITY_FLOOR.
          Sub-DN compression drift (e.g. PIL/cv2 round-trips, lossless resampling)
          is perceptually invisible and must NOT penalise quality scores.
        - A smoothly blended value between ``identity_value`` and ``worst_value``
          if both frames are flat and L2 is between FLAT_IDENTITY_FLOOR and
          FLAT_IDENTITY_FLOOR + FLAT_DEGRADATION_SCALE. The blend is linear in L2.
        - ``worst_value`` if both frames are flat and L2 >= FLAT_IDENTITY_FLOOR +
          FLAT_DEGRADATION_SCALE (colour difference is at or past perceptual JND
          for solid-colour patches), OR if exactly one frame is flat (unambiguous
          structural mismatch).
        - ``None`` if neither frame is flat — caller falls through to the existing
          structure-based computation.

    Formula (both-flat branch):
        blend = clamp((L2 - FLAT_IDENTITY_FLOOR) / FLAT_DEGRADATION_SCALE, 0, 1)
        score = identity_value * (1 - blend) + worst_value * blend
    """
    flat1 = _is_flat_frame(frame1)
    flat2 = _is_flat_frame(frame2)

    if not flat1 and not flat2:
        # Neither flat — fall through to structure-based metric.
        return None

    if flat1 and flat2:
        return _flat_colour_degradation(
            frame1, frame2, identity_value=identity_value, worst_value=worst_value
        )

    # Exactly one side flat → unambiguous structural mismatch.
    return worst_value


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

    Flat-content fallback: gradient + phase-congruency both go to zero on
    solid-colour frames, so the stability constants would make the metric
    silently report 1.0 regardless of pixel colour. ``_flat_content_fallback``
    short-circuits with honest smooth-degradation values in that regime —
    see docs/metrics-audit/2026-05-22/report.md and task note
    giflab-flat-mean-tol-recalibration.
    """
    f1, f2 = _resize_if_needed(frame1, frame2)

    fallback = _flat_content_fallback(f1, f2, identity_value=1.0, worst_value=0.0)
    if fallback is not None:
        return fallback

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
    """Gradient Magnitude Similarity Deviation (lower is better).

    Flat-content fallback: gradient magnitudes are zero on solid-colour
    frames, so the stability constant would silently yield 0.0 (no
    distortion) for any flat pair regardless of colour. We instead use
    smooth degradation from 0.0 to 0.5 as the colour distance grows —
    0.5 is in the same band as gmsd on heavily distorted real content
    (typically 0.1–0.4). See docs/metrics-audit/2026-05-22/report.md and
    task note giflab-flat-mean-tol-recalibration.
    """
    f1, f2 = _resize_if_needed(frame1, frame2)

    fallback = _flat_content_fallback(f1, f2, identity_value=0.0, worst_value=0.5)
    if fallback is not None:
        return fallback

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
    """Colour-Histogram Correlation (0-1, higher is better).

    Per-channel marginal histogram correlation: 32-bin histograms are built
    independently for R, G, B; Pearson correlation between matched channels
    is averaged, then mapped from [-1, 1] to [0, 1].

    Invariances (intentional, documented after the 2026-05-22 metrics audit):
        1. **Spatial invariance.** Only marginal pixel-value distributions
           are compared; spatial arrangement is invisible. A spatially
           scrambled copy of the input scores ~1.0 against the original.
        2. **Channel independence.** Per-channel correlation is averaged;
           joint-channel structure (e.g. R/G/B covariance, hue rotations
           that preserve per-channel marginals) is invisible.
        3. **Bin coarseness.** With 32 bins, each bin covers 8 intensity
           levels. Small intra-bin shifts are invisible; large degradations
           that collapse pixel values toward already-populous bins (e.g.
           extreme palette quantization, extreme blur) can *rebound* the
           correlation because bin overlap rises again. This makes chist
           non-monotonic across degradation strength on smooth/photographic
           content — see the audit report under
           `docs/metrics-audit/2026-05-22/report.md` ("chist" section).

    Implication: chist is a colour-fidelity signal, not a holistic quality
    signal. It is appropriately weighted at ~4% in `composite_quality`
    (see `config.ENHANCED_CHIST_WEIGHT`). Do not use it as a sole quality
    discriminator; pair with a spatially-aware metric (ssim, fsim, gmsd,
    lpips) for any ranking decision.
    """
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
    """Edge-Map Jaccard similarity (0-1, higher is better; NaN if undefined).

    Flat-content fallback: Canny finds no edges on solid-colour frames, so
    a naive ``union == 0`` branch would silently treat any flat pair as
    perfect. ``_flat_content_fallback`` returns smooth-degradation values:
    identity (1.0) for same-colour flats, blending to worst-case (0.0) for
    different-colour flats. See docs/metrics-audit/2026-05-22/report.md and
    task note giflab-flat-mean-tol-recalibration.

    Undefined-on-edgeless (NaN): for NON-flat content where Canny still finds
    no edges in EITHER frame (e.g. smooth gradients with no hard transitions),
    edge similarity is *undefined* — there are no edges to compare — so this
    returns ``float("nan")``, NOT a fabricated-perfect 1.0. The old 1.0 guard
    injected fake-perfect outliers on smooth-gradient content that rebounded
    the ``nanmedian`` aggregate upward at high lossy and pulled
    ``composite_quality`` non-monotonic (2026-06-03 audit; see
    docs/metrics-audit/2026-06-03/post-fix-verdict.md). NaN is dropped by the
    ``_MEDIAN_AGGREGATED_METRICS`` nanmedian aggregation, so edgeless frames no
    longer inflate the score; if every frame is edgeless the metric aggregates
    to NaN and ``composite_quality`` redistributes its weight. Note that
    compression which *introduces* banding edges absent from an edgeless
    original gives ``union > 0, intersection == 0`` → 0.0 (artifact, penalised),
    which is the genuine signal and is unaffected.

    Args:
        frame1: First frame (RGB or grayscale)
        frame2: Second frame (RGB or grayscale)
        threshold1: Lower Canny threshold
        threshold2: Upper Canny threshold
    """
    f1, f2 = _resize_if_needed(frame1, frame2)

    fallback = _flat_content_fallback(f1, f2, identity_value=1.0, worst_value=0.0)
    if fallback is not None:
        return fallback

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
        # No edges in either non-flat frame → undefined, not fabricated-perfect.
        return float("nan")
    return float(intersection / union)


def texture_similarity(frame1: np.ndarray, frame2: np.ndarray) -> float:
    """Texture-Histogram correlation using uniform LBP (0-1, higher is better).

    Compares two frames via a Local Binary Pattern histogram correlation.
    LBP (the local binary pattern) is a TRUE PAIR-COMPARISON metric, but it has
    known mathematical INVARIANCES that make some "obviously different" pairs
    report near 1.0:

    - Intensity inversion: a frame and its inverted-intensity counterpart
      produce nearly identical uniform-LBP histograms. White↔black
      pathological pairs from the audit corpus reported
      ``texture_similarity ≈ 0.9996`` for this reason.
    - Spatially-uniform regions: both frames being solid colours (any
      colours) produce identical near-degenerate histograms, so the raw
      corrcoef path reported ~1.0 regardless of colour.
    - Monotonic intensity transforms more generally: LBP encodes local
      ORDER relations between pixels, not absolute intensities, so any
      monotone intensity remapping preserves the pattern.

    Despite these invariances the metric is NOT single-stream — both frames
    are required and the LBP statistics are compared.

    COLOUR-BLIND — NOT a standalone perceptual-quality proxy. LBP is computed
    on a GREYSCALE conversion of each frame (``cv2.cvtColor(..., RGB2GRAY)``
    below), so it measures only the coarse spatial micro-texture pattern and
    is insensitive to colour. A pure colour-quantisation failure — where the
    palette is reorganised but the luminance pattern survives — leaves the LBP
    histogram almost unchanged, so ``texture_similarity`` stays near 1.0 even
    when the image "looks" badly degraded. The 2026-05-26 outlier deep-dive
    saw exactly this on two real GIFs (``8e172835`` Christmas stocking,
    ``XpkfMAWfUsWg`` gstatic data-viz): ``texture_similarity ≈ 0.999`` while
    ssim fell to 0.11-0.23, deltae rose to 25-66 and ssimulacra2 hit 0. Use
    this metric to detect grain removal / texture smoothing, NOT to judge
    whether colour fidelity survived; always pair it with a colour- and
    luminance-sensitive metric (ssim, deltae, ssimulacra2, lpips) for any
    quality decision. It contributes a small ``ENHANCED_TEXTURE_WEIGHT`` share
    of ``composite_quality`` precisely so that a surviving texture score
    cannot offset a catastrophic colour failure — see
    docs/metrics-audit/outlier-deep-dive-2026-05-26.md and task note
    giflab-texture-similarity-composite-weight-review.

    Flat- and near-flat-content handling (audit-fix, this metric was the last
    structure-based pair metric missing it; fsim/gmsd/edge_similarity/
    sharpness_similarity converted earlier):

    - Both frames solid colour → ``_flat_content_fallback`` returns honest
      smooth degradation by colour L2 (identity 1.0 for same-colour flats,
      blending to worst 0.0 for different-colour flats such as white-vs-black).
      This replaces the old ``np.std == 0 → return 1.0`` cliff that silently
      treated every solid pair as identity.
    - One flat, one textured → ``_flat_content_fallback`` returns worst (0.0).
    - NEAR-flat residual escape — a frame with a tiny gradient (per-channel
      std just above ``FLAT_STD_THRESHOLD``) passes the strict flat test yet
      still carries no real structure. Because LBP is intensity-inversion
      invariant, such a near-flat ramp and its 255-complement still collide at
      ~0.9995. We blend the LBP score toward the colour-distance degradation
      with a CONTINUOUS weight driven by the grayscale peak-to-peak (no cliff;
      see ``_TEXTURE_NEAR_FLAT_PTP_CEILING``): the weight reaches full strength
      only as both frames approach perfect flatness and is exactly zero for any
      frame with visible structure, so real content is scored by the unchanged
      LBP path.

    See docs/metrics-audit/2026-05-22/report.md and task note
    giflab-texture-similarity-flat-content-aggregation-and-cliff for context.
    """
    f1, f2 = _resize_if_needed(frame1, frame2)

    # Solid-colour pairs (both flat, or exactly one flat) get honest
    # smooth-degradation values — never the LBP-identity pathology.
    fallback = _flat_content_fallback(f1, f2, identity_value=1.0, worst_value=0.0)
    if fallback is not None:
        return fallback

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

    corr = np.corrcoef(hist1, hist2)[0, 1]
    lbp_score = float(np.clip((corr + 1) / 2.0, 0.0, 1.0))

    # Near-flat guard: neither frame is strictly flat (so _flat_content_fallback
    # returned None), but if BOTH grayscale frames are near-flat the LBP score
    # is corrupted by intensity-inversion invariance. Blend continuously toward
    # the colour-distance degradation; the weight is 0 for any textured frame.
    ptp_max = float(max(np.ptp(gray1), np.ptp(gray2)))
    near_flat_weight = max(
        0.0,
        min(
            1.0,
            (_TEXTURE_NEAR_FLAT_PTP_CEILING - ptp_max) / _TEXTURE_NEAR_FLAT_PTP_CEILING,
        ),
    )
    if near_flat_weight <= 0.0:
        return lbp_score

    colour_score = _flat_colour_degradation(f1, f2, identity_value=1.0, worst_value=0.0)
    return (1.0 - near_flat_weight) * lbp_score + near_flat_weight * colour_score


def sharpness_similarity(frame1: np.ndarray, frame2: np.ndarray) -> float:
    """Sharpness similarity based on Laplacian variance ratio (0-1, higher is better).

    Flat-content fallback: the previous ``var1 == 0 and var2 == 0 -> 1.0``
    branch silently treated white-vs-black as "identical sharpness".
    ``_flat_content_fallback`` now distinguishes matching-colour flats
    (identity 1.0 via smooth floor) from differing-colour flats (smooth
    degradation to worst 0.0) and asymmetric flat/non-flat pairs (worst 0.0).
    See docs/metrics-audit/2026-05-22/report.md and task note
    giflab-flat-mean-tol-recalibration.
    """
    f1, f2 = _resize_if_needed(frame1, frame2)

    fallback = _flat_content_fallback(f1, f2, identity_value=1.0, worst_value=0.0)
    if fallback is not None:
        return fallback

    if f1.ndim == 3:
        gray1 = cv2.cvtColor(f1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(f2, cv2.COLOR_RGB2GRAY)
    else:
        gray1, gray2 = f1, f2

    var1 = float(np.var(cv2.Laplacian(gray1, cv2.CV_64F)))
    var2 = float(np.var(cv2.Laplacian(gray2, cv2.CV_64F)))

    # Defensive: with the flat-content fallback above, at most one of the
    # variances can be ~0 here (the other side is non-flat). Preserve the
    # original zero-handling for numerical edge cases.
    if var1 == 0 and var2 == 0:
        return 1.0
    if max(var1, var2) == 0:
        return 0.0

    return float(min(var1, var2) / max(var1, var2))


# --------------------------------------------------------------------------- #


# Metrics whose per-frame distributions are heavy-tailed toward 1.0 on
# flat / sparse-edge content, causing ``np.max`` to be INCONCLUSIVE and
# ``np.mean`` to be non-monotonic under palette reduction. For these metrics,
# ``_aggregate_metric`` uses ``np.median`` as the primary aggregation instead
# of ``np.mean``.
#
# - ``edge_similarity``: smooth-gradient GIFs quantized to few colours develop
#   banding edges in the compressed stream that aren't in the original, so most
#   frame pairs score near-zero Jaccard.  However, when *neither* stream has any
#   detectable edges (both flat colour patches within the Canny window), the
#   ``union == 0`` guard in ``edge_similarity()`` returns 1.0.  A ``mean``
#   aggregation is dragged upward by these 1.0 outliers and produces
#   non-monotonic scores as the palette shrinks further (scores go 1.0 → ~0 →
#   0.02 → 0.04 at [256, 64, 16, 4] colours — audit 2026-05-22, report.md line
#   105–106).  ``np.median`` ignores the outliers and faithfully reflects the
#   typical frame quality (task: giflab-edge-similarity-max-aggregation-sparse-
#   edges, Wave 2 of giflab-rollout-2026-05-26).
# - ``texture_similarity``: LBP-histogram correlation is intensity-inversion
#   invariant, so flat / near-flat frame pairs (including a frame vs its
#   255-complement) collide near 1.0.  Even with the per-frame flat-content
#   fallback, a clip whose frames alternate between flat and textured content
#   produces a heavy 1.0 tail from the flat frames; ``np.mean`` is dragged
#   upward exactly as for edge_similarity, so median is the robust central
#   tendency here too (task: giflab-texture-similarity-flat-content-
#   aggregation-and-cliff).
_MEDIAN_AGGREGATED_METRICS: frozenset[str] = frozenset(
    ["edge_similarity", "texture_similarity"]
)


# Keys produced by the temporal-artifact and gradient-color detectors that are
# computed on the COMPRESSED stream alone (the original frames are accepted but
# never used by the detector). They read like original-vs-compressed pair
# signals but never were one, so they MUST be emitted only under a
# ``_compressed`` suffix — never as a bare key that downstream consumers
# (composite_quality, validation_checker, the storage CSV) would mistake for a
# pair comparison. See CLAUDE.md "Pair-wise over single-stream — and labelled
# honestly". ``flat_region_count`` is produced by BOTH
# ``calculate_enhanced_temporal_metrics`` and ``calculate_gradient_color_metrics``;
# both producer merges must gate on this same tuple. The bare keys were REMOVED
# (Wave 7) — this single source of truth is shared by every assembly path
# (``calculate_comprehensive_metrics_from_frames`` main + high-tier branches and
# ``calculate_selected_metrics``) so the key schema stays identical across paths.
_SINGLE_STREAM_TEMPORAL_KEYS: tuple[str, ...] = (
    "flicker_excess",
    "flicker_frame_ratio",
    "flat_flicker_ratio",
    "flat_region_count",
    "temporal_pumping_score",
    "quality_oscillation_frequency",
    "lpips_t_mean",
    "lpips_t_p95",
    "lpips_t_max",
)


def _merge_single_stream_metrics(
    target: dict[str, Any], producer_metrics: dict[str, Any]
) -> None:
    """Merge a temporal/gradient producer dict into ``target`` in place.

    Re-keys any single-stream key (``_SINGLE_STREAM_TEMPORAL_KEYS``) to its
    ``_compressed`` suffix so a single-stream value can never land in the result
    under a bare pair-shaped name. All other keys pass through unchanged. Numeric
    values are coerced to ``float`` to match the rest of the result schema; the
    rare non-numeric value (e.g. an error string) is passed through verbatim.

    Used by every assembly path so the emitted key schema is identical
    regardless of which path computed the metrics (CLAUDE.md "Same key shape
    across paths").
    """
    for key, value in producer_metrics.items():
        coerced = float(value) if isinstance(value, int | float) else value
        if key in _SINGLE_STREAM_TEMPORAL_KEYS:
            target[f"{key}_compressed"] = coerced
        else:
            target[key] = coerced


def _nan_fallback_dict(keys: list[str]) -> dict[str, float]:
    """Build a "no measurement" fallback dict mapping every key to NaN.

    Single source of truth for the sentinel-free error paths. Per
    CLAUDE.md "NaN over fabricated values", a metric that genuinely could
    not be computed must report ``float("nan")`` so downstream aggregators
    (``np.nanmean`` / composite_quality / validators) skip it honestly
    rather than mistaking a fabricated 0.5 / 50.0 for a real score.
    """
    return {key: float("nan") for key in keys}


# Canonical 5-key SSIMULACRA2 error/disabled-path result. Score keys are NaN
# ("not computed" on the normalised [0, 1] scale); ``_frame_count`` and
# ``_triggered`` are real bookkeeping values, not scores, so they stay 0.0.
# Mirrors Ssimulacra2Validator._nan_result() — both must emit the same shape.
def _default_ssimulacra2_fallback() -> dict[str, float | str]:
    """Return a fresh copy of the canonical SSIMULACRA2 fallback dict.

    Returns a new dict each call so callers can safely mutate / cast in place.
    All values are floats; the declared value type is ``float | str`` so the
    result is directly assignable to the ``dict[str, float | str]`` metric
    accumulators it feeds (dict value types are invariant).
    """
    fallback: dict[str, float | str] = {
        **_nan_fallback_dict(
            ["ssimulacra2_mean", "ssimulacra2_p95", "ssimulacra2_min"]
        )
    }
    fallback["ssimulacra2_frame_count"] = 0.0
    fallback["ssimulacra2_triggered"] = 0.0
    return fallback


# Canonical LPIPS fallback dict (score keys NaN; count/downscaled/device are
# bookkeeping). ``deep_perceptual_frame_count`` and ``deep_perceptual_device``
# are filled in by the caller because they depend on the actual frame list.
def _default_lpips_score_keys() -> dict[str, float]:
    """Return the three LPIPS score keys set to NaN ("not measured")."""
    return _nan_fallback_dict(
        ["lpips_quality_mean", "lpips_quality_p95", "lpips_quality_max"]
    )


def _aggregate_metric(
    values: list[float],
    metric_name: str,
    primary_agg: Any | None = None,
) -> dict[str, float]:
    """Aggregate frame-level metric values into descriptive statistics.

    Args:
        values: List of frame-level metric values
        metric_name: Name of the metric for key generation
        primary_agg: Callable that reduces a 1-D np.ndarray to a scalar.
            When ``None`` (default), the function resolves the aggregation
            automatically: metrics listed in ``_MEDIAN_AGGREGATED_METRICS``
            (``edge_similarity`` and ``texture_similarity``) use
            ``np.nanmedian``; all others use ``np.nanmean``.

    Returns:
        Dictionary with primary (median or mean), std, min, max for the metric.
        The primary key equals ``metric_name``; sub-keys are always the
        full min/max/std regardless of the primary aggregation function.

    NaN handling (audit-fix, NaN-over-sentinels): per-frame ``except`` blocks
    now append ``float("nan")`` when a frame's metric can't be computed, so
    ``values`` may contain NaN. Aggregation is NaN-aware (``np.nanmean`` etc.)
    so surviving frames carry the score honestly instead of being dragged
    toward a fabricated 0.0. When *every* frame is NaN (or the list is empty),
    the primary/min/max keys are NaN — "not measured" — rather than 0.0, which
    would silently look like a real worst-case score downstream.
    """
    if primary_agg is None:
        primary_agg = (
            np.nanmedian if metric_name in _MEDIAN_AGGREGATED_METRICS else np.nanmean
        )

    # Empty list OR all-NaN: nothing was measured. Return NaN (not 0.0) and
    # short-circuit before calling np.nanmean/nanmin (which would emit a
    # noisy "Mean of empty slice" RuntimeWarning on an all-NaN array).
    values_array = np.array(values, dtype=float)
    if values_array.size == 0 or bool(np.all(np.isnan(values_array))):
        nan = float("nan")
        return {
            metric_name: nan,
            f"{metric_name}_std": nan,
            f"{metric_name}_min": nan,
            f"{metric_name}_max": nan,
        }

    # Handle edge case of single (non-NaN, given the all-NaN guard above) frame
    if len(values) == 1:
        return {
            metric_name: float(values_array[0]),
            f"{metric_name}_std": 0.0,
            f"{metric_name}_min": float(values_array[0]),
            f"{metric_name}_max": float(values_array[0]),
        }

    return {
        metric_name: float(primary_agg(values_array)),
        f"{metric_name}_std": float(np.nanstd(values_array)),
        f"{metric_name}_min": float(np.nanmin(values_array)),
        f"{metric_name}_max": float(np.nanmax(values_array)),
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
                metric_values["mse"].append(float("nan"))

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
                metric_values["psnr"].append(float("nan"))

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
                # NOTE: defensive clamp to [0, 1] guards against numerical edge
                # cases only. skimage SSIM returns values in [-1, 1]; a slightly
                # negative result on totally dissimilar frames maps to 0 here.
                # This clamp is NOT the cause of the smooth_gradient lossy
                # "bump-up" investigated in docs/metrics-audit/2026-05-22 — raw
                # per-frame SSIM also bumps. The root cause is animately's
                # --lossy parameter saturating around level ~125 on
                # low-complexity gradient content: the audit grid samples
                # [20, 60, 100, 160], and the last two levels straddle the
                # saturation knee, producing two distinct outputs whose
                # local-window SSIM happens to differ by ~0.005. See sanity.py
                # LOSSY_LEVELS comment.
                metric_values["ssim"].append(max(0.0, min(1.0, frame_ssim)))
            except Exception as e:
                logger.warning(f"SSIM calculation failed: {e}")
                metric_values["ssim"].append(float("nan"))

    # Calculate advanced metrics (conditional)
    if selected_metrics.get("fsim", False):
        metric_values["fsim"] = []
        for orig_frame, comp_frame in aligned_pairs:
            try:
                frame_fsim = fsim(orig_frame, comp_frame)
                metric_values["fsim"].append(frame_fsim)
            except Exception as e:
                logger.warning(f"FSIM calculation failed: {e}")
                metric_values["fsim"].append(float("nan"))

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
                metric_values["edge_similarity"].append(float("nan"))

    if selected_metrics.get("texture_similarity", False):
        metric_values["texture_similarity"] = []
        for orig_frame, comp_frame in aligned_pairs:
            try:
                frame_texture = texture_similarity(orig_frame, comp_frame)
                metric_values["texture_similarity"].append(frame_texture)
            except Exception as e:
                logger.warning(f"Texture similarity calculation failed: {e}")
                metric_values["texture_similarity"].append(float("nan"))

    # Calculate expensive deep metrics conditionally
    results: dict[str, Any] = {}

    if selected_metrics.get("lpips", False) and getattr(
        config, "ENABLE_DEEP_PERCEPTUAL", True
    ):
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
            # NaN, not 0.5 — a fabricated midpoint silently inflates
            # composite_quality and corpus aggregates. See _nan_fallback_dict.
            results.update(_default_lpips_score_keys())

    if selected_metrics.get("ssimulacra2", False) and getattr(
        config, "ENABLE_SSIMULACRA2", True
    ):
        try:
            from .ssimulacra2_metrics import calculate_ssimulacra2_quality_metrics

            ssim2_metrics = calculate_ssimulacra2_quality_metrics(
                original_frames_resized, compressed_frames_resized, config
            )
            results.update(ssim2_metrics)
        except Exception as e:
            logger.warning(f"SSIMULACRA2 calculation failed: {e}")
            # Emit the full 5-key shape (not just _mean) so this error path
            # matches the main path and downstream consumers always see the
            # same schema. NaN = not computed; 50.0 was a raw-scale sentinel
            # that downstream read as normalised [0, 1].
            results.update(_default_ssimulacra2_fallback())

    if (
        selected_metrics.get("temporal_artifacts", False)
        and config.ENABLE_TEMPORAL_ARTIFACTS
    ):
        try:
            from .temporal_artifacts import calculate_enhanced_temporal_metrics

            temporal_metrics = calculate_enhanced_temporal_metrics(
                original_frames_resized, compressed_frames_resized, device=None
            )
            # Re-key single-stream temporal keys to ``_compressed`` so this
            # optimized path emits the SAME schema as the main from_frames
            # assembly — a bare ``flicker_excess`` etc. must never reach the
            # storage CSV / validation_checker, which now read ``_compressed``.
            _merge_single_stream_metrics(results, temporal_metrics)
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
            # ``flat_region_count`` is single-stream here too — gate the
            # gradient merge on the same tuple so the gradient-sourced
            # ``flat_region_count`` becomes ``_compressed`` rather than bare.
            _merge_single_stream_metrics(results, gradient_metrics)
        except Exception as e:
            logger.warning(f"Color gradients calculation failed: {e}")

    # Aggregate frame-level metrics
    # Add "_mean" suffix to match expected format for composite quality calculation
    # Note: _aggregate_metric auto-selects the primary aggregation function from
    # _MEDIAN_AGGREGATED_METRICS (e.g. edge_similarity uses median, others use mean).
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
    force_all_metrics: bool = False,
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
        # Function parameter takes precedence; fall back to env var for the
        # legacy override path used by the dataset pipeline.
        force_all_metrics = force_all_metrics or (
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

                            # Add gradient and color metrics to optimized
                            # results, re-keying the single-stream
                            # ``flat_region_count`` to ``_compressed`` so this
                            # high-tier fast branch emits the same schema as the
                            # main path (CLAUDE.md "Same key shape across paths").
                            _merge_single_stream_metrics(
                                optimized_results, gradient_color_metrics
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
                    # Default fallback metrics (canonical 5-key NaN shape).
                    default_ssimulacra2_metrics: dict[str, float | str] = {
                        **_default_ssimulacra2_fallback()
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
                                    default_ssim2_key,
                                    default_ssim2_value,
                                ) in default_ssimulacra2_metrics.items():
                                    # All default values are floats, safe to cast directly
                                    optimized_results[default_ssim2_key] = float(
                                        default_ssim2_value
                                    )
                        except Exception as e:
                            logger.warning(
                                f"SSIMULACRA2 metrics failed in optimized path: {e}, using defaults"
                            )
                            # Add default values
                            for (
                                default_ssim2_key,
                                default_ssim2_value,
                            ) in default_ssimulacra2_metrics.items():
                                # All default values are floats, safe to cast directly
                                optimized_results[default_ssim2_key] = float(
                                    default_ssim2_value
                                )
                    else:
                        logger.debug("SSIMULACRA2 metrics calculation disabled")
                        for (
                            default_ssim2_key,
                            default_ssim2_value,
                        ) in default_ssimulacra2_metrics.items():
                            # All default values are floats, safe to cast directly
                            optimized_results[default_ssim2_key] = float(
                                default_ssim2_value
                            )

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

                        # Audit-fix (Wave 7): bare ``temporal_consistency``
                        # removed; statistical siblings re-rooted onto
                        # ``_compressed`` (single-stream value computed on the
                        # compressed frames only — see
                        # calculate_comprehensive_metrics_from_frames).
                        optimized_results["temporal_consistency_compressed_std"] = 0.0
                        optimized_results[
                            "temporal_consistency_compressed_min"
                        ] = float(temporal_post)
                        optimized_results[
                            "temporal_consistency_compressed_max"
                        ] = float(temporal_post)
                        optimized_results["temporal_consistency_pre"] = float(
                            temporal_pre
                        )
                        optimized_results["temporal_consistency_post"] = float(
                            temporal_post
                        )
                        optimized_results["temporal_consistency_delta"] = float(
                            temporal_delta
                        )
                        # Honest single-stream labelling (see from_frames path).
                        optimized_results["temporal_consistency_compressed"] = float(
                            temporal_post
                        )
                        optimized_results["temporal_consistency_original"] = float(
                            temporal_pre
                        )
                    except Exception as e:
                        logger.warning(f"Temporal consistency calculation failed: {e}")
                        # Audit-fix [[giflab-optimized-temporal-failure-nan]]:
                        # emit NaN, NOT fabricated-perfect (1.0/0.0), when the
                        # temporal calc fails. The old defaults silently inflated
                        # composite_quality on exactly the runs that LOST temporal
                        # signal (``temporal_consistency_delta`` and the legacy
                        # ``temporal_consistency_compressed`` feed
                        # calculate_composite_quality). NaN propagates the loss
                        # honestly: ``_is_missing`` filters it and
                        # ``_resolve_composite_from_contributions`` redistributes
                        # the temporal weight, and the validator reports the data
                        # as "unavailable for validation".
                        optimized_results["temporal_consistency_pre"] = float("nan")
                        optimized_results["temporal_consistency_post"] = float("nan")
                        optimized_results["temporal_consistency_delta"] = float("nan")
                        optimized_results["temporal_consistency_compressed"] = float(
                            "nan"
                        )
                        optimized_results["temporal_consistency_original"] = float(
                            "nan"
                        )

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
                    # Normalize PSNR values. A failed frame is NaN ("not
                    # measured") in the parallel path; preserve it rather than
                    # passing it through ``max(0.0, min(nan, 1.0))``, which in
                    # Python's scalar min/max collapses NaN to 0.0 and would
                    # silently fabricate a worst-case score, defeating the
                    # NaN-aware aggregation in _aggregate_metric.
                    metric_values["psnr"] = [
                        value
                        if (isinstance(value, float) and math.isnan(value))
                        else max(0.0, min(value / float(config.PSNR_MAX_DB), 1.0))
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
                    # See defensive-clamp NOTE at the primary SSIM site (search
                    # "smooth_gradient lossy" in this file). Clamp is a guard
                    # for the [-1, 1] edge case, not the source of the audit's
                    # smooth_gradient SSIM bump-up.
                    metric_values["ssim"].append(max(0.0, min(1.0, frame_ssim)))
                except Exception as e:
                    logger.warning(f"SSIM calculation failed for frame: {e}")
                    metric_values["ssim"].append(float("nan"))

                # MS-SSIM calculation
                try:
                    frame_ms_ssim = calculate_ms_ssim(orig_frame, comp_frame)
                    metric_values["ms_ssim"].append(frame_ms_ssim)
                except Exception as e:
                    logger.warning(f"MS-SSIM calculation failed for frame: {e}")
                    metric_values["ms_ssim"].append(float("nan"))

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
                    metric_values["psnr"].append(float("nan"))
                    raw_metric_values["psnr"].append(float("nan"))

                # New metrics - MSE and RMSE
                try:
                    frame_mse = mse(orig_frame, comp_frame)
                    metric_values["mse"].append(frame_mse)

                    frame_rmse = rmse(orig_frame, comp_frame)
                    metric_values["rmse"].append(frame_rmse)
                except Exception as e:
                    logger.warning(f"MSE/RMSE calculation failed for frame: {e}")
                    metric_values["mse"].append(float("nan"))
                    metric_values["rmse"].append(float("nan"))

                # FSIM calculation
                try:
                    frame_fsim = fsim(orig_frame, comp_frame)
                    metric_values["fsim"].append(frame_fsim)
                except Exception as e:
                    logger.warning(f"FSIM calculation failed for frame: {e}")
                    metric_values["fsim"].append(float("nan"))

                # GMSD calculation
                try:
                    frame_gmsd = gmsd(orig_frame, comp_frame)
                    metric_values["gmsd"].append(frame_gmsd)
                except Exception as e:
                    logger.warning(f"GMSD calculation failed for frame: {e}")
                    metric_values["gmsd"].append(float("nan"))

                # Color histogram correlation
                try:
                    frame_chist = chist(orig_frame, comp_frame)
                    metric_values["chist"].append(frame_chist)
                except Exception as e:
                    logger.warning(f"Color histogram calculation failed for frame: {e}")
                    metric_values["chist"].append(float("nan"))

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
                    metric_values["edge_similarity"].append(float("nan"))

                # Texture similarity
                try:
                    frame_texture = texture_similarity(orig_frame, comp_frame)
                    metric_values["texture_similarity"].append(frame_texture)
                except Exception as e:
                    logger.warning(
                        f"Texture similarity calculation failed for frame: {e}"
                    )
                    metric_values["texture_similarity"].append(float("nan"))

                # Sharpness similarity
                try:
                    frame_sharpness = sharpness_similarity(orig_frame, comp_frame)
                    metric_values["sharpness_similarity"].append(frame_sharpness)
                except Exception as e:
                    logger.warning(
                        f"Sharpness similarity calculation failed for frame: {e}"
                    )
                    metric_values["sharpness_similarity"].append(float("nan"))

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

        # Enhanced temporal artifact metrics. Gated on ENABLE_TEMPORAL_ARTIFACTS
        # so callers that don't need temporal computation (e.g. the public
        # measure() API with only cheap metrics) skip the LPIPS model load.
        from .temporal_artifacts import zero_temporal_metrics

        _temporal_fallback = zero_temporal_metrics(len(compressed_frames_resized))
        enhanced_temporal_metrics: dict[str, float] = {}
        if config.ENABLE_TEMPORAL_ARTIFACTS:
            try:
                from .temporal_artifacts import calculate_enhanced_temporal_metrics

                enhanced_temporal_metrics = calculate_enhanced_temporal_metrics(
                    original_frames_resized, compressed_frames_resized, device=None
                )
            except ImportError as e:
                logger.warning(f"Enhanced temporal artifacts module not available: {e}")
                enhanced_temporal_metrics = dict(_temporal_fallback)
            except Exception as e:
                logger.error(f"Enhanced temporal artifacts calculation failed: {e}")
                enhanced_temporal_metrics = dict(_temporal_fallback)
        else:
            enhanced_temporal_metrics = dict(_temporal_fallback)

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

        # Default fallback metrics. Score keys are NaN ("not measured"); a 0.5
        # midpoint sentinel silently inflated composite_quality and corpus
        # aggregates. See _default_lpips_score_keys / _nan_fallback_dict.
        default_deep_perceptual_metrics: dict[str, float | str] = {
            **_default_lpips_score_keys(),
            "deep_perceptual_frame_count": float(len(compressed_frames_resized)),
            "deep_perceptual_downscaled": 0.0,
            "deep_perceptual_device": "fallback",
        }

        # Check if deep perceptual metrics should be calculated. Honour the
        # ENABLE_DEEP_PERCEPTUAL flag at the call site so callers opting out
        # (e.g. the public measure() API with no `lpips` in requested) get the
        # zero-valued fallback without the misleading "No LPIPS scores
        # obtained" warning that fires inside the disabled-LPIPS code path.
        should_calculate_deep_perceptual = getattr(
            config, "ENABLE_DEEP_PERCEPTUAL", True
        )

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
                    ssimulacra2_metrics = _default_ssimulacra2_fallback()

            except ImportError as e:
                logger.info(
                    f"SSIMULACRA2 metrics module not available: {e}. Using fallback values."
                )
                ssimulacra2_metrics = _default_ssimulacra2_fallback()

            except AttributeError as e:
                logger.warning(
                    f"SSIMULACRA2 metrics function not found: {e}. Module may be incomplete."
                )
                ssimulacra2_metrics = _default_ssimulacra2_fallback()

            except (ValueError, TypeError, RuntimeError) as e:
                logger.error(
                    f"Error calculating SSIMULACRA2 metrics: {e}. Using fallback values."
                )
                ssimulacra2_metrics = _default_ssimulacra2_fallback()

            except Exception as e:
                logger.error(
                    f"Unexpected error in SSIMULACRA2 metrics calculation: {e}. Using fallback values."
                )
                ssimulacra2_metrics = _default_ssimulacra2_fallback()
        else:
            logger.debug("SSIMULACRA2 metrics calculation disabled")
            ssimulacra2_metrics = _default_ssimulacra2_fallback()

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
        # Note: _aggregate_metric auto-selects the primary aggregation function from
        # _MEDIAN_AGGREGATED_METRICS (e.g. edge_similarity uses median, others use mean).
        for metric_name, values in metric_values.items():
            result.update(_aggregate_metric(values, metric_name))

        # Audit-fix [[giflab-composite-quality-bare-vs-mean-key-mismatch]]:
        # ``_aggregate_metric`` emits the primary statistic under the BARE key
        # (``ssim``, not ``ssim_mean``). But ``calculate_composite_quality``
        # (enhanced_metrics.py), the storage CSV/SQLite schema
        # (storage.QUALITY_METRIC_COLUMNS) and quality_validation all read the
        # ``{metric}_mean`` keys. On this main serial/parallel path those keys
        # were ABSENT, so ~90% of the composite weight (every structural /
        # signal term) was silently dropped and ``total_weight``
        # renormalisation hid it. Phase 6 (optimized_metrics.py) already emits
        # BOTH bare and ``_mean`` for ssim/mse/psnr (locked by
        # test_phase6_schema_contract); this brings the standard path to the
        # same key shape. The fix is purely additive — bare keys are retained
        # (public_api.measure() and several tests read them).
        #
        # Same-scale aliases: these ``_mean`` keys share the exact scale of
        # their bare key (mse/rmse are raw; ssim/fsim/gmsd/chist/edge/texture/
        # sharpness/ms_ssim are 0-1), matching Phase 6's ``_mean`` definitions.
        # Skip NaN values so an all-NaN (every-frame-failed) structural metric
        # keeps "key absent" semantics — composite then redistributes its
        # weight exactly as it did before this fix, rather than entering a NaN
        # into the weighted sum.
        _mean_alias_bases = (
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
        )
        for base in _mean_alias_bases:
            if base in result:
                base_value = result[base]
                if isinstance(base_value, int | float) and not (
                    isinstance(base_value, float) and math.isnan(base_value)
                ):
                    result[f"{base}_mean"] = float(base_value)

        # PSNR special case: the bare ``psnr`` key is normalised to 0-1
        # (divided by config.PSNR_MAX_DB above), but ``normalize_metric`` and
        # ``calculate_composite_quality`` expect ``psnr_mean`` in RAW dB (they
        # divide by 50 dB themselves). Aliasing the normalised bare value would
        # DOUBLE-normalise (e.g. 0.47 → /50 → ~0.009), tanking the 20%-weight
        # PSNR term. So compute ``psnr_mean`` from the raw-dB values held in
        # ``raw_metric_values["psnr"]`` (populated by both the parallel and
        # serial frame loops, independent of config.RAW_METRICS). This matches
        # calculate_selected_metrics and Phase 6, both of which keep psnr raw.
        #
        # NaN handling — OMIT, don't emit NaN: when every frame's PSNR failed
        # (raw list empty / all-NaN), ``psnr_mean`` is left ABSENT rather than
        # set to NaN. This mirrors the skip-NaN policy for the structural
        # aliases above and, crucially, avoids the
        # ``normalize_metric('psnr_mean', nan)`` trap: that function does
        # ``min(nan, 50.0)`` → ``nan`` then ``max(0.0, min(1.0, nan))`` →
        # ``1.0``, so a present-but-NaN ``psnr_mean`` would silently award the
        # FULL 20% PSNR weight as a PERFECT score — a fabricated value, worse
        # than the honest "key absent → weight redistributed" outcome. (The
        # separate ``psnr_raw`` block below still emits NaN because that key is
        # diagnostic-only and never read by the composite.)
        psnr_mean_raw_vals = np.array(raw_metric_values["psnr"], dtype=float)
        if psnr_mean_raw_vals.size > 0 and not bool(
            np.all(np.isnan(psnr_mean_raw_vals))
        ):
            result["psnr_mean"] = float(np.nanmean(psnr_mean_raw_vals))

        # Add temporal consistency (single value, not frame-level).
        #
        # Audit-fix (Wave 7): the legacy bare ``temporal_consistency`` key has
        # been REMOVED. It carried the post-compression value of a metric
        # computed on the COMPRESSED STREAM ONLY, yet its name read like an
        # original-vs-compressed pair signal — exactly the single-stream-
        # mislabelled-as-pair anti-pattern CLAUDE.md warns against. The honest
        # ``_compressed`` / ``_original`` suffixed keys are the replacement, and
        # the statistical siblings are re-rooted onto ``_compressed`` so the
        # ``X_min <= X_mean <= X_max`` family invariant still holds.
        result["temporal_consistency_compressed_std"] = 0.0
        result["temporal_consistency_compressed_min"] = float(temporal_post)
        result["temporal_consistency_compressed_max"] = float(temporal_post)

        # Provenance keys: pre, post (explicit) and delta (true pair signal).
        result["temporal_consistency_pre"] = float(temporal_pre)
        result["temporal_consistency_post"] = float(temporal_post)
        result["temporal_consistency_delta"] = float(temporal_delta)

        # Honest single-stream labelling: ``_compressed`` is the compressed-only
        # value (== post), ``_original`` is the original-only value (== pre).
        result["temporal_consistency_compressed"] = float(temporal_post)
        result["temporal_consistency_original"] = float(temporal_pre)

        # Add disposal artifact metrics. Same Wave-7 treatment: bare
        # ``disposal_artifacts`` removed; stats re-rooted onto ``_compressed``.
        result["disposal_artifacts_compressed_std"] = 0.0
        result["disposal_artifacts_compressed_min"] = float(disposal_artifacts_post)
        result["disposal_artifacts_compressed_max"] = float(disposal_artifacts_post)
        result["disposal_artifacts_pre"] = float(disposal_artifacts_pre)
        result["disposal_artifacts_post"] = float(disposal_artifacts_post)
        result["disposal_artifacts_delta"] = float(disposal_artifacts_delta)
        result["disposal_artifacts_compressed"] = float(disposal_artifacts_post)
        result["disposal_artifacts_original"] = float(disposal_artifacts_pre)

        # Audit-fix (Wave 7): all of the enhanced temporal metrics
        # (flicker_excess, flicker_frame_ratio, flat_flicker_ratio,
        # flat_region_count, temporal_pumping_score,
        # quality_oscillation_frequency, lpips_t_mean, lpips_t_p95,
        # lpips_t_max) are computed on COMPRESSED FRAMES ONLY (see
        # temporal_artifacts.calculate_enhanced_temporal_metrics — the
        # ``original_frames`` argument is accepted but never used in the
        # detector calls). They are emitted ONLY under their ``_compressed``
        # suffix so callers cannot mistake a single-stream value for an
        # original-vs-compressed pair signal. The legacy bare keys have been
        # REMOVED (Wave 7) — they read like pair signals but never were one.
        # The re-keying tuple + merge live at module scope
        # (``_SINGLE_STREAM_TEMPORAL_KEYS`` / ``_merge_single_stream_metrics``)
        # so every assembly path applies the identical schema.

        # Add enhanced temporal artifact metrics (Task 1.2). Suppress the bare
        # single-stream keys and re-key them to ``_compressed`` sourced directly
        # from the producer dict — never let the bare key land in ``result``.
        _merge_single_stream_metrics(result, enhanced_temporal_metrics)

        # Add enhanced gradient and color artifact metrics (Task 1.3 & 1.4).
        # ``flat_region_count`` is also a single-stream key produced here — it
        # must follow the same ``_compressed``-only treatment as above.
        _merge_single_stream_metrics(result, gradient_color_metrics)

        # Add deep perceptual metrics (Task 2.2)
        for deep_key, deep_value in deep_perceptual_metrics.items():
            if isinstance(deep_value, int | float):
                result[deep_key] = float(deep_value)
            else:
                result[deep_key] = str(deep_value)

        # Add SSIMULACRA2 metrics (Phase 3.2)
        # (fresh loop-var names: ssim2_value is bound to a float-valued dict
        # earlier in this function and mypy pins the loop-variable type)
        for s2_key, s2_value in ssimulacra2_metrics.items():
            # Convert values to appropriate types for storage
            result[s2_key] = (
                float(s2_value)
                if isinstance(s2_value, int | float)
                else str(s2_value)
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

        # Frame-drop alignment warning ([[giflab-alignment-warning-threshold]]).
        # Surface imperfect frame-drop alignment (the silent
        # ``alignment_accuracy=0.976`` case) as a float flag (1.0 = warn,
        # 0.0 = no warn). NaN-honest: only a REAL alignment value below the
        # configured threshold warns. The value is NEVER set when:
        #   - alignment was not measured (no file_metadata -> timing_metrics == {},
        #     so the key is absent),
        #   - it is the missing-data NaN default (timing_validation.py),
        #   - it is the documented -1.0 failure sentinel (the timing-validation
        #     except branches above).
        # A genuine 0.0 (fully misaligned) DOES warn — distinct from NaN.
        alignment_accuracy = result.get("alignment_accuracy")
        alignment_warning = 0.0
        if isinstance(alignment_accuracy, int | float):
            alignment_value = float(alignment_accuracy)
            is_missing = math.isnan(alignment_value)
            is_failure_sentinel = alignment_value == -1.0
            if (
                not is_missing
                and not is_failure_sentinel
                and alignment_value < config.ALIGNMENT_WARNING_THRESHOLD
            ):
                alignment_warning = 1.0
                logger.warning(
                    "Imperfect frame-drop alignment: alignment_accuracy=%.3f "
                    "< threshold %.3f (original frames=%d, compressed frames=%d)",
                    alignment_value,
                    config.ALIGNMENT_WARNING_THRESHOLD,
                    int(original_frame_count),
                    int(compressed_frame_count),
                )
        result["alignment_warning"] = alignment_warning

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
                # Wave 7: bare ``temporal_consistency`` removed; the explicit
                # provenance variants keep their raw copies below.
                "temporal_consistency_pre",
                "temporal_consistency_post",
                "temporal_consistency_delta",
            ]

            for metric_name in raw_equivalent_metrics:
                result[f"{metric_name}_raw"] = result[metric_name]

            # Handle PSNR separately: use un-scaled mean value. NaN-aware so a
            # frame whose PSNR failed (now appended as NaN) doesn't drag the
            # raw mean toward 0; empty / all-NaN -> NaN ("not measured").
            psnr_raw_vals = np.array(raw_metric_values["psnr"], dtype=float)
            if psnr_raw_vals.size > 0 and not bool(np.all(np.isnan(psnr_raw_vals))):
                result["psnr_raw"] = float(np.nanmean(psnr_raw_vals))
            else:
                result["psnr_raw"] = float("nan")

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


# Sentinel raw values used to probe each composite-input metric's polarity.
# Each pair is (low_raw, high_raw) chosen STRICTLY INSIDE the metric's
# documented monotonic range so the composite's response is monotone over the
# pair. See ``normalize_metric`` / ``calculate_composite_quality`` for the
# per-metric transforms these exercise end-to-end.
_POLARITY_PROBE_SENTINELS: dict[str, tuple[float, float]] = {
    "ssim_mean": (0.1, 0.9),
    "ms_ssim_mean": (0.1, 0.9),
    "fsim_mean": (0.1, 0.9),
    "edge_similarity_mean": (0.1, 0.9),
    "chist_mean": (0.1, 0.9),
    "sharpness_similarity_mean": (0.1, 0.9),
    "texture_similarity_mean": (0.1, 0.9),
    "ssimulacra2_mean": (0.1, 0.9),
    "psnr_mean": (5.0, 45.0),
    "mse_mean": (1.0, 10000.0),
    "gmsd_mean": (0.05, 0.45),
    "banding_score_mean": (0.1, 0.9),
    "deltae_mean": (1.0, 9.0),
    "lpips_quality_mean": (0.1, 0.9),
    "temporal_consistency_delta": (0.1, 0.9),
    "temporal_consistency_compressed": (0.1, 0.9),
}

# Per-stem sibling-statistic suffixes copied alongside the chosen ``_mean`` /
# delta value so the aggregate invariant ``X_min <= X_mean <= X_max`` holds for
# whichever pass won that stem. Retained as documentation of the *statistical*
# siblings, but the merge now copies EVERY ``{stem}*`` companion from the
# chosen pass (see ``_merge_worst_of_dual_composite``) so no companion can
# drift to the losing pass's provenance.
_SIBLING_SUFFIXES = ("_std", "_min", "_max", "_raw")

# NOTE on the same-scale BARE alias: for ssim/ms_ssim/mse/fsim/gmsd/chist/
# edge_similarity/texture_similarity/sharpness_similarity the bare key equals the
# ``_mean`` key (``result[base] == result[f"{base}_mean"]`` — a load-bearing
# contract the public ``measure()`` surface projects; see metrics.py ~3506-3545
# and ``test_from_frames_mean_aliases_match_bare``). The worst-of merge does NOT
# need a hand-maintained list of these stems: ``_copy_stem_family`` copies the
# bare key (``key == stem``) along with the whole family from the winning pass,
# so the alias is preserved generically. PSNR's bare key is normalised 0-1 while
# ``psnr_mean`` is raw dB; that scale split is preserved automatically because
# each pass's bare ``psnr`` is that pass's OWN normalised value, so the family
# copy carries both consistently.

# Cache of derived polarity maps, keyed by the two config flags that determine
# the sign/membership of the composite-input set. The polarity (the SIGN of each
# single-metric monotone composite) is config-derived and stable per run; the
# weight MAGNITUDES that ``USE_ENHANCED_COMPOSITE_QUALITY`` toggles never flip a
# monotone single-metric composite's direction, and ``USE_TEMPORAL_DELTA_FOR_
# COMPOSITE`` only swaps WHICH temporal key is in the set. Memoising on this
# tuple avoids re-running the ~30-call probe (~15 keys x 2 sentinels) on every
# transparent-GIF metrics call in a batch run.
_POLARITY_CACHE: dict[tuple[bool, bool], dict[str, int]] = {}


def _composite_metric_polarity(config: MetricsConfig) -> dict[str, int]:
    """Derive each composite-input metric's polarity by probing the END-TO-END
    composite, not ``normalize_metric``.

    For each composite-input ``_mean`` / delta key the active config consumes,
    build a minimal metrics dict containing ONLY that key (plus nothing else),
    evaluate ``calculate_composite_quality`` at a low and a high raw sentinel,
    and read off the direction:

    * composite(high) > composite(low)  -> higher RAW improves quality
      -> polarity ``+1`` (the WORST value is the MINIMUM raw).
    * composite(high) < composite(low)  -> higher RAW degrades quality
      -> polarity ``-1`` (the WORST value is the MAXIMUM raw).

    Because only the probed key is present, weight redistribution makes the
    composite a monotone function of that single metric's normalized
    contribution, so the composite's sign of change equals the per-metric
    transform's sign — captured through WHATEVER path (inline block or
    ``normalize_metric``) the composite actually uses. This is invariant to
    where the transform lives, which is why probing the end-to-end composite is
    correct for ``lpips_quality_mean`` / ``ssimulacra2_mean`` /
    ``temporal_consistency_delta`` (none of which have a ``normalize_metric``
    branch) where probing ``normalize_metric`` directly would derive the
    OPPOSITE direction.

    Returns:
        Mapping of composite-input key -> polarity (+1 worst=MIN, -1 worst=MAX),
        limited to the keys the composite consumes in the active config mode
        (the temporal key follows ``USE_TEMPORAL_DELTA_FOR_COMPOSITE``).
    """
    from .enhanced_metrics import calculate_composite_quality

    use_temporal_delta = getattr(config, "USE_TEMPORAL_DELTA_FOR_COMPOSITE", True)
    use_enhanced = getattr(config, "USE_ENHANCED_COMPOSITE_QUALITY", True)

    # Memoise per (temporal-delta, enhanced) flag tuple: polarity is config-
    # derived and stable per run, so the ~30-call probe should run at most once
    # per config mode rather than on every transparent-GIF metrics call.
    cache_key = (bool(use_temporal_delta), bool(use_enhanced))
    cached = _POLARITY_CACHE.get(cache_key)
    if cached is not None:
        return dict(cached)

    # The keys the composite actually reads in the active config mode.
    candidate_keys = [
        "ssim_mean",
        "ms_ssim_mean",
        "psnr_mean",
        "mse_mean",
        "fsim_mean",
        "edge_similarity_mean",
        "gmsd_mean",
        "chist_mean",
        "sharpness_similarity_mean",
        "texture_similarity_mean",
        "lpips_quality_mean",
        "ssimulacra2_mean",
        "banding_score_mean",
        "deltae_mean",
    ]
    # Temporal: True -> delta (worst=MAX); False -> temporal_consistency_compressed
    # (single-stream compressed value, higher-better via the standard branch,
    # worst=MIN). Wave 7 renamed the bare key to ``_compressed``.
    candidate_keys.append(
        "temporal_consistency_delta"
        if use_temporal_delta
        else "temporal_consistency_compressed"
    )

    polarity: dict[str, int] = {}
    for key in candidate_keys:
        low_raw, high_raw = _POLARITY_PROBE_SENTINELS[key]
        composite_low = calculate_composite_quality({key: low_raw}, config)
        composite_high = calculate_composite_quality({key: high_raw}, config)
        # Both finite by construction (single present metric, valid sentinels).
        if composite_high >= composite_low:
            polarity[key] = 1  # higher raw better -> worst is the minimum
        else:
            polarity[key] = -1  # higher raw worse -> worst is the maximum

    # Store a copy so callers can mutate the returned dict without poisoning the
    # cache, and return a fresh copy for the same reason.
    _POLARITY_CACHE[cache_key] = dict(polarity)
    return polarity


def _worst_of(value_a: float, value_b: float, polarity: int) -> tuple[float, bool]:
    """Return (worst_value, chose_b) for two raw values under *polarity*.

    NaN-aware with nanmin / nanmax semantics: one finite + one NaN -> the
    finite value (a real measurement beats a non-measurement); both NaN ->
    NaN (honest propagation, never coerced to 0/1). ``chose_b`` reports whether
    the BLACK (second) pass supplied the chosen value, so the caller can copy
    that pass's sibling statistics.
    """
    a_nan = isinstance(value_a, float) and math.isnan(value_a)
    b_nan = isinstance(value_b, float) and math.isnan(value_b)

    if a_nan and b_nan:
        return float("nan"), False
    if a_nan:
        return value_b, True
    if b_nan:
        return value_a, False

    if polarity == 1:
        # worst = minimum raw
        return (value_b, True) if value_b < value_a else (value_a, False)
    # polarity == -1: worst = maximum raw
    return (value_b, True) if value_b > value_a else (value_a, False)


def _companion_stem_for(key: str) -> str:
    """Return the COMPANION prefix whose ``{stem}*`` family follows a stem's
    winning pass.

    For an ``X_mean`` key the family is ``X*`` (so the bare ``X`` alias, the
    ``X_std`` / ``X_min`` / ``X_max`` / ``X_raw`` stats AND non-sibling
    companions such as ``X_p95``, ``X_first`` / ``X_last`` / ``X_middle`` /
    ``X_positional_variance``, ``X_pct_gt*`` all move together). The temporal
    candidate keys (``temporal_consistency_compressed`` /
    ``temporal_consistency_delta`` — Wave 7 renamed the bare key) collapse to
    the shared ``temporal_consistency`` family so the whole pre/post/original/
    compressed/delta cluster (including the re-rooted
    ``temporal_consistency_compressed_std`` / ``_min`` / ``_max``, which still
    start with ``temporal_consistency_``) follows whichever background won the
    temporal comparison — never half-white, half-black.
    """
    if key in ("temporal_consistency_compressed", "temporal_consistency_delta"):
        return "temporal_consistency"
    return key[: -len("_mean")] if key.endswith("_mean") else key


def _copy_stem_family(
    src: dict[str, float | str],
    dst: dict[str, float | str],
    stem: str,
) -> None:
    """Overwrite every ``{stem}`` / ``{stem}_*`` key in *dst* with *src*'s value.

    This copies the BARE alias (``key == stem``), the statistical siblings AND
    every non-sibling companion (``_p95`` / ``_pre`` / ``_post`` / ``_first`` /
    ``_pct_gt*`` / …) so a stem's full key family carries a single, consistent
    provenance after the merge. Stems never prefix-collide across metric
    families (``ssim_`` vs ``ssimulacra2``, ``mse_`` vs ``ms_ssim``, etc.), so
    matching on ``key == stem or key.startswith(stem + "_")`` is exact.
    """
    prefix = stem + "_"
    for k, v in src.items():
        if k == stem or k.startswith(prefix):
            dst[k] = v


def _merge_worst_of_dual_composite(
    white: dict[str, float | str],
    black: dict[str, float | str],
    config: MetricsConfig,
) -> dict[str, float | str]:
    """Merge white- and black-composite metric dicts via per-metric worst-of.

    Strategy A: for every composite-input ``_mean`` / delta key the composite
    consumes, pick the value that is WORSE for quality (direction derived from
    the live end-to-end composite via ``_composite_metric_polarity``). When the
    worst value comes from the BLACK pass, copy that stem's ENTIRE key family
    (``{stem}`` bare alias + ``_std`` / ``_min`` / ``_max`` / ``_raw`` siblings +
    every non-sibling companion such as ``_p95`` / ``_pre`` / ``_post`` /
    ``_first`` / ``_pct_gt*``) from BLACK; otherwise the white family (already in
    the base) is kept.

    Why the bare-alias copy is load-bearing: the public ``measure()`` surface
    projects the BARE metric keys (``ssim``, ``gmsd``, ``fsim``, … via
    ``public_api._PUBLIC_TO_INTERNAL_METRIC_KEY``), NOT the ``_mean`` keys. The
    bare key is also a documented same-scale alias of ``_mean`` (metrics.py
    ~3506-3545, ``test_from_frames_mean_aliases_match_bare``). If the merge
    updated only ``X_mean`` and left bare ``X`` at the optimistic white value,
    ``measure().ssim`` / ``.gmsd`` / … would report the WHITE-only score on a
    transparent dark-content GIF — directly contradicting the worst-of contract.
    Copying the whole family from the winning pass keeps bare == ``_mean`` and
    keeps every companion (``ssimulacra2_p95``, the temporal cluster, the
    positional ssim stats) on a single provenance. PSNR's scale split survives
    because each pass's bare ``psnr`` is that pass's own normalised value
    (``psnr_mean / PSNR_MAX_DB``); copying black's bare ``psnr`` alongside black's
    raw-dB ``psnr_mean`` keeps the normalised-bare / raw-dB-``_mean`` relationship
    intact at the worst-of value.

    The base is the WHITE dict, so every non-composite key (frame counts,
    kilobytes, compression_ratio, string keys) is authoritative from white —
    these are identical across passes (same files/metadata). The merge is purely
    per-``_mean`` worst-of and NEVER asserts equal frame counts between passes:
    content-based alignment can legitimately yield different pair counts per
    background, and the ``_mean`` values remain comparable regardless.
    ``render_ms``, ``composite_quality`` and ``efficiency`` are intentionally NOT
    finalised here: the caller overwrites render_ms at the file level and re-runs
    ``process_metrics_with_enhanced_quality`` to recompute composite_quality AND
    efficiency from the merged worst-of values.

    NaN-aware throughout (see ``_worst_of``): a missing measurement never
    coerces to a fabricated best/worst case.
    """
    merged: dict[str, float | str] = dict(white)

    polarity = _composite_metric_polarity(config)

    for key, pol in polarity.items():
        white_present = key in white
        black_present = key in black

        if not white_present and not black_present:
            continue

        stem = _companion_stem_for(key)

        if white_present and not black_present:
            # Present only on white — already in base; nothing to overwrite.
            continue
        if black_present and not white_present:
            # Present only on black — take its whole family (no white family
            # exists to keep aliases/companions consistent otherwise).
            _copy_stem_family(black, merged, stem)
            continue

        # Present on both: per-metric worst-of.
        white_val = white[key]
        black_val = black[key]
        if not isinstance(white_val, int | float) or not isinstance(
            black_val, int | float
        ):
            # Non-numeric (shouldn't happen for these keys) — keep white.
            continue

        worst_value, chose_black = _worst_of(float(white_val), float(black_val), pol)
        merged[key] = worst_value
        if chose_black:
            # Copy the chosen (black) pass's WHOLE family — bare alias, sibling
            # stats and every non-sibling companion — so bare == _mean holds,
            # X_min <= X_mean <= X_max holds, and no companion (ssimulacra2_p95,
            # the temporal cluster, the positional ssim stats) drifts to the
            # losing white pass. ``merged[key]`` is re-set to ``worst_value``
            # afterwards in case ``_worst_of`` returned a NaN-resolved value that
            # differs from the raw ``black[key]`` the family copy just wrote.
            _copy_stem_family(black, merged, stem)
            merged[key] = worst_value

    return merged


def calculate_comprehensive_metrics(
    original_path: Path,
    compressed_path: Path,
    config: MetricsConfig | None = None,
    frame_reduction_context: bool = False,
    force_all_metrics: bool = False,
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
        # File-level wall-clock timer for render_ms. This wraps the probe +
        # both extractions + both from_frames passes + merge, so on a
        # transparent GIF the reported render_ms genuinely ~doubles (matches
        # the documented dual-pass cost). On an opaque GIF only one pass runs,
        # so render_ms tracks the single-pass total.
        start_time = time.perf_counter()

        # WHITE pass first (default background). On opaque GIFs this is the
        # ONLY extraction — zero added cost. has_alpha survives a warm cache
        # hit (persisted through the cache), so the dual pass still triggers on
        # transparent GIFs even when the white entry was already cached.
        white_original = extract_gif_frames(original_path, config.SSIM_MAX_FRAMES)
        white_compressed = extract_gif_frames(compressed_path, config.SSIM_MAX_FRAMES)

        # Extract metadata for file-specific operations
        try:
            from .meta import extract_gif_metadata

            original_metadata = extract_gif_metadata(original_path)
            compressed_metadata = extract_gif_metadata(compressed_path)
            original_frame_count = original_metadata.orig_frames
            compressed_frame_count = compressed_metadata.orig_frames
        except Exception:
            # Fallback to extracted frames count
            original_frame_count = len(white_original.frames)
            compressed_frame_count = len(white_compressed.frames)

        # Prepare file metadata for the frame-based function. The file sizes
        # (and therefore compression_ratio / kilobytes) do NOT depend on the
        # compositing background, so both passes share the same metadata.
        file_metadata = {
            "original_path": original_path,
            "compressed_path": compressed_path,
            "original_frame_count": original_frame_count,
            "compressed_frame_count": compressed_frame_count,
            "original_size_bytes": original_path.stat().st_size,
            "compressed_size_bytes": compressed_path.stat().st_size,
        }

        # Either GIF carrying transparency triggers the dual-composite path:
        # white-only compositing biases every pixel metric in favour of
        # dark-content GIFs (the difference is swamped against white). A black
        # second pass surfaces that difference; worst-of merging stops a
        # compressor from gaming the score by picking a friendly background.
        needs_dual = white_original.has_alpha or white_compressed.has_alpha

        white_result = calculate_comprehensive_metrics_from_frames(
            white_original.frames,
            white_compressed.frames,
            config=config,
            frame_reduction_context=frame_reduction_context,
            file_metadata=file_metadata,
            force_all_metrics=force_all_metrics,
        )

        if not needs_dual:
            # Opaque GIF: single white pass, exactly as before. Re-time at the
            # file level so render_ms is consistent with the dual path's
            # file-level timing (negligible wrapper overhead over from_frames).
            white_result["render_ms"] = min(
                int((time.perf_counter() - start_time) * 1000), 86400000
            )
            return white_result

        # Transparent GIF: also composite onto BLACK and merge worst-of.
        black_original = extract_gif_frames(
            original_path, config.SSIM_MAX_FRAMES, alpha_background=(0, 0, 0)
        )
        black_compressed = extract_gif_frames(
            compressed_path, config.SSIM_MAX_FRAMES, alpha_background=(0, 0, 0)
        )

        black_result = calculate_comprehensive_metrics_from_frames(
            black_original.frames,
            black_compressed.frames,
            config=config,
            frame_reduction_context=frame_reduction_context,
            file_metadata=file_metadata,
            force_all_metrics=force_all_metrics,
        )

        merged = _merge_worst_of_dual_composite(white_result, black_result, config)

        # Re-run the enhanced-quality processor on the MERGED dict so
        # composite_quality AND efficiency are recomputed from the
        # pessimistic per-metric worst-of values (not carried from either
        # pass). compression_ratio is identical across passes (file-size
        # derived), so efficiency is consistent and reflects the lower
        # composite.
        from .enhanced_metrics import process_metrics_with_enhanced_quality

        merged = process_metrics_with_enhanced_quality(merged, config)

        # File-level wall-clock total (probe + both extractions + both
        # from_frames + merge).
        merged["render_ms"] = min(
            int((time.perf_counter() - start_time) * 1000), 86400000
        )

        return merged

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
