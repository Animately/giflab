"""
Integration module for ValidationCache with metrics calculation functions.

This module provides wrappers and utilities to integrate the ValidationCache
with various metric calculation functions like SSIM, MS-SSIM, LPIPS, etc.
"""

import hashlib
import json
import logging
import time
from typing import Any, Optional, Union

import numpy as np

from ..config import VALIDATION_CACHE
from .validation_cache import ValidationCache, get_validation_cache

logger = logging.getLogger(__name__)

def _get_metric_config(metric_type: str) -> dict[str, Any]:
    """
    Get relevant configuration for a specific metric type.
    
    Args:
        metric_type: type of metric
        
    Returns:
        Configuration dictionary for the metric
    """
    config = {}
    
    if metric_type == "ms_ssim":
        config["scales"] = 5  # Default MS-SSIM scales
    elif metric_type == "lpips":
        config["net"] = "alex"  # Default LPIPS network
        config["version"] = "0.1"
    elif metric_type == "gradient_color":
        config["enable_gradient"] = True
        config["enable_color"] = True
    elif metric_type == "ssimulacra2":
        config["enable_gpu"] = False  # Default to CPU for consistency
    
    return config

def calculate_ms_ssim_cached(
    frame1: np.ndarray,
    frame2: np.ndarray,
    scales: int = 5,
    use_validation_cache: bool = True,
    frame_indices: tuple[int, int] | None = None,
) -> float:
    """
    Calculate MS-SSIM with ValidationCache support.
    
    Args:
        frame1: First frame
        frame2: Second frame
        scales: Number of scales for MS-SSIM
        use_validation_cache: Whether to use validation cache
        frame_indices: Optional frame indices for cache key
        
    Returns:
        MS-SSIM value
    """
    if not use_validation_cache or not VALIDATION_CACHE.get("enabled", True) or not VALIDATION_CACHE.get("cache_ms_ssim", True):
        # Cache disabled, calculate directly
        from ..metrics import calculate_ms_ssim
        return calculate_ms_ssim(frame1, frame2, scales=scales)
    
    cache = get_validation_cache()
    config = {"scales": scales}
    
    # Try to get from cache
    cached_value = cache.get(
        frame1, frame2, "ms_ssim", config, frame_indices
    )
    
    if cached_value is not None:
        return float(cached_value)
    
    # Calculate and cache
    from ..metrics import calculate_ms_ssim
    value = calculate_ms_ssim(frame1, frame2, scales=scales)
    
    cache.put(
        frame1, frame2, "ms_ssim", value, config, frame_indices
    )
    
    return value

def calculate_ssim_cached(
    frame1: np.ndarray,
    frame2: np.ndarray,
    use_validation_cache: bool = True,
    frame_indices: tuple[int, int] | None = None,
) -> float:
    """
    Calculate SSIM with ValidationCache support.
    
    Args:
        frame1: First frame
        frame2: Second frame
        use_validation_cache: Whether to use validation cache
        frame_indices: Optional frame indices for cache key
        
    Returns:
        SSIM value
    """
    if not use_validation_cache or not VALIDATION_CACHE.get("enabled", True) or not VALIDATION_CACHE.get("cache_ssim", True):
        # Cache disabled, calculate directly
        from skimage.metrics import structural_similarity as ssim
        
        # Convert to grayscale if needed
        if len(frame1.shape) == 3:
            import cv2
            frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
            frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
        else:
            frame1_gray = frame1
            frame2_gray = frame2
        
        return ssim(frame1_gray, frame2_gray, data_range=255)
    
    cache = get_validation_cache()
    config = {}
    
    # Try to get from cache
    cached_value = cache.get(
        frame1, frame2, "ssim", config, frame_indices
    )
    
    if cached_value is not None:
        return float(cached_value)
    
    # Calculate and cache
    import cv2
    from skimage.metrics import structural_similarity as ssim
    
    # Convert to grayscale if needed
    if len(frame1.shape) == 3:
        frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
        frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
    else:
        frame1_gray = frame1
        frame2_gray = frame2
    
    value = ssim(frame1_gray, frame2_gray, data_range=255)
    
    cache.put(
        frame1, frame2, "ssim", value, config, frame_indices
    )
    
    return value

def calculate_lpips_cached(
    frame1: np.ndarray,
    frame2: np.ndarray,
    net: str = "alex",
    version: str = "0.1",
    use_validation_cache: bool = True,
    frame_indices: tuple[int, int] | None = None,
) -> float:
    """
    Calculate LPIPS with ValidationCache support.
    
    Args:
        frame1: First frame
        frame2: Second frame
        net: LPIPS network type
        version: LPIPS version
        use_validation_cache: Whether to use validation cache
        frame_indices: Optional frame indices for cache key
        
    Returns:
        LPIPS value
    """
    if not use_validation_cache or not VALIDATION_CACHE.get("enabled", True) or not VALIDATION_CACHE.get("cache_lpips", True):
        # Cache disabled, calculate directly
        from ..deep_perceptual_metrics import calculate_deep_perceptual_quality_metrics
        result = calculate_deep_perceptual_quality_metrics([frame1], [frame2], config={"device": "auto"})
        return result.get("lpips_quality_mean", 0.5)
    
    cache = get_validation_cache()
    config = {"net": net, "version": version}
    
    # Try to get from cache
    cached_value = cache.get(
        frame1, frame2, "lpips", config, frame_indices
    )
    
    if cached_value is not None:
        return float(cached_value)
    
    # Calculate and cache
    from ..deep_perceptual_metrics import calculate_deep_perceptual_quality_metrics
    result = calculate_deep_perceptual_quality_metrics([frame1], [frame2], config={"device": "auto"})
    value = result.get("lpips_quality_mean", 0.5)
    
    cache.put(
        frame1, frame2, "lpips", value, config, frame_indices
    )
    
    return value

def calculate_gradient_color_cached(
    frames1: list[np.ndarray],
    frames2: list[np.ndarray],
    use_validation_cache: bool = True,
) -> dict[str, Any]:
    """
    Calculate gradient and color artifacts with ValidationCache support.
    
    Args:
        frames1: First set of frames
        frames2: Second set of frames
        use_validation_cache: Whether to use validation cache
        
    Returns:
        Dictionary of gradient and color metrics
    """
    if not use_validation_cache or not VALIDATION_CACHE.get("enabled", True) or not VALIDATION_CACHE.get("cache_gradient_color", True):
        # Cache disabled, calculate directly
        from ..gradient_color_artifacts import calculate_gradient_color_metrics
        return calculate_gradient_color_metrics(frames1, frames2)
    
    cache = get_validation_cache()
    
    # For gradient_color, we cache the entire result as it processes all frames together
    # Create a composite hash for all frames
    all_frames_hash = hashlib.md5()
    for f1, f2 in zip(frames1, frames2, strict=True):
        all_frames_hash.update(cache.get_frame_hash(f1).encode())
        all_frames_hash.update(cache.get_frame_hash(f2).encode())

    cache_key = all_frames_hash.hexdigest()[:32]
    config = {"enable_gradient": True, "enable_color": True}
    
    # Try to get from cache using a synthetic lookup
    # We'll use the first frame pair as the lookup key with the composite hash
    if frames1 and frames2:
        # Create a special cache key using the composite hash
        special_key = cache.generate_cache_key(
            cache_key, cache_key, "gradient_color", config
        )
        
        # Check if we have this in cache
        with cache._lock:
            if special_key in cache._memory_cache:
                entry = cache._memory_cache[special_key]
                if time.time() - entry.timestamp < cache.ttl_seconds:
                    cache._memory_cache.move_to_end(special_key)
                    cache._stats.hits += 1
                    logger.debug(f"Cache hit (memory): gradient_color for {len(frames1)} frames")
                    return entry.value
    
    # Calculate and cache
    from ..gradient_color_artifacts import calculate_gradient_color_metrics
    result = calculate_gradient_color_metrics(frames1, frames2)
    
    # Store in cache with the composite key
    if frames1 and frames2:
        from ..caching.validation_cache import ValidationResult
        
        entry = ValidationResult(
            metric_type="gradient_color",
            value=result,
            config_hash=hashlib.md5(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16],
            timestamp=time.time(),
            metadata={"num_frames": len(frames1)},
        )
        
        with cache._lock:
            cache._add_to_memory(special_key, entry)
            cache._store_to_disk(special_key, entry, cache_key, cache_key)
    
    return result

def calculate_ssimulacra2_cached(
    frames1: list[np.ndarray],
    frames2: list[np.ndarray],
    use_validation_cache: bool = True,
) -> dict[str, Any]:
    """
    Calculate SSIMulacra2 metrics with ValidationCache support.
    
    Args:
        frames1: First set of frames
        frames2: Second set of frames
        use_validation_cache: Whether to use validation cache
        
    Returns:
        Dictionary of SSIMulacra2 metrics
    """
    if not use_validation_cache or not VALIDATION_CACHE.get("enabled", True) or not VALIDATION_CACHE.get("cache_ssimulacra2", True):
        # Cache disabled, calculate directly
        from ..ssimulacra2_metrics import calculate_ssimulacra2_quality_metrics
        return calculate_ssimulacra2_quality_metrics(frames1, frames2)
    
    cache = get_validation_cache()
    
    # Similar to gradient_color, cache the entire result
    all_frames_hash = hashlib.md5()
    for f1, f2 in zip(frames1, frames2, strict=True):
        all_frames_hash.update(cache.get_frame_hash(f1).encode())
        all_frames_hash.update(cache.get_frame_hash(f2).encode())

    cache_key = all_frames_hash.hexdigest()[:32]
    config = {"enable_gpu": False}
    
    # Try to get from cache
    if frames1 and frames2:
        special_key = cache.generate_cache_key(
            cache_key, cache_key, "ssimulacra2", config
        )
        
        with cache._lock:
            if special_key in cache._memory_cache:
                entry = cache._memory_cache[special_key]
                if time.time() - entry.timestamp < cache.ttl_seconds:
                    cache._memory_cache.move_to_end(special_key)
                    cache._stats.hits += 1
                    logger.debug(f"Cache hit (memory): ssimulacra2 for {len(frames1)} frames")
                    return entry.value
    
    # Calculate and cache
    from ..ssimulacra2_metrics import calculate_ssimulacra2_quality_metrics
    result = calculate_ssimulacra2_quality_metrics(frames1, frames2)
    
    # Store in cache
    if frames1 and frames2:
        from ..caching.validation_cache import ValidationResult
        
        entry = ValidationResult(
            metric_type="ssimulacra2",
            value=result,
            config_hash=hashlib.md5(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16],
            timestamp=time.time(),
            metadata={"num_frames": len(frames1)},
        )
        
        with cache._lock:
            cache._add_to_memory(special_key, entry)
            cache._store_to_disk(special_key, entry, cache_key, cache_key)
    
    return result

def integrate_validation_cache_with_metrics() -> None:
    """
    Monkey-patch the metrics module to use cached versions.
    
    This function should be called during application initialization
    to enable validation caching transparently.
    """
    if not VALIDATION_CACHE.get("enabled", True):
        logger.info("ValidationCache is disabled in configuration")
        return
    
    try:
        import sys

        from .. import metrics
        
        # Store original functions
        if not hasattr(metrics, '_original_calculate_ms_ssim'):
            metrics._original_calculate_ms_ssim = metrics.calculate_ms_ssim
        
        # Replace with cached versions
        def wrapped_ms_ssim(frame1, frame2, scales=5, use_cache=True):
            if use_cache and VALIDATION_CACHE.get("cache_ms_ssim", True):
                return calculate_ms_ssim_cached(
                    frame1, frame2, scales, use_validation_cache=True
                )
            else:
                return metrics._original_calculate_ms_ssim(
                    frame1, frame2, scales, use_cache
                )
        
        metrics.calculate_ms_ssim = wrapped_ms_ssim
        
        logger.info("ValidationCache integrated with metrics module")
        
    except Exception as e:
        logger.warning(f"Failed to integrate ValidationCache: {e}")
