"""Caching systems for GIF validation performance optimization."""

from .frame_cache import (
    CacheStats,
    FrameCache,
    FrameCacheEntry,
    get_frame_cache,
    reset_frame_cache,
)
from .metrics_integration import (
    calculate_gradient_color_cached,
    calculate_lpips_cached,
    calculate_ms_ssim_cached,
    calculate_ssim_cached,
    calculate_ssimulacra2_cached,
    integrate_validation_cache_with_metrics,
)
from .validation_cache import (
    ValidationCache,
    ValidationCacheStats,
    ValidationResult,
    get_validation_cache,
    reset_validation_cache,
)

__all__ = [
    # Frame cache
    "CacheStats",
    "FrameCache",
    "FrameCacheEntry",
    "get_frame_cache",
    "reset_frame_cache",
    # Validation cache
    "ValidationCache",
    "ValidationCacheStats",
    "ValidationResult",
    "get_validation_cache",
    "reset_validation_cache",
    # Metrics integration
    "calculate_ms_ssim_cached",
    "calculate_ssim_cached",
    "calculate_lpips_cached",
    "calculate_gradient_color_cached",
    "calculate_ssimulacra2_cached",
    "integrate_validation_cache_with_metrics",
]
