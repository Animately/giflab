"""
Thread-safe model caching for deep learning models.

This module provides a singleton cache for expensive deep learning models
like LPIPS to prevent memory leaks from repeated instantiation.
"""

import gc
import logging
import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Check for PyTorch availability
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import lpips

    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False


class LPIPSModelCache:
    """Thread-safe singleton cache for LPIPS models.

    This cache ensures that expensive LPIPS models are loaded only once
    and reused across multiple calls, preventing memory leaks.
    """

    _instance: Optional["LPIPSModelCache"] = None
    _lock = threading.Lock()
    _models: dict[str, Any] = {}
    _ref_counts: dict[str, int] = {}

    # Feature flag for enabling cache (can be disabled for debugging)
    USE_CACHE = os.getenv("GIFLAB_USE_MODEL_CACHE", "true").lower() == "true"

    def __new__(cls) -> "LPIPSModelCache":
        """Ensure only one instance exists (singleton pattern)."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def get_model(
        cls,
        net: str = "alex",
        version: str = "0.1",
        spatial: bool = False,
        device: str = "cpu",
    ) -> Any:
        """Get or create a cached LPIPS model.

        Args:
            net: Network architecture ('alex', 'vgg', 'squeeze')
            version: Model version
            spatial: Whether to use spatial mode
            device: PyTorch device ('cpu', 'cuda', etc.)

        Returns:
            Cached LPIPS model instance
        """
        if not cls.USE_CACHE:
            # Cache disabled - return new model each time (old behavior)
            logger.debug("Model cache disabled, creating new LPIPS model")
            if not LPIPS_AVAILABLE:
                return None
            model = lpips.LPIPS(net=net, version=version, spatial=spatial)
            if TORCH_AVAILABLE and device != "cpu":
                model = model.to(device)
            model.eval()
            return model

        # Generate cache key
        key = f"{net}_{version}_{spatial}_{device}"

        # Double-checked locking pattern for thread safety
        if key not in cls._models:
            with cls._lock:
                if key not in cls._models:
                    if not LPIPS_AVAILABLE:
                        logger.warning("LPIPS not available, cannot create model")
                        return None

                    logger.info(f"Creating new LPIPS model for cache key: {key}")
                    try:
                        # Create model
                        model = lpips.LPIPS(net=net, version=version, spatial=spatial)

                        # Move to appropriate device
                        if TORCH_AVAILABLE and device != "cpu":
                            if device == "cuda" and torch.cuda.is_available():
                                model = model.to(device)
                            elif device.startswith("cuda:"):
                                # Specific GPU device
                                model = model.to(device)

                        # Set to evaluation mode
                        model.eval()

                        # Store in cache
                        cls._models[key] = model
                        cls._ref_counts[key] = 0

                        logger.debug(f"Successfully cached LPIPS model: {key}")

                    except Exception as e:
                        logger.error(f"Failed to create LPIPS model: {e}")
                        return None

        # Increment reference count
        cls._ref_counts[key] = cls._ref_counts.get(key, 0) + 1
        logger.debug(f"Model cache hit for {key}, ref count: {cls._ref_counts[key]}")

        return cls._models[key]

    @classmethod
    def cleanup(cls, force: bool = False) -> None:
        """Clean up cached models to free memory.

        Args:
            force: If True, cleanup all models regardless of reference count
        """
        with cls._lock:
            if not cls._models:
                return

            logger.info(f"Cleaning up {len(cls._models)} cached models")

            # Identify models to cleanup
            models_to_remove = []
            for key, ref_count in cls._ref_counts.items():
                if force or ref_count == 0:
                    models_to_remove.append(key)

            # Remove models
            for key in models_to_remove:
                if key in cls._models:
                    model = cls._models[key]
                    del model  # Explicitly delete model
                    del cls._models[key]
                    if key in cls._ref_counts:
                        del cls._ref_counts[key]
                    logger.debug(f"Removed model from cache: {key}")

            # Force garbage collection
            gc.collect()

            # Clear PyTorch cache if available
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.debug("Cleared PyTorch CUDA cache")

    @classmethod
    def release_model(
        cls,
        net: str = "alex",
        version: str = "0.1",
        spatial: bool = False,
        device: str = "cpu",
    ) -> None:
        """Decrement reference count for a model.

        This should be called when done using a model to allow cleanup.
        """
        if not cls.USE_CACHE:
            return

        key = f"{net}_{version}_{spatial}_{device}"
        with cls._lock:
            if key in cls._ref_counts:
                cls._ref_counts[key] = max(0, cls._ref_counts[key] - 1)
                logger.debug(f"Released model {key}, ref count: {cls._ref_counts[key]}")

    @classmethod
    def get_cache_info(cls) -> dict[str, Any]:
        """Get information about the current cache state.

        Returns:
            Dictionary with cache statistics
        """
        with cls._lock:
            return {
                "enabled": cls.USE_CACHE,
                "models_cached": len(cls._models),
                "total_references": sum(cls._ref_counts.values()),
                "model_keys": list(cls._models.keys()),
                "ref_counts": dict(cls._ref_counts),
            }


@contextmanager
def lpips_model_context(
    net: str = "alex",
    version: str = "0.1",
    spatial: bool = False,
    device: str = "cpu",
) -> Iterator[Any]:
    """Context manager for LPIPS model usage with automatic cleanup.

    Example:
        with lpips_model_context() as model:
            if model:
                score = model(tensor1, tensor2)
    """
    model = None
    try:
        model = LPIPSModelCache.get_model(net, version, spatial, device)
        yield model
    finally:
        if model is not None:
            LPIPSModelCache.release_model(net, version, spatial, device)


def cleanup_model_cache(force: bool = False) -> None:
    """Convenience function to cleanup the model cache.

    Args:
        force: If True, cleanup all models regardless of reference count
    """
    LPIPSModelCache.cleanup(force=force)


def get_model_cache_info() -> dict[str, Any]:
    """Get current model cache statistics.

    Returns:
        Dictionary with cache information
    """
    return LPIPSModelCache.get_cache_info()
