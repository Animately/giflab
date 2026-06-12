"""
Integration module for instrumenting GifLab optimization systems with metrics.

All instrumented wrappers are ``(*args, **kwargs)`` passthroughs: they must
never restate the wrapped method's signature, because signature drift between
a wrapper and the real method turns instrumentation into a TypeError factory
(bug class fixed 2026-06-12; pinned by
tests/functional/test_monitoring_instrumentation.py).
"""

import functools
import logging
import time
from typing import Any

from .decorators import MetricTracker
from .metrics_collector import get_metrics_collector

logger = logging.getLogger(__name__)


def instrument_frame_cache() -> None:
    """
    Instrument FrameCache with performance metrics.

    Metrics collected:
    - cache.frame.hits/misses
    - cache.frame.evictions
    - cache.frame.memory_usage_mb
    - cache.frame.disk_usage_mb
    - cache.frame.operation.duration
    """
    try:
        from ..caching.frame_cache import FrameCache

        collector = get_metrics_collector()

        # Wrap get method
        original_get = FrameCache.get

        @functools.wraps(original_get)
        def instrumented_get(*args: Any, **kwargs: Any) -> Any:
            start_time = time.perf_counter()

            result = original_get(*args, **kwargs)

            duration = time.perf_counter() - start_time
            collector.record_timer(
                "cache.frame.operation.duration", duration, tags={"operation": "get"}
            )

            # Track hit/miss
            if result is not None:
                collector.record_counter("cache.frame.hits", 1.0)
            else:
                collector.record_counter("cache.frame.misses", 1.0)

            # Track memory usage
            cache_self = args[0] if args else None
            if cache_self is not None and hasattr(cache_self, "_memory_bytes"):
                collector.record_gauge(
                    "cache.frame.memory_usage_mb",
                    cache_self._memory_bytes / (1024 * 1024),
                )

            return result

        # Runtime class patching is invisible to the type system; deliberate.
        FrameCache.get = instrumented_get  # type: ignore[method-assign]

        # Wrap put method
        original_put = FrameCache.put

        @functools.wraps(original_put)
        def instrumented_put(*args: Any, **kwargs: Any) -> Any:
            start_time = time.perf_counter()

            result = original_put(*args, **kwargs)

            duration = time.perf_counter() - start_time
            collector.record_timer(
                "cache.frame.operation.duration", duration, tags={"operation": "put"}
            )

            # Track frame count (signature: put(self, file_path, frames, ...))
            frames = kwargs.get("frames", args[2] if len(args) > 2 else None)
            if frames:
                collector.record_histogram(
                    "cache.frame.entry_size", len(frames), tags={"type": "frame_count"}
                )

            return result

        # Runtime class patching is invisible to the type system; deliberate.
        FrameCache.put = instrumented_put  # type: ignore[method-assign]

        # Wrap evict method if it exists
        if hasattr(FrameCache, "_evict_if_needed"):
            original_evict = FrameCache._evict_if_needed

            @functools.wraps(original_evict)
            def instrumented_evict(*args: Any, **kwargs: Any) -> Any:
                evicted = original_evict(*args, **kwargs)
                if evicted > 0:
                    collector.record_counter("cache.frame.evictions", evicted)
                return evicted

            # Runtime class patching (no ignore needed: inside the hasattr
            # guard mypy types the attribute as Any).
            FrameCache._evict_if_needed = instrumented_evict

        logger.info("FrameCache instrumented with metrics")

    except ImportError:
        logger.debug("FrameCache not available for instrumentation")
    except Exception as e:
        logger.error(f"Error instrumenting FrameCache: {e}")


def instrument_validation_cache() -> None:
    """
    Instrument ValidationCache with performance metrics.

    Metrics collected:
    - cache.validation.hits/misses (tagged by metric_type)
    - cache.validation.memory_usage_mb
    - cache.validation.operation.duration
    - cache.validation.entries_by_type
    """
    try:
        from ..caching.validation_cache import ValidationCache

        collector = get_metrics_collector()

        # Wrap get method
        original_get = ValidationCache.get

        @functools.wraps(original_get)
        def instrumented_get(*args: Any, **kwargs: Any) -> Any:
            start_time = time.perf_counter()

            result = original_get(*args, **kwargs)

            duration = time.perf_counter() - start_time
            # Signature: get(self, frame1, frame2, metric_type, ...)
            metric_type = kwargs.get(
                "metric_type", args[3] if len(args) > 3 else "unknown"
            )
            tags = {"metric_type": metric_type, "operation": "get"}

            collector.record_timer(
                "cache.validation.operation.duration", duration, tags=tags
            )

            # Track hit/miss
            if result is not None:
                collector.record_counter(
                    "cache.validation.hits", 1.0, tags={"metric_type": metric_type}
                )
            else:
                collector.record_counter(
                    "cache.validation.misses", 1.0, tags={"metric_type": metric_type}
                )

            return result

        # Runtime class patching is invisible to the type system; deliberate.
        ValidationCache.get = instrumented_get  # type: ignore[method-assign]

        # Wrap put method
        original_put = ValidationCache.put

        @functools.wraps(original_put)
        def instrumented_put(*args: Any, **kwargs: Any) -> Any:
            start_time = time.perf_counter()

            ret_val = original_put(*args, **kwargs)

            duration = time.perf_counter() - start_time
            # Signature: put(self, frame1, frame2, metric_type, value, ...)
            metric_type = kwargs.get(
                "metric_type", args[3] if len(args) > 3 else "unknown"
            )
            value = kwargs.get("value", args[4] if len(args) > 4 else None)
            collector.record_timer(
                "cache.validation.operation.duration",
                duration,
                tags={"metric_type": metric_type, "operation": "put"},
            )

            # Track result values
            if isinstance(value, int | float):
                collector.record_histogram(f"validation.{metric_type}.values", value)

            return ret_val

        # Runtime class patching is invisible to the type system; deliberate.
        ValidationCache.put = instrumented_put  # type: ignore[method-assign]

        # Wrap get_stats method
        if hasattr(ValidationCache, "get_stats"):
            original_get_stats = ValidationCache.get_stats

            @functools.wraps(original_get_stats)
            def instrumented_get_stats(*args: Any, **kwargs: Any) -> Any:
                stats = original_get_stats(*args, **kwargs)

                # Report stats as gauges. ValidationCacheStats fields are
                # hits/misses/memory_entries/disk_entries/memory_bytes/
                # disk_bytes/evictions (+ hit_rate property).
                if stats:
                    collector.record_gauge("cache.validation.hit_rate", stats.hit_rate)
                    # "total_entries" reports entries actually resident
                    # (memory + disk). Operations are already emitted honestly
                    # as the cache.validation.hits/misses counters above.
                    collector.record_gauge(
                        "cache.validation.total_entries",
                        stats.memory_entries + stats.disk_entries,
                    )
                    collector.record_gauge(
                        "cache.validation.memory_usage_mb",
                        stats.memory_bytes / (1024 * 1024),
                    )

                return stats

            # Runtime class patching is invisible to the type system; deliberate.
            ValidationCache.get_stats = instrumented_get_stats  # type: ignore[method-assign]

        logger.info("ValidationCache instrumented with metrics")

    except ImportError:
        logger.debug("ValidationCache not available for instrumentation")
    except Exception as e:
        logger.error(f"Error instrumenting ValidationCache: {e}")


def instrument_resize_cache() -> None:
    """
    Instrument ResizedFrameCache with performance metrics.

    Metrics collected:
    - cache.resize.hits/misses
    - cache.resize.buffer_pool.reuse_rate
    - cache.resize.operation.duration
    - cache.resize.memory_usage_mb
    """
    try:
        from ..caching.resized_frame_cache import FrameBufferPool, ResizedFrameCache

        collector = get_metrics_collector()

        # Instrument ResizedFrameCache.
        # Type-erase the original binding: instrumentation wrappers must not
        # assume the wrapped signature/return type (ResizedFrameCache.get
        # returns ndarray, never None -- without erasure mypy proves the
        # miss-counter branch unreachable).
        original_get: Any = ResizedFrameCache.get

        @functools.wraps(original_get)
        def instrumented_get(*args: Any, **kwargs: Any) -> Any:
            start_time = time.perf_counter()

            result = original_get(*args, **kwargs)

            duration = time.perf_counter() - start_time
            collector.record_timer(
                "cache.resize.operation.duration", duration, tags={"operation": "get"}
            )

            if result is not None:
                collector.record_counter("cache.resize.hits", 1.0)
            else:
                collector.record_counter("cache.resize.misses", 1.0)

            # Track memory usage
            cache_self = args[0] if args else None
            if cache_self is not None and hasattr(cache_self, "_current_memory"):
                collector.record_gauge(
                    "cache.resize.memory_usage_mb",
                    cache_self._current_memory / (1024 * 1024),
                )

            return result

        # Runtime class patching is invisible to the type system; deliberate.
        ResizedFrameCache.get = instrumented_get  # type: ignore[method-assign]

        # Instrument FrameBufferPool
        if hasattr(FrameBufferPool, "get_buffer"):
            original_get_buffer = FrameBufferPool.get_buffer

            @functools.wraps(original_get_buffer)
            def instrumented_get_buffer(*args: Any, **kwargs: Any) -> Any:
                result = original_get_buffer(*args, **kwargs)

                # Track buffer pool efficiency from the pool's real counters
                # (same formula as FrameBufferPool.get_stats).
                pool_self = args[0] if args else None
                if pool_self is not None and hasattr(pool_self, "_stats"):
                    pool_stats = pool_self._stats
                    reuse_rate = pool_stats["reuses"] / max(
                        1, pool_stats["allocations"] + pool_stats["reuses"]
                    )
                    collector.record_gauge(
                        "cache.resize.buffer_pool.reuse_rate", reuse_rate
                    )

                return result

            # Runtime class patching is invisible to the type system; deliberate.
            FrameBufferPool.get_buffer = instrumented_get_buffer  # type: ignore[method-assign]

        logger.info("ResizedFrameCache instrumented with metrics")

    except ImportError:
        logger.debug("ResizedFrameCache not available for instrumentation")
    except Exception as e:
        logger.error(f"Error instrumenting ResizedFrameCache: {e}")


def instrument_sampling() -> None:
    """
    Instrument frame sampling system with performance metrics.

    Metrics collected:
    - sampling.frames_sampled_ratio
    - sampling.confidence_interval_width
    - sampling.strategy_usage (counter by strategy)
    - sampling.speedup_factor
    """
    try:
        from ..sampling.frame_sampler import FrameSampler

        collector = get_metrics_collector()

        # Wrap sample method
        original_sample = FrameSampler.sample

        @functools.wraps(original_sample)
        def instrumented_sample(*args: Any, **kwargs: Any) -> Any:
            start_time = time.perf_counter()

            result = original_sample(*args, **kwargs)

            duration = time.perf_counter() - start_time

            if result:
                # Track sampling metrics
                collector.record_gauge(
                    "sampling.frames_sampled_ratio", result.sampling_rate
                )

                if result.confidence_interval:
                    ci_width = (
                        result.confidence_interval[1] - result.confidence_interval[0]
                    )
                    collector.record_gauge(
                        "sampling.confidence_interval_width", ci_width
                    )

                # Track strategy usage
                sampler_self = args[0] if args else None
                strategy_name = result.strategy_used or type(sampler_self).__name__
                collector.record_counter(
                    "sampling.strategy_usage", 1.0, tags={"strategy": strategy_name}
                )

                # Estimate speedup
                if result.sampling_rate < 1.0:
                    speedup = 1.0 / result.sampling_rate
                    collector.record_gauge("sampling.speedup_factor", speedup)

                # Track timing
                collector.record_timer(
                    "sampling.operation.duration",
                    duration,
                    tags={"strategy": strategy_name},
                )

            return result

        # Runtime class patching is invisible to the type system; deliberate.
        FrameSampler.sample = instrumented_sample  # type: ignore[method-assign]

        logger.info("Frame sampling instrumented with metrics")

    except ImportError:
        logger.debug("Frame sampling not available for instrumentation")
    except Exception as e:
        logger.error(f"Error instrumenting frame sampling: {e}")


def instrument_lazy_imports() -> None:
    """
    Instrument lazy import system with performance metrics.

    Metrics collected:
    - lazy_import.load_time_ms (by module)
    - lazy_import.load_count (by module)
    - lazy_import.fallback_used (counter)
    """
    try:
        from ..lazy_imports import LazyModule

        collector = get_metrics_collector()

        # Wrap _load_module method
        original_load = LazyModule._load_module

        @functools.wraps(original_load)
        def instrumented_load(*args: Any, **kwargs: Any) -> Any:
            start_time = time.perf_counter()

            result = original_load(*args, **kwargs)

            duration = time.perf_counter() - start_time

            module_self = args[0] if args else None
            module_name = getattr(module_self, "_module_name", "unknown")

            # Track module load time
            collector.record_timer(
                "lazy_import.load_time", duration, tags={"module": module_name}
            )

            # Track load count
            collector.record_counter(
                "lazy_import.load_count", 1.0, tags={"module": module_name}
            )

            # Track fallback usage
            if (
                result is None
                and getattr(module_self, "_fallback_value", None) is not None
            ):
                collector.record_counter(
                    "lazy_import.fallback_used", 1.0, tags={"module": module_name}
                )

            return result

        # Runtime class patching is invisible to the type system; deliberate.
        LazyModule._load_module = instrumented_load  # type: ignore[method-assign]

        logger.info("Lazy imports instrumented with metrics")

    except ImportError:
        logger.debug("Lazy imports not available for instrumentation")
    except Exception as e:
        logger.error(f"Error instrumenting lazy imports: {e}")


def instrument_metrics_calculation() -> None:
    """
    Instrument core metrics calculation functions.

    Metrics collected:
    - metrics.calculation.duration (by metric type)
    - metrics.frame_count (histogram)
    - metrics.quality_scores (histogram by metric)
    """
    try:
        from .. import metrics

        get_metrics_collector()
        tracker = MetricTracker("metrics")

        # Instrument calculate_comprehensive_metrics_from_frames
        if hasattr(metrics, "calculate_comprehensive_metrics_from_frames"):
            original_calc = metrics.calculate_comprehensive_metrics_from_frames

            @functools.wraps(original_calc)
            def instrumented_calc(*args: Any, **kwargs: Any) -> Any:
                # Signature: (original_frames, compressed_frames, ...)
                original_frames = kwargs.get(
                    "original_frames", args[0] if len(args) > 0 else []
                )
                compressed_frames = kwargs.get(
                    "compressed_frames", args[1] if len(args) > 1 else []
                )
                with tracker.timer("calculation.total"):
                    # Track frame counts
                    tracker.histogram(
                        "frame_count", len(original_frames), tags={"type": "original"}
                    )
                    tracker.histogram(
                        "frame_count",
                        len(compressed_frames),
                        tags={"type": "compressed"},
                    )

                    result = original_calc(*args, **kwargs)

                    # Track quality scores
                    for metric_name, value in result.items():
                        if isinstance(
                            value, int | float
                        ) and not metric_name.startswith("_"):
                            tracker.histogram(
                                "quality_scores", value, tags={"metric": metric_name}
                            )

                    return result

            # Deliberate runtime module patching for instrumentation.
            metrics.calculate_comprehensive_metrics_from_frames = instrumented_calc

        logger.info("Metrics calculation instrumented")

    except ImportError:
        logger.debug("Metrics module not available for instrumentation")
    except Exception as e:
        logger.error(f"Error instrumenting metrics calculation: {e}")


def instrument_all_systems() -> None:
    """
    Instrument all GifLab optimization systems with metrics.

    This is the main entry point for enabling monitoring.
    """
    logger.info("Instrumenting all GifLab optimization systems")

    # Check if monitoring is enabled
    try:
        from ..config import MONITORING

        if not MONITORING.get("enabled", True):
            logger.info("Monitoring disabled in configuration")
            return
    except ImportError:
        pass

    # Instrument each system
    instrument_frame_cache()
    instrument_validation_cache()
    instrument_resize_cache()
    instrument_sampling()
    instrument_lazy_imports()
    instrument_metrics_calculation()

    # Initialize memory monitoring if enabled
    try:
        systems_config = MONITORING.get("systems", {})
        if systems_config.get("memory_pressure", True):
            from .memory_integration import initialize_memory_monitoring

            if initialize_memory_monitoring():
                logger.info("Memory monitoring integration initialized")
            else:
                logger.warning("Memory monitoring initialization failed")
    except Exception as e:
        logger.error(f"Error initializing memory monitoring: {e}")

    logger.info("All systems instrumented successfully")


def remove_instrumentation() -> None:
    """
    Remove instrumentation from all systems (mainly for testing).

    Note: This requires keeping references to original methods,
    which is not implemented in this basic version.
    """
    logger.warning("Remove instrumentation not fully implemented")
    # In a production system, we would store original method references
    # and restore them here
