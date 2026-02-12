"""
Integration module for instrumenting GifLab optimization systems with metrics.
"""

import functools
import logging
import time
from collections.abc import Callable
from typing import Any

from .decorators import MetricTracker
from .metrics_collector import get_metrics_collector

logger = logging.getLogger(__name__)


def instrument_frame_cache():
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
        def instrumented_get(self, file_path):
            start_time = time.perf_counter()
            
            result = original_get(self, file_path)
            
            duration = time.perf_counter() - start_time
            collector.record_timer(
                "cache.frame.operation.duration",
                duration,
                tags={"operation": "get"}
            )
            
            # Track hit/miss
            if result is not None:
                collector.record_counter("cache.frame.hits", 1.0)
            else:
                collector.record_counter("cache.frame.misses", 1.0)
            
            # Track memory usage
            if hasattr(self, "_memory_bytes"):
                collector.record_gauge(
                    "cache.frame.memory_usage_mb",
                    self._memory_bytes / (1024 * 1024)
                )
            
            return result
        
        FrameCache.get = instrumented_get
        
        # Wrap put method
        original_put = FrameCache.put
        
        @functools.wraps(original_put)
        def instrumented_put(self, file_path, frames):
            start_time = time.perf_counter()
            
            result = original_put(self, file_path, frames)
            
            duration = time.perf_counter() - start_time
            collector.record_timer(
                "cache.frame.operation.duration",
                duration,
                tags={"operation": "put"}
            )
            
            # Track frame count
            if frames:
                collector.record_histogram(
                    "cache.frame.entry_size",
                    len(frames),
                    tags={"type": "frame_count"}
                )
            
            return result
        
        FrameCache.put = instrumented_put
        
        # Wrap evict method if it exists
        if hasattr(FrameCache, "_evict_if_needed"):
            original_evict = FrameCache._evict_if_needed
            
            @functools.wraps(original_evict)
            def instrumented_evict(self):
                evicted = original_evict(self)
                if evicted > 0:
                    collector.record_counter("cache.frame.evictions", evicted)
                return evicted
            
            FrameCache._evict_if_needed = instrumented_evict
        
        logger.info("FrameCache instrumented with metrics")
        
    except ImportError:
        logger.debug("FrameCache not available for instrumentation")
    except Exception as e:
        logger.error(f"Error instrumenting FrameCache: {e}")


def instrument_validation_cache():
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
        def instrumented_get(self, frame1, frame2, metric_type, config=None, frame_indices=None):
            start_time = time.perf_counter()
            
            result = original_get(self, frame1, frame2, metric_type, config, frame_indices)
            
            duration = time.perf_counter() - start_time
            tags = {"metric_type": metric_type, "operation": "get"}
            
            collector.record_timer(
                "cache.validation.operation.duration",
                duration,
                tags=tags
            )
            
            # Track hit/miss
            if result is not None:
                collector.record_counter("cache.validation.hits", 1.0, tags={"metric_type": metric_type})
            else:
                collector.record_counter("cache.validation.misses", 1.0, tags={"metric_type": metric_type})
            
            return result
        
        ValidationCache.get = instrumented_get
        
        # Wrap put method
        original_put = ValidationCache.put
        
        @functools.wraps(original_put)
        def instrumented_put(self, frame1, frame2, metric_type, result, config=None, frame_indices=None):
            start_time = time.perf_counter()
            
            ret_val = original_put(self, frame1, frame2, metric_type, result, config, frame_indices)
            
            duration = time.perf_counter() - start_time
            collector.record_timer(
                "cache.validation.operation.duration",
                duration,
                tags={"metric_type": metric_type, "operation": "put"}
            )
            
            # Track result values
            if isinstance(result, int | float):
                collector.record_histogram(
                    f"validation.{metric_type}.values",
                    result
                )
            
            return ret_val
        
        ValidationCache.put = instrumented_put
        
        # Wrap get_stats method
        if hasattr(ValidationCache, "get_stats"):
            original_get_stats = ValidationCache.get_stats
            
            @functools.wraps(original_get_stats)
            def instrumented_get_stats(self):
                stats = original_get_stats(self)
                
                # Report stats as gauges
                if stats:
                    collector.record_gauge("cache.validation.hit_rate", stats.hit_rate)
                    collector.record_gauge("cache.validation.total_entries", stats.total_gets + stats.total_puts)
                    
                    if hasattr(stats, "memory_usage_mb"):
                        collector.record_gauge("cache.validation.memory_usage_mb", stats.memory_usage_mb)
                
                return stats
            
            ValidationCache.get_stats = instrumented_get_stats
        
        logger.info("ValidationCache instrumented with metrics")
        
    except ImportError:
        logger.debug("ValidationCache not available for instrumentation")
    except Exception as e:
        logger.error(f"Error instrumenting ValidationCache: {e}")


def instrument_resize_cache():
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
        
        # Instrument ResizedFrameCache
        original_get = ResizedFrameCache.get
        
        @functools.wraps(original_get)
        def instrumented_get(self, frame_hash, target_size, interpolation):
            start_time = time.perf_counter()
            
            result = original_get(self, frame_hash, target_size, interpolation)
            
            duration = time.perf_counter() - start_time
            collector.record_timer(
                "cache.resize.operation.duration",
                duration,
                tags={"operation": "get"}
            )
            
            if result is not None:
                collector.record_counter("cache.resize.hits", 1.0)
            else:
                collector.record_counter("cache.resize.misses", 1.0)
            
            # Track memory usage
            if hasattr(self, "_current_memory"):
                collector.record_gauge(
                    "cache.resize.memory_usage_mb",
                    self._current_memory / (1024 * 1024)
                )
            
            return result
        
        ResizedFrameCache.get = instrumented_get
        
        # Instrument FrameBufferPool
        if hasattr(FrameBufferPool, "get_buffer"):
            original_get_buffer = FrameBufferPool.get_buffer
            
            @functools.wraps(original_get_buffer)
            def instrumented_get_buffer(self, shape):
                result = original_get_buffer(self, shape)
                
                # Track buffer pool efficiency
                if hasattr(self, "_reuse_count") and hasattr(self, "_total_requests"):
                    if self._total_requests > 0:
                        reuse_rate = self._reuse_count / self._total_requests
                        collector.record_gauge("cache.resize.buffer_pool.reuse_rate", reuse_rate)
                
                return result
            
            FrameBufferPool.get_buffer = instrumented_get_buffer
        
        logger.info("ResizedFrameCache instrumented with metrics")
        
    except ImportError:
        logger.debug("ResizedFrameCache not available for instrumentation")
    except Exception as e:
        logger.error(f"Error instrumenting ResizedFrameCache: {e}")


def instrument_sampling():
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
        def instrumented_sample(self, frames, **kwargs):
            start_time = time.perf_counter()
            
            result = original_sample(self, frames, **kwargs)
            
            duration = time.perf_counter() - start_time
            
            if result:
                # Track sampling metrics
                collector.record_gauge(
                    "sampling.frames_sampled_ratio",
                    result.sampling_rate
                )
                
                if result.confidence_interval:
                    ci_width = result.confidence_interval[1] - result.confidence_interval[0]
                    collector.record_gauge(
                        "sampling.confidence_interval_width",
                        ci_width
                    )
                
                # Track strategy usage
                strategy_name = result.strategy_used or self.__class__.__name__
                collector.record_counter(
                    "sampling.strategy_usage",
                    1.0,
                    tags={"strategy": strategy_name}
                )
                
                # Estimate speedup
                if result.sampling_rate < 1.0:
                    speedup = 1.0 / result.sampling_rate
                    collector.record_gauge("sampling.speedup_factor", speedup)
                
                # Track timing
                collector.record_timer(
                    "sampling.operation.duration",
                    duration,
                    tags={"strategy": strategy_name}
                )
            
            return result
        
        FrameSampler.sample = instrumented_sample
        
        logger.info("Frame sampling instrumented with metrics")
        
    except ImportError:
        logger.debug("Frame sampling not available for instrumentation")
    except Exception as e:
        logger.error(f"Error instrumenting frame sampling: {e}")


def instrument_lazy_imports():
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
        def instrumented_load(self):
            start_time = time.perf_counter()
            
            result = original_load(self)
            
            duration = time.perf_counter() - start_time
            
            # Track module load time
            collector.record_timer(
                "lazy_import.load_time",
                duration,
                tags={"module": self._module_name}
            )
            
            # Track load count
            collector.record_counter(
                "lazy_import.load_count",
                1.0,
                tags={"module": self._module_name}
            )
            
            # Track fallback usage
            if result is None and self._fallback_value is not None:
                collector.record_counter(
                    "lazy_import.fallback_used",
                    1.0,
                    tags={"module": self._module_name}
                )
            
            return result
        
        LazyModule._load_module = instrumented_load
        
        logger.info("Lazy imports instrumented with metrics")
        
    except ImportError:
        logger.debug("Lazy imports not available for instrumentation")
    except Exception as e:
        logger.error(f"Error instrumenting lazy imports: {e}")


def instrument_metrics_calculation():
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
            def instrumented_calc(original_frames, compressed_frames, config=None):
                with tracker.timer("calculation.total"):
                    # Track frame counts
                    tracker.histogram("frame_count", len(original_frames), tags={"type": "original"})
                    tracker.histogram("frame_count", len(compressed_frames), tags={"type": "compressed"})
                    
                    result = original_calc(original_frames, compressed_frames, config)
                    
                    # Track quality scores
                    for metric_name, value in result.items():
                        if isinstance(value, int | float) and not metric_name.startswith("_"):
                            tracker.histogram(
                                "quality_scores",
                                value,
                                tags={"metric": metric_name}
                            )
                    
                    return result
            
            metrics.calculate_comprehensive_metrics_from_frames = instrumented_calc
        
        logger.info("Metrics calculation instrumented")
        
    except ImportError:
        logger.debug("Metrics module not available for instrumentation")
    except Exception as e:
        logger.error(f"Error instrumenting metrics calculation: {e}")


def instrument_all_systems():
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


def remove_instrumentation():
    """
    Remove instrumentation from all systems (mainly for testing).
    
    Note: This requires keeping references to original methods,
    which is not implemented in this basic version.
    """
    logger.warning("Remove instrumentation not fully implemented")
    # In a production system, we would store original method references
    # and restore them here
