"""
Decorators for easy metric instrumentation.
"""

import functools
import time
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any, Optional

from .metrics_collector import get_metrics_collector


def track_timing(
    metric_name: str = None,
    tags: dict[str, str] = None,
    include_args: bool = False,
):
    """
    Decorator to track function execution time.
    
    Args:
        metric_name: Custom metric name (defaults to function name)
        tags: Additional tags to add to metric
        include_args: Whether to include function args as tags
    
    Example:
        @track_timing(metric_name="cache.lookup", tags={"cache": "frame"})
        def get_frame(key):
            return cache[key]
    """
    def decorator(func: Callable) -> Callable:
        name = metric_name or f"{func.__module__}.{func.__name__}.duration"
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            collector = get_metrics_collector()
            
            # Build tags
            metric_tags = tags.copy() if tags else {}
            if include_args and args:
                metric_tags["arg0"] = str(args[0])[:50]  # Limit tag value length
            
            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                metric_tags["status"] = "success"
                return result
            except Exception as e:
                metric_tags["status"] = "error"
                metric_tags["error_type"] = type(e).__name__
                raise
            finally:
                duration = time.perf_counter() - start_time
                collector.record_timer(name, duration, metric_tags)
        
        return wrapper
    return decorator

def track_counter(
    metric_name: str = None,
    value: float = 1.0,
    tags: dict[str, str] = None,
    on_success_only: bool = True,
):
    """
    Decorator to track function call counts.
    
    Args:
        metric_name: Custom metric name (defaults to function name)
        value: Counter increment value
        tags: Additional tags to add to metric
        on_success_only: Only increment on successful execution
    
    Example:
        @track_counter(metric_name="cache.hits", tags={"cache": "frame"})
        def cache_hit():
            return cached_value
    """
    def decorator(func: Callable) -> Callable:
        name = metric_name or f"{func.__module__}.{func.__name__}.calls"
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            collector = get_metrics_collector()
            metric_tags = tags.copy() if tags else {}
            
            try:
                result = func(*args, **kwargs)
                metric_tags["status"] = "success"
                collector.record_counter(name, value, metric_tags)
                return result
            except Exception as e:
                metric_tags["status"] = "error"
                metric_tags["error_type"] = type(e).__name__
                if not on_success_only:
                    collector.record_counter(name, value, metric_tags)
                raise
        
        return wrapper
    return decorator

def track_gauge(
    metric_name: str,
    value_func: Callable[[Any], float],
    tags: dict[str, str] = None,
):
    """
    Decorator to track gauge values from function results.
    
    Args:
        metric_name: Metric name for gauge
        value_func: Function to extract gauge value from result
        tags: Additional tags to add to metric
    
    Example:
        @track_gauge(
            metric_name="cache.size",
            value_func=lambda result: len(result),
            tags={"cache": "frame"}
        )
        def get_cache_contents():
            return cache.items()
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            collector = get_metrics_collector()
            
            result = func(*args, **kwargs)
            
            try:
                gauge_value = value_func(result)
                collector.record_gauge(metric_name, gauge_value, tags)
            except Exception:
                pass  # Silently ignore extraction errors
            
            return result
        
        return wrapper
    return decorator

def track_histogram(
    metric_name: str,
    value_func: Callable[[Any], float],
    tags: dict[str, str] = None,
):
    """
    Decorator to track histogram values from function results.
    
    Args:
        metric_name: Metric name for histogram
        value_func: Function to extract value from result
        tags: Additional tags to add to metric
    
    Example:
        @track_histogram(
            metric_name="frame.size",
            value_func=lambda frame: frame.nbytes,
            tags={"operation": "extract"}
        )
        def extract_frame(gif_path):
            return load_frame(gif_path)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            collector = get_metrics_collector()
            
            result = func(*args, **kwargs)
            
            try:
                hist_value = value_func(result)
                collector.record_histogram(metric_name, hist_value, tags)
            except Exception:
                pass  # Silently ignore extraction errors
            
            return result
        
        return wrapper
    return decorator

@contextmanager
def timer_context(metric_name: str, tags: dict[str, str] = None):
    """
    Context manager for timing code blocks.
    
    Example:
        with timer_context("processing.batch", tags={"batch_size": "100"}):
            process_batch(items)
    """
    collector = get_metrics_collector()
    start_time = time.perf_counter()
    
    try:
        yield
    finally:
        duration = time.perf_counter() - start_time
        collector.record_timer(metric_name, duration, tags)

@contextmanager
def counter_context(
    metric_name: str,
    value: float = 1.0,
    tags: dict[str, str] = None,
    on_success_only: bool = True,
):
    """
    Context manager for counting operations.
    
    Example:
        with counter_context("files.processed", tags={"type": "gif"}):
            process_file(path)
    """
    collector = get_metrics_collector()
    success = False
    
    try:
        yield
        success = True
    finally:
        if success or not on_success_only:
            metric_tags = tags.copy() if tags else {}
            metric_tags["status"] = "success" if success else "error"
            collector.record_counter(metric_name, value, metric_tags)

class MetricTracker:
    """
    Class-based metric tracker for more complex instrumentation.
    
    Example:
        tracker = MetricTracker("validation")
        
        with tracker.timer("total"):
            for frame in frames:
                with tracker.timer("frame", tags={"index": str(i)}):
                    result = validate_frame(frame)
                    tracker.histogram("quality", result.quality)
                    tracker.counter("processed")
    """
    
    def __init__(self, prefix: str, tags: dict[str, str] = None):
        """
        Initialize metric tracker.
        
        Args:
            prefix: Prefix for all metrics
            tags: Default tags for all metrics
        """
        self.prefix = prefix
        self.default_tags = tags or {}
        self.collector = get_metrics_collector()
    
    def timer(self, name: str, tags: dict[str, str] = None):
        """Create timer context."""
        metric_name = f"{self.prefix}.{name}"
        metric_tags = {**self.default_tags, **(tags or {})}
        return timer_context(metric_name, metric_tags)
    
    def counter(
        self,
        name: str,
        value: float = 1.0,
        tags: dict[str, str] = None
    ):
        """Record counter."""
        metric_name = f"{self.prefix}.{name}"
        metric_tags = {**self.default_tags, **(tags or {})}
        self.collector.record_counter(metric_name, value, metric_tags)
    
    def gauge(self, name: str, value: float, tags: dict[str, str] = None):
        """Record gauge."""
        metric_name = f"{self.prefix}.{name}"
        metric_tags = {**self.default_tags, **(tags or {})}
        self.collector.record_gauge(metric_name, value, metric_tags)
    
    def histogram(self, name: str, value: float, tags: dict[str, str] = None):
        """Record histogram."""
        metric_name = f"{self.prefix}.{name}"
        metric_tags = {**self.default_tags, **(tags or {})}
        self.collector.record_histogram(metric_name, value, metric_tags)
    
    def record_error(self, error: Exception, tags: dict[str, str] = None):
        """Record error occurrence."""
        metric_tags = {
            **self.default_tags,
            **(tags or {}),
            "error_type": type(error).__name__,
        }
        self.collector.record_counter(f"{self.prefix}.errors", 1.0, metric_tags)

def track_cache_operation(cache_name: str):
    """
    Specialized decorator for cache operations.
    
    Automatically tracks hits, misses, and timing.
    
    Example:
        @track_cache_operation("frame_cache")
        def get(self, key):
            # Returns (value, hit) tuple
            if key in self.cache:
                return self.cache[key], True
            return None, False
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            collector = get_metrics_collector()
            
            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                
                # Expect (value, hit) tuple for cache operations
                if isinstance(result, tuple) and len(result) == 2:
                    value, hit = result
                    metric_name = f"cache.{cache_name}.{'hits' if hit else 'misses'}"
                    collector.record_counter(metric_name, 1.0)
                    
                    # Return just the value
                    return value
                else:
                    return result
                    
            finally:
                duration = time.perf_counter() - start_time
                collector.record_timer(
                    f"cache.{cache_name}.operation.duration",
                    duration,
                    tags={"operation": func.__name__}
                )
        
        return wrapper
    return decorator
