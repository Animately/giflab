"""
Memory monitoring integration for GifLab caching systems.

This module integrates the memory monitoring system with existing caches,
providing automatic memory tracking, pressure detection, and eviction policies.
"""

import logging
import time
from collections.abc import Callable
from typing import Optional

from ..config import MONITORING
from .memory_monitor import (
    MemoryPressureLevel,
    get_cache_memory_tracker,
    get_memory_pressure_manager,
    get_system_memory_monitor,
    start_memory_monitoring,
)
from .metrics_collector import get_metrics_collector

logger = logging.getLogger(__name__)


class MemoryPressureIntegration:
    """Integrates memory pressure monitoring with existing monitoring systems."""

    def __init__(self):
        self._initialized = False
        self._last_eviction_time = 0.0
        self._eviction_cooldown = 30.0  # seconds

    def initialize(self) -> bool:
        """Initialize memory pressure monitoring integration."""
        if self._initialized:
            return True

        try:
            # Check if memory pressure monitoring is enabled
            config = MONITORING.get("memory_pressure", {})
            if not config.get("enabled", True):
                logger.info("Memory pressure monitoring disabled via configuration")
                return False

            # Initialize monitoring components
            get_system_memory_monitor()
            get_cache_memory_tracker()
            pressure_manager = get_memory_pressure_manager()

            # Configure eviction cooldown from config
            hysteresis = config.get("hysteresis", {})
            self._eviction_cooldown = hysteresis.get("eviction_cooldown", 30.0)

            # Start system monitoring
            start_memory_monitoring()

            # Set up periodic pressure checking if enabled
            if config.get("auto_eviction", True):
                self._setup_pressure_checking(pressure_manager)

            self._initialized = True
            logger.info("Memory pressure monitoring integration initialized")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize memory pressure monitoring: {e}")
            return False

    def _setup_pressure_checking(self, pressure_manager) -> None:
        """Set up periodic memory pressure checking."""
        # This would typically integrate with the existing monitoring loop
        # For now, we'll register the pressure manager for use by other systems
        logger.debug("Memory pressure checking configured")

    def check_and_handle_pressure(self) -> bool:
        """Check memory pressure and handle eviction if needed."""
        if not self._initialized:
            return False

        try:
            # Check cooldown period
            current_time = time.time()
            if current_time - self._last_eviction_time < self._eviction_cooldown:
                return False

            pressure_manager = get_memory_pressure_manager()
            should_evict, target_mb = pressure_manager.check_memory_pressure()

            if should_evict and target_mb and target_mb > 0:
                logger.warning(
                    f"Executing memory pressure eviction: target={target_mb:.1f}MB"
                )

                freed_mb = pressure_manager.execute_eviction(target_mb)
                self._last_eviction_time = current_time

                # Record metrics
                collector = get_metrics_collector()
                collector.record_counter("memory.pressure.evictions", 1)
                collector.record_histogram("memory.pressure.freed_mb", freed_mb)

                logger.info(
                    f"Memory pressure eviction completed: freed={freed_mb:.1f}MB"
                )
                return True

        except Exception as e:
            logger.error(f"Error handling memory pressure: {e}")

        return False


def instrument_cache_with_memory_tracking(
    cache_type: str,
    get_size_callback: Callable[[], float],
    eviction_callback: Callable[[float], float] | None = None,
) -> None:
    """
    Instrument a cache with memory tracking and eviction support.

    Args:
        cache_type: Type identifier for the cache (e.g., "frame_cache")
        get_size_callback: Function that returns current cache size in MB
        eviction_callback: Function that evicts memory and returns MB freed
    """
    try:
        cache_tracker = get_cache_memory_tracker()
        pressure_manager = get_memory_pressure_manager()

        # Register eviction callback if provided
        if eviction_callback:
            pressure_manager.register_eviction_callback(cache_type, eviction_callback)

        # Set up periodic size updates
        def update_cache_size():
            try:
                size_mb = get_size_callback()
                cache_tracker.update_cache_size(cache_type, size_mb)

                # Also record as metric
                collector = get_metrics_collector()
                collector.record_gauge(f"cache.{cache_type}.memory_usage_mb", size_mb)

            except Exception as e:
                logger.error(f"Error updating cache size for {cache_type}: {e}")

        # Update size immediately
        update_cache_size()

        logger.debug(f"Cache {cache_type} instrumented with memory tracking")

    except Exception as e:
        logger.error(
            f"Failed to instrument cache {cache_type} with memory tracking: {e}"
        )


def instrument_frame_cache_memory() -> None:
    """Instrument FrameCache with memory tracking and eviction."""
    try:
        # Import conditionally to avoid circular dependencies
        from ..caching.frame_cache import get_frame_cache
        from ..config import ENABLE_EXPERIMENTAL_CACHING

        if not ENABLE_EXPERIMENTAL_CACHING:
            logger.debug("Frame cache not enabled, skipping memory instrumentation")
            return

        frame_cache = get_frame_cache()

        def get_frame_cache_size() -> float:
            """Get frame cache memory usage in MB."""
            if hasattr(frame_cache, "_memory_bytes"):
                return frame_cache._memory_bytes / (1024 * 1024)
            return 0.0

        def evict_frame_cache(target_mb: float) -> float:
            """Evict from frame cache and return MB freed."""
            initial_size = get_frame_cache_size()

            # Evict entries until target is met or cache is empty
            evicted_count = 0
            while get_frame_cache_size() > (initial_size - target_mb):
                if hasattr(frame_cache, "_evict_oldest"):
                    evicted = frame_cache._evict_oldest()
                    if not evicted:
                        break
                    evicted_count += 1
                else:
                    # Fallback: clear entire cache if no selective eviction
                    if hasattr(frame_cache, "clear"):
                        frame_cache.clear()
                    break

            final_size = get_frame_cache_size()
            freed_mb = initial_size - final_size

            logger.info(
                f"Frame cache eviction: freed={freed_mb:.1f}MB, evicted={evicted_count} entries"
            )
            return freed_mb

        instrument_cache_with_memory_tracking(
            "frame_cache", get_frame_cache_size, evict_frame_cache
        )

    except ImportError:
        logger.debug("Frame cache not available for memory instrumentation")
    except Exception as e:
        logger.error(f"Error instrumenting frame cache memory: {e}")


def instrument_all_caches_memory() -> None:
    """Instrument all available caches with memory tracking."""
    logger.info("Instrumenting caches with memory tracking")

    # Instrument frame cache
    instrument_frame_cache_memory()

    # Add other cache types as they become available
    # instrument_resize_cache_memory()
    # instrument_validation_cache_memory()

    logger.info("Cache memory instrumentation completed")


def setup_memory_pressure_alerts() -> None:
    """Set up memory pressure alerts with the existing alerting system."""
    try:
        from .alerting import AlertRule, get_alert_manager

        alert_manager = get_alert_manager()
        config = MONITORING.get("memory_pressure", {})
        thresholds = config.get("thresholds", {})

        # Create system memory pressure alert rule
        system_memory_rule = AlertRule(
            name="system.memory.pressure",
            metric_pattern="system.memory.usage_percent",
            condition=lambda x: x >= thresholds.get("warning", 0.70),
            warning_threshold=thresholds.get("warning", 0.70),
            critical_threshold=thresholds.get("critical", 0.80),
            message_template="System memory usage {value:.1%} exceeds threshold",
            tags={"source": "memory_monitoring", "type": "system"},
        )
        alert_manager.add_rule(system_memory_rule)

        # Create process memory pressure alert rule
        process_memory_rule = AlertRule(
            name="process.memory.pressure",
            metric_pattern="process.memory.usage_mb",
            condition=lambda x: x >= 500.0,  # 500MB process memory threshold
            warning_threshold=500.0,
            critical_threshold=1000.0,
            message_template="Process memory usage {value:.0f}MB exceeds threshold",
            tags={"source": "memory_monitoring", "type": "process"},
        )
        alert_manager.add_rule(process_memory_rule)

        logger.info("Memory pressure alerts configured")

    except Exception as e:
        logger.error(f"Failed to setup memory pressure alerts: {e}")


def integrate_memory_monitoring_with_metrics() -> None:
    """Integrate memory monitoring with the existing metrics system."""
    try:
        system_monitor = get_system_memory_monitor()
        collector = get_metrics_collector()

        def record_memory_metrics():
            """Record current memory statistics as metrics."""
            stats = system_monitor.get_current_stats()
            if stats:
                collector.record_gauge(
                    "system.memory.usage_percent", stats.system_memory_percent
                )
                collector.record_gauge("system.memory.usage_mb", stats.system_memory_mb)
                collector.record_gauge(
                    "system.memory.available_mb", stats.system_available_mb
                )
                collector.record_gauge(
                    "process.memory.usage_mb", stats.process_memory_mb
                )
                collector.record_gauge(
                    "process.memory.usage_percent", stats.process_memory_percent
                )

                # Record pressure level as numeric value
                pressure_values = {
                    MemoryPressureLevel.NORMAL: 0,
                    MemoryPressureLevel.WARNING: 1,
                    MemoryPressureLevel.CRITICAL: 2,
                    MemoryPressureLevel.EMERGENCY: 3,
                }
                collector.record_gauge(
                    "system.memory.pressure_level",
                    pressure_values[stats.pressure_level],
                )

        # Record metrics immediately
        record_memory_metrics()

        logger.debug("Memory monitoring integrated with metrics system")

    except Exception as e:
        logger.error(f"Failed to integrate memory monitoring with metrics: {e}")


# Global integration instance
_memory_integration: MemoryPressureIntegration | None = None


def get_memory_integration() -> MemoryPressureIntegration:
    """Get singleton memory pressure integration instance."""
    global _memory_integration
    if _memory_integration is None:
        _memory_integration = MemoryPressureIntegration()
    return _memory_integration


def initialize_memory_monitoring() -> bool:
    """Initialize complete memory monitoring integration."""
    try:
        # Initialize core integration
        integration = get_memory_integration()
        if not integration.initialize():
            return False

        # Set up cache instrumentation
        instrument_all_caches_memory()

        # Set up alerts
        setup_memory_pressure_alerts()

        # Integrate with metrics
        integrate_memory_monitoring_with_metrics()

        logger.info("Memory monitoring fully initialized")
        return True

    except Exception as e:
        logger.error(f"Failed to initialize memory monitoring: {e}")
        return False
