"""System-wide memory monitoring for GifLab caching and performance systems.

This module provides comprehensive memory monitoring infrastructure designed to prevent
memory exhaustion while maximizing cache effectiveness. It includes cross-platform
memory tracking, automatic pressure detection, and intelligent cache eviction.

Key Components:
    SystemMemoryMonitor: Cross-platform system and process memory monitoring
    MemoryPressureManager: Automatic cache eviction based on memory pressure
    CacheMemoryTracker: Centralized tracking of cache memory usage
    ConservativeEvictionPolicy: Smart eviction policy with hysteresis

Architecture Overview:
    The memory monitoring system uses a three-tier approach:
    1. Collection: SystemMemoryMonitor gathers real-time memory statistics
    2. Analysis: MemoryPressureManager evaluates pressure using configurable policies
    3. Action: Automatic cache eviction via registered callbacks in priority order

Memory Pressure Levels:
    - NORMAL (< 70%): No action required, optimal operating conditions
    - WARNING (70-80%): Proactive monitoring, prepare for potential eviction
    - CRITICAL (80-95%): Active cache eviction to prevent performance degradation
    - EMERGENCY (> 95%): Aggressive eviction to prevent system instability

Eviction Strategy:
    Cache eviction follows a priority order designed to preserve the most valuable data:
    1. validation_cache: Fast to rebuild, highest eviction priority
    2. resize_cache: Medium rebuild cost, medium priority
    3. frame_cache: Expensive to rebuild, lowest eviction priority (most valuable)

Cross-Platform Support:
    Uses psutil when available for accurate memory statistics on all platforms.
    Provides graceful fallback behavior when psutil is unavailable, ensuring
    the system remains functional with limited monitoring capabilities.

Thread Safety:
    All components are designed for thread-safe operation in concurrent environments.
    Internal locking ensures consistent state during memory pressure events.

Integration Points:
    - Cache systems register eviction callbacks for automatic pressure management
    - Monitoring systems receive memory pressure alerts and statistics
    - CLI tools provide real-time memory status and diagnostic information
    - Effectiveness monitoring correlates memory pressure with cache performance

Performance Characteristics:
    - Memory collection: < 1ms per operation with psutil
    - Pressure evaluation: < 0.1ms per check
    - Eviction coordination: < 1ms overhead (excluding cache-specific eviction time)
    - Background monitoring: Configurable interval (default 5s) with minimal CPU impact

Configuration:
    Memory monitoring behavior is controlled via configuration in giflab.config:
    - Pressure thresholds (70%/80%/95% default)
    - Eviction targets (15%/30%/50% of process memory)
    - Monitoring intervals and hysteresis settings
    - Enable/disable flags for different monitoring components

Usage Patterns:
    Basic monitoring:
        >>> monitor = SystemMemoryMonitor()
        >>> monitor.start_monitoring()
        >>> stats = monitor.get_current_stats()
        >>> monitor.stop_monitoring()
    
    Pressure management:
        >>> manager = MemoryPressureManager(monitor, tracker)
        >>> manager.register_eviction_callback("frame_cache", cache.evict)
        >>> should_evict, target_mb = manager.check_memory_pressure()
        >>> if should_evict:
        >>>     freed_mb = manager.execute_eviction(target_mb)
    
    Cache tracking:
        >>> tracker = CacheMemoryTracker()
        >>> tracker.update_cache_size("frame_cache", 150.0)
        >>> usage = tracker.get_total_cache_usage()

Error Handling:
    The module is designed for robust operation in production environments:
    - Graceful degradation when psutil unavailable
    - Continue operation if individual cache eviction callbacks fail
    - Comprehensive logging for troubleshooting and monitoring
    - Safe defaults that prevent system instability

Dependencies:
    Required:
        - threading: For thread-safe operation and background monitoring
        - logging: For operational logging and debugging
        - dataclasses: For type-safe data structures
        - enum: For memory pressure level definitions
    
    Optional:
        - psutil: For accurate cross-platform memory monitoring
          (Falls back to limited monitoring without psutil)
        - cache_effectiveness: For advanced cache effectiveness analysis
          (Tracking continues without effectiveness monitoring)

See Also:
    - docs/technical/memory-monitoring-architecture.md: Detailed architecture documentation
    - docs/guides/cli-dependency-troubleshooting.md: CLI memory monitoring usage
    - src/giflab/config.py: Memory monitoring configuration options
    - tests/test_memory_monitoring.py: Comprehensive test coverage and usage examples

Authors:
    GifLab Memory Safety Infrastructure (Phase 3.1)
    
Version:
    Added in Phase 3.1 as part of critical memory safety improvements
    Enhanced in Phase 3.2 with cache effectiveness integration
"""

import logging
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Protocol

from ..lazy_imports import check_import_available

logger = logging.getLogger(__name__)

# Lazy import psutil for cross-platform memory monitoring
_psutil = None
_psutil_available = None

def _get_psutil():
    """Lazy import of psutil with availability checking."""
    global _psutil, _psutil_available
    
    if _psutil_available is None:
        _psutil_available = check_import_available("psutil")
        if _psutil_available:
            try:
                import psutil
                _psutil = psutil
            except ImportError:
                _psutil_available = False
                logger.warning("psutil import failed, memory monitoring will be limited")
    
    return _psutil if _psutil_available else None

class MemoryPressureLevel(Enum):
    """Memory pressure severity levels."""
    NORMAL = "normal"        # < 70% usage
    WARNING = "warning"      # 70-80% usage
    CRITICAL = "critical"    # 80-95% usage
    EMERGENCY = "emergency"  # > 95% usage

@dataclass
class MemoryStats:
    """Memory usage statistics."""
    process_memory_mb: float
    process_memory_percent: float
    system_memory_mb: float
    system_memory_percent: float
    system_available_mb: float
    total_system_mb: float
    pressure_level: MemoryPressureLevel
    timestamp: float

@dataclass
class CacheMemoryUsage:
    """Memory usage breakdown by cache type."""
    frame_cache_mb: float = 0.0
    resize_cache_mb: float = 0.0
    validation_cache_mb: float = 0.0
    total_cache_mb: float = 0.0
    

class EvictionPolicy(Protocol):
    """Protocol for cache eviction policies."""
    
    def should_evict(self, memory_stats: MemoryStats, cache_usage: CacheMemoryUsage) -> bool:
        """Determine if eviction should occur based on memory state."""
        ...
    
    def get_eviction_target_mb(self, memory_stats: MemoryStats) -> float:
        """Calculate target memory to free via eviction."""
        ...

class ConservativeEvictionPolicy:
    """Conservative eviction policy that maintains system stability."""
    
    def __init__(self, warning_threshold: float = 0.8, critical_threshold: float = 0.95):
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
    
    def should_evict(self, memory_stats: MemoryStats, cache_usage: CacheMemoryUsage) -> bool:
        """Evict when system memory pressure exceeds warning threshold."""
        return memory_stats.system_memory_percent >= self.warning_threshold
    
    def get_eviction_target_mb(self, memory_stats: MemoryStats) -> float:
        """Calculate memory to free - more aggressive as pressure increases."""
        if memory_stats.system_memory_percent >= self.critical_threshold:
            # Critical: free 50% of cache memory
            return memory_stats.process_memory_mb * 0.3
        elif memory_stats.system_memory_percent >= self.warning_threshold:
            # Warning: free 25% of cache memory
            return memory_stats.process_memory_mb * 0.15
        return 0.0

class SystemMemoryMonitor:
    """Cross-platform system memory monitoring with pressure detection.
    
    Provides real-time monitoring of system and process memory usage with automatic
    pressure level detection. Uses psutil when available, with graceful fallback
    for environments where psutil is not installed.
    
    The monitor can run in background mode with continuous updates, or be used
    synchronously for on-demand memory statistics collection.
    
    Memory pressure levels are calculated based on system memory usage:
    - NORMAL: < 70% system memory used
    - WARNING: 70-80% system memory used
    - CRITICAL: 80-95% system memory used
    - EMERGENCY: > 95% system memory used
    
    Attributes:
        update_interval (float): Seconds between memory checks in background mode.
            Default: 5.0 seconds.
    
    Example:
        >>> monitor = SystemMemoryMonitor(update_interval=10.0)
        >>> monitor.start_monitoring()  # Start background monitoring
        >>> stats = monitor.get_current_stats()
        >>> print(f"System memory: {stats.system_memory_percent:.1%}")
        >>> monitor.stop_monitoring()
        
        >>> # Or use synchronously
        >>> stats = monitor.collect_memory_stats()
        >>> if stats.pressure_level != MemoryPressureLevel.NORMAL:
        >>>     print(f"Memory pressure: {stats.pressure_level.value}")
    
    Thread Safety:
        All public methods are thread-safe. Internal state is protected by RLock.
        
    Performance:
        Memory collection overhead is typically < 1ms per call with psutil.
        Fallback mode has minimal overhead but provides limited information.
    """
    
    def __init__(self, update_interval: float = 5.0):
        """Initialize the system memory monitor.
        
        Args:
            update_interval (float): Seconds between memory checks in background
                monitoring mode. Must be > 0. Default: 5.0 seconds.
                
        Raises:
            ValueError: If update_interval <= 0.
        """
        if update_interval <= 0:
            raise ValueError("update_interval must be positive")
            
        self.update_interval = update_interval
        self._lock = threading.RLock()
        self._current_stats: MemoryStats | None = None
        self._monitoring_active = False
        self._monitor_thread: threading.Thread | None = None
        
        # Check psutil availability for cross-platform memory monitoring
        self._psutil = _get_psutil()
        if not self._psutil:
            logger.warning("psutil unavailable - memory monitoring will use fallbacks")
    
    def start_monitoring(self) -> None:
        """Start background memory monitoring in a daemon thread.
        
        Creates a background thread that continuously monitors memory usage
        at the configured update_interval. The thread is marked as daemon
        so it won't prevent program shutdown.
        
        Monitoring thread updates internal stats accessible via get_current_stats().
        
        This method is idempotent - calling it multiple times has no effect
        if monitoring is already active.
        
        Thread Safety:
            Safe to call from multiple threads. Uses internal locking.
            
        Example:
            >>> monitor = SystemMemoryMonitor()
            >>> monitor.start_monitoring()
            >>> # Background monitoring now active
            >>> time.sleep(10)  # Monitor runs in background
            >>> current = monitor.get_current_stats()
            >>> monitor.stop_monitoring()
        """
        with self._lock:
            if self._monitoring_active:
                return
                
            self._monitoring_active = True
            self._monitor_thread = threading.Thread(
                target=self._monitoring_loop,
                name="MemoryMonitor",
                daemon=True  # Don't block program shutdown
            )
            self._monitor_thread.start()
            logger.info("System memory monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop background memory monitoring gracefully.
        
        Signals the monitoring thread to stop and waits up to 1 second
        for it to terminate. If the thread doesn't stop within the timeout,
        it will be left to terminate naturally (as a daemon thread).
        
        This method is idempotent - safe to call even if monitoring
        is not active.
        
        Thread Safety:
            Safe to call from multiple threads. Uses internal locking.
            
        Example:
            >>> monitor.start_monitoring()
            >>> # ... monitoring runs ...
            >>> monitor.stop_monitoring()  # Clean shutdown
        """
        with self._lock:
            self._monitoring_active = False
            if self._monitor_thread and self._monitor_thread.is_alive():
                self._monitor_thread.join(timeout=1.0)
            logger.info("System memory monitoring stopped")
    
    def get_current_stats(self) -> MemoryStats | None:
        """Get the most recent memory statistics from background monitoring.
        
        Returns the latest MemoryStats collected by the background monitoring
        thread. If background monitoring is not active, returns None.
        
        For immediate/fresh statistics, use collect_memory_stats() instead.
        
        Returns:
            Optional[MemoryStats]: Latest memory statistics, or None if
                background monitoring is not active or no stats collected yet.
                
        Thread Safety:
            Thread-safe. Returns a reference to the current stats object.
            
        Example:
            >>> monitor.start_monitoring()
            >>> time.sleep(6)  # Wait for first collection
            >>> stats = monitor.get_current_stats()
            >>> if stats:
            >>>     print(f"Process memory: {stats.process_memory_mb:.1f}MB")
        """
        with self._lock:
            return self._current_stats
    
    def collect_memory_stats(self) -> MemoryStats:
        """Collect current memory statistics synchronously.
        
        Immediately collects fresh memory usage statistics for both the
        current process and system-wide memory usage. This method always
        returns current data, unlike get_current_stats() which returns
        cached background monitoring data.
        
        Uses psutil when available for accurate cross-platform memory
        information. Falls back to minimal stats when psutil unavailable.
        
        Returns:
            MemoryStats: Complete memory statistics including:
                - Process memory usage (MB and percentage)
                - System memory usage (MB and percentage)
                - Available system memory (MB)
                - Total system memory (MB)
                - Memory pressure level (NORMAL/WARNING/CRITICAL/EMERGENCY)
                - Collection timestamp
                
        Performance:
            Typically < 1ms with psutil, < 0.1ms with fallback.
            
        Example:
            >>> stats = monitor.collect_memory_stats()
            >>> print(f"System: {stats.system_memory_percent:.1%}")
            >>> print(f"Process: {stats.process_memory_mb:.1f}MB")
            >>> print(f"Pressure: {stats.pressure_level.value}")
        """
        timestamp = time.time()
        
        if self._psutil:
            try:
                # Get system memory info using psutil
                virtual_mem = self._psutil.virtual_memory()
                system_memory_mb = virtual_mem.used / (1024 * 1024)
                system_memory_percent = virtual_mem.percent / 100.0
                system_available_mb = virtual_mem.available / (1024 * 1024)
                total_system_mb = virtual_mem.total / (1024 * 1024)
                
                # Get current process memory info
                process = self._psutil.Process()
                process_info = process.memory_info()
                process_memory_mb = process_info.rss / (1024 * 1024)  # Resident Set Size
                process_memory_percent = (process_memory_mb / total_system_mb) * 100.0
                
            except Exception as e:
                logger.error(f"Failed to collect memory stats with psutil: {e}")
                return self._get_fallback_stats(timestamp)
        else:
            return self._get_fallback_stats(timestamp)
        
        # Calculate memory pressure level based on system usage percentage
        pressure_level = self._calculate_pressure_level(system_memory_percent)
        
        return MemoryStats(
            process_memory_mb=process_memory_mb,
            process_memory_percent=process_memory_percent,
            system_memory_mb=system_memory_mb,
            system_memory_percent=system_memory_percent,
            system_available_mb=system_available_mb,
            total_system_mb=total_system_mb,
            pressure_level=pressure_level,
            timestamp=timestamp
        )
    
    def _get_fallback_stats(self, timestamp: float) -> MemoryStats:
        """Fallback memory stats when psutil is unavailable.
        
        Provides minimal MemoryStats when psutil cannot be used for memory
        monitoring. Sets conservative defaults that won't trigger pressure
        management but indicate monitoring limitations.
        
        Args:
            timestamp (float): Unix timestamp for the statistics.
            
        Returns:
            MemoryStats: Minimal stats with safe defaults:
                - All memory values set to 0.0
                - Available memory set to infinity (no pressure)
                - Pressure level set to NORMAL
                - Timestamp preserved
                
        Note:
            This fallback ensures the monitoring system continues to function
            even without psutil, but provides no actionable memory information.
        """
        logger.debug("Using fallback memory statistics")
        return MemoryStats(
            process_memory_mb=0.0,
            process_memory_percent=0.0,
            system_memory_mb=0.0,
            system_memory_percent=0.0,
            system_available_mb=float('inf'),  # Assume unlimited when unknown
            total_system_mb=0.0,
            pressure_level=MemoryPressureLevel.NORMAL,
            timestamp=timestamp
        )
    
    def _calculate_pressure_level(self, system_percent: float) -> MemoryPressureLevel:
        """Calculate memory pressure level from system usage percentage.
        
        Converts system memory usage percentage into discrete pressure levels
        using conservative thresholds designed to trigger eviction before
        system instability occurs.
        
        Thresholds are based on typical system behavior:
        - 70%: Early warning for proactive cache management
        - 80%: Critical level where performance degrades
        - 95%: Emergency level approaching system limits
        
        Args:
            system_percent (float): System memory usage as percentage (0.0-1.0).
                Values > 1.0 are treated as 100%.
                
        Returns:
            MemoryPressureLevel: Appropriate pressure level:
                - NORMAL: < 70% usage
                - WARNING: 70-80% usage
                - CRITICAL: 80-95% usage
                - EMERGENCY: > 95% usage
                
        Example:
            >>> level = monitor._calculate_pressure_level(0.75)
            >>> assert level == MemoryPressureLevel.WARNING
            >>> level = monitor._calculate_pressure_level(0.90)
            >>> assert level == MemoryPressureLevel.CRITICAL
        """
        if system_percent >= 0.95:
            return MemoryPressureLevel.EMERGENCY
        elif system_percent >= 0.80:
            return MemoryPressureLevel.CRITICAL
        elif system_percent >= 0.70:
            return MemoryPressureLevel.WARNING
        else:
            return MemoryPressureLevel.NORMAL
    
    def _monitoring_loop(self) -> None:
        """Background monitoring loop executed in daemon thread.
        
        Continuously collects memory statistics at the configured update_interval
        until monitoring is stopped. Updates internal _current_stats and logs
        memory pressure warnings when pressure levels are elevated.
        
        This is an internal method called by start_monitoring(). It handles
        exceptions gracefully to prevent the monitoring thread from crashing.
        
        Error Handling:
            Logs errors and continues monitoring. Uses update_interval as
            sleep duration even after errors to maintain consistent timing.
            
        Logging:
            Logs WARNING messages when memory pressure is detected.
            Logs ERROR messages for collection failures.
            
        Performance:
            Designed for long-running operation with minimal overhead.
            Sleep between collections to avoid excessive CPU usage.
        """
        while self._monitoring_active:
            try:
                stats = self.collect_memory_stats()
                with self._lock:
                    self._current_stats = stats
                
                # Log pressure level changes for operational awareness
                if stats.pressure_level != MemoryPressureLevel.NORMAL:
                    logger.warning(
                        f"Memory pressure detected: {stats.pressure_level.value} "
                        f"(System: {stats.system_memory_percent:.1%}, "
                        f"Process: {stats.process_memory_mb:.1f}MB)"
                    )
                
                time.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Error in memory monitoring loop: {e}")
                time.sleep(self.update_interval)  # Continue monitoring despite errors

class CacheMemoryTracker:
    """Tracks memory usage across all GifLab cache systems with effectiveness monitoring.
    
    Provides centralized tracking of memory consumption across different cache types
    (frame cache, resize cache, validation cache) and integrates with cache effectiveness
    monitoring when available.
    
    The tracker maintains real-time memory usage statistics for each cache type and
    supports cache operation recording for performance analysis and optimization.
    
    Supported Cache Types:
        - frame_cache: GIF frame data cache
        - resize_cache: Resized frame cache
        - validation_cache: Validation result cache
        - Custom cache types via update_cache_size()
    
    Integration Features:
        - Automatic cache effectiveness monitoring integration
        - Operation recording for hit/miss ratio analysis
        - Memory pressure correlation tracking
        - System-wide effectiveness reporting
    
    Attributes:
        _cache_sizes (dict[str, float]): Memory usage per cache type in MB.
        _effectiveness_monitor: Optional effectiveness monitoring integration.
        _effectiveness_enabled (bool): Whether effectiveness monitoring is active.
    
    Example:
        >>> tracker = CacheMemoryTracker()
        >>>
        >>> # Update cache sizes as caches grow/shrink
        >>> tracker.update_cache_size("frame_cache", 150.5)
        >>> tracker.update_cache_size("resize_cache", 75.2)
        >>>
        >>> # Get total usage for memory pressure management
        >>> usage = tracker.get_total_cache_usage()
        >>> print(f"Total cache memory: {usage.total_cache_mb:.1f}MB")
        >>>
        >>> # Record cache operations for effectiveness analysis
        >>> tracker.record_cache_operation("frame_cache", "hit", "video_123_frame_5")
        >>> tracker.record_cache_operation("resize_cache", "miss", "resized_100x100_video_456")
    
    Thread Safety:
        All public methods are thread-safe using internal RLock.
        Safe for concurrent access from multiple cache systems.
        
    Performance:
        Minimal overhead for basic tracking (~0.1ms per operation).
        Effectiveness monitoring adds negligible overhead when enabled.
    """
    
    def __init__(self):
        """Initialize the cache memory tracker.
        
        Sets up internal data structures and attempts to initialize cache
        effectiveness monitoring if available. Effectiveness monitoring
        integration is optional and fails gracefully if not available.
        
        Thread Safety:
            Creates thread-safe internal structures.
            
        Example:
            >>> tracker = CacheMemoryTracker()
            >>> # Tracker is ready for use immediately
            >>> tracker.update_cache_size("frame_cache", 0.0)
        """
        self._lock = threading.RLock()
        self._cache_sizes: dict[str, float] = {}
        
        # Cache effectiveness monitoring integration (optional)
        self._effectiveness_monitor = None
        self._effectiveness_enabled = False
        self._init_effectiveness_monitoring()
    
    def update_cache_size(self, cache_type: str, size_mb: float) -> None:
        """Update memory usage for a specific cache type.
        
        Updates the tracked memory usage for the specified cache type.
        This should be called whenever a cache's memory footprint changes
        due to additions, evictions, or clearing.
        
        Args:
            cache_type (str): Unique identifier for the cache type.
                Common values: "frame_cache", "resize_cache", "validation_cache"
            size_mb (float): Current memory usage in megabytes. Must be >= 0.
                
        Thread Safety:
            Thread-safe. Multiple caches can update simultaneously.
            
        Performance:
            Very fast operation (~0.1ms). Safe to call frequently.
            
        Example:
            >>> tracker = CacheMemoryTracker()
            >>>
            >>> # Cache grew by adding new entries
            >>> tracker.update_cache_size("frame_cache", 120.5)
            >>>
            >>> # Cache shrunk due to eviction
            >>> tracker.update_cache_size("frame_cache", 95.2)
            >>>
            >>> # Cache was cleared
            >>> tracker.update_cache_size("frame_cache", 0.0)
        """
        with self._lock:
            self._cache_sizes[cache_type] = size_mb
    
    def get_total_cache_usage(self) -> CacheMemoryUsage:
        """Get total memory usage across all tracked caches.
        
        Returns a CacheMemoryUsage object containing per-cache and total
        memory usage statistics. Used by memory pressure management to
        determine eviction targets.
        
        Returns:
            CacheMemoryUsage: Object containing:
                - frame_cache_mb: Memory used by frame cache
                - resize_cache_mb: Memory used by resize cache
                - validation_cache_mb: Memory used by validation cache
                - total_cache_mb: Sum of all cache memory usage
                
        Thread Safety:
            Thread-safe. Returns consistent snapshot of current usage.
            
        Performance:
            Fast operation (~0.1ms). Safe to call frequently for monitoring.
            
        Example:
            >>> usage = tracker.get_total_cache_usage()
            >>> print(f"Frame cache: {usage.frame_cache_mb:.1f}MB")
            >>> print(f"Resize cache: {usage.resize_cache_mb:.1f}MB")
            >>> print(f"Validation cache: {usage.validation_cache_mb:.1f}MB")
            >>> print(f"Total: {usage.total_cache_mb:.1f}MB")
            >>>
            >>> # Check if total usage exceeds threshold
            >>> if usage.total_cache_mb > 500.0:
            >>>     print("Cache memory usage is high, consider eviction")
        """
        with self._lock:
            return CacheMemoryUsage(
                frame_cache_mb=self._cache_sizes.get("frame_cache", 0.0),
                resize_cache_mb=self._cache_sizes.get("resize_cache", 0.0),
                validation_cache_mb=self._cache_sizes.get("validation_cache", 0.0),
                total_cache_mb=sum(self._cache_sizes.values())
            )
    
    def reset_cache_size(self, cache_type: str) -> None:
        """Reset memory usage for a specific cache type to zero.
        
        Removes the specified cache type from memory tracking. This should
        be called when a cache is completely cleared or disabled.
        
        Args:
            cache_type (str): Cache type to reset. If the cache type is not
                currently tracked, this operation has no effect.
                
        Thread Safety:
            Thread-safe. Safe to call concurrently with other operations.
            
        Example:
            >>> tracker.update_cache_size("validation_cache", 50.0)
            >>> usage = tracker.get_total_cache_usage()
            >>> print(usage.validation_cache_mb)  # 50.0
            >>>
            >>> tracker.reset_cache_size("validation_cache")
            >>> usage = tracker.get_total_cache_usage()
            >>> print(usage.validation_cache_mb)  # 0.0
        """
        with self._lock:
            self._cache_sizes.pop(cache_type, None)
    
    def _init_effectiveness_monitoring(self) -> None:
        """Initialize cache effectiveness monitoring if available.
        
        Attempts to initialize integration with cache effectiveness monitoring
        system. This is optional functionality that enhances cache analysis
        capabilities when the effectiveness monitoring module is available.
        
        Fails gracefully if effectiveness monitoring is not installed or
        not enabled, allowing the memory tracker to function normally
        without effectiveness features.
        
        Internal Method:
            This method is called during __init__ and should not be called
            directly by external code.
            
        Logging:
            - DEBUG: Success/failure of effectiveness monitoring initialization
        """
        try:
            from .cache_effectiveness import (
                get_cache_effectiveness_monitor,
                is_cache_effectiveness_monitoring_enabled,
            )
            if is_cache_effectiveness_monitoring_enabled():
                self._effectiveness_monitor = get_cache_effectiveness_monitor()
                self._effectiveness_enabled = True
                logger.debug("Cache effectiveness monitoring initialized")
        except ImportError:
            logger.debug("Cache effectiveness monitoring not available")
    
    def record_cache_operation(self,
                              cache_type: str,
                              operation: str,
                              key: str,
                              data_size_mb: float = 0.0,
                              processing_time_ms: float = 0.0) -> None:
        """Record a cache operation for effectiveness analysis.
        
        Records cache operations (hit, miss, put, evict, expire) for effectiveness
        monitoring and performance analysis. This data is used to calculate
        hit ratios, eviction rates, and memory pressure correlations.
        
        Only records operations when effectiveness monitoring is enabled.
        Fails gracefully when effectiveness monitoring is unavailable.
        
        Args:
            cache_type (str): type of cache where operation occurred.
            operation (str): type of operation performed. Valid values:
                - "hit": Successful cache lookup
                - "miss": Failed cache lookup
                - "put": Data stored in cache
                - "evict": Data removed due to memory pressure
                - "expire": Data removed due to TTL expiration
            key (str): Cache key involved in the operation. Used for
                pattern analysis and debugging.
            data_size_mb (float, optional): Size of data involved in MB.
                Defaults to 0.0. Used for memory impact analysis.
            processing_time_ms (float, optional): Time taken for operation in
                milliseconds. Defaults to 0.0. Used for performance analysis.
                
        Thread Safety:
            Thread-safe. Safe to call from cache operations.
            
        Performance:
            Minimal overhead when effectiveness monitoring enabled (~0.1ms).
            Zero overhead when effectiveness monitoring disabled.
            
        Example:
            >>> # Record successful cache hit
            >>> tracker.record_cache_operation(
            ...     cache_type="frame_cache",
            ...     operation="hit",
            ...     key="video_123_frame_10",
            ...     processing_time_ms=0.5
            ... )
            >>>
            >>> # Record cache miss with subsequent put
            >>> tracker.record_cache_operation("resize_cache", "miss", "100x100_video_456")
            >>> tracker.record_cache_operation(
            ...     cache_type="resize_cache",
            ...     operation="put",
            ...     key="100x100_video_456",
            ...     data_size_mb=2.5,
            ...     processing_time_ms=15.2
            ... )
            >>>
            >>> # Record memory pressure eviction
            >>> tracker.record_cache_operation(
            ...     cache_type="frame_cache",
            ...     operation="evict",
            ...     key="old_video_frame_batch",
            ...     data_size_mb=45.0
            ... )
                
        Error Handling:
            Logs debug messages for operation recording failures but does not
            raise exceptions. Cache operations continue normally even if
            effectiveness recording fails.
        """
        if not self._effectiveness_enabled or not self._effectiveness_monitor:
            return
        
        try:
            from .cache_effectiveness import CacheOperationType
            
            # Map string operations to enum values for type safety
            operation_mapping = {
                'hit': CacheOperationType.HIT,
                'miss': CacheOperationType.MISS,
                'put': CacheOperationType.PUT,
                'evict': CacheOperationType.EVICT,
                'expire': CacheOperationType.EXPIRE
            }
            
            if operation in operation_mapping:
                # Get current memory pressure level for correlation analysis
                memory_pressure_level = None
                if hasattr(self, '_current_memory_stats'):
                    memory_pressure_level = getattr(self._current_memory_stats, 'pressure_level', None)
                
                self._effectiveness_monitor.record_operation(
                    cache_type=cache_type,
                    operation=operation_mapping[operation],
                    key=key,
                    data_size_mb=data_size_mb,
                    processing_time_ms=processing_time_ms,
                    memory_pressure_level=memory_pressure_level
                )
        except Exception as e:
            logger.debug(f"Failed to record cache operation for effectiveness monitoring: {e}")
    
    def get_cache_effectiveness_stats(self, cache_type: str) -> Any | None:
        """Get effectiveness statistics for a specific cache type.
        
        Retrieves detailed effectiveness statistics for the specified cache
        type, including hit ratios, eviction rates, and performance metrics.
        
        Args:
            cache_type (str): Cache type to get statistics for.
            
        Returns:
            Optional[Any]: Effectiveness statistics object or None if
                effectiveness monitoring is not available or no data exists
                for the specified cache type.
                
        Thread Safety:
            Thread-safe. Safe to call concurrently.
            
        Example:
            >>> stats = tracker.get_cache_effectiveness_stats("frame_cache")
            >>> if stats:
            >>>     print(f"Hit ratio: {stats.hit_ratio:.2%}")
            >>>     print(f"Eviction rate: {stats.eviction_rate_per_hour:.1f}/hr")
        """
        if not self._effectiveness_enabled or not self._effectiveness_monitor:
            return None
        
        try:
            return self._effectiveness_monitor.get_cache_effectiveness(cache_type)
        except Exception as e:
            logger.debug(f"Failed to get cache effectiveness stats: {e}")
            return None
    
    def get_system_effectiveness_summary(self) -> dict[str, Any]:
        """Get system-wide cache effectiveness summary.
        
        Retrieves a comprehensive summary of cache effectiveness across all
        cache types, including overall hit ratios, memory efficiency, and
        recommendations for optimization.
        
        Returns:
            dict[str, Any]: Summary dictionary containing:
                - When effectiveness monitoring enabled: comprehensive stats
                - When effectiveness monitoring disabled: {"effectiveness_monitoring": "disabled"}
                - On error: {"error": "error_description"}
                
        Thread Safety:
            Thread-safe. Safe to call for monitoring dashboards.
            
        Performance:
            Aggregates data across all cache types. May take 1-10ms depending
            on cache activity and monitoring history.
            
        Example:
            >>> summary = tracker.get_system_effectiveness_summary()
            >>> if "effectiveness_monitoring" in summary:
            >>>     print("Effectiveness monitoring not available")
            >>> elif "error" in summary:
            >>>     print(f"Error getting stats: {summary['error']}")
            >>> else:
            >>>     print(f"Overall hit ratio: {summary['overall_hit_ratio']:.2%}")
            >>>     print(f"Memory efficiency: {summary['memory_efficiency']:.1f}")
            >>>     for rec in summary.get('recommendations', []):
            >>>         print(f"Recommendation: {rec}")
        """
        if not self._effectiveness_enabled or not self._effectiveness_monitor:
            return {"effectiveness_monitoring": "disabled"}
        
        try:
            return self._effectiveness_monitor.get_system_effectiveness_summary()
        except Exception as e:
            logger.debug(f"Failed to get system effectiveness summary: {e}")
            return {"error": str(e)}

class MemoryPressureManager:
    """Manages memory pressure detection and automatic cache eviction.
    
    Coordinates between system memory monitoring and cache memory tracking to
    automatically evict cached data when memory pressure reaches configured
    thresholds. Uses pluggable eviction policies for flexible pressure management.
    
    The manager maintains a registry of eviction callbacks for different cache
    types and executes eviction in priority order when memory pressure is detected.
    
    Eviction Strategy:
        1. Check memory pressure using configured EvictionPolicy
        2. Calculate target memory to free based on pressure level
        3. Execute eviction callbacks in priority order:
           - validation_cache (highest priority - clear first)
           - resize_cache (medium priority)
           - frame_cache (lowest priority - most valuable data)
        4. Record eviction events for effectiveness monitoring
    
    Attributes:
        system_monitor (SystemMemoryMonitor): Source of system memory statistics.
        cache_tracker (CacheMemoryTracker): Source of cache memory usage data.
        eviction_policy (EvictionPolicy): Policy determining when/how much to evict.
    
    Example:
        >>> monitor = SystemMemoryMonitor()
        >>> tracker = CacheMemoryTracker()
        >>> manager = MemoryPressureManager(monitor, tracker)
        >>>
        >>> # Register cache eviction callbacks
        >>> manager.register_eviction_callback("frame_cache", frame_cache.evict)
        >>> manager.register_eviction_callback("resize_cache", resize_cache.clear)
        >>>
        >>> # Check and handle memory pressure
        >>> should_evict, target_mb = manager.check_memory_pressure()
        >>> if should_evict:
        >>>     freed_mb = manager.execute_eviction(target_mb)
        >>>     print(f"Freed {freed_mb:.1f}MB to reduce memory pressure")
    
    Thread Safety:
        All public methods are thread-safe. Eviction callback registration
        and execution are protected by internal locking.
        
    Integration:
        Designed to integrate with existing cache systems and monitoring
        infrastructure. Works with any cache that can provide eviction callbacks.
    """
    
    def __init__(self,
                 system_monitor: SystemMemoryMonitor,
                 cache_tracker: CacheMemoryTracker,
                 eviction_policy: EvictionPolicy | None = None):
        """Initialize the memory pressure manager.
        
        Args:
            system_monitor (SystemMemoryMonitor): Monitor providing system
                memory statistics for pressure detection.
            cache_tracker (CacheMemoryTracker): Tracker providing cache memory
                usage information for eviction targeting.
            eviction_policy (Optional[EvictionPolicy]): Policy determining when
                and how much to evict. Defaults to ConservativeEvictionPolicy.
                
        Example:
            >>> monitor = SystemMemoryMonitor(update_interval=5.0)
            >>> tracker = CacheMemoryTracker()
            >>> policy = ConservativeEvictionPolicy()
            >>> manager = MemoryPressureManager(monitor, tracker, policy)
        """
        self.system_monitor = system_monitor
        self.cache_tracker = cache_tracker
        self.eviction_policy = eviction_policy or ConservativeEvictionPolicy()
        self._eviction_callbacks: dict[str, Any] = {}
        self._lock = threading.RLock()
    
    def register_eviction_callback(self, cache_type: str, callback) -> None:
        """Register callback function for cache eviction.
        
        Registers a callback function that will be called during memory pressure
        eviction. The callback should accept a target_mb parameter and return
        the actual amount of memory freed.
        
        Args:
            cache_type (str): Unique identifier for the cache type. Used for
                eviction priority ordering and logging. Common types:
                - "frame_cache": GIF frame cache (lowest priority)
                - "resize_cache": Resized frame cache (medium priority)
                - "validation_cache": Validation result cache (highest priority)
            callback (Callable[[float], float]): Function to call for eviction.
                Must accept target_mb (float) parameter and return freed_mb (float).
                
        Thread Safety:
            Thread-safe. Uses internal locking for callback registration.
            
        Example:
            >>> def frame_cache_evict(target_mb: float) -> float:
            ...     freed = 0.0
            ...     # Evict cache entries until target_mb freed
            ...     return freed
            >>>
            >>> manager.register_eviction_callback("frame_cache", frame_cache_evict)
            
        Note:
            Callbacks are executed in priority order during eviction:
            validation_cache → resize_cache → frame_cache
        """
        with self._lock:
            self._eviction_callbacks[cache_type] = callback
    
    def check_memory_pressure(self) -> tuple[bool, float | None]:
        """Check if current memory usage requires cache eviction.
        
        Evaluates current system memory usage and cache memory consumption
        against the configured eviction policy to determine if memory pressure
        eviction should be triggered.
        
        Uses the configured EvictionPolicy to make eviction decisions, allowing
        for different pressure management strategies (conservative, aggressive,
        adaptive, etc.).
        
        Returns:
            tuple[bool, Optional[float]]: A tuple containing:
                - should_evict (bool): True if eviction should be performed
                - target_mb_to_free (Optional[float]): Amount of memory to free
                  in MB, or None if no eviction needed
                  
        Thread Safety:
            Thread-safe. Accesses memory statistics atomically.
            
        Performance:
            Fast operation typically < 1ms. Only collects statistics,
            does not perform any eviction.
            
        Example:
            >>> should_evict, target_mb = manager.check_memory_pressure()
            >>> if should_evict:
            >>>     print(f"Memory pressure detected, need to free {target_mb:.1f}MB")
            >>>     freed_mb = manager.execute_eviction(target_mb)
            >>> else:
            >>>     print("Memory usage within acceptable limits")
                  
        Logging:
            Logs WARNING when memory pressure eviction is triggered, including
            target amount, total cache usage, and system memory percentage.
        """
        memory_stats = self.system_monitor.get_current_stats()
        if not memory_stats:
            return False, None
        
        cache_usage = self.cache_tracker.get_total_cache_usage()
        
        should_evict = self.eviction_policy.should_evict(memory_stats, cache_usage)
        target_mb = None
        
        if should_evict:
            target_mb = self.eviction_policy.get_eviction_target_mb(memory_stats)
            logger.warning(
                f"Memory pressure eviction triggered: "
                f"target={target_mb:.1f}MB, "
                f"cache_total={cache_usage.total_cache_mb:.1f}MB, "
                f"system={memory_stats.system_memory_percent:.1%}"
            )
        
        return should_evict, target_mb
    
    def execute_eviction(self, target_mb: float) -> float:
        """Execute cache eviction to free the specified amount of memory.
        
        Calls registered eviction callbacks in priority order to free approximately
        the target amount of memory. Continues eviction across cache types until
        the target is met or all callbacks have been executed.
        
        Eviction Priority Order (high to low priority):
        1. validation_cache - Fast to rebuild, clears quickly
        2. resize_cache - Medium cost to rebuild
        3. frame_cache - Expensive to rebuild, evict last
        
        Records eviction events in the cache tracker for effectiveness monitoring
        and performance analysis.
        
        Args:
            target_mb (float): Target amount of memory to free in megabytes.
                Must be > 0. Eviction continues until this amount is freed
                or all callbacks are exhausted.
                
        Returns:
            float: Actual amount of memory freed in MB. May be less than
                target_mb if insufficient cache data available, or more if
                cache eviction granularity prevents exact targeting.
                
        Thread Safety:
            Thread-safe. Uses internal locking during callback execution.
            Individual cache callbacks are responsible for their own thread safety.
            
        Error Handling:
            Continues eviction even if individual callbacks fail. Logs errors
            for failed callbacks but doesn't interrupt the eviction process.
            
        Performance:
            Performance depends on cache callback implementations. The manager
            itself adds minimal overhead (<1ms) for coordination.
            
        Example:
            >>> # After detecting memory pressure
            >>> should_evict, target_mb = manager.check_memory_pressure()
            >>> if should_evict:
            >>>     freed_mb = manager.execute_eviction(target_mb)
            >>>     efficiency = (freed_mb / target_mb) * 100
            >>>     print(f"Eviction efficiency: {efficiency:.1f}%")
            >>>
            >>>     if freed_mb < target_mb * 0.8:  # Less than 80% efficiency
            >>>         print("Warning: Eviction target not fully met")
                
        Logging:
            - INFO: Successful eviction from each cache type
            - ERROR: Failed eviction callbacks
            - Records detailed eviction events for monitoring
        """
        freed_mb = 0.0
        
        with self._lock:
            # Priority order: validation_cache -> resize_cache -> frame_cache
            # Higher priority caches are cheaper to rebuild
            eviction_order = ["validation_cache", "resize_cache", "frame_cache"]
            
            for cache_type in eviction_order:
                if target_mb <= 0 or cache_type not in self._eviction_callbacks:
                    continue
                
                callback = self._eviction_callbacks[cache_type]
                try:
                    # Execute eviction callback with remaining target
                    cache_freed = callback(target_mb)
                    freed_mb += cache_freed
                    target_mb -= cache_freed
                    
                    # Record eviction event for effectiveness monitoring
                    # This helps track eviction patterns and cache effectiveness
                    self.cache_tracker.record_cache_operation(
                        cache_type=cache_type,
                        operation='evict',
                        key=f"eviction_batch_{time.time()}",
                        data_size_mb=cache_freed
                    )
                    
                    logger.info(
                        f"Evicted {cache_freed:.1f}MB from {cache_type}, "
                        f"total_freed={freed_mb:.1f}MB"
                    )
                    
                except Exception as e:
                    logger.error(f"Failed to evict from {cache_type}: {e}")
                    # Continue with other caches even if one fails
        
        return freed_mb

# Global instances for singleton access
_system_monitor: SystemMemoryMonitor | None = None
_cache_tracker: CacheMemoryTracker | None = None
_pressure_manager: MemoryPressureManager | None = None
_monitor_lock = threading.RLock()

def get_system_memory_monitor() -> SystemMemoryMonitor:
    """Get singleton system memory monitor."""
    global _system_monitor
    with _monitor_lock:
        if _system_monitor is None:
            _system_monitor = SystemMemoryMonitor()
        return _system_monitor

def get_cache_memory_tracker() -> CacheMemoryTracker:
    """Get singleton cache memory tracker."""
    global _cache_tracker
    with _monitor_lock:
        if _cache_tracker is None:
            _cache_tracker = CacheMemoryTracker()
        return _cache_tracker

def get_memory_pressure_manager() -> MemoryPressureManager:
    """Get singleton memory pressure manager."""
    global _pressure_manager
    with _monitor_lock:
        if _pressure_manager is None:
            system_monitor = get_system_memory_monitor()
            cache_tracker = get_cache_memory_tracker()
            _pressure_manager = MemoryPressureManager(system_monitor, cache_tracker)
        return _pressure_manager

def start_memory_monitoring() -> None:
    """Start system-wide memory monitoring."""
    monitor = get_system_memory_monitor()
    monitor.start_monitoring()

def stop_memory_monitoring() -> None:
    """Stop system-wide memory monitoring."""
    global _system_monitor
    if _system_monitor:
        _system_monitor.stop_monitoring()

def is_memory_monitoring_available() -> bool:
    """Check if memory monitoring is available (requires psutil)."""
    return _get_psutil() is not None
