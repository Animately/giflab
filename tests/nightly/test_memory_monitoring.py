"""
Tests for memory monitoring and pressure detection systems.

Tests cover:
- Memory statistics collection
- Pressure level detection
- Cache memory tracking
- Eviction policies
- Integration with monitoring system
"""

import threading
import time
import unittest.mock
from unittest.mock import MagicMock, patch

import pytest

from giflab.monitoring.memory_monitor import (
    MemoryStats,
    MemoryPressureLevel,
    CacheMemoryUsage,
    SystemMemoryMonitor,
    CacheMemoryTracker,
    MemoryPressureManager,
    ConservativeEvictionPolicy,
    get_system_memory_monitor,
    get_cache_memory_tracker,
    get_memory_pressure_manager,
    is_memory_monitoring_available,
)
from giflab.monitoring.memory_integration import (
    MemoryPressureIntegration,
    get_memory_integration,
    instrument_cache_with_memory_tracking,
)


class TestMemoryStats:
    """Test MemoryStats dataclass functionality."""
    
    def test_memory_stats_creation(self):
        """Test creating MemoryStats with all fields."""
        stats = MemoryStats(
            process_memory_mb=100.0,
            process_memory_percent=1.0,
            system_memory_mb=8000.0,
            system_memory_percent=0.75,
            system_available_mb=2000.0,
            total_system_mb=10000.0,
            pressure_level=MemoryPressureLevel.WARNING,
            timestamp=time.time()
        )
        
        assert stats.process_memory_mb == 100.0
        assert stats.system_memory_percent == 0.75
        assert stats.pressure_level == MemoryPressureLevel.WARNING
    
    def test_pressure_level_enum(self):
        """Test MemoryPressureLevel enum values."""
        assert MemoryPressureLevel.NORMAL.value == "normal"
        assert MemoryPressureLevel.WARNING.value == "warning"
        assert MemoryPressureLevel.CRITICAL.value == "critical"
        assert MemoryPressureLevel.EMERGENCY.value == "emergency"


class TestConservativeEvictionPolicy:
    """Test conservative eviction policy logic."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.policy = ConservativeEvictionPolicy(
            warning_threshold=0.8,
            critical_threshold=0.95
        )
    
    def test_no_eviction_below_threshold(self):
        """Test no eviction when memory usage is normal."""
        stats = MemoryStats(
            process_memory_mb=100.0,
            process_memory_percent=1.0,
            system_memory_mb=6000.0,
            system_memory_percent=0.6,  # Below warning threshold
            system_available_mb=4000.0,
            total_system_mb=10000.0,
            pressure_level=MemoryPressureLevel.NORMAL,
            timestamp=time.time()
        )
        
        cache_usage = CacheMemoryUsage(total_cache_mb=50.0)
        
        assert not self.policy.should_evict(stats, cache_usage)
        assert self.policy.get_eviction_target_mb(stats) == 0.0
    
    def test_eviction_at_warning_level(self):
        """Test eviction at warning level."""
        stats = MemoryStats(
            process_memory_mb=100.0,
            process_memory_percent=1.0,
            system_memory_mb=8500.0,
            system_memory_percent=0.85,  # Above warning threshold
            system_available_mb=1500.0,
            total_system_mb=10000.0,
            pressure_level=MemoryPressureLevel.WARNING,
            timestamp=time.time()
        )
        
        cache_usage = CacheMemoryUsage(total_cache_mb=50.0)
        
        assert self.policy.should_evict(stats, cache_usage)
        target_mb = self.policy.get_eviction_target_mb(stats)
        assert target_mb == 100.0 * 0.15  # 15% of process memory
    
    def test_eviction_at_critical_level(self):
        """Test higher eviction target at critical level."""
        stats = MemoryStats(
            process_memory_mb=200.0,
            process_memory_percent=2.0,
            system_memory_mb=9600.0,
            system_memory_percent=0.96,  # Above critical threshold
            system_available_mb=400.0,
            total_system_mb=10000.0,
            pressure_level=MemoryPressureLevel.CRITICAL,
            timestamp=time.time()
        )
        
        cache_usage = CacheMemoryUsage(total_cache_mb=100.0)
        
        assert self.policy.should_evict(stats, cache_usage)
        target_mb = self.policy.get_eviction_target_mb(stats)
        assert target_mb == 200.0 * 0.3  # 30% of process memory


class TestSystemMemoryMonitor:
    """Test system memory monitoring functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.monitor = SystemMemoryMonitor(update_interval=0.1)
    
    def teardown_method(self):
        """Clean up after tests."""
        self.monitor.stop_monitoring()
    
    def test_pressure_level_calculation(self):
        """Test memory pressure level calculation."""
        # Test normal level
        level = self.monitor._calculate_pressure_level(0.5)
        assert level == MemoryPressureLevel.NORMAL
        
        # Test warning level
        level = self.monitor._calculate_pressure_level(0.75)
        assert level == MemoryPressureLevel.WARNING
        
        # Test critical level
        level = self.monitor._calculate_pressure_level(0.85)
        assert level == MemoryPressureLevel.CRITICAL
        
        # Test emergency level
        level = self.monitor._calculate_pressure_level(0.98)
        assert level == MemoryPressureLevel.EMERGENCY
    
    def test_fallback_stats_when_psutil_unavailable(self):
        """Test fallback behavior when psutil is not available."""
        # Force the monitor to use fallback by setting _psutil to None
        original_psutil = self.monitor._psutil
        self.monitor._psutil = None
        
        try:
            stats = self.monitor.collect_memory_stats()
            
            assert stats.process_memory_mb == 0.0
            assert stats.system_memory_percent == 0.0
            assert stats.system_available_mb == float('inf')
            assert stats.pressure_level == MemoryPressureLevel.NORMAL
        finally:
            self.monitor._psutil = original_psutil
    
    def test_memory_stats_collection_with_psutil(self):
        """Test memory stats collection when psutil is available."""
        # Mock psutil directly on the monitor instance
        mock_psutil = MagicMock()
        original_psutil = self.monitor._psutil
        self.monitor._psutil = mock_psutil
        
        try:
            # Mock virtual memory
            mock_virtual_mem = MagicMock()
            mock_virtual_mem.used = 8 * 1024 * 1024 * 1024  # 8GB
            mock_virtual_mem.percent = 80.0
            mock_virtual_mem.available = 2 * 1024 * 1024 * 1024  # 2GB
            mock_virtual_mem.total = 10 * 1024 * 1024 * 1024  # 10GB
            mock_psutil.virtual_memory.return_value = mock_virtual_mem
            
            # Mock process memory
            mock_process = MagicMock()
            mock_memory_info = MagicMock()
            mock_memory_info.rss = 100 * 1024 * 1024  # 100MB
            mock_process.memory_info.return_value = mock_memory_info
            mock_psutil.Process.return_value = mock_process
            
            stats = self.monitor.collect_memory_stats()
            
            assert abs(stats.system_memory_mb - 8192.0) < 1.0  # ~8GB
            assert abs(stats.system_memory_percent - 0.8) < 0.01
            assert abs(stats.process_memory_mb - 100.0) < 1.0  # ~100MB
            assert stats.pressure_level == MemoryPressureLevel.CRITICAL
        finally:
            self.monitor._psutil = original_psutil
    
    def test_monitoring_start_stop(self):
        """Test starting and stopping monitoring."""
        assert not self.monitor._monitoring_active
        
        self.monitor.start_monitoring()
        assert self.monitor._monitoring_active
        assert self.monitor._monitor_thread.is_alive()
        
        # Wait briefly for monitoring to collect stats
        time.sleep(0.2)
        stats = self.monitor.get_current_stats()
        assert stats is not None
        
        self.monitor.stop_monitoring()
        assert not self.monitor._monitoring_active


class TestCacheMemoryTracker:
    """Test cache memory tracking functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.tracker = CacheMemoryTracker()
    
    def test_cache_size_tracking(self):
        """Test updating and retrieving cache sizes."""
        # Initially empty
        usage = self.tracker.get_total_cache_usage()
        assert usage.total_cache_mb == 0.0
        
        # Update cache sizes
        self.tracker.update_cache_size("frame_cache", 100.0)
        self.tracker.update_cache_size("resize_cache", 50.0)
        self.tracker.update_cache_size("validation_cache", 25.0)
        
        usage = self.tracker.get_total_cache_usage()
        assert usage.frame_cache_mb == 100.0
        assert usage.resize_cache_mb == 50.0
        assert usage.validation_cache_mb == 25.0
        assert usage.total_cache_mb == 175.0
    
    def test_cache_size_reset(self):
        """Test resetting specific cache sizes."""
        self.tracker.update_cache_size("frame_cache", 100.0)
        self.tracker.update_cache_size("resize_cache", 50.0)
        
        self.tracker.reset_cache_size("frame_cache")
        
        usage = self.tracker.get_total_cache_usage()
        assert usage.frame_cache_mb == 0.0
        assert usage.resize_cache_mb == 50.0
        assert usage.total_cache_mb == 50.0
    
    def test_thread_safety(self):
        """Test thread-safe cache tracking."""
        def update_cache_worker(cache_type: str, size: float):
            for i in range(10):
                self.tracker.update_cache_size(cache_type, size + i)
                time.sleep(0.01)
        
        threads = [
            threading.Thread(target=update_cache_worker, args=("cache_1", 10.0)),
            threading.Thread(target=update_cache_worker, args=("cache_2", 20.0)),
        ]
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Verify final state is consistent
        usage = self.tracker.get_total_cache_usage()
        assert usage.total_cache_mb > 0


class TestMemoryPressureManager:
    """Test memory pressure management and eviction."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.system_monitor = SystemMemoryMonitor()
        self.cache_tracker = CacheMemoryTracker()
        self.eviction_policy = ConservativeEvictionPolicy()
        self.manager = MemoryPressureManager(
            self.system_monitor,
            self.cache_tracker,
            self.eviction_policy
        )
    
    def test_eviction_callback_registration(self):
        """Test registering eviction callbacks."""
        callback = MagicMock(return_value=50.0)
        
        self.manager.register_eviction_callback("test_cache", callback)
        
        assert "test_cache" in self.manager._eviction_callbacks
        assert self.manager._eviction_callbacks["test_cache"] == callback
    
    def test_memory_pressure_check_no_pressure(self):
        """Test pressure check when memory usage is normal."""
        # Mock normal memory stats
        normal_stats = MemoryStats(
            process_memory_mb=100.0,
            process_memory_percent=1.0,
            system_memory_mb=6000.0,
            system_memory_percent=0.6,
            system_available_mb=4000.0,
            total_system_mb=10000.0,
            pressure_level=MemoryPressureLevel.NORMAL,
            timestamp=time.time()
        )
        
        with patch.object(self.system_monitor, 'get_current_stats', return_value=normal_stats):
            should_evict, target_mb = self.manager.check_memory_pressure()
            
            assert not should_evict
            assert target_mb is None
    
    def test_memory_pressure_check_with_pressure(self):
        """Test pressure check when eviction is needed."""
        # Mock high memory stats
        high_stats = MemoryStats(
            process_memory_mb=200.0,
            process_memory_percent=2.0,
            system_memory_mb=8500.0,
            system_memory_percent=0.85,
            system_available_mb=1500.0,
            total_system_mb=10000.0,
            pressure_level=MemoryPressureLevel.WARNING,
            timestamp=time.time()
        )
        
        with patch.object(self.system_monitor, 'get_current_stats', return_value=high_stats):
            should_evict, target_mb = self.manager.check_memory_pressure()
            
            assert should_evict
            assert target_mb == 200.0 * 0.15  # 15% of process memory
    
    def test_eviction_execution(self):
        """Test executing eviction across multiple caches."""
        # Register mock eviction callbacks
        frame_callback = MagicMock(return_value=20.0)
        resize_callback = MagicMock(return_value=15.0)
        validation_callback = MagicMock(return_value=10.0)
        
        self.manager.register_eviction_callback("frame_cache", frame_callback)
        self.manager.register_eviction_callback("resize_cache", resize_callback)
        self.manager.register_eviction_callback("validation_cache", validation_callback)
        
        # Execute eviction
        freed_mb = self.manager.execute_eviction(40.0)
        
        # Verify callbacks were called in priority order
        validation_callback.assert_called_once()
        resize_callback.assert_called_once()
        frame_callback.assert_called_once()
        
        assert freed_mb == 45.0  # Total freed across all caches


class TestMemoryIntegration:
    """Test memory monitoring integration."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.integration = MemoryPressureIntegration()
    
    @patch('giflab.monitoring.memory_integration.MONITORING')
    @patch('giflab.monitoring.memory_integration.start_memory_monitoring')
    def test_integration_initialization(self, mock_start_monitoring, mock_config):
        """Test memory integration initialization."""
        mock_config.get.return_value = {"enabled": True, "auto_eviction": True}
        
        result = self.integration.initialize()
        
        assert result
        assert self.integration._initialized
        mock_start_monitoring.assert_called_once()
    
    @patch('giflab.monitoring.memory_integration.MONITORING')
    def test_integration_disabled_by_config(self, mock_config):
        """Test integration disabled via configuration."""
        mock_config.get.return_value = {"enabled": False}
        
        result = self.integration.initialize()
        
        assert not result
        assert not self.integration._initialized
    
    def test_cache_instrumentation(self):
        """Test instrumenting cache with memory tracking."""
        size_callback = MagicMock(return_value=100.0)
        eviction_callback = MagicMock(return_value=50.0)
        
        with patch('giflab.monitoring.memory_integration.get_cache_memory_tracker') as mock_tracker:
            with patch('giflab.monitoring.memory_integration.get_memory_pressure_manager') as mock_manager:
                mock_tracker_instance = MagicMock()
                mock_manager_instance = MagicMock()
                mock_tracker.return_value = mock_tracker_instance
                mock_manager.return_value = mock_manager_instance
                
                instrument_cache_with_memory_tracking(
                    "test_cache",
                    size_callback,
                    eviction_callback
                )
                
                # Verify registration calls
                mock_manager_instance.register_eviction_callback.assert_called_with(
                    "test_cache", eviction_callback
                )
                mock_tracker_instance.update_cache_size.assert_called_with(
                    "test_cache", 100.0
                )


class TestSingletonAccess:
    """Test singleton access functions."""
    
    def test_get_system_memory_monitor(self):
        """Test singleton access to system memory monitor."""
        monitor1 = get_system_memory_monitor()
        monitor2 = get_system_memory_monitor()
        
        assert monitor1 is monitor2
        assert isinstance(monitor1, SystemMemoryMonitor)
    
    def test_get_cache_memory_tracker(self):
        """Test singleton access to cache memory tracker.""" 
        tracker1 = get_cache_memory_tracker()
        tracker2 = get_cache_memory_tracker()
        
        assert tracker1 is tracker2
        assert isinstance(tracker1, CacheMemoryTracker)
    
    def test_get_memory_pressure_manager(self):
        """Test singleton access to memory pressure manager."""
        manager1 = get_memory_pressure_manager()
        manager2 = get_memory_pressure_manager()
        
        assert manager1 is manager2
        assert isinstance(manager1, MemoryPressureManager)


class TestMemoryMonitoringAvailability:
    """Test memory monitoring availability detection."""
    
    @patch('giflab.monitoring.memory_monitor._get_psutil')
    def test_availability_with_psutil(self, mock_get_psutil):
        """Test availability when psutil is available."""
        mock_get_psutil.return_value = MagicMock()
        
        assert is_memory_monitoring_available()
    
    @patch('giflab.monitoring.memory_monitor._get_psutil')
    def test_availability_without_psutil(self, mock_get_psutil):
        """Test availability when psutil is not available."""
        mock_get_psutil.return_value = None
        
        assert not is_memory_monitoring_available()


@pytest.mark.integration
class TestMemoryMonitoringIntegration:
    """Integration tests for complete memory monitoring system."""
    
    def test_end_to_end_monitoring_cycle(self):
        """Test complete monitoring cycle from collection to eviction."""
        # This test would require careful setup and teardown
        # to avoid interfering with the actual system
        pass
    
    def test_memory_pressure_with_real_cache(self):
        """Test memory pressure detection with real cache usage.""" 
        # This test would create actual cache load and verify
        # that pressure detection and eviction work correctly
        pass


if __name__ == "__main__":
    pytest.main([__file__])