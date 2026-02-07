"""
Comprehensive tests for the performance monitoring infrastructure.
"""

import json
import sqlite3
import tempfile
import time
import threading
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest

from giflab.monitoring import (
    MetricsCollector,
    MetricType,
    get_metrics_collector,
    reset_metrics_collector,
    InMemoryBackend,
    SQLiteBackend,
    track_timing,
    track_counter,
    track_gauge,
    track_histogram,
    Alert,
    AlertLevel,
    AlertManager,
    check_alerts,
)
from giflab.monitoring.metrics_collector import (
    MetricPoint,
    MetricSummary,
    RingBuffer,
    MetricsAggregator,
)
from giflab.monitoring.decorators import MetricTracker, timer_context


class TestMetricPoint:
    """Tests for MetricPoint data class."""
    
    def test_metric_point_creation(self):
        """Test creating a metric point."""
        point = MetricPoint(
            name="test.metric",
            value=42.0,
            metric_type=MetricType.GAUGE,
            timestamp=time.time(),
            tags={"env": "test"},
        )
        
        assert point.name == "test.metric"
        assert point.value == 42.0
        assert point.metric_type == MetricType.GAUGE
        assert "env" in point.tags
        assert point.tags["env"] == "test"


class TestRingBuffer:
    """Tests for RingBuffer class."""
    
    def test_ring_buffer_append(self):
        """Test appending to ring buffer."""
        buffer = RingBuffer(max_size=3)
        
        for i in range(5):
            point = MetricPoint(
                name=f"metric_{i}",
                value=i,
                metric_type=MetricType.COUNTER,
                timestamp=time.time(),
            )
            buffer.append(point)
        
        # Should only have last 3 items
        all_points = buffer.get_all()
        assert len(all_points) == 3
        assert all_points[0].name == "metric_2"
        assert all_points[-1].name == "metric_4"
    
    def test_ring_buffer_get_recent(self):
        """Test getting recent items from buffer."""
        buffer = RingBuffer()
        now = time.time()
        
        # Add points at different times
        for i in range(5):
            point = MetricPoint(
                name=f"metric_{i}",
                value=i,
                metric_type=MetricType.COUNTER,
                timestamp=now - (10 - i * 2),  # 10, 8, 6, 4, 2 seconds ago
            )
            buffer.append(point)
        
        # Get points from last 5 seconds
        recent = buffer.get_recent(5)
        assert len(recent) == 2  # Only last 2 points are within 5 seconds
        assert recent[0].name == "metric_3"
        assert recent[1].name == "metric_4"
    
    def test_ring_buffer_thread_safety(self):
        """Test thread safety of ring buffer."""
        buffer = RingBuffer(max_size=1000)
        errors = []
        
        def writer(thread_id):
            try:
                for i in range(100):
                    point = MetricPoint(
                        name=f"thread_{thread_id}_metric_{i}",
                        value=i,
                        metric_type=MetricType.COUNTER,
                        timestamp=time.time(),
                    )
                    buffer.append(point)
            except Exception as e:
                errors.append(e)
        
        # Start multiple writer threads
        threads = []
        for i in range(10):
            t = threading.Thread(target=writer, args=(i,))
            threads.append(t)
            t.start()
        
        # Wait for all threads
        for t in threads:
            t.join()
        
        assert len(errors) == 0
        assert len(buffer.get_all()) <= 1000


class TestMetricsAggregator:
    """Tests for MetricsAggregator class."""
    
    def test_counter_aggregation(self):
        """Test counter metric aggregation."""
        aggregator = MetricsAggregator(flush_interval=10.0)
        
        # Add counter values
        aggregator.add_counter("requests", 1.0, tags={"endpoint": "/api"})
        aggregator.add_counter("requests", 2.0, tags={"endpoint": "/api"})
        aggregator.add_counter("requests", 1.0, tags={"endpoint": "/health"})
        
        # Flush and check results
        points = aggregator.flush()
        
        # Should have 2 points (one per tag set)
        assert len(points) == 2
        
        api_point = next(p for p in points if p.tags.get("endpoint") == "/api")
        assert api_point.value == 3.0  # 1 + 2
        
        health_point = next(p for p in points if p.tags.get("endpoint") == "/health")
        assert health_point.value == 1.0
    
    def test_gauge_aggregation(self):
        """Test gauge metric aggregation."""
        aggregator = MetricsAggregator()
        
        # Set gauge values (last value wins)
        aggregator.set_gauge("memory.usage", 100.0)
        aggregator.set_gauge("memory.usage", 150.0)
        aggregator.set_gauge("memory.usage", 125.0)
        
        points = aggregator.flush()
        assert len(points) == 1
        assert points[0].value == 125.0  # Last value
    
    def test_histogram_aggregation(self):
        """Test histogram metric aggregation."""
        aggregator = MetricsAggregator()
        
        # Add histogram values
        values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        for v in values:
            aggregator.add_histogram("response.size", v)
        
        points = aggregator.flush()
        
        # Should have multiple percentile points
        point_names = [p.name for p in points]
        assert "response.size.min" in point_names
        assert "response.size.p50" in point_names
        assert "response.size.p95" in point_names
        assert "response.size.p99" in point_names
        assert "response.size.max" in point_names
        assert "response.size.mean" in point_names
        
        # Check values
        min_point = next(p for p in points if p.name == "response.size.min")
        assert min_point.value == 10
        
        max_point = next(p for p in points if p.name == "response.size.max")
        assert max_point.value == 100
        
        mean_point = next(p for p in points if p.name == "response.size.mean")
        assert mean_point.value == 55  # Mean of 10-100


class TestMetricsCollector:
    """Tests for MetricsCollector class."""
    
    def test_collector_initialization(self):
        """Test collector initialization."""
        backend = InMemoryBackend()
        collector = MetricsCollector(
            backend=backend,
            buffer_size=100,
            flush_interval=5.0,
            enabled=True,
            sampling_rate=1.0,
        )
        
        assert collector.backend == backend
        assert collector.enabled
        assert collector.sampling_rate == 1.0
        
        collector.shutdown()
    
    def test_record_counter(self):
        """Test recording counter metrics."""
        collector = MetricsCollector()
        
        collector.record_counter("test.counter", 1.0)
        collector.record_counter("test.counter", 2.0)
        collector.record_counter("test.counter", 3.0, tags={"type": "api"})
        
        collector.flush()
        
        # Get summary
        summaries = collector.get_summary(name="test.counter", window_seconds=60)
        assert len(summaries) > 0
        
        collector.shutdown()
    
    def test_record_gauge(self):
        """Test recording gauge metrics."""
        collector = MetricsCollector()
        
        collector.record_gauge("memory.usage", 100.0)
        collector.record_gauge("memory.usage", 150.0)
        collector.record_gauge("memory.usage", 125.0)
        
        collector.flush()
        
        summaries = collector.get_summary(name="memory.usage", window_seconds=60)
        assert len(summaries) > 0
        # Last value should be reflected
        assert any(s.max == 125.0 for s in summaries)
        
        collector.shutdown()
    
    def test_record_timer(self):
        """Test recording timer metrics."""
        collector = MetricsCollector()
        
        # Record some durations
        durations = [0.01, 0.02, 0.03, 0.04, 0.05]  # seconds
        for d in durations:
            collector.record_timer("operation.duration", d)
        
        collector.flush()
        
        summaries = collector.get_summary(name="operation.duration", window_seconds=60)
        assert len(summaries) > 0
        
        collector.shutdown()
    
    def test_sampling_rate(self):
        """Test metric sampling rate."""
        collector = MetricsCollector(sampling_rate=0.0)  # Sample nothing
        
        for _ in range(100):
            collector.record_counter("sampled.metric", 1.0)
        
        collector.flush()
        summaries = collector.get_summary(name="sampled.metric")
        assert len(summaries) == 0  # No metrics should be recorded
        
        collector.shutdown()


class TestBackends:
    """Tests for metric storage backends."""
    
    def test_in_memory_backend(self):
        """Test in-memory backend operations."""
        backend = InMemoryBackend(max_size=5)
        
        # Write metrics
        for i in range(10):
            metric = MetricPoint(
                name=f"metric_{i}",
                value=i,
                metric_type=MetricType.COUNTER,
                timestamp=time.time(),
            )
            backend.write(metric)
        
        # Should only store last 5
        metrics = backend.query()
        assert len(metrics) == 5
        assert metrics[0].name == "metric_5"
        
        # Test query filters
        metrics = backend.query(name="metric_7")
        assert len(metrics) == 1
        assert metrics[0].value == 7
    
    def test_sqlite_backend(self):
        """Test SQLite backend operations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_metrics.db"
            backend = SQLiteBackend(db_path=db_path, retention_days=1.0)
            
            # Write batch of metrics
            metrics = []
            for i in range(10):
                metrics.append(MetricPoint(
                    name=f"metric_{i % 3}",  # 3 different metric names
                    value=i,
                    metric_type=MetricType.GAUGE,
                    timestamp=time.time(),
                    tags={"env": "test"},
                ))
            
            backend.write_batch(metrics)
            
            # Query all metrics
            results = backend.query()
            assert len(results) == 10
            
            # Query by name
            results = backend.query(name="metric_0")
            assert len(results) == 4  # metric_0, metric_3, metric_6, metric_9
            
            # Check stats
            stats = backend.get_stats()
            assert stats["current_count"] == 10
            assert stats["backend"] == "sqlite"
            assert db_path.exists()
            
            # Clear and verify
            backend.clear()
            results = backend.query()
            assert len(results) == 0


class TestDecorators:
    """Tests for metric tracking decorators."""
    
    def test_track_timing_decorator(self):
        """Test timing decorator."""
        collector = MetricsCollector()
        
        @track_timing(metric_name="test.function.duration")
        def slow_function():
            time.sleep(0.01)
            return "done"
        
        # Reset collector to use our instance
        with patch("giflab.monitoring.decorators.get_metrics_collector", return_value=collector):
            result = slow_function()
            assert result == "done"
        
        collector.flush()
        summaries = collector.get_summary(name="test.function.duration")
        assert len(summaries) > 0
        # Should have recorded the duration
        assert any(s.mean >= 0.01 for s in summaries)
        
        collector.shutdown()
    
    def test_track_counter_decorator(self):
        """Test counter decorator."""
        collector = MetricsCollector()
        
        call_count = 0
        
        @track_counter(metric_name="function.calls")
        def counted_function():
            nonlocal call_count
            call_count += 1
            return call_count
        
        with patch("giflab.monitoring.decorators.get_metrics_collector", return_value=collector):
            for _ in range(5):
                counted_function()
        
        collector.flush()
        summaries = collector.get_summary(name="function.calls")
        assert len(summaries) > 0
        assert any(s.sum == 5 for s in summaries)
        
        collector.shutdown()
    
    def test_timer_context_manager(self):
        """Test timer context manager."""
        collector = MetricsCollector()
        
        with patch("giflab.monitoring.decorators.get_metrics_collector", return_value=collector):
            with timer_context("operation.time", tags={"type": "test"}):
                time.sleep(0.01)
        
        collector.flush()
        summaries = collector.get_summary(name="operation.time")
        assert len(summaries) > 0
        
        collector.shutdown()
    
    def test_metric_tracker(self):
        """Test MetricTracker class."""
        collector = MetricsCollector()
        
        with patch("giflab.monitoring.decorators.get_metrics_collector", return_value=collector):
            tracker = MetricTracker("test_operation", tags={"env": "test"})
            
            with tracker.timer("total"):
                tracker.counter("items_processed", 10)
                tracker.gauge("queue_size", 5)
                tracker.histogram("item_size", 1024)
        
        collector.flush()
        
        # Check that metrics were recorded
        summaries = collector.get_summary()
        metric_names = [s.name for s in summaries]
        
        assert any("test_operation.total" in name for name in metric_names)
        assert any("test_operation.items_processed" in name for name in metric_names)
        
        collector.shutdown()


class TestAlerting:
    """Tests for alerting system."""
    
    def test_alert_creation(self):
        """Test creating alerts."""
        alert = Alert(
            level=AlertLevel.WARNING,
            system="cache",
            metric="hit_rate",
            message="Cache hit rate low",
            value=0.35,
            threshold=0.40,
        )
        
        assert alert.level == AlertLevel.WARNING
        assert alert.system == "cache"
        assert alert.value == 0.35
        assert alert.threshold == 0.40
        assert alert.timestamp > 0
    
    def test_alert_manager_rules(self):
        """Test alert manager with rules."""
        manager = AlertManager()
        
        # Check default rules loaded
        assert len(manager.rules) > 0
        
        # Simulate low cache hit rate
        collector = MetricsCollector()
        
        # Record hits and misses
        for _ in range(3):
            collector.record_counter("cache.frame.hits", 1.0)
        for _ in range(7):
            collector.record_counter("cache.frame.misses", 1.0)
        
        collector.flush()
        
        with patch("giflab.monitoring.alerting.get_metrics_collector", return_value=collector):
            alerts = manager.evaluate_metrics(window_seconds=60)
        
        # Should have triggered alert for low hit rate (30%)
        assert len(alerts) > 0
        cache_alerts = [a for a in alerts if "cache" in a.system and "hit_rate" in a.metric]
        assert len(cache_alerts) > 0
        
        collector.shutdown()
    
    def test_alert_history(self):
        """Test alert history tracking."""
        manager = AlertManager()
        
        # Add some alerts to history
        for i in range(5):
            alert = Alert(
                level=AlertLevel.INFO,
                system="test",
                metric="test_metric",
                message=f"Test alert {i}",
                value=i,
                threshold=0,
                timestamp=time.time() - i * 3600,  # Each hour apart
            )
            manager.alert_history.append(alert)
        
        # Get last 2 hours of history
        history = manager.get_alert_history(hours=2.5)
        assert len(history) == 3  # Alerts 0, 1, 2
    
    def test_check_alerts_function(self):
        """Test convenience check_alerts function."""
        collector = MetricsCollector()
        
        # Simulate metrics
        collector.record_counter("cache.frame.hits", 10.0)
        collector.record_counter("cache.frame.misses", 90.0)  # 10% hit rate
        collector.flush()
        
        with patch("giflab.monitoring.alerting.get_metrics_collector", return_value=collector):
            alerts = check_alerts(window_seconds=60, notify=False)
        
        # Should detect low hit rate
        assert any("hit rate" in str(a).lower() for a in alerts)
        
        collector.shutdown()


class TestIntegration:
    """Integration tests for monitoring system."""
    
    @pytest.mark.slow
    def test_end_to_end_monitoring(self):
        """Test end-to-end monitoring workflow."""
        # Reset singleton
        reset_metrics_collector()
        
        # Get collector
        collector = get_metrics_collector()
        
        # Simulate cache operations
        for i in range(100):
            if i % 3 == 0:
                collector.record_counter("cache.frame.hits", 1.0)
            else:
                collector.record_counter("cache.frame.misses", 1.0)
            
            collector.record_timer("cache.frame.operation.duration", 0.001 * (i % 10))
            collector.record_gauge("cache.frame.memory_usage_mb", 50 + i % 50)
        
        # Flush metrics
        collector.flush()
        
        # Get summaries
        summaries = collector.get_summary(window_seconds=60)
        assert len(summaries) > 0
        
        # Check alerts
        alerts = check_alerts(window_seconds=60, notify=False)
        # Should have alert for low hit rate (~33%)
        assert any(a.level == AlertLevel.WARNING for a in alerts)
        
        collector.shutdown()
    
    @pytest.mark.slow
    def test_concurrent_metric_recording(self):
        """Test concurrent metric recording from multiple threads."""
        collector = MetricsCollector()
        errors = []
        
        def record_metrics(thread_id):
            try:
                for i in range(100):
                    collector.record_counter(f"thread.{thread_id}.counter", 1.0)
                    collector.record_gauge(f"thread.{thread_id}.gauge", i)
                    collector.record_timer(f"thread.{thread_id}.timer", 0.001 * i)
                    
                    if i % 10 == 0:
                        collector.flush()
            except Exception as e:
                errors.append(e)
        
        # Start threads
        threads = []
        for i in range(5):
            t = threading.Thread(target=record_metrics, args=(i,))
            threads.append(t)
            t.start()
        
        # Wait for completion
        for t in threads:
            t.join()
        
        assert len(errors) == 0
        
        # Final flush
        collector.flush()
        
        # Check we have metrics from all threads
        summaries = collector.get_summary()
        thread_metrics = set()
        for s in summaries:
            if "thread" in s.name:
                thread_id = s.name.split(".")[1]
                thread_metrics.add(thread_id)
        
        assert len(thread_metrics) == 5  # All threads recorded metrics
        
        collector.shutdown()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])