"""
Tests for cache effectiveness monitoring and analysis.

Trimmed test suite covering representative scenarios:
- Cache effectiveness metrics collection and analysis
- Performance baseline framework functionality
- Effectiveness analysis and recommendation generation
- Integration with memory monitoring systems

Phase 3.2 Implementation: Ensure cache effectiveness monitoring is robust and reliable.
"""

import time
from unittest.mock import MagicMock, Mock, patch

import pytest

from src.giflab.monitoring.baseline_framework import (
    BaselineStatistics,
    BaselineTestMode,
    PerformanceBaselineFramework,
    PerformanceMeasurement,
    WorkloadScenario,
    baseline_performance_test,
)
from src.giflab.monitoring.cache_effectiveness import (
    BaselineComparison,
    CacheEffectivenessMonitor,
    CacheEffectivenessStats,
    CacheOperationType,
    get_cache_effectiveness_monitor,
)
from src.giflab.monitoring.effectiveness_analysis import (
    CacheEffectivenessAnalysis,
    CacheEffectivenessAnalyzer,
    CacheRecommendation,
    analyze_cache_effectiveness,
)
from src.giflab.monitoring.memory_monitor import MemoryPressureLevel


class TestCacheEffectivenessMonitor:
    """Test cache effectiveness monitoring functionality."""

    def test_monitor_initialization(self):
        """Test cache effectiveness monitor initialization."""
        monitor = CacheEffectivenessMonitor(
            max_operations_history=1000,
            time_window_minutes=30
        )

        assert monitor.max_operations_history == 1000
        assert monitor.time_window_minutes == 30
        assert len(monitor._operations) == 0
        assert len(monitor._cache_stats) == 0

    def test_record_cache_operation_hit(self):
        """Test recording cache hit operations."""
        monitor = CacheEffectivenessMonitor()

        monitor.record_operation(
            cache_type="frame_cache",
            operation=CacheOperationType.HIT,
            key="test_key_1",
            processing_time_ms=50.0
        )

        assert len(monitor._operations) == 1
        assert "frame_cache" in monitor._cache_stats

        stats = monitor._cache_stats["frame_cache"]
        assert stats.total_operations == 1
        assert stats.hits == 1
        assert stats.misses == 0

    def test_calculate_hit_rate(self):
        """Test hit rate calculation."""
        monitor = CacheEffectivenessMonitor()

        # Record mixed operations
        for i in range(8):  # 8 hits
            monitor.record_operation("cache", CacheOperationType.HIT, f"key_{i}")

        for i in range(2):  # 2 misses
            monitor.record_operation("cache", CacheOperationType.MISS, f"key_miss_{i}")

        effectiveness = monitor.get_cache_effectiveness("cache")
        assert effectiveness is not None
        assert effectiveness.hit_rate == 0.8  # 8/(8+2)
        assert effectiveness.hits == 8
        assert effectiveness.misses == 2

    def test_baseline_comparison(self):
        """Test baseline performance comparison."""
        monitor = CacheEffectivenessMonitor()

        # Record baseline (non-cached) times (need at least 10 samples)
        baseline_times = [100, 110, 105, 95, 120, 115, 108, 102, 98, 125, 103, 107]
        for time_ms in baseline_times:
            monitor.record_baseline_performance("operation_a", time_ms)

        # Record cached times (need at least 10 samples)
        cached_times = [50, 55, 45, 60, 40, 48, 52, 58, 42, 46, 53, 49]
        for time_ms in cached_times:
            monitor.record_cached_performance("operation_a", time_ms)

        comparison = monitor.get_baseline_comparison("operation_a")
        assert comparison is not None
        assert comparison.operation_type == "operation_a"
        assert comparison.performance_improvement > 0.4  # Should be ~50% improvement
        assert comparison.sample_size_cached == 12
        assert comparison.sample_size_non_cached == 12

    def test_system_effectiveness_summary(self):
        """Test system-wide effectiveness summary."""
        monitor = CacheEffectivenessMonitor()

        # Record operations for multiple cache types
        for cache_type in ["frame_cache", "resize_cache"]:
            for i in range(10):
                monitor.record_operation(cache_type, CacheOperationType.HIT, f"key_{i}")
            for i in range(5):
                monitor.record_operation(cache_type, CacheOperationType.MISS, f"miss_{i}")

        summary = monitor.get_system_effectiveness_summary()
        assert summary["total_operations"] == 30  # 15 ops * 2 cache types
        assert summary["overall_hit_rate"] == 20/30  # 20 hits out of 30 total
        assert summary["cache_types"] == 2


class TestPerformanceBaselineFramework:
    """Test performance baseline framework functionality."""

    def test_framework_initialization(self):
        """Test baseline framework initialization."""
        framework = PerformanceBaselineFramework(
            test_mode=BaselineTestMode.AB_TESTING,
            ab_split_ratio=0.2
        )

        assert framework.test_mode == BaselineTestMode.AB_TESTING
        assert framework.ab_split_ratio == 0.2
        assert len(framework._measurements) == 0

    def test_record_performance_measurement(self):
        """Test recording performance measurements."""
        framework = PerformanceBaselineFramework()

        framework.record_performance(
            operation_type="gif_processing",
            processing_time_ms=100.0,
            cache_enabled=True,
            memory_usage_mb=50.0,
            metadata={"file_size": "large"}
        )

        measurements = framework._measurements["gif_processing"]
        assert len(measurements) == 1

        measurement = measurements[0]
        assert measurement.operation_type == "gif_processing"
        assert measurement.processing_time_ms == 100.0
        assert measurement.cache_enabled is True
        assert measurement.metadata["file_size"] == "large"

    def test_baseline_statistics_calculation(self):
        """Test baseline statistics calculation."""
        framework = PerformanceBaselineFramework(min_samples_for_analysis=15)  # Lower threshold for testing

        # Record cached measurements
        for i in range(20):
            framework.record_performance("test_op", 50.0 + i, True)

        # Record non-cached measurements
        for i in range(20):
            framework.record_performance("test_op", 100.0 + i, False)

        stats = framework.get_baseline_statistics("test_op")
        assert stats is not None
        assert stats.cached_samples == 20
        assert stats.non_cached_samples == 20
        assert stats.performance_improvement > 0.4  # Should be significant improvement
        assert stats.min_samples_met is True


class TestCacheEffectivenessAnalyzer:
    """Test cache effectiveness analysis and recommendation generation."""

    @patch('src.giflab.monitoring.effectiveness_analysis.get_cache_effectiveness_monitor')
    @patch('src.giflab.monitoring.effectiveness_analysis.get_baseline_framework')
    @patch('src.giflab.monitoring.effectiveness_analysis.get_cache_memory_tracker')
    def test_insufficient_data_analysis(self, mock_memory, mock_baseline, mock_effectiveness):
        """Test analysis with insufficient data."""
        # Mock insufficient data scenario
        mock_effectiveness.return_value.get_all_cache_stats.return_value = {}
        mock_effectiveness.return_value.get_system_effectiveness_summary.return_value = {
            "total_operations": 10,  # Too few operations
            "overall_hit_rate": 0.0,
            "monitoring_duration_hours": 0.5  # Too short duration
        }
        mock_baseline.return_value.get_all_baseline_statistics.return_value = {}
        mock_baseline.return_value.generate_performance_report.return_value = {"performance_analysis": {}}
        mock_memory.return_value.get_system_effectiveness_summary.return_value = {}

        analyzer = CacheEffectivenessAnalyzer()
        analysis = analyzer.analyze_cache_effectiveness()

        assert analysis.recommendation == CacheRecommendation.INSUFFICIENT_DATA
        assert analysis.confidence_score < 0.2
        assert len(analysis.optimization_recommendations) > 0

    @patch('src.giflab.monitoring.effectiveness_analysis.get_cache_effectiveness_monitor')
    @patch('src.giflab.monitoring.effectiveness_analysis.get_baseline_framework')
    @patch('src.giflab.monitoring.effectiveness_analysis.get_cache_memory_tracker')
    def test_excellent_performance_analysis(self, mock_memory, mock_baseline, mock_effectiveness):
        """Test analysis with excellent cache performance."""
        # Mock excellent performance data
        mock_cache_stats = {
            "frame_cache": Mock(
                hit_rate=0.9,
                total_operations=1000,
                hits=900,
                misses=100,
                evictions=10,
                puts=200,
                total_data_cached_mb=100.0,
                cache_turnover_rate=0.05
            )
        }

        mock_effectiveness.return_value.get_all_cache_stats.return_value = mock_cache_stats
        mock_effectiveness.return_value.get_system_effectiveness_summary.return_value = {
            "total_operations": 1000,
            "overall_hit_rate": 0.9,
            "monitoring_duration_hours": 5.0
        }

        mock_baseline_stats = {
            "gif_processing": Mock(
                operation_type="gif_processing",
                performance_improvement=0.4,  # 40% improvement
                statistical_significance=True,
                min_samples_met=True,
                cached_samples=100,
                non_cached_samples=100
            )
        }

        mock_baseline.return_value.get_all_baseline_statistics.return_value = mock_baseline_stats
        mock_baseline.return_value.generate_performance_report.return_value = {
            "performance_analysis": {"average_improvement": 0.4}
        }

        mock_memory.return_value.get_system_effectiveness_summary.return_value = {
            "efficiency_score": 0.9
        }

        analyzer = CacheEffectivenessAnalyzer()
        analysis = analyzer.analyze_cache_effectiveness()

        assert analysis.recommendation == CacheRecommendation.ENABLE_PRODUCTION
        assert analysis.confidence_score > 0.8
        assert analysis.overall_hit_rate == 0.9
        assert analysis.average_performance_improvement == 0.4


class TestIntegration:
    """Test integration between cache effectiveness components."""

    @patch('src.giflab.monitoring.effectiveness_analysis.get_cache_effectiveness_monitor')
    @patch('src.giflab.monitoring.effectiveness_analysis.get_baseline_framework')
    @patch('src.giflab.monitoring.effectiveness_analysis.get_cache_memory_tracker')
    def test_end_to_end_analysis(self, mock_memory, mock_baseline, mock_effectiveness):
        """Test end-to-end cache effectiveness analysis."""
        # Setup mock data representing a realistic scenario
        mock_cache_stats = {
            "frame_cache": Mock(
                hit_rate=0.7,
                total_operations=500,
                hits=350,
                misses=150,
                evictions=25,
                puts=100,
                total_data_cached_mb=75.0,
                cache_turnover_rate=0.25
            )
        }

        mock_effectiveness.return_value.get_all_cache_stats.return_value = mock_cache_stats
        mock_effectiveness.return_value.get_system_effectiveness_summary.return_value = {
            "total_operations": 500,
            "overall_hit_rate": 0.7,
            "monitoring_duration_hours": 3.0
        }

        mock_baseline_stats = {
            "frame_processing": Mock(
                operation_type="frame_processing",
                performance_improvement=0.25,  # 25% improvement
                statistical_significance=True,
                min_samples_met=True,
                cached_samples=50,
                non_cached_samples=50
            )
        }

        mock_baseline.return_value.get_all_baseline_statistics.return_value = mock_baseline_stats
        mock_baseline.return_value.generate_performance_report.return_value = {
            "performance_analysis": {"average_improvement": 0.25}
        }

        mock_memory.return_value.get_system_effectiveness_summary.return_value = {
            "efficiency_score": 0.75
        }

        # Run analysis
        analysis = analyze_cache_effectiveness()

        # Verify results
        assert analysis.recommendation in [CacheRecommendation.ENABLE_PRODUCTION, CacheRecommendation.ENABLE_WITH_MONITORING]
        assert analysis.confidence_score > 0.6
        assert analysis.overall_hit_rate == 0.7
        assert analysis.average_performance_improvement == 0.25
        assert len(analysis.optimization_recommendations) >= 0


if __name__ == "__main__":
    pytest.main([__file__])
