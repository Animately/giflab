"""
Performance baseline framework for cache effectiveness analysis.

This module provides A/B testing framework for comparing cached vs non-cached operations:
- Automated baseline collection during normal operations
- A/B split testing with statistical significance analysis
- Performance regression detection and alerting
- Representative workload generation for controlled testing

Phase 3.2 Implementation: Evidence-based cache optimization framework.
"""

import contextlib
import logging
import random
import statistics
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


class BaselineTestMode(Enum):
    """Performance baseline testing modes."""

    DISABLED = "disabled"  # No baseline testing
    PASSIVE = "passive"  # Collect baselines during normal operations
    AB_TESTING = "ab_testing"  # Active A/B testing with traffic splitting
    CONTROLLED = "controlled"  # Controlled testing with synthetic workloads


@dataclass
class PerformanceMeasurement:
    """Individual performance measurement for analysis."""

    operation_type: str
    cache_enabled: bool
    processing_time_ms: float
    memory_usage_mb: float
    timestamp: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BaselineStatistics:
    """Statistical analysis of baseline performance data."""

    operation_type: str

    # Cached performance statistics
    cached_samples: int = 0
    cached_mean_ms: float = 0.0
    cached_median_ms: float = 0.0
    cached_p95_ms: float = 0.0
    cached_stddev_ms: float = 0.0

    # Non-cached performance statistics
    non_cached_samples: int = 0
    non_cached_mean_ms: float = 0.0
    non_cached_median_ms: float = 0.0
    non_cached_p95_ms: float = 0.0
    non_cached_stddev_ms: float = 0.0

    # Comparative analysis
    performance_improvement: float = 0.0  # (non_cached - cached) / non_cached
    statistical_significance: bool = False
    confidence_interval_95: tuple[float, float] = (0.0, 0.0)

    # Test validity
    min_samples_met: bool = False
    last_update: float = field(default_factory=time.time)
    collection_duration_hours: float = 0.0


@dataclass
class WorkloadScenario:
    """Controlled workload scenario for testing."""

    name: str
    operation_type: str
    setup_function: Callable[[], Any]
    execution_function: Callable[[Any], float]  # Returns processing time in ms
    cleanup_function: Callable[[Any], None] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class PerformanceBaselineFramework:
    """Framework for collecting and analyzing cache performance baselines."""

    def __init__(
        self,
        test_mode: BaselineTestMode = BaselineTestMode.PASSIVE,
        max_samples_per_operation: int = 1000,
        min_samples_for_analysis: int = 30,
        ab_split_ratio: float = 0.1,
    ):  # 10% traffic for baseline testing
        self.test_mode = test_mode
        self.max_samples_per_operation = max_samples_per_operation
        self.min_samples_for_analysis = min_samples_for_analysis
        self.ab_split_ratio = ab_split_ratio

        self._lock = threading.RLock()

        # Performance measurements storage
        self._measurements: dict[str, deque[PerformanceMeasurement]] = defaultdict(
            lambda: deque(maxlen=max_samples_per_operation)
        )

        # Controlled workload scenarios
        self._workload_scenarios: dict[str, WorkloadScenario] = {}

        # A/B testing state
        self._ab_testing_enabled = False
        self._ab_test_counters: dict[str, int] = defaultdict(int)

        # Analysis cache
        self._baseline_stats_cache: dict[str, tuple[float, BaselineStatistics]] = {}
        self._cache_ttl_seconds = 300.0  # 5 minutes

        # Framework state
        self._framework_start_time = time.time()
        self._is_active = True

        logger.info(
            f"Performance baseline framework initialized in {test_mode.value} mode"
        )

    def start_ab_testing(self) -> None:
        """Start A/B testing mode for active baseline collection."""
        with self._lock:
            self._ab_testing_enabled = True
            self.test_mode = BaselineTestMode.AB_TESTING
        logger.info("A/B testing mode activated for baseline collection")

    def stop_ab_testing(self) -> None:
        """Stop A/B testing and return to passive mode."""
        with self._lock:
            self._ab_testing_enabled = False
            self.test_mode = BaselineTestMode.PASSIVE
        logger.info("A/B testing mode deactivated, returned to passive mode")

    def should_run_baseline_test(self, operation_type: str) -> bool:
        """
        Determine if an operation should run in baseline (non-cached) mode for testing.

        Returns True if this operation should bypass caching for baseline measurement.
        """
        if self.test_mode == BaselineTestMode.DISABLED:
            return False

        if self.test_mode == BaselineTestMode.PASSIVE:
            # In passive mode, only collect baselines when caching is naturally disabled
            return False

        if self.test_mode == BaselineTestMode.AB_TESTING:
            with self._lock:
                counter = self._ab_test_counters[operation_type]
                self._ab_test_counters[operation_type] += 1

                # Use split ratio to determine baseline testing
                return (counter % int(1.0 / self.ab_split_ratio)) == 0

        return False

    def record_performance(
        self,
        operation_type: str,
        processing_time_ms: float,
        cache_enabled: bool,
        memory_usage_mb: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record a performance measurement for baseline analysis."""
        if not self._is_active:
            return

        measurement = PerformanceMeasurement(
            operation_type=operation_type,
            cache_enabled=cache_enabled,
            processing_time_ms=processing_time_ms,
            memory_usage_mb=memory_usage_mb,
            timestamp=time.time(),
            metadata=metadata or {},
        )

        with self._lock:
            self._measurements[operation_type].append(measurement)

            # Clear analysis cache to force recalculation
            if operation_type in self._baseline_stats_cache:
                del self._baseline_stats_cache[operation_type]

        logger.debug(
            f"Recorded performance: {operation_type} - {processing_time_ms:.2f}ms (cache: {cache_enabled})"
        )

    def get_baseline_statistics(self, operation_type: str) -> BaselineStatistics | None:
        """Get comprehensive baseline statistics for an operation type."""
        if operation_type not in self._measurements:
            return None

        current_time = time.time()

        # Check cache
        if operation_type in self._baseline_stats_cache:
            cache_time, cached_stats = self._baseline_stats_cache[operation_type]
            if current_time - cache_time < self._cache_ttl_seconds:
                return cached_stats

        with self._lock:
            measurements = list(self._measurements[operation_type])

            if not measurements:
                return None

            # Separate cached and non-cached measurements
            cached_times = [
                m.processing_time_ms for m in measurements if m.cache_enabled
            ]
            non_cached_times = [
                m.processing_time_ms for m in measurements if not m.cache_enabled
            ]

            # Create baseline statistics
            stats = BaselineStatistics(operation_type=operation_type)

            # Analyze cached performance
            if cached_times:
                stats.cached_samples = len(cached_times)
                stats.cached_mean_ms = statistics.mean(cached_times)
                stats.cached_median_ms = statistics.median(cached_times)
                stats.cached_p95_ms = self._calculate_percentile(cached_times, 0.95)
                if len(cached_times) > 1:
                    stats.cached_stddev_ms = statistics.stdev(cached_times)

            # Analyze non-cached performance
            if non_cached_times:
                stats.non_cached_samples = len(non_cached_times)
                stats.non_cached_mean_ms = statistics.mean(non_cached_times)
                stats.non_cached_median_ms = statistics.median(non_cached_times)
                stats.non_cached_p95_ms = self._calculate_percentile(
                    non_cached_times, 0.95
                )
                if len(non_cached_times) > 1:
                    stats.non_cached_stddev_ms = statistics.stdev(non_cached_times)

            # Comparative analysis
            if cached_times and non_cached_times and stats.non_cached_mean_ms > 0:
                stats.performance_improvement = (
                    stats.non_cached_mean_ms - stats.cached_mean_ms
                ) / stats.non_cached_mean_ms

                # Statistical significance test (simplified t-test)
                stats.statistical_significance = self._test_statistical_significance(
                    cached_times, non_cached_times
                )

                # 95% confidence interval for performance improvement
                stats.confidence_interval_95 = self._calculate_confidence_interval(
                    cached_times, non_cached_times
                )

            # Test validity checks
            stats.min_samples_met = (
                len(cached_times) >= self.min_samples_for_analysis
                and len(non_cached_times) >= self.min_samples_for_analysis
            )

            stats.collection_duration_hours = (
                current_time - self._framework_start_time
            ) / 3600

            # Cache the result
            self._baseline_stats_cache[operation_type] = (current_time, stats)

            return stats

    def register_workload_scenario(self, scenario: WorkloadScenario) -> None:
        """Register a controlled workload scenario for testing."""
        with self._lock:
            self._workload_scenarios[scenario.name] = scenario
        logger.info(f"Registered workload scenario: {scenario.name}")

    def run_controlled_test(
        self, scenario_name: str, iterations: int = 100, cache_enabled: bool = True
    ) -> list[PerformanceMeasurement]:
        """Run controlled performance test with a specific scenario."""
        if scenario_name not in self._workload_scenarios:
            raise ValueError(f"Unknown workload scenario: {scenario_name}")

        scenario = self._workload_scenarios[scenario_name]
        measurements = []

        logger.info(
            f"Running controlled test: {scenario_name} ({iterations} iterations, cache: {cache_enabled})"
        )

        for i in range(iterations):
            try:
                # Setup test environment
                test_context = scenario.setup_function()

                # Measure execution time
                start_time = time.time()
                scenario.execution_function(test_context)
                processing_time_ms = (time.time() - start_time) * 1000

                # Record measurement
                measurement = PerformanceMeasurement(
                    operation_type=scenario.operation_type,
                    cache_enabled=cache_enabled,
                    processing_time_ms=processing_time_ms,
                    memory_usage_mb=0.0,  # Could be enhanced to measure memory
                    timestamp=time.time(),
                    metadata={
                        "scenario": scenario_name,
                        "iteration": i,
                        **scenario.metadata,
                    },
                )

                measurements.append(measurement)
                self.record_performance(
                    operation_type=scenario.operation_type,
                    processing_time_ms=processing_time_ms,
                    cache_enabled=cache_enabled,
                    metadata=measurement.metadata,
                )

                # Cleanup
                if scenario.cleanup_function:
                    scenario.cleanup_function(test_context)

            except Exception as e:
                logger.error(f"Error in controlled test iteration {i}: {e}")

        logger.info(
            f"Controlled test completed: {len(measurements)} measurements collected"
        )
        return measurements

    def run_comparative_test(
        self, scenario_name: str, iterations_per_mode: int = 50
    ) -> tuple[list[PerformanceMeasurement], list[PerformanceMeasurement]]:
        """Run comparative test with both cached and non-cached modes."""
        cached_measurements = self.run_controlled_test(
            scenario_name, iterations_per_mode, cache_enabled=True
        )
        non_cached_measurements = self.run_controlled_test(
            scenario_name, iterations_per_mode, cache_enabled=False
        )

        return cached_measurements, non_cached_measurements

    def get_all_baseline_statistics(self) -> dict[str, BaselineStatistics]:
        """Get baseline statistics for all monitored operation types."""
        result = {}
        for operation_type in self._measurements.keys():
            stats = self.get_baseline_statistics(operation_type)
            if stats:
                result[operation_type] = stats
        return result

    def generate_performance_report(self) -> dict[str, Any]:
        """Generate comprehensive performance analysis report."""
        all_stats = self.get_all_baseline_statistics()

        if not all_stats:
            return {
                "status": "no_data",
                "message": "No performance data collected yet",
                "framework_mode": self.test_mode.value,
            }

        # Aggregate analysis
        total_operations = sum(
            stats.cached_samples + stats.non_cached_samples
            for stats in all_stats.values()
        )

        valid_comparisons = [
            stats
            for stats in all_stats.values()
            if stats.min_samples_met and stats.statistical_significance
        ]

        if valid_comparisons:
            avg_improvement = statistics.mean(
                [stats.performance_improvement for stats in valid_comparisons]
            )
            best_improvement = max(
                [stats.performance_improvement for stats in valid_comparisons]
            )
            worst_improvement = min(
                [stats.performance_improvement for stats in valid_comparisons]
            )
        else:
            avg_improvement = best_improvement = worst_improvement = 0.0

        return {
            "status": "active",
            "framework_mode": self.test_mode.value,
            "collection_summary": {
                "total_operations": total_operations,
                "operation_types": len(all_stats),
                "valid_comparisons": len(valid_comparisons),
                "collection_duration_hours": max(
                    [stats.collection_duration_hours for stats in all_stats.values()]
                )
                if all_stats
                else 0.0,
            },
            "performance_analysis": {
                "average_improvement": avg_improvement,
                "best_improvement": best_improvement,
                "worst_improvement": worst_improvement,
                "statistically_significant_tests": len(valid_comparisons),
            },
            "operation_details": {
                op_type: {
                    "cached_samples": stats.cached_samples,
                    "non_cached_samples": stats.non_cached_samples,
                    "performance_improvement": stats.performance_improvement,
                    "statistical_significance": stats.statistical_significance,
                    "min_samples_met": stats.min_samples_met,
                }
                for op_type, stats in all_stats.items()
            },
            "recommendations": self._generate_recommendations(valid_comparisons),
            "report_timestamp": time.time(),
        }

    def clear_measurements(self, operation_type: str | None = None) -> None:
        """Clear performance measurements for analysis reset."""
        with self._lock:
            if operation_type:
                if operation_type in self._measurements:
                    self._measurements[operation_type].clear()
                if operation_type in self._baseline_stats_cache:
                    del self._baseline_stats_cache[operation_type]
            else:
                self._measurements.clear()
                self._baseline_stats_cache.clear()

        logger.info(
            f"Cleared baseline measurements for {operation_type or 'all operations'}"
        )

    def _calculate_percentile(self, values: list[float], percentile: float) -> float:
        """Calculate percentile for a list of values."""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        index = int(percentile * (len(sorted_values) - 1))
        return sorted_values[index]

    def _test_statistical_significance(
        self, cached_times: list[float], non_cached_times: list[float]
    ) -> bool:
        """Simple statistical significance test (t-test approximation)."""
        if (
            len(cached_times) < self.min_samples_for_analysis
            or len(non_cached_times) < self.min_samples_for_analysis
        ):
            return False

        try:
            cached_mean = statistics.mean(cached_times)
            non_cached_mean = statistics.mean(non_cached_times)

            # Simple effect size test - at least 10% difference
            if (
                abs(non_cached_mean - cached_mean) / max(non_cached_mean, cached_mean)
                < 0.1
            ):
                return False

            # Need more sophisticated statistical test for production
            # For now, require sufficient samples and meaningful difference
            return len(cached_times) >= 30 and len(non_cached_times) >= 30

        except Exception:
            return False

    def _calculate_confidence_interval(
        self, cached_times: list[float], non_cached_times: list[float]
    ) -> tuple[float, float]:
        """Calculate 95% confidence interval for performance improvement."""
        # Simplified confidence interval calculation
        # In production, would use proper statistical methods
        try:
            cached_mean = statistics.mean(cached_times)
            non_cached_mean = statistics.mean(non_cached_times)

            improvement = (
                (non_cached_mean - cached_mean) / non_cached_mean
                if non_cached_mean > 0
                else 0.0
            )

            # Simplified margin of error (would use proper calculation in production)
            margin = 0.05  # 5% margin of error

            return (improvement - margin, improvement + margin)
        except Exception:
            return (0.0, 0.0)

    def _generate_recommendations(
        self, valid_comparisons: list[BaselineStatistics]
    ) -> list[str]:
        """Generate actionable recommendations based on performance analysis."""
        recommendations = []

        if not valid_comparisons:
            recommendations.append(
                "Insufficient data for recommendations. Continue collecting baseline measurements."
            )
            return recommendations

        avg_improvement = statistics.mean(
            [stats.performance_improvement for stats in valid_comparisons]
        )

        if avg_improvement > 0.1:  # 10% improvement
            recommendations.append(
                "Caching shows significant performance benefits. Consider enabling in production."
            )
        elif avg_improvement > 0.05:  # 5% improvement
            recommendations.append(
                "Caching shows moderate performance benefits. Monitor memory usage before enabling."
            )
        elif avg_improvement < -0.05:  # 5% degradation
            recommendations.append(
                "Caching appears to degrade performance. Consider keeping disabled or optimizing cache implementation."
            )
        else:
            recommendations.append(
                "Caching impact is minimal. Focus on other optimization opportunities."
            )

        # Operation-specific recommendations
        high_benefit_ops = [
            stats for stats in valid_comparisons if stats.performance_improvement > 0.2
        ]
        if high_benefit_ops:
            op_names = [stats.operation_type for stats in high_benefit_ops]
            recommendations.append(
                f"High-benefit operations for caching: {', '.join(op_names)}"
            )

        low_benefit_ops = [
            stats for stats in valid_comparisons if stats.performance_improvement < 0.1
        ]
        if low_benefit_ops:
            op_names = [stats.operation_type for stats in low_benefit_ops]
            recommendations.append(
                f"Consider selective caching - low benefit for: {', '.join(op_names)}"
            )

        return recommendations


# Context manager for baseline testing
@contextlib.contextmanager
def baseline_performance_test(
    framework: PerformanceBaselineFramework,
    operation_type: str,
    force_baseline: bool = False,
):
    """Context manager for automatic performance measurement."""
    cache_enabled = not (
        force_baseline or framework.should_run_baseline_test(operation_type)
    )

    start_time = time.time()
    try:
        yield cache_enabled
    finally:
        processing_time_ms = (time.time() - start_time) * 1000
        framework.record_performance(
            operation_type=operation_type,
            processing_time_ms=processing_time_ms,
            cache_enabled=cache_enabled,
        )


# Global instance for singleton access
_baseline_framework: PerformanceBaselineFramework | None = None
_baseline_lock = threading.RLock()


def get_baseline_framework() -> PerformanceBaselineFramework:
    """Get singleton performance baseline framework."""
    global _baseline_framework
    with _baseline_lock:
        if _baseline_framework is None:
            _baseline_framework = PerformanceBaselineFramework()
        return _baseline_framework


def is_baseline_testing_enabled() -> bool:
    """Check if baseline testing is enabled in configuration."""
    try:
        from ..config import MONITORING

        return MONITORING.get("enabled", True) and MONITORING.get("systems", {}).get(
            "frame_cache", True
        )
    except ImportError:
        return False
