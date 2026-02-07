"""
Tests for Phase 7: Performance Regression Detection System

Trimmed test suite covering representative scenarios:
- PerformanceBaseline statistical calculations
- RegressionDetector threshold detection
- PerformanceHistory data management
- ContinuousMonitor background monitoring
- End-to-end integration testing
"""

import json
import os
import tempfile
import time
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call

import pytest

import src.giflab.monitoring.performance_regression as _pr_module
import src.giflab.monitoring.alerting as _alerting_module
import src.giflab.monitoring.metrics_collector as _metrics_module

from src.giflab.monitoring.performance_regression import (
    PerformanceBaseline,
    RegressionAlert,
    RegressionDetector,
    PerformanceHistory,
    ContinuousMonitor,
    create_performance_monitor
)
from src.giflab.monitoring.alerting import Alert, AlertLevel
from src.giflab.benchmarks.phase_4_3_benchmarking import BenchmarkResult, BenchmarkScenario


class TestPerformanceBaseline(unittest.TestCase):
    """Test PerformanceBaseline statistical calculations."""

    def setUp(self):
        self.baseline = PerformanceBaseline(
            scenario_name="test_scenario",
            mean_processing_time=10.0,
            std_processing_time=1.0,
            mean_memory_usage=100.0,
            std_memory_usage=10.0,
            sample_count=5,
            last_updated=datetime(2025, 1, 12, 12, 0, 0),
            confidence_level=0.95
        )

    def test_baseline_creation(self):
        """Test baseline creation with valid parameters."""
        self.assertEqual(self.baseline.scenario_name, "test_scenario")
        self.assertEqual(self.baseline.mean_processing_time, 10.0)
        self.assertEqual(self.baseline.sample_count, 5)
        self.assertEqual(self.baseline.confidence_level, 0.95)

    def test_control_limits_calculation(self):
        """Test statistical control limits calculation."""
        time_lower, time_upper, memory_lower, memory_upper = self.baseline.get_control_limits()

        # For 95% confidence level, z-score is 1.96
        expected_time_lower = max(0, 10.0 - 1.96 * 1.0)
        expected_time_upper = 10.0 + 1.96 * 1.0
        expected_memory_lower = max(0, 100.0 - 1.96 * 10.0)
        expected_memory_upper = 100.0 + 1.96 * 10.0

        self.assertAlmostEqual(time_lower, expected_time_lower, places=2)
        self.assertAlmostEqual(time_upper, expected_time_upper, places=2)
        self.assertAlmostEqual(memory_lower, expected_memory_lower, places=2)
        self.assertAlmostEqual(memory_upper, expected_memory_upper, places=2)


class TestRegressionAlert(unittest.TestCase):
    """Test RegressionAlert creation."""

    def test_alert_creation(self):
        """Test regression alert creation."""
        detection_time = datetime.now()
        alert = RegressionAlert(
            scenario="test_scenario",
            metric_type="processing_time",
            current_value=15.0,
            baseline_mean=10.0,
            baseline_std=1.0,
            regression_severity=0.50,
            detection_time=detection_time,
            confidence_level=0.95
        )

        self.assertEqual(alert.scenario, "test_scenario")
        self.assertEqual(alert.metric_type, "processing_time")
        self.assertEqual(alert.regression_severity, 0.50)
        self.assertEqual(alert.detection_time, detection_time)


class TestPerformanceHistory(unittest.TestCase):
    """Test PerformanceHistory data management."""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.history = PerformanceHistory(
            history_path=self.temp_dir,
            max_history_days=7
        )

        # Mock benchmark result
        self.mock_result = Mock(spec=BenchmarkResult)
        self.mock_result.processing_time = 10.0
        self.mock_result.mean_memory_usage = 100.0
        self.mock_result.success_rate = 1.0
        self.mock_result.total_files = 5

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_record_benchmark(self):
        """Test recording benchmark results."""
        self.history.record_benchmark("test_scenario", self.mock_result)

        history_file = self.temp_dir / "test_scenario_history.jsonl"
        self.assertTrue(history_file.exists())

        with open(history_file, 'r') as f:
            line = f.readline().strip()
            record = json.loads(line)

        self.assertEqual(record['processing_time'], 10.0)
        self.assertEqual(record['memory_usage'], 100.0)
        self.assertEqual(record['success_rate'], 1.0)
        self.assertEqual(record['total_files'], 5)
        self.assertIn('timestamp', record)


class TestRegressionDetector(unittest.TestCase):
    """Test RegressionDetector baseline management and regression detection."""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.baseline_path = self.temp_dir / "baselines.json"
        self.detector = RegressionDetector(
            baseline_path=self.baseline_path,
            regression_threshold=0.10,  # 10%
            confidence_level=0.95
        )

        # Create mock benchmark results
        self.mock_results = []
        for i in range(5):
            result = Mock(spec=BenchmarkResult)
            result.processing_time = 10.0 + i * 0.5
            result.mean_memory_usage = 100.0 + i * 2.0
            self.mock_results.append(result)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_baseline_creation_and_persistence(self):
        """Test creating and persisting baselines."""
        self.detector.update_baseline("test_scenario", self.mock_results)

        # Check baseline was created
        self.assertIn("test_scenario", self.detector.baselines)
        baseline = self.detector.baselines["test_scenario"]

        self.assertEqual(baseline.scenario_name, "test_scenario")
        self.assertAlmostEqual(baseline.mean_processing_time, 11.0, places=1)  # Mean of 10-12
        self.assertEqual(baseline.sample_count, 5)

        # Check persistence
        self.assertTrue(self.baseline_path.exists())

        # Create new detector to test loading
        new_detector = RegressionDetector(
            baseline_path=self.baseline_path,
            regression_threshold=0.10,
            confidence_level=0.95
        )

        self.assertIn("test_scenario", new_detector.baselines)
        loaded_baseline = new_detector.baselines["test_scenario"]
        self.assertAlmostEqual(loaded_baseline.mean_processing_time, baseline.mean_processing_time, places=2)

    def test_regression_detection_processing_time(self):
        """Test regression detection for processing time."""
        # Create baseline
        self.detector.update_baseline("perf_scenario", self.mock_results)

        # Create regressed result (50% slower)
        regressed_result = Mock(spec=BenchmarkResult)
        regressed_result.processing_time = 16.5  # 50% slower than 11.0 baseline
        regressed_result.mean_memory_usage = 100.0

        alerts = self.detector.detect_regressions("perf_scenario", regressed_result)

        self.assertEqual(len(alerts), 1)
        alert = alerts[0]
        self.assertEqual(alert.scenario, "perf_scenario")
        self.assertEqual(alert.metric_type, "processing_time")
        self.assertGreater(alert.regression_severity, 0.10)  # Should exceed 10% threshold
        self.assertAlmostEqual(alert.current_value, 16.5, places=1)

    def test_no_regression_detection(self):
        """Test that no alerts are generated for good performance."""
        # Create baseline
        self.detector.update_baseline("good_scenario", self.mock_results)

        # Create good result (within baseline range)
        good_result = Mock(spec=BenchmarkResult)
        good_result.processing_time = 10.5  # Well within baseline
        good_result.mean_memory_usage = 95.0  # Slightly better than baseline

        alerts = self.detector.detect_regressions("good_scenario", good_result)

        self.assertEqual(len(alerts), 0)


class TestContinuousMonitor(unittest.TestCase):
    """Test ContinuousMonitor background monitoring."""

    def setUp(self):
        # Create mock components
        self.mock_benchmarker = Mock()
        self.mock_detector = Mock()
        self.mock_history = Mock()
        self.mock_alert_manager = Mock()
        self.mock_metrics_collector = Mock()

        # Setup mock detector baselines
        self.mock_detector.baselines = {"test_scenario": Mock()}

        # Setup mock alert manager with iterable alerts
        self.mock_alert_manager.alerts = []

        self.monitor = ContinuousMonitor(
            benchmarker=self.mock_benchmarker,
            detector=self.mock_detector,
            history=self.mock_history,
            alert_manager=self.mock_alert_manager,
            metrics_collector=self.mock_metrics_collector,
            monitoring_interval=1  # 1 second for fast testing
        )

    def test_monitor_start_stop(self):
        """Test starting and stopping monitoring."""
        # Start monitoring
        self.monitor.start_monitoring()

        self.assertTrue(self.monitor.enabled)
        self.assertIsNotNone(self.monitor.monitoring_thread)
        self.assertTrue(self.monitor.monitoring_thread.is_alive())

        # Stop monitoring
        self.monitor.stop_monitoring()

        self.assertFalse(self.monitor.enabled)

    def test_regression_alert_sending(self):
        """Test sending regression alerts."""
        # Setup mock benchmark results with regression
        mock_result = Mock(spec=BenchmarkResult)
        mock_result.processing_time = 15.0  # Regressed
        mock_result.mean_memory_usage = 100.0
        self.mock_benchmarker.run_scenario.return_value = [mock_result]

        # Setup regression alert
        mock_alert = RegressionAlert(
            scenario="test_scenario",
            metric_type="processing_time",
            current_value=15.0,
            baseline_mean=10.0,
            baseline_std=1.0,
            regression_severity=0.50,
            detection_time=datetime.now(),
            confidence_level=0.95
        )
        self.mock_detector.detect_regressions.return_value = [mock_alert]

        # Mock alert manager
        self.mock_alert_manager.alerts = []

        # Run monitoring check
        scenario = self.monitor.monitoring_scenarios[0]
        self.monitor._run_monitoring_check(scenario)

        # Verify alert was added
        self.assertEqual(len(self.mock_alert_manager.alerts), 1)
        sent_alert = self.mock_alert_manager.alerts[0]
        self.assertIn("performance_regression", sent_alert.system)
        self.assertEqual(sent_alert.level, AlertLevel.CRITICAL)  # 50% regression


class TestMonitorIntegration(unittest.TestCase):
    """Integration tests for the complete monitoring system."""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_end_to_end_baseline_and_detection(self):
        """Test end-to-end baseline creation and regression detection."""
        # Create real components (with temp directories)
        baseline_path = self.temp_dir / "baselines.json"
        history_path = self.temp_dir / "history"

        detector = RegressionDetector(
            baseline_path=baseline_path,
            regression_threshold=0.20,  # 20% threshold
            confidence_level=0.95
        )

        history = PerformanceHistory(
            history_path=history_path,
            max_history_days=7
        )

        # Create baseline with mock results
        baseline_results = []
        for i in range(3):
            result = Mock(spec=BenchmarkResult)
            result.processing_time = 10.0 + i * 0.5
            result.mean_memory_usage = 100.0 + i * 2.0
            baseline_results.append(result)

        detector.update_baseline("integration_test", baseline_results)

        # Test good performance (no regression)
        good_result = MagicMock()
        good_result.processing_time = 10.5
        good_result.mean_memory_usage = 101.0
        good_result.success_rate = 1.0
        good_result.total_files = 5
        good_result.phase6_enabled = False

        history.record_benchmark("integration_test", good_result)
        alerts = detector.detect_regressions("integration_test", good_result)

        self.assertEqual(len(alerts), 0)  # No regression

        # Small delay to ensure different timestamps for trend calculation
        time.sleep(0.05)

        # Test regressed performance
        bad_result = MagicMock()
        bad_result.processing_time = 14.0  # ~27% regression
        bad_result.mean_memory_usage = 130.0  # ~28% regression
        bad_result.success_rate = 1.0
        bad_result.total_files = 5
        bad_result.phase6_enabled = False

        history.record_benchmark("integration_test", bad_result)
        alerts = detector.detect_regressions("integration_test", bad_result)

        self.assertGreater(len(alerts), 0)  # Should have regressions

        # Verify alert details
        time_alert = next((a for a in alerts if a.metric_type == "processing_time"), None)
        memory_alert = next((a for a in alerts if a.metric_type == "memory_usage"), None)

        self.assertIsNotNone(time_alert)
        self.assertIsNotNone(memory_alert)
        self.assertGreater(time_alert.regression_severity, 0.20)
        self.assertGreater(memory_alert.regression_severity, 0.20)

        # Small delay to ensure different timestamps for trend calculation
        time.sleep(0.05)

        # Test history records
        records = history.get_recent_history("integration_test", days=1)
        self.assertEqual(len(records), 2)  # good + bad


if __name__ == '__main__':
    # Configure logging for tests
    import logging
    logging.basicConfig(level=logging.WARNING)

    # Run tests
    unittest.main(verbosity=2)
