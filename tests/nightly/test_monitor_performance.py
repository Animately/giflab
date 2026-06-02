"""Unit tests for the test performance monitor."""

import json

# Import the monitor class
import sys
import time

import pytest

sys.path.append("scripts")
from monitor_test_performance import DEFAULT_PYTEST_ARGS, TestPerformanceMonitor


class TestMonitorPerformance:
    """Test the TestPerformanceMonitor class."""

    @pytest.mark.fast
    def test_default_thresholds(self):
        """Test that default thresholds are loaded correctly."""
        monitor = TestPerformanceMonitor()

        assert monitor.config["thresholds"]["fast"] == 10
        assert monitor.config["thresholds"]["integration"] == 300
        assert monitor.config["thresholds"]["full"] == 1800

    @pytest.mark.fast
    def test_custom_config_loading(self, tmp_path):
        """Test loading custom configuration from file."""
        config_file = tmp_path / "test_config.json"
        custom_config = {
            "thresholds": {"fast": 5, "integration": 120},
            "alert_on_regression": False,
        }

        with open(config_file, "w") as f:
            json.dump(custom_config, f)

        monitor = TestPerformanceMonitor(config_file)

        assert monitor.config["thresholds"]["fast"] == 5
        assert monitor.config["thresholds"]["integration"] == 120
        assert (
            monitor.config["thresholds"].get("full", 1800) == 1800
        )  # Default preserved
        assert monitor.config["alert_on_regression"] is False

    @pytest.mark.fast
    def test_check_performance_within_threshold(self):
        """Test performance check when within threshold."""
        monitor = TestPerformanceMonitor()

        # Fast test taking 5s should pass (threshold is 10s)
        assert monitor.check_performance("fast", 5.0) is True

        # Integration test taking 120s should pass (threshold is 300s)
        assert monitor.check_performance("integration", 120.0) is True

    @pytest.mark.fast
    def test_check_performance_exceeds_threshold(self):
        """Test performance check when exceeding threshold."""
        monitor = TestPerformanceMonitor()

        # Fast test taking 15s should fail (threshold is 10s)
        assert monitor.check_performance("fast", 15.0) is False

        # Integration test taking 400s should fail (threshold is 300s)
        assert monitor.check_performance("integration", 400.0) is False

    @pytest.mark.fast
    def test_check_performance_applies_regression_tolerance(self):
        """regression_tolerance widens the pass band to target × tolerance.

        Regression guard for the wired-up tolerance knob: the config carried a
        regression_tolerance for ages but check_performance ignored it, leaving a
        hard cliff that failed the fast tier on sub-percent CI jitter (~10.0s vs a
        10s target).
        """
        monitor = TestPerformanceMonitor()
        monitor.config["thresholds"] = {"fast": 10}
        monitor.config["regression_tolerance"] = 1.5

        # Within the tolerance band (10 < 12 ≤ 15) → acceptable, not a regression
        assert monitor.check_performance("fast", 12.0) is True
        # At the effective ceiling (10 × 1.5 = 15) → still acceptable
        assert monitor.check_performance("fast", 15.0) is True
        # Beyond the tolerance band → genuine regression
        assert monitor.check_performance("fast", 15.1) is False

    @pytest.mark.fast
    def test_check_performance_defaults_to_strict_without_tolerance(self):
        """With no regression_tolerance configured, the target is the hard limit."""
        monitor = TestPerformanceMonitor()
        monitor.config["thresholds"] = {"fast": 10}
        monitor.config.pop("regression_tolerance", None)

        assert monitor.check_performance("fast", 10.0) is True
        assert monitor.check_performance("fast", 10.1) is False

    @pytest.mark.fast
    def test_check_performance_unknown_tier(self):
        """Test performance check for unknown test tier."""
        monitor = TestPerformanceMonitor()

        # Unknown tier should always pass (infinite threshold)
        assert monitor.check_performance("unknown", 9999.0) is True

    @pytest.mark.fast
    def test_record_performance_disabled(self, tmp_path):
        """Test that performance recording can be disabled."""
        config_file = tmp_path / "no_history_config.json"
        config = {"save_history": False}

        with open(config_file, "w") as f:
            json.dump(config, f)

        monitor = TestPerformanceMonitor(config_file)
        monitor.history_file = tmp_path / "history.json"

        # Record performance - should not create file
        monitor.record_performance("fast", 5.0, True)

        assert not monitor.history_file.exists()

    @pytest.mark.fast
    def test_record_performance_creates_history(self, tmp_path):
        """Test that performance recording creates history file."""
        config_file = tmp_path / "history_config.json"
        config = {"save_history": True, "thresholds": {"fast": 10}}

        with open(config_file, "w") as f:
            json.dump(config, f)

        monitor = TestPerformanceMonitor(config_file)
        monitor.history_file = tmp_path / "history.json"

        # Record performance
        monitor.record_performance("fast", 5.0, True)

        assert monitor.history_file.exists()

        # Check history content
        with open(monitor.history_file) as f:
            history = json.load(f)

        assert len(history) == 1
        record = history[0]
        assert record["test_tier"] == "fast"
        assert record["duration"] == 5.0
        assert record["success"] is True
        assert record["threshold"] == 10
        assert record["threshold_met"] is True
        assert "timestamp" in record

    @pytest.mark.fast
    def test_record_performance_appends_to_existing(self, tmp_path):
        """Test that performance recording appends to existing history."""
        config_file = tmp_path / "history_config.json"
        config = {"save_history": True, "thresholds": {"fast": 10}}

        with open(config_file, "w") as f:
            json.dump(config, f)

        monitor = TestPerformanceMonitor(config_file)
        monitor.history_file = tmp_path / "history.json"

        # Create initial history
        initial_history = [
            {
                "timestamp": time.time() - 3600,
                "test_tier": "fast",
                "duration": 8.0,
                "success": True,
                "threshold": 10,
                "threshold_met": True,
            }
        ]

        with open(monitor.history_file, "w") as f:
            json.dump(initial_history, f)

        # Record new performance
        monitor.record_performance("fast", 12.0, False)

        # Check history was appended
        with open(monitor.history_file) as f:
            history = json.load(f)

        assert len(history) == 2
        assert history[0]["duration"] == 8.0  # Original record
        assert history[1]["duration"] == 12.0  # New record
        assert history[1]["threshold_met"] is False  # Exceeded threshold

    @pytest.mark.fast
    def test_config_file_not_found(self, tmp_path):
        """Test graceful handling of missing config file."""
        missing_config = tmp_path / "missing.json"

        # Should not raise exception, should use defaults
        monitor = TestPerformanceMonitor(missing_config)

        assert monitor.config["thresholds"]["fast"] == 10
        assert monitor.config["alert_on_regression"] is True

    @pytest.mark.fast
    def test_invalid_config_file(self, tmp_path):
        """Test graceful handling of invalid config file."""
        invalid_config = tmp_path / "invalid.json"

        # Write invalid JSON
        with open(invalid_config, "w") as f:
            f.write("{ invalid json }")

        # Should not raise exception, should use defaults
        monitor = TestPerformanceMonitor(invalid_config)

        assert monitor.config["thresholds"]["fast"] == 10
        assert monitor.config["alert_on_regression"] is True

    @pytest.mark.fast
    def test_hang_guard_timeout_exceeds_regression_threshold(self):
        """The subprocess kill-timeout must be decoupled from the regression target.

        Regression guard for the bug that turned the perf check red: the monitor
        passed ``thresholds[tier]`` straight into ``subprocess.run(timeout=...)``,
        so a fast suite that legitimately ran to ~its target was *killed* mid-run
        with TimeoutExpired and reported as a failure — the kill-timeout WAS the
        regression threshold. The hang guard exists only to reap a genuinely
        deadlocked pytest, so it must sit comfortably above the regression band,
        never coincide with it.
        """
        monitor = TestPerformanceMonitor()

        for tier in ("fast", "integration", "full"):
            hang_guard = monitor._hang_guard_timeout(tier)
            regression = monitor._effective_threshold(tier)
            assert hang_guard > regression, (
                f"{tier}: hang-guard {hang_guard}s must exceed regression "
                f"threshold {regression}s so a slow-but-healthy run is never killed"
            )

    @pytest.mark.fast
    def test_hang_guard_timeout_respects_floor(self):
        """A tiny configured threshold must not shrink the hang guard below the floor.

        With a very small target (e.g. a 1s fast tier), ``effective × 3`` would be
        only a few seconds — enough that a slow-but-healthy CI runner gets killed.
        The floor guarantees a genuinely hung process is still reaped while a
        healthy-but-slow run survives.
        """
        monitor = TestPerformanceMonitor()
        monitor.config["thresholds"] = {"fast": 1}
        monitor.config["regression_tolerance"] = 1.5

        # effective = 1 × 1.5 = 1.5; × 3 = 4.5 → far below the floor
        assert monitor._hang_guard_timeout("fast") >= 120

    @pytest.mark.fast
    def test_default_fast_args_scoped_to_smoke_functional(self):
        """The fast tier must collect the SAME set as the canonical Fast Tests gate.

        Regression guard for the mis-scoped tier: the monitor (and test-matrix
        Lightning) ran ``-m fast tests/``, an explicit path that overrides the
        pyproject ``testpaths`` and swept ~65 fast-marked integration/nightly
        tests — including the flaky LPIPS-memory and overhead-ratio tests — into a
        tier meant to mirror ``ci.yml``'s bare ``-m fast`` (smoke+functional only).
        Lock the scope to smoke+functional and forbid a bare ``tests/`` token.
        """
        fast_args = DEFAULT_PYTEST_ARGS["fast"]

        assert "tests/smoke" in fast_args
        assert "tests/functional" in fast_args
        # A bare ``tests/`` token re-introduces the integration/nightly sweep.
        assert "tests/" not in fast_args

    @pytest.mark.fast
    def test_default_integration_args_scoped_to_test_ci_boundary(self):
        """The integration tier must mirror ``make test-ci`` (no nightly sweep)."""
        integration_args = DEFAULT_PYTEST_ARGS["integration"]

        assert "tests/smoke" in integration_args
        assert "tests/functional" in integration_args
        assert "tests/integration" in integration_args
        # Must not pull in nightly via a bare ``tests/`` token.
        assert "tests/" not in integration_args
        assert "tests/nightly" not in integration_args
