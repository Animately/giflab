#!/usr/bin/env python3
"""
Test Performance Monitor for GifLab

Monitors test execution times and alerts on performance regressions.
Supports integration with CI/CD pipelines and local development.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

# Default pytest arguments for each tier.
#
# These are scoped to EXACTLY the testpaths each tier is meant to cover, never a
# bare ``tests/``. A bare ``tests/`` overrides the pyproject ``testpaths`` and
# sweeps fast-/not-slow-marked integration *and* nightly tests into the tier —
# including the flaky LPIPS-memory and overhead-ratio benchmarks — which is what
# turned the performance check red. The reference is ci.yml's canonical gates:
#   * fast        → bare ``-m fast`` (pyproject testpaths = smoke + functional)
#   * integration → ``make test-ci`` (smoke + functional + integration)
# We name the paths explicitly here because the monitor runs pytest from a
# subprocess that may not inherit the same testpaths defaulting.
DEFAULT_PYTEST_ARGS = {
    "fast": [
        "-m",
        "fast",
        "tests/smoke",
        "tests/functional",
        "-n",
        "auto",
        # xdist only honours xdist_group markers (which tests/conftest.py maps
        # the `serial` marker onto) under loadgroup distribution; the default
        # `load` dist silently scatters serial-marked tests across workers.
        "--dist",
        "loadgroup",
        "--tb=short",
    ],
    "integration": [
        "-m",
        "not slow",
        "tests/smoke",
        "tests/functional",
        "tests/integration",
        "-n",
        "4",
        "--dist",
        "loadgroup",
        "--tb=short",
        "--durations=10",
    ],
    "full": ["tests/", "--tb=short", "--durations=20", "--maxfail=10"],
}

# Lower bound for the subprocess hang guard (seconds). Ensures a genuinely
# deadlocked pytest is still reaped while a slow-but-healthy run on a small CI
# runner is never killed, even when a tier's configured target is tiny.
HANG_GUARD_FLOOR_SECONDS = 120


class TestPerformanceMonitor:
    """Monitor and validate test performance against defined thresholds."""

    # Performance thresholds (in seconds)
    THRESHOLDS = {
        "fast": 10,  # Lightning-fast development tests
        "integration": 300,  # Integration tests (5 minutes)
        "full": 1800,  # Full test suite (30 minutes)
    }

    def __init__(self, config_path: Path | None = None):
        """Initialize the performance monitor.

        Args:
            config_path: Optional path to custom configuration file
        """
        self.config = self._load_config(config_path)
        self.history_file = Path("test-performance-history.json")

    def _load_config(self, config_path: Path | None) -> dict:
        """Load configuration from file or use defaults."""
        default_config = {
            "thresholds": self.THRESHOLDS.copy(),
            "alert_on_regression": True,
            "save_history": True,
            "slack_webhook_url": None,  # Optional Slack integration
            "email_alerts": None,  # Optional email integration
        }

        if config_path and config_path.exists():
            try:
                with open(config_path) as f:
                    custom_config = json.load(f)
                default_config.update(custom_config)
            except Exception as e:
                print(f"⚠️  Warning: Could not load config from {config_path}: {e}")

        return default_config

    def run_timed_test(
        self, test_tier: str, pytest_args: list
    ) -> tuple[int, float, str]:
        """Run tests with timing and capture output.

        Args:
            test_tier: The test tier ('fast', 'integration', 'full')
            pytest_args: Arguments to pass to pytest

        Returns:
            Tuple of (exit_code, duration_seconds, output)
        """
        print(f"⚡ Running {test_tier} tests with performance monitoring...")

        start_time = time.time()

        try:
            # Run pytest with timing. The subprocess timeout is a *hang guard*
            # (reaps a deadlocked pytest), deliberately decoupled from the
            # regression threshold — see _hang_guard_timeout. Using the
            # regression threshold here would kill a slow-but-healthy run and
            # report it as a failure instead of a (recoverable) regression.
            result = subprocess.run(
                ["poetry", "run", "pytest"] + pytest_args,
                capture_output=True,
                text=True,
                timeout=self._hang_guard_timeout(test_tier),
            )

            end_time = time.time()
            duration = end_time - start_time

            return result.returncode, duration, result.stdout + result.stderr

        except subprocess.TimeoutExpired:
            end_time = time.time()
            duration = end_time - start_time
            error_msg = f"❌ Tests timed out after {duration:.1f}s"
            return 1, duration, error_msg

    def _effective_threshold(self, test_tier: str) -> float:
        """Duration above which a run counts as a regression.

        The ``thresholds`` value is the *target* time; ``regression_tolerance``
        (default 1.0) sets how far past the target a run may drift before it is
        flagged. Without this, a tier that legitimately sits right at its target
        trips a false regression on sub-percent CI jitter — the ``fast`` tier at
        ~10.0s against a 10s target is exactly that case. ``regression_tolerance``
        was already in the config but was never applied; this wires it up.
        """
        target = self.config["thresholds"].get(test_tier, float("inf"))
        tolerance = self.config.get("regression_tolerance", 1.0)
        return target * tolerance

    def _hang_guard_timeout(self, test_tier: str) -> float:
        """Wall-clock ceiling for killing a *deadlocked* pytest subprocess.

        This is intentionally NOT the regression threshold. The regression
        threshold (``_effective_threshold``) decides whether a *completed* run
        was too slow; this guard only exists to reap a pytest that has hung and
        will never finish. Coupling the two — as the original code did by passing
        ``thresholds[tier]`` straight into ``subprocess.run(timeout=...)`` — meant
        a suite that ran right up to its target was killed mid-run with
        TimeoutExpired and reported as a hard failure rather than a recoverable
        regression. That is the bug that turned the performance check red.

        We sit the guard at ``effective × 3`` (plenty of headroom for CI runners
        with fewer ``-n auto`` workers than a dev box) but never below a fixed
        floor, so a tier with a tiny configured target still tolerates a
        slow-but-healthy run while a genuine deadlock is still reaped.
        """
        effective = self._effective_threshold(test_tier)
        if effective == float("inf"):
            # Unknown tier: no meaningful target. Fall back to a generous,
            # finite hang guard rather than blocking forever.
            return 3600.0
        return max(effective * 3, float(HANG_GUARD_FLOOR_SECONDS))

    def check_performance(self, test_tier: str, duration: float) -> bool:
        """Check if performance meets threshold requirements.

        Args:
            test_tier: The test tier being evaluated
            duration: Test execution time in seconds

        Returns:
            True if performance is acceptable, False otherwise
        """
        return duration <= self._effective_threshold(test_tier)

    def record_performance(self, test_tier: str, duration: float, success: bool):
        """Record performance data to history file.

        Args:
            test_tier: The test tier that was run
            duration: Test execution time in seconds
            success: Whether tests passed
        """
        if not self.config.get("save_history", False):
            return

        # Load existing history
        history = []
        if self.history_file.exists():
            try:
                with open(self.history_file) as f:
                    history = json.load(f)
            except Exception:
                history = []

        # Add new record
        record = {
            "timestamp": time.time(),
            "test_tier": test_tier,
            "duration": duration,
            "success": success,
            "threshold": self.config["thresholds"].get(test_tier),
            "threshold_met": self.check_performance(test_tier, duration),
        }

        history.append(record)

        # Keep only last 100 records per tier
        history = history[-100:]

        # Save updated history
        try:
            with open(self.history_file, "w") as f:
                json.dump(history, f, indent=2)
        except Exception as e:
            print(f"⚠️  Warning: Could not save performance history: {e}")

    def generate_report(
        self, test_tier: str, duration: float, exit_code: int, output: str
    ):
        """Generate a comprehensive performance report.

        Args:
            test_tier: The test tier that was run
            duration: Test execution time in seconds
            exit_code: Test exit code (0 = success)
            output: Test output/logs
        """
        threshold = self.config["thresholds"].get(test_tier, float("inf"))
        effective = self._effective_threshold(test_tier)
        performance_ok = self.check_performance(test_tier, duration)
        tests_passed = exit_code == 0

        print("\n" + "=" * 60)
        print(f"📊 TEST PERFORMANCE REPORT - {test_tier.upper()} TIER")
        print("=" * 60)
        print(f"⏱️  Execution Time: {duration:.1f}s")
        print(f"🎯 Target: ≤{threshold}s (regression above {effective:.0f}s)")
        print(f"✅ Performance Target: {'✅ MET' if performance_ok else '❌ EXCEEDED'}")
        print(f"🧪 Test Results: {'✅ PASSED' if tests_passed else '❌ FAILED'}")

        if not performance_ok:
            print("\n🚨 PERFORMANCE REGRESSION DETECTED!")
            print(f"   Regression threshold: ≤{effective:.0f}s")
            print(f"   Actual: {duration:.1f}s")
            print(
                f"   Overage: +{duration - effective:.1f}s ({((duration / effective - 1) * 100):.1f}% over)"
            )
            print("\n💡 RECOMMENDATIONS:")
            print("   • Review recent changes that might impact test performance")
            print("   • Check if test data size has grown unexpectedly")
            print("   • Verify mock patterns are working correctly")
            print("   • Consider if parallel execution is functioning properly")

        if not tests_passed:
            # Surface the captured pytest output: without this, CI logs show
            # only "❌ FAILED" with no indication of which test failed.
            tail_lines = output.strip().splitlines()[-50:]
            print(f"\n🔍 PYTEST OUTPUT (last {len(tail_lines)} lines):")
            for line in tail_lines:
                print(f"   {line}")

        # Show trend if history exists
        self._show_performance_trend(test_tier)

        print("=" * 60)

    def _show_performance_trend(self, test_tier: str):
        """Show recent performance trend for the test tier."""
        if not self.history_file.exists():
            return

        try:
            with open(self.history_file) as f:
                history = json.load(f)

            # Filter to current test tier and last 5 runs
            tier_history = [r for r in history if r["test_tier"] == test_tier][-5:]

            if len(tier_history) < 2:
                return

            print(f"\n📈 RECENT PERFORMANCE TREND ({test_tier}):")
            for _i, record in enumerate(tier_history):
                timestamp = time.strftime(
                    "%m/%d %H:%M", time.localtime(record["timestamp"])
                )
                duration = record["duration"]
                status = "✅" if record["threshold_met"] else "❌"
                print(f"   {timestamp}: {duration:5.1f}s {status}")

        except Exception:
            pass  # Silently skip trend display on error

    def send_alerts(self, test_tier: str, duration: float, threshold: float):
        """Send alerts for performance regressions.

        Args:
            test_tier: The test tier that regressed
            duration: Actual execution time
            threshold: Expected threshold
        """
        if not self.config.get("alert_on_regression", False):
            return

        message = (
            f"🚨 GifLab Test Performance Regression\n"
            f"Tier: {test_tier}\n"
            f"Expected: ≤{threshold}s\n"
            f"Actual: {duration:.1f}s\n"
            f"Overage: +{duration - threshold:.1f}s"
        )

        # Slack webhook (if configured)
        slack_url = self.config.get("slack_webhook_url")
        if slack_url:
            try:
                import requests

                payload = {"text": message}
                requests.post(slack_url, json=payload, timeout=10)
                print("📱 Slack alert sent")
            except Exception as e:
                print(f"⚠️  Could not send Slack alert: {e}")

        # Email alerts could be implemented here
        email_config = self.config.get("email_alerts")
        if email_config:
            print("📧 Email alerts not yet implemented")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Monitor GifLab test performance")
    parser.add_argument(
        "tier", choices=["fast", "integration", "full"], help="Test tier to monitor"
    )
    parser.add_argument("--config", type=Path, help="Path to configuration file")
    parser.add_argument(
        "--pytest-args", nargs=argparse.REMAINDER, help="Arguments to pass to pytest"
    )

    args = parser.parse_args()

    pytest_args = args.pytest_args or DEFAULT_PYTEST_ARGS.get(args.tier, [])

    # Set environment variables for test tier
    env_vars = {
        "fast": {
            "GIFLAB_ULTRA_FAST": "1",
            "GIFLAB_MAX_PIPES": "3",
            "GIFLAB_MOCK_ALL_ENGINES": "1",
        },
        "integration": {"GIFLAB_MAX_PIPES": "10"},
        "full": {"GIFLAB_FULL_MATRIX": "1"},
    }

    # Apply environment variables
    import os

    for key, value in env_vars.get(args.tier, {}).items():
        os.environ[key] = value

    # Initialize monitor and run tests
    monitor = TestPerformanceMonitor(args.config)
    exit_code, duration, output = monitor.run_timed_test(args.tier, pytest_args)

    # Record performance and generate report
    monitor.record_performance(args.tier, duration, exit_code == 0)
    monitor.generate_report(args.tier, duration, exit_code, output)

    # Send alerts if performance regression detected
    threshold = monitor.config["thresholds"].get(args.tier, float("inf"))
    if duration > threshold:
        monitor.send_alerts(args.tier, duration, threshold)

    # Exit with same code as tests
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
