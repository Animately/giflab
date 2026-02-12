"""
Alerting rules and threshold monitoring for GifLab optimization systems.
"""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from ..config import MONITORING
from .metrics_collector import MetricSummary, MetricType, get_metrics_collector

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class Alert:
    """Single alert instance."""

    level: AlertLevel
    system: str
    metric: str
    message: str
    value: float
    threshold: float
    timestamp: float = field(default_factory=time.time)
    tags: dict[str, str] = field(default_factory=dict)

    def __str__(self) -> str:
        """String representation of alert."""
        ts = datetime.fromtimestamp(self.timestamp).strftime("%Y-%m-%d %H:%M:%S")
        return (
            f"[{self.level.value.upper()}] {ts} - {self.system}/{self.metric}: "
            f"{self.message} (value={self.value:.2f}, threshold={self.threshold:.2f})"
        )


@dataclass
class AlertRule:
    """Definition of an alert rule."""

    name: str
    metric_pattern: str
    condition: Callable[[float], bool]
    warning_threshold: float | None = None
    critical_threshold: float | None = None
    message_template: str = "{metric} exceeded threshold"
    tags: dict[str, str] = field(default_factory=dict)
    window_seconds: int = 300
    min_occurrences: int = 1

    def evaluate(self, value: float) -> Alert | None:
        """Evaluate rule against a value and return alert if triggered."""
        if self.critical_threshold is not None and value >= self.critical_threshold:
            return Alert(
                level=AlertLevel.CRITICAL,
                system=self.name.split(".")[0],
                metric=self.name,
                message=self.message_template.format(
                    metric=self.name, value=value, threshold=self.critical_threshold
                ),
                value=value,
                threshold=self.critical_threshold,
                tags=self.tags,
            )
        elif self.warning_threshold is not None and value >= self.warning_threshold:
            return Alert(
                level=AlertLevel.WARNING,
                system=self.name.split(".")[0],
                metric=self.name,
                message=self.message_template.format(
                    metric=self.name, value=value, threshold=self.warning_threshold
                ),
                value=value,
                threshold=self.warning_threshold,
                tags=self.tags,
            )
        return None


class AlertManager:
    """
    Manages alert rules and evaluates metrics against thresholds.
    """

    def __init__(self):
        """Initialize alert manager with default rules."""
        self.rules: list[AlertRule] = []
        self.alerts: list[Alert] = []
        self.alert_history: list[Alert] = []
        self.max_history = 1000

        # Load default rules from config
        self._load_default_rules()

    def _load_default_rules(self):
        """Load default alert rules from configuration."""
        alerts_config = MONITORING.get("alerts", {})

        # Cache hit rate rules
        for cache_type in ["frame", "validation", "resize"]:
            self.add_rule(
                AlertRule(
                    name=f"cache.{cache_type}.hit_rate",
                    metric_pattern=f"cache.{cache_type}.hits",
                    condition=lambda x: x < 1.0,  # Will calculate hit rate
                    warning_threshold=1
                    - alerts_config.get("cache_hit_rate_warning", 0.4),
                    critical_threshold=1
                    - alerts_config.get("cache_hit_rate_critical", 0.2),
                    message_template=f"{cache_type.capitalize()} cache hit rate {{value:.1%}} below threshold",
                )
            )

        # Memory usage rules
        memory_limits = {
            "frame": 500,  # MB
            "validation": 100,
            "resize": 200,
        }

        for cache_type, limit in memory_limits.items():
            self.add_rule(
                AlertRule(
                    name=f"cache.{cache_type}.memory_usage",
                    metric_pattern=f"cache.{cache_type}.memory_usage_mb",
                    condition=lambda x, lim=limit: x / lim > 0.8,
                    warning_threshold=limit
                    * alerts_config.get("memory_usage_warning", 0.8),
                    critical_threshold=limit
                    * alerts_config.get("memory_usage_critical", 0.95),
                    message_template=f"{cache_type.capitalize()} cache memory usage {{value:.1f}}MB ({{value:.1%}} of {limit}MB limit)",
                )
            )

        # Eviction rate spike rule
        spike_threshold = alerts_config.get("eviction_rate_spike", 3.0)
        for cache_type in ["frame", "validation", "resize"]:
            self.add_rule(
                AlertRule(
                    name=f"cache.{cache_type}.eviction_spike",
                    metric_pattern=f"cache.{cache_type}.evictions",
                    condition=lambda x: x > spike_threshold,
                    warning_threshold=spike_threshold,
                    message_template=f"{cache_type.capitalize()} cache eviction rate spike ({{value:.1f}}x normal)",
                    window_seconds=60,
                )
            )

        # Response time degradation
        degradation_threshold = alerts_config.get("response_time_degradation", 1.5)
        for operation in ["frame", "validation", "resize"]:
            self.add_rule(
                AlertRule(
                    name=f"cache.{operation}.response_time",
                    metric_pattern=f"cache.{operation}.operation.duration",
                    condition=lambda x: x > 0.1,  # 100ms baseline
                    warning_threshold=0.1 * degradation_threshold,
                    message_template=f"{operation.capitalize()} operation taking {{value:.1f}}ms ({{value:.1f}}x baseline)",
                )
            )

    def add_rule(self, rule: AlertRule):
        """Add a new alert rule."""
        self.rules.append(rule)

    def remove_rule(self, name: str):
        """Remove an alert rule by name."""
        self.rules = [r for r in self.rules if r.name != name]

    def evaluate_metrics(
        self,
        window_seconds: int = 300,
        clear_existing: bool = True,
    ) -> list[Alert]:
        """
        Evaluate all rules against current metrics.

        Args:
            window_seconds: Time window for metric evaluation
            clear_existing: Whether to clear existing alerts

        Returns:
            list of triggered alerts
        """
        if clear_existing:
            self.alerts.clear()

        collector = get_metrics_collector()
        summaries = collector.get_summary(window_seconds=window_seconds)

        # Group metrics by type for easier processing
        metrics_by_name = {}
        for summary in summaries:
            key = summary.name
            metrics_by_name[key] = summary

        # Evaluate each rule
        for rule in self.rules:
            # Special handling for hit rate calculations
            if "hit_rate" in rule.name:
                cache_type = rule.name.split(".")[1]
                hits_key = f"cache.{cache_type}.hits"
                misses_key = f"cache.{cache_type}.misses"

                if hits_key in metrics_by_name and misses_key in metrics_by_name:
                    hits = metrics_by_name[hits_key].sum
                    misses = metrics_by_name[misses_key].sum
                    total = hits + misses

                    if total > 0:
                        hit_rate = hits / total
                        # Invert for threshold comparison (we alert on low hit rate)
                        alert = rule.evaluate(1 - hit_rate)
                        if alert:
                            alert.value = hit_rate  # Show actual hit rate
                            self.alerts.append(alert)
                            self.alert_history.append(alert)

            # Direct metric evaluation
            elif rule.metric_pattern in metrics_by_name:
                summary = metrics_by_name[rule.metric_pattern]
                value = summary.p95 if "duration" in rule.name else summary.mean

                alert = rule.evaluate(value)
                if alert:
                    self.alerts.append(alert)
                    self.alert_history.append(alert)

        # Trim history
        if len(self.alert_history) > self.max_history:
            self.alert_history = self.alert_history[-self.max_history :]

        return self.alerts

    def get_active_alerts(self, level: AlertLevel | None = None) -> list[Alert]:
        """
        Get currently active alerts.

        Args:
            level: Filter by alert level

        Returns:
            list of active alerts
        """
        if level is None:
            return self.alerts.copy()
        return [a for a in self.alerts if a.level == level]

    def get_alert_history(
        self,
        hours: float = 24,
        level: AlertLevel | None = None,
    ) -> list[Alert]:
        """
        Get alert history.

        Args:
            hours: Number of hours of history to retrieve
            level: Filter by alert level

        Returns:
            list of historical alerts
        """
        cutoff = time.time() - (hours * 3600)
        history = [a for a in self.alert_history if a.timestamp >= cutoff]

        if level is not None:
            history = [a for a in history if a.level == level]

        return history

    def clear_alerts(self):
        """Clear all active alerts."""
        self.alerts.clear()

    def get_alert_summary(self) -> dict[str, Any]:
        """
        Get summary of alert status.

        Returns:
            Dictionary with alert statistics
        """
        active_by_level = {
            AlertLevel.INFO: 0,
            AlertLevel.WARNING: 0,
            AlertLevel.CRITICAL: 0,
        }

        for alert in self.alerts:
            active_by_level[alert.level] += 1

        # Calculate alert rate (alerts per hour)
        if self.alert_history:
            oldest = min(a.timestamp for a in self.alert_history)
            newest = max(a.timestamp for a in self.alert_history)
            duration_hours = (newest - oldest) / 3600
            alert_rate = (
                len(self.alert_history) / duration_hours if duration_hours > 0 else 0
            )
        else:
            alert_rate = 0

        return {
            "active_alerts": len(self.alerts),
            "critical_count": active_by_level[AlertLevel.CRITICAL],
            "warning_count": active_by_level[AlertLevel.WARNING],
            "info_count": active_by_level[AlertLevel.INFO],
            "alert_rate_per_hour": alert_rate,
            "history_size": len(self.alert_history),
            "rules_count": len(self.rules),
        }


class AlertNotifier:
    """
    Handles alert notifications to various channels.
    """

    def __init__(self):
        """Initialize notifier."""
        self.handlers: dict[str, Callable[[Alert], None]] = {}

        # Register default handlers
        self.register_handler("log", self._log_handler)
        self.register_handler("console", self._console_handler)

    def register_handler(self, name: str, handler: Callable[[Alert], None]):
        """
        Register a notification handler.

        Args:
            name: Handler name
            handler: Function that takes an Alert and sends notification
        """
        self.handlers[name] = handler

    def notify(self, alert: Alert, handlers: list[str] | None = None):
        """
        Send alert notification.

        Args:
            alert: Alert to send
            handlers: list of handler names to use (None = all)
        """
        if handlers is None:
            handlers = list(self.handlers.keys())

        for handler_name in handlers:
            if handler_name in self.handlers:
                try:
                    self.handlers[handler_name](alert)
                except Exception as e:
                    logger.error(f"Error in alert handler {handler_name}: {e}")

    def _log_handler(self, alert: Alert):
        """Log alert to logger."""
        if alert.level == AlertLevel.CRITICAL:
            logger.critical(str(alert))
        elif alert.level == AlertLevel.WARNING:
            logger.warning(str(alert))
        else:
            logger.info(str(alert))

    def _console_handler(self, alert: Alert):
        """Print alert to console."""
        from rich.console import Console

        console = Console()

        color_map = {
            AlertLevel.INFO: "blue",
            AlertLevel.WARNING: "yellow",
            AlertLevel.CRITICAL: "red",
        }

        color = color_map[alert.level]
        console.print(f"[{color}]{alert}[/{color}]")


# Global instances
_alert_manager: AlertManager | None = None
_alert_notifier: AlertNotifier | None = None


def get_alert_manager() -> AlertManager:
    """Get singleton alert manager instance."""
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = AlertManager()
    return _alert_manager


def get_alert_notifier() -> AlertNotifier:
    """Get singleton alert notifier instance."""
    global _alert_notifier
    if _alert_notifier is None:
        _alert_notifier = AlertNotifier()
    return _alert_notifier


def check_alerts(
    window_seconds: int = 300,
    notify: bool = True,
    handlers: list[str] | None = None,
) -> list[Alert]:
    """
    Convenience function to check for alerts and optionally notify.

    Args:
        window_seconds: Time window for evaluation
        notify: Whether to send notifications
        handlers: Notification handlers to use

    Returns:
        list of triggered alerts
    """
    manager = get_alert_manager()
    notifier = get_alert_notifier()

    alerts = manager.evaluate_metrics(window_seconds=window_seconds)

    if notify and alerts:
        for alert in alerts:
            notifier.notify(alert, handlers=handlers)

    return alerts
