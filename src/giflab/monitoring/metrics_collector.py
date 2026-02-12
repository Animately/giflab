"""
Core metrics collection infrastructure with pluggable backends.
"""

import logging
import statistics
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional, Union

if TYPE_CHECKING:
    from .backends import MetricsBackend

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics that can be collected."""

    COUNTER = "counter"  # Monotonically increasing value
    GAUGE = "gauge"  # Point-in-time value
    HISTOGRAM = "histogram"  # Distribution of values
    TIMER = "timer"  # Duration measurements


@dataclass
class MetricPoint:
    """Single metric data point."""

    name: str
    value: float
    metric_type: MetricType
    timestamp: float
    tags: dict[str, str] = field(default_factory=dict)


@dataclass
class MetricSummary:
    """Summary statistics for a metric over a time window."""

    name: str
    metric_type: MetricType
    count: int
    sum: float
    min: float
    max: float
    mean: float
    median: float
    p95: float
    p99: float
    stddev: float
    tags: dict[str, str] = field(default_factory=dict)
    window_start: float = field(default_factory=time.time)
    window_end: float = field(default_factory=time.time)


class RingBuffer:
    """Thread-safe ring buffer for storing recent metrics."""

    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.lock = threading.RLock()

    def append(self, item: MetricPoint):
        """Add item to buffer."""
        with self.lock:
            self.buffer.append(item)

    def get_all(self) -> list[MetricPoint]:
        """Get all items in buffer."""
        with self.lock:
            return list(self.buffer)

    def get_recent(self, seconds: float) -> list[MetricPoint]:
        """Get items from last N seconds."""
        cutoff = time.time() - seconds
        with self.lock:
            return [p for p in self.buffer if p.timestamp >= cutoff]

    def clear(self):
        """Clear all items."""
        with self.lock:
            self.buffer.clear()


class MetricsAggregator:
    """Aggregates metrics for efficient batch processing."""

    def __init__(self, flush_interval: float = 10.0):
        self.flush_interval = flush_interval
        self.counters = defaultdict(float)
        self.gauges = defaultdict(float)
        self.histograms = defaultdict(list)
        self.timers = defaultdict(list)
        self.lock = threading.RLock()
        self.last_flush = time.time()

    def add_counter(self, name: str, value: float, tags: dict[str, str] = None):
        """Increment counter."""
        key = self._make_key(name, tags)
        with self.lock:
            self.counters[key] += value

    def set_gauge(self, name: str, value: float, tags: dict[str, str] = None):
        """Set gauge value."""
        key = self._make_key(name, tags)
        with self.lock:
            self.gauges[key] = value

    def add_histogram(self, name: str, value: float, tags: dict[str, str] = None):
        """Add value to histogram."""
        key = self._make_key(name, tags)
        with self.lock:
            self.histograms[key].append(value)

    def add_timer(self, name: str, duration: float, tags: dict[str, str] = None):
        """Add timing measurement."""
        key = self._make_key(name, tags)
        with self.lock:
            self.timers[key].append(duration)

    def should_flush(self) -> bool:
        """Check if aggregator should be flushed."""
        return time.time() - self.last_flush >= self.flush_interval

    def flush(self) -> list[MetricPoint]:
        """Flush all aggregated metrics."""
        with self.lock:
            points = []
            timestamp = time.time()

            # Flush counters
            for key, value in self.counters.items():
                name, tags = self._parse_key(key)
                points.append(
                    MetricPoint(
                        name=name,
                        value=value,
                        metric_type=MetricType.COUNTER,
                        timestamp=timestamp,
                        tags=tags,
                    )
                )

            # Flush gauges
            for key, value in self.gauges.items():
                name, tags = self._parse_key(key)
                points.append(
                    MetricPoint(
                        name=name,
                        value=value,
                        metric_type=MetricType.GAUGE,
                        timestamp=timestamp,
                        tags=tags,
                    )
                )

            # Flush histograms (store summary stats)
            for key, values in self.histograms.items():
                if values:
                    name, tags = self._parse_key(key)
                    # Store key percentiles
                    for percentile, value in self._calculate_percentiles(
                        values
                    ).items():
                        points.append(
                            MetricPoint(
                                name=f"{name}.{percentile}",
                                value=value,
                                metric_type=MetricType.HISTOGRAM,
                                timestamp=timestamp,
                                tags=tags,
                            )
                        )

            # Flush timers (store as histograms)
            for key, durations in self.timers.items():
                if durations:
                    name, tags = self._parse_key(key)
                    for percentile, value in self._calculate_percentiles(
                        durations
                    ).items():
                        points.append(
                            MetricPoint(
                                name=f"{name}.{percentile}",
                                value=value * 1000,  # Convert to ms
                                metric_type=MetricType.TIMER,
                                timestamp=timestamp,
                                tags=tags,
                            )
                        )

            # Clear aggregated data
            self.counters.clear()
            self.gauges.clear()
            self.histograms.clear()
            self.timers.clear()
            self.last_flush = timestamp

            return points

    def _make_key(self, name: str, tags: dict[str, str] | None) -> str:
        """Create unique key from name and tags."""
        if not tags:
            return name
        tag_str = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
        return f"{name}|{tag_str}"

    def _parse_key(self, key: str) -> tuple[str, dict[str, str]]:
        """Parse key back into name and tags."""
        if "|" not in key:
            return key, {}
        name, tag_str = key.split("|", 1)
        tags = {}
        if tag_str:
            for pair in tag_str.split(","):
                k, v = pair.split("=", 1)
                tags[k] = v
        return name, tags

    def _calculate_percentiles(self, values: list[float]) -> dict[str, float]:
        """Calculate percentiles for a list of values."""
        if not values:
            return {}

        sorted_values = sorted(values)
        return {
            "min": sorted_values[0],
            "p50": statistics.median(sorted_values),
            "p95": self._percentile(sorted_values, 0.95),
            "p99": self._percentile(sorted_values, 0.99),
            "max": sorted_values[-1],
            "mean": statistics.mean(values),
            "count": len(values),
        }

    def _percentile(self, sorted_values: list[float], p: float) -> float:
        """Calculate percentile from sorted values."""
        if not sorted_values:
            return 0.0
        k = (len(sorted_values) - 1) * p
        f = int(k)
        c = f + 1 if f < len(sorted_values) - 1 else f
        return sorted_values[f] + (k - f) * (sorted_values[c] - sorted_values[f])


class MetricsCollector:
    """
    Main metrics collector that coordinates metric collection and storage.

    This collector provides:
    - Thread-safe metric recording
    - Automatic aggregation and batching
    - Multiple backend support
    - Efficient in-memory buffering
    - Configurable sampling rates
    """

    def __init__(
        self,
        backend: Optional["MetricsBackend"] = None,
        buffer_size: int = 10000,
        flush_interval: float = 10.0,
        enabled: bool = True,
        sampling_rate: float = 1.0,
    ):
        """
        Initialize metrics collector.

        Args:
            backend: Backend for storing metrics (defaults to InMemoryBackend)
            buffer_size: Size of in-memory ring buffer
            flush_interval: Seconds between automatic flushes
            enabled: Whether metrics collection is enabled
            sampling_rate: Fraction of metrics to collect (1.0 = all)
        """
        from .backends import InMemoryBackend

        self.backend = backend or InMemoryBackend()
        self.buffer = RingBuffer(max_size=buffer_size)
        self.aggregator = MetricsAggregator(flush_interval=flush_interval)
        self.enabled = enabled
        self.sampling_rate = sampling_rate
        self.lock = threading.RLock()
        self._shutdown = False

        # Start background flush thread
        self._flush_thread = threading.Thread(target=self._flush_loop, daemon=True)
        self._flush_thread.start()

    def record_counter(
        self, name: str, value: float = 1.0, tags: dict[str, str] = None
    ):
        """Record counter increment."""
        if not self._should_sample():
            return

        self.aggregator.add_counter(name, value, tags)
        if self.aggregator.should_flush():
            self.flush()

    def record_gauge(self, name: str, value: float, tags: dict[str, str] = None):
        """Record gauge value."""
        if not self._should_sample():
            return

        self.aggregator.set_gauge(name, value, tags)
        if self.aggregator.should_flush():
            self.flush()

    def record_histogram(self, name: str, value: float, tags: dict[str, str] = None):
        """Record histogram value."""
        if not self._should_sample():
            return

        self.aggregator.add_histogram(name, value, tags)
        if self.aggregator.should_flush():
            self.flush()

    def record_timer(self, name: str, duration: float, tags: dict[str, str] = None):
        """Record timer duration in seconds."""
        if not self._should_sample():
            return

        self.aggregator.add_timer(name, duration, tags)
        if self.aggregator.should_flush():
            self.flush()

    def flush(self):
        """Flush aggregated metrics to backend."""
        try:
            points = self.aggregator.flush()
            if points:
                # Add to ring buffer
                for point in points:
                    self.buffer.append(point)

                # Send to backend
                self.backend.write_batch(points)
        except Exception as e:
            logger.error(f"Error flushing metrics: {e}")

    def get_summary(
        self,
        name: str = None,
        metric_type: MetricType = None,
        window_seconds: float = 300,
        tags: dict[str, str] = None,
    ) -> list[MetricSummary]:
        """
        Get summary statistics for metrics.

        Args:
            name: Filter by metric name (prefix match)
            metric_type: Filter by metric type
            window_seconds: Time window to summarize
            tags: Filter by tags

        Returns:
            list of metric summaries
        """
        # Get recent points from buffer
        points = self.buffer.get_recent(window_seconds)

        # Filter points
        if name:
            points = [p for p in points if p.name.startswith(name)]
        if metric_type:
            points = [p for p in points if p.metric_type == metric_type]
        if tags:
            points = [
                p for p in points if all(p.tags.get(k) == v for k, v in tags.items())
            ]

        # Group by metric name and tags
        grouped = defaultdict(list)
        for point in points:
            key = (point.name, frozenset(point.tags.items()))
            grouped[key].append(point)

        # Calculate summaries
        summaries = []
        for (metric_name, tag_items), metric_points in grouped.items():
            if not metric_points:
                continue

            values = [p.value for p in metric_points]
            summaries.append(
                MetricSummary(
                    name=metric_name,
                    metric_type=metric_points[0].metric_type,
                    count=len(values),
                    sum=sum(values),
                    min=min(values),
                    max=max(values),
                    mean=statistics.mean(values),
                    median=statistics.median(values),
                    p95=self.aggregator._percentile(sorted(values), 0.95),
                    p99=self.aggregator._percentile(sorted(values), 0.99),
                    stddev=statistics.stdev(values) if len(values) > 1 else 0,
                    tags=dict(tag_items),
                    window_start=min(p.timestamp for p in metric_points),
                    window_end=max(p.timestamp for p in metric_points),
                )
            )

        return summaries

    def get_metrics(
        self, start_time: float = None, end_time: float = None, **filters
    ) -> list[MetricPoint]:
        """Get raw metric points from backend."""
        return self.backend.query(start_time, end_time, **filters)

    def clear(self):
        """Clear all metrics."""
        self.buffer.clear()
        self.backend.clear()

    def shutdown(self):
        """Shutdown metrics collector."""
        self._shutdown = True
        self.flush()
        if self._flush_thread.is_alive():
            self._flush_thread.join(timeout=1.0)

    def _should_sample(self) -> bool:
        """Check if metric should be sampled."""
        if not self.enabled:
            return False
        if self.sampling_rate >= 1.0:
            return True
        import random

        return random.random() < self.sampling_rate

    def _flush_loop(self):
        """Background thread for periodic flushing."""
        while not self._shutdown:
            time.sleep(1.0)
            if self.aggregator.should_flush():
                self.flush()


# Global singleton instance
_metrics_collector: MetricsCollector | None = None
_lock = threading.Lock()


def get_metrics_collector() -> MetricsCollector:
    """Get singleton metrics collector instance."""
    global _metrics_collector
    if _metrics_collector is None:
        with _lock:
            if _metrics_collector is None:
                from ..config import MONITORING
                from .backends import create_backend

                backend = create_backend(MONITORING.get("backend", "memory"))
                _metrics_collector = MetricsCollector(
                    backend=backend,
                    buffer_size=MONITORING.get("buffer_size", 10000),
                    flush_interval=MONITORING.get("flush_interval", 10.0),
                    enabled=MONITORING.get("enabled", True),
                    sampling_rate=MONITORING.get("sampling_rate", 1.0),
                )
    return _metrics_collector


def reset_metrics_collector():
    """Reset singleton metrics collector (mainly for testing)."""
    global _metrics_collector
    with _lock:
        if _metrics_collector:
            _metrics_collector.shutdown()
        _metrics_collector = None
