"""
Pluggable backends for metrics storage.
"""

import json
import logging
import sqlite3
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from pathlib import Path
from typing import Any, Optional

from .metrics_collector import MetricPoint, MetricType

logger = logging.getLogger(__name__)

class MetricsBackend(ABC):
    """Abstract base class for metrics storage backends."""
    
    @abstractmethod
    def write(self, metric: MetricPoint):
        """Write single metric point."""
        pass
    
    @abstractmethod
    def write_batch(self, metrics: list[MetricPoint]):
        """Write batch of metric points."""
        pass
    
    @abstractmethod
    def query(
        self,
        start_time: float = None,
        end_time: float = None,
        name: str = None,
        metric_type: MetricType = None,
        tags: dict[str, str] = None,
        limit: int = 1000,
    ) -> list[MetricPoint]:
        """Query metrics with filters."""
        pass
    
    @abstractmethod
    def clear(self):
        """Clear all stored metrics."""
        pass
    
    @abstractmethod
    def get_stats(self) -> dict[str, Any]:
        """Get backend statistics."""
        pass

class InMemoryBackend(MetricsBackend):
    """
    Simple in-memory metrics backend using a deque.
    Suitable for development and testing.
    """
    
    def __init__(self, max_size: int = 100000):
        self.max_size = max_size
        self.metrics = deque(maxlen=max_size)
        self.lock = threading.RLock()
        self.total_written = 0
    
    def write(self, metric: MetricPoint):
        """Write single metric point."""
        with self.lock:
            self.metrics.append(metric)
            self.total_written += 1
    
    def write_batch(self, metrics: list[MetricPoint]):
        """Write batch of metric points."""
        with self.lock:
            self.metrics.extend(metrics)
            self.total_written += len(metrics)
    
    def query(
        self,
        start_time: float = None,
        end_time: float = None,
        name: str = None,
        metric_type: MetricType = None,
        tags: dict[str, str] = None,
        limit: int = 1000,
    ) -> list[MetricPoint]:
        """Query metrics with filters."""
        with self.lock:
            results = []
            for metric in self.metrics:
                # Apply filters
                if start_time and metric.timestamp < start_time:
                    continue
                if end_time and metric.timestamp > end_time:
                    continue
                if name and not metric.name.startswith(name):
                    continue
                if metric_type and metric.metric_type != metric_type:
                    continue
                if tags and not all(metric.tags.get(k) == v for k, v in tags.items()):
                    continue
                
                results.append(metric)
                if len(results) >= limit:
                    break
            
            return results
    
    def clear(self):
        """Clear all stored metrics."""
        with self.lock:
            self.metrics.clear()
            self.total_written = 0
    
    def get_stats(self) -> dict[str, Any]:
        """Get backend statistics."""
        with self.lock:
            return {
                "backend": "memory",
                "current_size": len(self.metrics),
                "max_size": self.max_size,
                "total_written": self.total_written,
                "oldest_timestamp": self.metrics[0].timestamp if self.metrics else None,
                "newest_timestamp": self.metrics[-1].timestamp if self.metrics else None,
            }

class SQLiteBackend(MetricsBackend):
    """
    SQLite-based metrics backend for persistent storage.
    Provides efficient querying and long-term retention.
    """
    
    def __init__(
        self,
        db_path: Path | None = None,
        retention_days: float = 7.0,
        max_size_mb: float = 100.0,
    ):
        """
        Initialize SQLite backend.
        
        Args:
            db_path: Path to SQLite database (defaults to ~/.giflab_cache/metrics.db)
            retention_days: Days to retain metrics before cleanup
            max_size_mb: Maximum database size in MB
        """
        if db_path is None:
            cache_dir = Path.home() / ".giflab_cache"
            cache_dir.mkdir(exist_ok=True)
            db_path = cache_dir / "metrics.db"
        
        self.db_path = Path(db_path)
        self.retention_days = retention_days
        self.max_size_mb = max_size_mb
        self.lock = threading.RLock()
        
        # Statistics
        self.total_written = 0
        self.last_cleanup = time.time()
        
        # Initialize database
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema."""
        with self.lock:
            conn = sqlite3.connect(str(self.db_path))
            try:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL,
                        value REAL NOT NULL,
                        metric_type TEXT NOT NULL,
                        timestamp REAL NOT NULL,
                        tags TEXT,
                        created_at REAL DEFAULT (strftime('%s', 'now'))
                    )
                """)
                
                # Create indexes for efficient querying
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_metrics_timestamp
                    ON metrics(timestamp DESC)
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_metrics_name_timestamp
                    ON metrics(name, timestamp DESC)
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_metrics_type_timestamp
                    ON metrics(metric_type, timestamp DESC)
                """)
                
                conn.commit()
            finally:
                conn.close()
    
    def write(self, metric: MetricPoint):
        """Write single metric point."""
        self.write_batch([metric])
    
    def write_batch(self, metrics: list[MetricPoint]):
        """Write batch of metric points."""
        if not metrics:
            return
        
        with self.lock:
            conn = sqlite3.connect(str(self.db_path))
            try:
                # Prepare batch data
                batch_data = []
                for metric in metrics:
                    tags_json = json.dumps(metric.tags) if metric.tags else None
                    batch_data.append((
                        metric.name,
                        metric.value,
                        metric.metric_type.value,
                        metric.timestamp,
                        tags_json,
                    ))
                
                # Batch insert
                conn.executemany("""
                    INSERT INTO metrics (name, value, metric_type, timestamp, tags)
                    VALUES (?, ?, ?, ?, ?)
                """, batch_data)
                
                conn.commit()
                self.total_written += len(metrics)
                
                # Periodic cleanup
                if time.time() - self.last_cleanup > 3600:  # Every hour
                    self._cleanup()
                
            finally:
                conn.close()
    
    def query(
        self,
        start_time: float = None,
        end_time: float = None,
        name: str = None,
        metric_type: MetricType = None,
        tags: dict[str, str] = None,
        limit: int = 1000,
    ) -> list[MetricPoint]:
        """Query metrics with filters."""
        with self.lock:
            conn = sqlite3.connect(str(self.db_path))
            try:
                # Build query
                query = "SELECT name, value, metric_type, timestamp, tags FROM metrics WHERE 1=1"
                params = []
                
                if start_time:
                    query += " AND timestamp >= ?"
                    params.append(start_time)
                if end_time:
                    query += " AND timestamp <= ?"
                    params.append(end_time)
                if name:
                    query += " AND name LIKE ?"
                    params.append(f"{name}%")
                if metric_type:
                    query += " AND metric_type = ?"
                    params.append(metric_type.value)
                
                query += " ORDER BY timestamp DESC LIMIT ?"
                params.append(limit)
                
                cursor = conn.execute(query, params)
                results = []
                
                for row in cursor:
                    metric_tags = json.loads(row[4]) if row[4] else {}
                    
                    # Filter by tags if specified
                    if tags and not all(metric_tags.get(k) == v for k, v in tags.items()):
                        continue
                    
                    results.append(MetricPoint(
                        name=row[0],
                        value=row[1],
                        metric_type=MetricType(row[2]),
                        timestamp=row[3],
                        tags=metric_tags,
                    ))
                
                return results
                
            finally:
                conn.close()
    
    def clear(self):
        """Clear all stored metrics."""
        with self.lock:
            conn = sqlite3.connect(str(self.db_path), isolation_level=None)
            try:
                conn.execute("DELETE FROM metrics")
                conn.execute("VACUUM")
                self.total_written = 0
            finally:
                conn.close()
    
    def get_stats(self) -> dict[str, Any]:
        """Get backend statistics."""
        with self.lock:
            conn = sqlite3.connect(str(self.db_path))
            try:
                # Get counts and timestamps
                cursor = conn.execute("""
                    SELECT
                        COUNT(*) as count,
                        MIN(timestamp) as oldest,
                        MAX(timestamp) as newest
                    FROM metrics
                """)
                row = cursor.fetchone()
                
                # Get database size
                db_size = self.db_path.stat().st_size if self.db_path.exists() else 0
                
                return {
                    "backend": "sqlite",
                    "db_path": str(self.db_path),
                    "current_count": row[0] if row else 0,
                    "oldest_timestamp": row[1] if row else None,
                    "newest_timestamp": row[2] if row else None,
                    "total_written": self.total_written,
                    "db_size_mb": db_size / (1024 * 1024),
                    "max_size_mb": self.max_size_mb,
                    "retention_days": self.retention_days,
                }
            finally:
                conn.close()
    
    def _cleanup(self):
        """Clean up old metrics and manage database size."""
        cutoff_time = time.time() - (self.retention_days * 86400)

        conn = sqlite3.connect(str(self.db_path), isolation_level=None)
        try:
            # Delete old metrics
            conn.execute("DELETE FROM metrics WHERE timestamp < ?", (cutoff_time,))

            # Check database size
            db_size_mb = self.db_path.stat().st_size / (1024 * 1024)
            if db_size_mb > self.max_size_mb:
                # Delete oldest 10% of metrics
                conn.execute("""
                    DELETE FROM metrics
                    WHERE id IN (
                        SELECT id FROM metrics
                        ORDER BY timestamp ASC
                        LIMIT (SELECT COUNT(*) / 10 FROM metrics)
                    )
                """)

            conn.execute("VACUUM")
            self.last_cleanup = time.time()
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
        finally:
            conn.close()

class StatsDBackend(MetricsBackend):
    """
    StatsD backend for integration with external monitoring systems.
    Requires statsd library: pip install statsd
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 8125,
        prefix: str = "giflab",
        buffer_size: int = 1000,
    ):
        """
        Initialize StatsD backend.
        
        Args:
            host: StatsD server host
            port: StatsD server port
            prefix: Metric name prefix
            buffer_size: Local buffer size for fallback
        """
        self.host = host
        self.port = port
        self.prefix = prefix
        
        # Local buffer for when StatsD is unavailable
        self.fallback_buffer = deque(maxlen=buffer_size)
        self.client = None
        
        try:
            import statsd
            self.client = statsd.StatsClient(host, port, prefix=prefix)
            self.available = True
        except ImportError:
            logger.warning("statsd library not available, using fallback buffer")
            self.available = False
    
    def write(self, metric: MetricPoint):
        """Write single metric point."""
        if not self.available:
            self.fallback_buffer.append(metric)
            return
        
        try:
            # Format metric name with tags
            name = self._format_name(metric.name, metric.tags)
            
            if metric.metric_type == MetricType.COUNTER:
                self.client.incr(name, metric.value)
            elif metric.metric_type == MetricType.GAUGE:
                self.client.gauge(name, metric.value)
            elif metric.metric_type == MetricType.TIMER:
                self.client.timing(name, metric.value * 1000)  # Convert to ms
            elif metric.metric_type == MetricType.HISTOGRAM:
                # StatsD doesn't have native histogram, use gauge
                self.client.gauge(name, metric.value)
                
        except Exception as e:
            logger.error(f"Error sending to StatsD: {e}")
            self.fallback_buffer.append(metric)
    
    def write_batch(self, metrics: list[MetricPoint]):
        """Write batch of metric points."""
        for metric in metrics:
            self.write(metric)
    
    def query(self, **kwargs) -> list[MetricPoint]:
        """StatsD backend doesn't support querying."""
        # Return fallback buffer contents if requested
        return list(self.fallback_buffer)
    
    def clear(self):
        """Clear fallback buffer."""
        self.fallback_buffer.clear()
    
    def get_stats(self) -> dict[str, Any]:
        """Get backend statistics."""
        return {
            "backend": "statsd",
            "host": self.host,
            "port": self.port,
            "prefix": self.prefix,
            "available": self.available,
            "fallback_buffer_size": len(self.fallback_buffer),
        }
    
    def _format_name(self, name: str, tags: dict[str, str]) -> str:
        """Format metric name with tags for StatsD."""
        if not tags:
            return name
        # Simple tag encoding for StatsD
        tag_str = ".".join(f"{k}_{v}" for k, v in sorted(tags.items()))
        return f"{name}.{tag_str}"

def create_backend(backend_type: str, **kwargs) -> MetricsBackend:
    """
    Factory function to create metrics backend.
    
    Args:
        backend_type: type of backend ("memory", "sqlite", "statsd")
        **kwargs: Backend-specific configuration
    
    Returns:
        MetricsBackend instance
    """
    if backend_type == "memory":
        return InMemoryBackend(**kwargs)
    elif backend_type == "sqlite":
        return SQLiteBackend(**kwargs)
    elif backend_type == "statsd":
        return StatsDBackend(**kwargs)
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")
