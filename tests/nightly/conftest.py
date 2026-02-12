"""Nightly test configuration.

Nightly tests include memory leak detection, performance benchmarks,
stress tests, and golden regression tests. No time limit.
These produce data, not just pass/fail.
"""

import pytest


@pytest.fixture
def memory_tracker():
    """Track memory allocations for leak detection."""
    import tracemalloc

    tracemalloc.start()
    snapshot_before = tracemalloc.take_snapshot()
    yield {"before": snapshot_before}
    tracemalloc.take_snapshot()
    tracemalloc.stop()


@pytest.fixture
def process_memory():
    """Get current process memory usage via psutil."""
    import os

    import psutil

    process = psutil.Process(os.getpid())

    class MemoryMonitor:
        def current_mb(self):
            return process.memory_info().rss / (1024 * 1024)

    return MemoryMonitor()
