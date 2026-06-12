"""Tests for monitoring instrumentation wrappers (giflab.monitoring.integration).

These tests pin two real bugs found during the 2026-06-12 mypy cleanup:

* Bug 2 — argument-signature drift: the instrumented wrappers for
  ``FrameCache.get``/``FrameCache.put`` (and ``ValidationCache.put``'s
  ``metadata`` parameter) accepted fewer arguments than the real methods, so
  any caller using the full signature got a ``TypeError`` once instrumentation
  was enabled.
* Bug 3 — result-attribute drift: ``instrumented_get_stats`` read
  ``ValidationCacheStats.total_gets``/``total_puts`` (fields that have never
  existed -> ``AttributeError``), and two hasattr-guarded gauges
  (``cache.validation.memory_usage_mb``, keyed on the non-existent
  ``memory_usage_mb`` attribute) could never fire.
"""

from pathlib import Path

import numpy as np
import pytest
from giflab.caching.frame_cache import FrameCache
from giflab.caching.validation_cache import ValidationCache
from giflab.monitoring.backends import InMemoryBackend
from giflab.monitoring.integration import (
    instrument_frame_cache,
    instrument_validation_cache,
)
from giflab.monitoring.metrics_collector import MetricsCollector
from PIL import Image


@pytest.fixture
def restore_instrumented_methods():
    """Snapshot and restore the class methods the instrumentation patches.

    ``remove_instrumentation()`` is a documented no-op, so without this
    fixture the patched methods would leak into every other test in the
    worker.
    """
    frame_attrs = {
        name: getattr(FrameCache, name)
        for name in ("get", "put", "_evict_if_needed")
        if hasattr(FrameCache, name)
    }
    validation_attrs = {
        name: getattr(ValidationCache, name)
        for name in ("get", "put", "get_stats")
        if hasattr(ValidationCache, name)
    }
    yield
    for name, fn in frame_attrs.items():
        setattr(FrameCache, name, fn)
    for name, fn in validation_attrs.items():
        setattr(ValidationCache, name, fn)


@pytest.fixture
def collector(monkeypatch):
    """An isolated in-memory collector wired into the instrumentation module."""
    test_collector = MetricsCollector(
        backend=InMemoryBackend(),
        flush_interval=3600.0,  # never auto-flush mid-test
    )
    monkeypatch.setattr(
        "giflab.monitoring.integration.get_metrics_collector",
        lambda: test_collector,
    )
    yield test_collector
    test_collector.shutdown()


@pytest.fixture
def sample_gif(tmp_path):
    gif_path = tmp_path / "test.gif"
    frames = [
        Image.new("RGB", (32, 32), color=(i * 25, i * 25, i * 25)) for i in range(5)
    ]
    frames[0].save(
        gif_path, save_all=True, append_images=frames[1:], duration=100, loop=0
    )
    return gif_path


@pytest.fixture
def frame_cache(tmp_path):
    cache = FrameCache(
        memory_limit_mb=10,
        disk_path=tmp_path / "frame_cache.db",
        disk_limit_mb=50,
        ttl_seconds=3600,
        enabled=True,
    )
    yield cache
    cache.clear()


@pytest.fixture
def validation_cache(tmp_path):
    cache = ValidationCache(
        memory_limit_mb=10,
        disk_path=tmp_path / "validation_cache.db",
        disk_limit_mb=50,
        ttl_seconds=3600,
        enabled=True,
    )
    yield cache
    cache.clear()


class TestFrameCacheInstrumentationSignature:
    """Bug 2: instrumented wrappers must accept the real methods' signatures."""

    def test_instrumented_get_accepts_full_signature(
        self, restore_instrumented_methods, collector, frame_cache, sample_gif
    ):
        instrument_frame_cache()

        # The real FrameCache.get signature is
        # (file_path, max_frames=None, alpha_background=None).
        result = frame_cache.get(
            sample_gif, max_frames=5, alpha_background=(0, 0, 0)
        )
        assert result is None  # cold cache -> miss, but no TypeError

        # Miss must have been counted through the wrapper.
        assert collector.aggregator.counters.get("cache.frame.misses", 0) >= 1.0

    def test_instrumented_put_accepts_full_signature(
        self, restore_instrumented_methods, collector, frame_cache, sample_gif
    ):
        instrument_frame_cache()

        frames = [
            np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8) for _ in range(5)
        ]
        # The real FrameCache.put signature is
        # (file_path, frames, frame_count, dimensions, duration_ms,
        #  alpha_background=None, has_alpha=False).
        frame_cache.put(
            sample_gif,
            frames,
            frame_count=5,
            dimensions=(32, 32),
            duration_ms=500,
            has_alpha=False,
        )

        # The round-trip through the wrapper must now hit.
        result = frame_cache.get(sample_gif, max_frames=5)
        assert result is not None


class TestValidationCacheInstrumentation:
    """Bug 2 (metadata param) + bug 3 (stats attribute drift)."""

    def test_instrumented_put_accepts_metadata_kwarg(
        self, restore_instrumented_methods, collector, validation_cache
    ):
        instrument_validation_cache()

        rng = np.random.default_rng(42)
        frame1 = rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)
        frame2 = rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)

        # The real ValidationCache.put signature ends with metadata=None.
        validation_cache.put(
            frame1, frame2, "ssim", 0.95, metadata={"source": "test"}
        )
        assert validation_cache.get(frame1, frame2, "ssim") == 0.95

    def test_instrumented_get_stats_emits_honest_gauges(
        self, restore_instrumented_methods, collector, validation_cache
    ):
        instrument_validation_cache()

        rng = np.random.default_rng(7)
        frame1 = rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)
        frame2 = rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)
        validation_cache.put(frame1, frame2, "ssim", 0.9)
        validation_cache.get(frame1, frame2, "ssim")

        # On the buggy code this raised AttributeError (no .total_gets).
        stats = validation_cache.get_stats()

        gauges = collector.aggregator.gauges
        # total_entries must report entries actually resident (memory + disk),
        # not operations (hits/misses are separate counters already).
        assert (
            gauges.get("cache.validation.total_entries")
            == stats.memory_entries + stats.disk_entries
        )
        # memory_usage_mb gauge previously never fired (dead hasattr guard on
        # the non-existent stats.memory_usage_mb attribute).
        assert gauges.get("cache.validation.memory_usage_mb") == pytest.approx(
            stats.memory_bytes / (1024 * 1024)
        )
        assert gauges.get("cache.validation.hit_rate") == pytest.approx(
            stats.hit_rate
        )
