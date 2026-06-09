"""Worker-reaping regression tests for the process-pool metrics path.

Companion to PR #53 (which fixed the broken-pool *deadlock*). PR #53 only
guaranteed teardown on two paths: a clean run (the ``else`` branch) and a
``BrokenProcessPool`` (the ``except`` branch). Any *other* abnormal exit from
``_process_with_pool`` — a ``KeyboardInterrupt`` / ``MemoryError`` /
``SystemExit`` raised while collecting results, or a generic error from
``as_completed`` — skipped both branches, so ``executor.shutdown()`` never ran
and the spawned worker processes were orphaned (re-parented to launchd,
~100 MB RSS each, accumulating across sessions). See task note
``giflab-processpool-worker-leak-on-crash``.

These tests pin the contract: *however* ``_process_with_pool`` exits, every
worker process it spawned must be dead by the time control leaves the method,
and the live-executor registry that the ``atexit`` reaper drains must be empty.
"""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor
from unittest.mock import patch

import numpy as np
import pytest
from giflab import parallel_metrics
from giflab.parallel_metrics import (
    _LIVE_PROCESS_EXECUTORS,
    ParallelConfig,
    ParallelMetricsCalculator,
    _reap_live_process_executors,
)


def _executor_processes(executor) -> list:
    """Worker processes still tracked by *executor*.

    After a clean ``shutdown(wait=True)`` the stdlib clears
    ``_processes`` to ``None`` once every worker has exited — that is itself
    the leak-free state, so treat ``None`` as "no live workers".
    """
    processes = getattr(executor, "_processes", None)
    return list(processes.values()) if processes else []


def _wait_until_dead(processes, timeout: float = 5.0) -> list[int]:
    """Return the PIDs from *processes* that are still alive after *timeout*."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        alive = [p.pid for p in processes if p.is_alive()]
        if not alive:
            return []
        time.sleep(0.05)
    return [p.pid for p in processes if p.is_alive()]


def _make_pairs(n: int) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(0)
    return [
        (
            rng.integers(0, 255, (16, 16, 3), dtype=np.uint8),
            rng.integers(0, 255, (16, 16, 3), dtype=np.uint8),
        )
        for _ in range(n)
    ]


class _ExecutorCapture:
    """Patch ``ProcessPoolExecutor`` so the test can grab the live instance."""

    def __init__(self) -> None:
        self.instances: list[ProcessPoolExecutor] = []

    def __call__(self, *args, **kwargs):
        executor = ProcessPoolExecutor(*args, **kwargs)
        self.instances.append(executor)
        return executor


@pytest.mark.serial
class TestProcessPoolWorkerReaping:
    """The process pool must never orphan workers, on any exit path."""

    def _calc(self) -> ParallelMetricsCalculator:
        return ParallelMetricsCalculator(
            ParallelConfig(max_workers=2, use_process_pool=True)
        )

    def test_clean_run_reaps_workers(self):
        """A successful pooled run leaves no live workers and an empty registry."""
        capture = _ExecutorCapture()
        calc = self._calc()
        with patch.object(parallel_metrics, "ProcessPoolExecutor", capture):
            calc._process_with_pool(
                calc._create_chunks(_make_pairs(4), 1),
                {"mse": None},
                None,
            )
        assert capture.instances, "expected the process pool to be constructed"
        for executor in capture.instances:
            # After a clean shutdown(wait=True) the stdlib clears _processes to
            # None once the workers have exited; _executor_processes() reads
            # that as "no live workers".
            assert _wait_until_dead(_executor_processes(executor)) == []
        assert not _LIVE_PROCESS_EXECUTORS, "registry must be drained after a run"

    def test_abrupt_exception_during_collection_reaps_workers(self):
        """A non-BrokenProcessPool error mid-collection must still reap workers.

        This is the leak the task documents: ``KeyboardInterrupt`` (and friends)
        skip both the ``except BrokenProcessPool`` and the ``else`` branch, so
        the pre-fix code never called ``shutdown()`` and the workers survived.
        """
        capture = _ExecutorCapture()
        calc = self._calc()
        # Snapshot the live worker Process objects at the moment of failure —
        # after submit() has spawned them but before _process_with_pool tears
        # the executor down (which may clear _processes).
        snapshot: list = []

        def boom(_futures):  # noqa: ANN001 - test stub
            executor = capture.instances[-1]
            snapshot.extend(_executor_processes(executor))
            raise KeyboardInterrupt("simulated abnormal parent exit")

        with patch.object(parallel_metrics, "ProcessPoolExecutor", capture):
            with patch.object(parallel_metrics, "as_completed", boom):
                with pytest.raises(KeyboardInterrupt):
                    calc._process_with_pool(
                        calc._create_chunks(_make_pairs(4), 1),
                        {"mse": None},
                        None,
                    )

        assert capture.instances, "expected the process pool to be constructed"
        assert snapshot, "expected worker processes to have spawned before failure"
        leaked = _wait_until_dead(snapshot)
        assert not leaked, f"workers orphaned after abnormal exit: {leaked}"
        assert not _LIVE_PROCESS_EXECUTORS, "registry must be drained after a run"

    def test_atexit_reaper_terminates_registered_executor(self):
        """The atexit reaper must terminate any executor still in the registry.

        Simulates the parent dying mid-pool: an executor is live in the
        registry and the reaper (installed via ``atexit``) must kill its
        workers rather than leaving them to re-parent to launchd.
        """
        executor = ProcessPoolExecutor(max_workers=2)
        try:
            # Force the workers to actually start.
            assert executor.submit(abs, -1).result() == 1
            _LIVE_PROCESS_EXECUTORS.add(executor)
            procs = _executor_processes(executor)
            assert procs, "expected spawned worker processes"

            _reap_live_process_executors()

            assert _wait_until_dead(procs) == []
            assert executor not in _LIVE_PROCESS_EXECUTORS
        finally:
            _LIVE_PROCESS_EXECUTORS.discard(executor)
            executor.shutdown(wait=False, cancel_futures=True)
