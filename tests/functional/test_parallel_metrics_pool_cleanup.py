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

import json
import os
import signal
import subprocess
import sys
import textwrap
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


def _pid_alive(pid: int) -> bool:
    """True if *pid* is a live, non-zombie process."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # signal 0 also succeeds for zombies; treat a zombie as dead.
    try:
        with open(f"/proc/{pid}/stat") as fh:
            return fh.read().split(") ", 1)[1].split(" ", 1)[0] != "Z"
    except OSError:
        # No /proc (macOS) — fall back to ps for the process state.
        out = subprocess.run(
            ["ps", "-o", "state=", "-p", str(pid)],
            capture_output=True,
            text=True,
        )
        state = out.stdout.strip()
        return bool(state) and not state.startswith("Z")


def _executor_processes(executor) -> list:
    """Worker processes still tracked by *executor*.

    After a clean ``shutdown(wait=True)`` the stdlib clears
    ``_processes`` to ``None`` once every worker has exited — that is itself
    the leak-free state, so treat ``None`` as "no live workers".
    """
    processes = getattr(executor, "_processes", None)
    return list(processes.values()) if processes else []


def _wait_until_dead(processes, timeout: float = 15.0) -> list[int]:
    """Return the PIDs from *processes* that are still alive after *timeout*.

    Liveness is probed at the OS level (``_pid_alive``), NOT via
    ``Process.is_alive()``: once the reaper's ``waitpid`` has reaped a worker
    ahead of the executor's own machinery, multiprocessing's internal
    ``poll()`` swallows the ``ChildProcessError`` and ``is_alive()`` reports
    True forever — a dead, fully-reaped worker then reads as an "orphan"
    (false positive observed ~2/15 runs under parallel load).

    Polls and returns early, so the generous timeout only costs time when
    workers genuinely leak; 5s proved too tight for reaping spawned workers
    on loaded 2-core CI runners.
    """
    pids = [p.pid for p in processes]
    deadline = time.time() + timeout
    while time.time() < deadline:
        alive = [pid for pid in pids if _pid_alive(pid)]
        if not alive:
            return []
        time.sleep(0.05)
    return [pid for pid in pids if _pid_alive(pid)]


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


# A child program that imports parallel_metrics (installing the SIGTERM
# reaper), starts a registered pool with live workers, prints their PIDs as
# JSON on a marker line, then idles until SIGTERM arrives. We send a plain
# ``kill`` (SIGTERM) — the default kill signal a process manager sends first —
# and assert the workers it spawned are dead, proving the SIGTERM path is
# genuinely covered (not just claimed) and does not orphan workers the way a
# SIG_DFL terminate would.
#
# It is written to a real module file (not passed via ``-c``) so that the
# ``multiprocessing`` *spawn* start method can re-import it cleanly: spawn
# re-imports the main module in every worker, which raises
# ``RuntimeError: An attempt has been made to start a new process before ...``
# unless the pool work sits under an ``if __name__ == "__main__"`` guard.
_SIGTERM_CHILD = textwrap.dedent(
    """
    import json, time
    from concurrent.futures import ProcessPoolExecutor
    from giflab import parallel_metrics

    if __name__ == "__main__":
        executor = ProcessPoolExecutor(max_workers=2)
        # Force the workers to actually spawn and become idle.
        assert executor.submit(abs, -1).result() == 1
        with parallel_metrics._LIVE_EXECUTORS_LOCK:
            parallel_metrics._LIVE_PROCESS_EXECUTORS.add(executor)
        pids = [p.pid for p in executor._processes.values()]
        print("WORKER_PIDS:" + json.dumps(pids), flush=True)
        # Idle in the foreground so the parent can deliver SIGTERM mid-pool.
        time.sleep(60)
    """
)


@pytest.mark.serial
class TestProcessPoolSigtermReaping:
    """A plain ``kill`` (SIGTERM) must reap the pool's workers, not orphan them."""

    def test_sigterm_reaps_registered_workers(self, tmp_path):
        child = tmp_path / "sigterm_child.py"
        child.write_text(_SIGTERM_CHILD)
        proc = subprocess.Popen(
            [sys.executable, str(child)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            worker_pids: list[int] = []
            deadline = time.time() + 30.0
            assert proc.stdout is not None
            while time.time() < deadline:
                line = proc.stdout.readline()
                if not line:
                    if proc.poll() is not None:
                        break
                    continue
                if line.startswith("WORKER_PIDS:"):
                    worker_pids = json.loads(line[len("WORKER_PIDS:") :])
                    break
            assert worker_pids, "child never reported worker PIDs; stderr:\n" + (
                proc.stderr.read() if proc.stderr else ""
            )
            # All workers must be alive before we kill the parent.
            assert all(_pid_alive(pid) for pid in worker_pids)

            # Plain kill == SIGTERM. The handler must reap then terminate.
            proc.send_signal(signal.SIGTERM)
            proc.wait(timeout=30)

            # Give the OS a moment to finish tearing the workers down.
            deadline = time.time() + 10.0
            while time.time() < deadline:
                still_alive = [pid for pid in worker_pids if _pid_alive(pid)]
                if not still_alive:
                    break
                time.sleep(0.1)
            still_alive = [pid for pid in worker_pids if _pid_alive(pid)]
            assert not still_alive, (
                f"workers orphaned after SIGTERM: {still_alive} "
                "(SIGTERM was not genuinely covered)"
            )
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=10)
