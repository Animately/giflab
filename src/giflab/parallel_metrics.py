"""Parallel processing utilities for GIF metrics calculation.

This module provides parallelization infrastructure for frame-level metrics
to significantly reduce processing time for multi-frame GIFs.
"""

import atexit
import logging
import multiprocessing as mp
import os
import signal
import threading
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from functools import partial
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# --- Worker-leak guard --------------------------------------------------------
#
# ``ProcessPoolExecutor`` only reaps its worker processes when ``shutdown()`` is
# called. If the parent exits abnormally while a pool is live — an uncaught
# error or ``KeyboardInterrupt`` propagating out of ``_process_with_pool``, or a
# fatal signal — the workers are never told to stop and re-parent to the init
# process (launchd / PID 1), lingering at ~100 MB RSS each and accumulating
# across sessions (observed: a 2-day-old batch of orphans). See task
# ``giflab-processpool-worker-leak-on-crash``.
#
# This registry tracks every executor that is currently mid-flight inside
# ``_process_with_pool``. Three mechanisms drain it, one per way the parent can
# go down:
#
#   * Normal interpreter exit (a ``sys.exit`` / falling off ``main`` / an
#     uncaught exception that propagates all the way out) — the ``atexit`` hook
#     below.
#   * ``SIGTERM`` (plain ``kill``, the first signal process managers send) —
#     a dedicated handler installed by ``_install_sigterm_reaper`` below.
#     Without it ``SIGTERM`` defaults to ``SIG_DFL`` (immediate terminate),
#     which runs *no* ``atexit`` hooks, so the workers would orphan.
#   * ``SIGINT`` (Ctrl-C) — *not* handled here. Python's default ``SIGINT``
#     disposition raises ``KeyboardInterrupt`` into ``_process_with_pool``,
#     where the ``except BaseException`` branch tears the pool down inline.
#
# ``stdlib``'s own ``concurrent.futures`` atexit hook joins workers with
# ``wait=True`` and can itself deadlock on a *broken* pool — the exact hang
# PR #53 fixed — so every drain path here deliberately *terminates* rather
# than joins.
#
# Note: ``giflab.multiprocessing_support`` installs its own ``SIGTERM``/
# ``SIGINT`` handlers, but (a) that module is not imported on the metrics path
# (it is absent from ``sys.modules`` after ``import giflab.parallel_metrics``),
# and (b) its handler only *logs and returns* — it neither exits nor re-raises,
# so it would not trigger any ``atexit``-style reaping even if loaded, and it
# would actively swallow the ``KeyboardInterrupt`` the ``SIGINT`` path relies
# on. We therefore do our own signal handling rather than depend on it.
_LIVE_PROCESS_EXECUTORS: set[ProcessPoolExecutor] = set()
_LIVE_EXECUTORS_LOCK = threading.Lock()


def _pool_worker_pids(executor: ProcessPoolExecutor) -> list[int]:
    """Snapshot the worker OS PIDs an *executor* currently tracks.

    Must be read BEFORE ``executor.shutdown()``: the stdlib clears
    ``_processes`` to ``None`` partway through ``shutdown``, so a snapshot taken
    afterwards finds nothing to reap and the workers orphan. Filters on a
    non-None pid only — NOT on ``is_alive()``: ``Process.is_alive`` routes
    through ``Popen.poll()``, which is unreliable from inside our signal
    handler or a half-torn-down pool; ``os.kill`` later treats an already-dead
    PID as a graceful no-op, so a populated-pid check is the honest gate.
    """
    processes = getattr(executor, "_processes", None) or {}
    return [proc.pid for proc in list(processes.values()) if getattr(proc, "pid", None)]


def _terminate_pool_workers(executor: ProcessPoolExecutor) -> None:
    """Force-kill any worker processes still alive under *executor*.

    Snapshots the worker PIDs (see ``_pool_worker_pids`` for why this must
    precede ``shutdown``) and signals them via ``_terminate_worker_pids``.
    Best-effort: an executor that never started its workers is a no-op.
    """
    _terminate_worker_pids(_pool_worker_pids(executor))


def _terminate_worker_pids(pids: list[int]) -> None:
    """Force-kill the given worker *pids*, escalating ``SIGTERM`` -> ``SIGKILL``.

    We signal the worker OS processes directly with ``os.kill`` on their PIDs —
    not via ``Process.terminate()`` / ``Process.kill()``. Those route through
    the worker's ``Popen`` object, which (verified empirically) becomes a no-op
    when called from inside our ``SIGTERM`` handler or while the pool is being
    torn down: the SIGTERM is delivered to the parent but the worker survives
    and re-parents to launchd — the exact orphan this guard exists to prevent.
    ``os.kill(pid, ...)`` hits the kernel directly and always lands.

    Escalation: ``SIGTERM`` first (lets the worker exit cleanly), a short
    bounded reap, then an uncatchable ``SIGKILL`` for any survivor. Reaping is a
    non-blocking ``os.waitpid(..., WNOHANG)`` against PIDs we are the parent of,
    so the workers do not linger as zombies. Best-effort: an already-exited PID
    is a no-op.
    """
    if not pids:
        return

    def _send(pid: int, sig: int) -> None:
        try:
            os.kill(pid, sig)
        except OSError:  # already reaped / not ours — graceful no-op
            pass

    # SIGTERM first.
    for pid in pids:
        _send(pid, signal.SIGTERM)
    survivors = _reap_pids(pids, timeout=2.0)
    # SIGKILL anything still standing.
    for pid in survivors:
        _send(pid, signal.SIGKILL)
    _reap_pids(survivors, timeout=2.0)


def _reap_pids(pids: list[int], timeout: float) -> list[int]:
    """Wait up to *timeout* for *pids* to die; return the PIDs still alive.

    Reaps children we own with a non-blocking ``waitpid`` so they do not linger
    as zombies; for PIDs we are not the parent of (the registry can outlive the
    direct parent relationship), liveness is probed with signal ``0``.
    """
    deadline = time.monotonic() + timeout
    remaining = list(pids)
    while remaining and time.monotonic() < deadline:
        still: list[int] = []
        for pid in remaining:
            try:
                reaped_pid, _ = os.waitpid(pid, os.WNOHANG)
                if reaped_pid == 0:
                    # Still running and we are its parent.
                    still.append(pid)
            except ChildProcessError:
                # Not our child (or already reaped) — fall back to signal 0.
                try:
                    os.kill(pid, 0)
                    still.append(pid)
                except OSError:
                    pass  # gone
            except OSError:
                pass  # gone
        remaining = still
        if remaining:
            time.sleep(0.02)
    return remaining


def _reap_live_process_executors() -> None:
    """Tear down every still-registered process pool, terminating its workers.

    Invoked from ``atexit`` and from the ``SIGTERM`` handler so that an abnormal
    parent exit cannot leave orphaned ``multiprocessing.spawn`` workers behind.

    Order matters: we snapshot the worker PIDs and kill them by PID FIRST, then
    call ``executor.shutdown(wait=False)`` for the management-thread cleanup.
    Doing shutdown first would clear ``_processes`` to ``None`` before we could
    read the PIDs (the original leak), and ``wait=False`` is required regardless
    to avoid the broken-pool join deadlock PR #53 fixed.
    """
    with _LIVE_EXECUTORS_LOCK:
        executors = list(_LIVE_PROCESS_EXECUTORS)
        _LIVE_PROCESS_EXECUTORS.clear()
    for executor in executors:
        # Snapshot + kill the workers BEFORE shutdown nulls ``_processes``.
        _terminate_worker_pids(_pool_worker_pids(executor))
        try:
            executor.shutdown(wait=False, cancel_futures=True)
        except Exception:  # pragma: no cover - defensive
            pass


# Reap on normal interpreter exit (``sys.exit``, falling off ``main``, or an
# uncaught exception propagating out). This does NOT cover ``SIGTERM``: a
# ``SIG_DFL`` terminate runs no ``atexit`` hooks, so ``SIGTERM`` needs its own
# handler (installed just below). It also does not cover ``SIGINT``/Ctrl-C —
# that surfaces as ``KeyboardInterrupt`` inside ``_process_with_pool`` and is
# torn down there. ``SIGKILL`` / ``kill -9`` is uncatchable by design — nothing
# in Python can reap on -9.
atexit.register(_reap_live_process_executors)


def _install_sigterm_reaper() -> None:
    """Install a ``SIGTERM`` handler that reaps live pools, then terminates.

    A plain ``kill`` (``SIGTERM``) is the default signal a process manager — or
    a human — sends first, and the task's reproduction is an explicit kill.
    Without a handler, ``SIGTERM`` runs the ``SIG_DFL`` immediate terminate,
    which fires *no* ``atexit`` hooks, so every live ``ProcessPoolExecutor``'s
    workers would orphan (re-parent to launchd / PID 1). This handler closes
    that gap: it drains the live-executor registry, chains to any handler that
    was already installed (so we do not silently clobber e.g.
    ``multiprocessing_support``'s, if it loaded first), then restores the
    default disposition and re-raises ``SIGTERM`` so the process still exits
    with the conventional ``128 + SIGTERM`` status rather than swallowing the
    signal.

    Only installed when the current ``SIGTERM`` disposition is the default or a
    plain previously-registered handler — never inside a worker (where the main
    interpreter's signal machinery is not in play) and never if signals are
    unavailable on this platform.
    """
    try:
        previous = signal.getsignal(signal.SIGTERM)
    except (ValueError, OSError):  # pragma: no cover - non-main-thread / no signals
        return

    # Idempotent: if our reaper is already the SIGTERM handler (e.g. a module
    # reload re-ran this), do nothing rather than chain a copy onto itself.
    if getattr(previous, "_giflab_sigterm_reaper", False):
        return

    def _handler(signum: int, frame: Any) -> None:
        try:
            _reap_live_process_executors()
        finally:
            # Chain to a real previously-installed handler if there was one,
            # then fall through to the default terminate. ``SIG_IGN`` /
            # ``SIG_DFL`` are sentinels, not callables — skip them.
            if callable(previous):
                previous(signum, frame)
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
            os.kill(os.getpid(), signum)

    _handler._giflab_sigterm_reaper = True  # type: ignore[attr-defined]
    try:
        signal.signal(signal.SIGTERM, _handler)
    except (ValueError, OSError):  # pragma: no cover - not on the main thread
        # ``signal.signal`` only works on the main thread of the main
        # interpreter; if this module is first imported off-thread we simply
        # rely on the atexit hook for the recoverable cases.
        pass


_install_sigterm_reaper()


@dataclass
class ParallelConfig:
    """Configuration for parallel processing."""

    max_workers: int | None = None
    chunk_strategy: str = "adaptive"  # "adaptive", "fixed", "dynamic"
    min_chunk_size: int = 1
    max_chunk_size: int = 50
    use_process_pool: bool = (
        # Default to threads. The frame metrics are dominated by cv2/numpy ops
        # that release the GIL, so a thread pool parallelises them without the
        # ProcessPool's fragility: a worker that dies abruptly (a native
        # segfault/OOM on some GIF content) breaks the pool, and CPython's
        # broken-pool teardown then DEADLOCKS both the metrics call and
        # interpreter exit (see _process_with_pool). A thread pool cannot break
        # that way. Opt back in with GIFLAB_USE_PROCESS_POOL=true if you have
        # benchmarked a real gain for your workload.
        False
    )
    enable_profiling: bool = False

    def __post_init__(self) -> None:
        """Initialize configuration from environment variables."""
        # Get max workers from environment or use CPU count
        if self.max_workers is None:
            env_workers = os.environ.get("GIFLAB_MAX_PARALLEL_WORKERS")
            if env_workers:
                try:
                    self.max_workers = int(env_workers)
                except ValueError:
                    logger.warning(
                        f"Invalid GIFLAB_MAX_PARALLEL_WORKERS: {env_workers}"
                    )
                    self.max_workers = mp.cpu_count()
            else:
                self.max_workers = mp.cpu_count()

        # Ensure reasonable bounds
        self.max_workers = max(1, min(self.max_workers, mp.cpu_count() * 2))

        # Get other settings from environment
        self.chunk_strategy = os.environ.get(
            "GIFLAB_CHUNK_STRATEGY", self.chunk_strategy
        )
        self.enable_profiling = (
            os.environ.get("GIFLAB_ENABLE_PROFILING", "false").lower() == "true"
        )
        # Explicit opt-in to the process pool (default is the safer thread
        # pool). Only an explicitly-set env var overrides a value passed to the
        # constructor.
        env_pool = os.environ.get("GIFLAB_USE_PROCESS_POOL")
        if env_pool is not None:
            self.use_process_pool = env_pool.lower() == "true"


class ParallelMetricsCalculator:
    """Calculates frame-level metrics in parallel for improved performance."""

    def __init__(self, config: ParallelConfig | None = None):
        """Initialize the parallel calculator.

        Args:
            config: Parallel processing configuration
        """
        self.config = config or ParallelConfig()
        self._profiling_data: dict[str, Any] = {}

    def calculate_frame_metrics_parallel(
        self,
        aligned_pairs: list[tuple[np.ndarray, np.ndarray]],
        metric_functions: dict[str, Callable | None],
        config: Any = None,
    ) -> dict[str, list[float]]:
        """Calculate frame-level metrics in parallel.

        Args:
            aligned_pairs: List of (original, compressed) frame pairs
            metric_functions: Dictionary mapping metric names to calculation functions
            config: Optional metrics configuration object

        Returns:
            Dictionary mapping metric names to lists of per-frame values
        """
        if not aligned_pairs:
            return {name: [] for name in metric_functions}

        start_time = time.perf_counter()

        # Determine chunk size based on strategy
        chunk_size = self._determine_chunk_size(len(aligned_pairs))

        # Split work into chunks
        chunks = self._create_chunks(aligned_pairs, chunk_size)

        if self.config.enable_profiling:
            logger.info(
                f"Parallel processing: {len(aligned_pairs)} pairs, "
                f"{len(chunks)} chunks, {self.config.max_workers} workers"
            )

        # Process chunks in parallel
        if self.config.use_process_pool:
            results = self._process_with_pool(chunks, metric_functions, config)
        else:
            results = self._process_with_threads(chunks, metric_functions, config)

        # Aggregate results maintaining order
        aggregated = self._aggregate_results(results, list(metric_functions.keys()))

        if self.config.enable_profiling:
            elapsed = time.perf_counter() - start_time
            self._profiling_data["parallel_time"] = elapsed
            self._profiling_data["frames_processed"] = len(aligned_pairs)
            self._profiling_data["chunks_created"] = len(chunks)
            logger.info(f"Parallel processing completed in {elapsed:.3f}s")

        return aggregated

    def _determine_chunk_size(self, total_items: int) -> int:
        """Determine optimal chunk size based on strategy.

        Args:
            total_items: Total number of items to process

        Returns:
            Optimal chunk size
        """
        if self.config.chunk_strategy == "fixed":
            # Fixed chunk size
            return self.config.min_chunk_size

        elif self.config.chunk_strategy == "dynamic":
            # Dynamic based on worker count
            max_workers = self.config.max_workers or mp.cpu_count()
            chunk_size = max(1, total_items // (max_workers * 4))
            return max(
                self.config.min_chunk_size, min(chunk_size, self.config.max_chunk_size)
            )

        else:  # adaptive (default)
            # Adaptive based on total items and workers
            max_workers = self.config.max_workers or mp.cpu_count()
            if total_items <= max_workers:
                # Few items: one per worker
                return 1
            elif total_items <= max_workers * 10:
                # Moderate items: small chunks for better load balancing
                return max(1, total_items // (max_workers * 3))
            else:
                # Many items: larger chunks to reduce overhead
                chunk_size = total_items // (max_workers * 2)
                return max(
                    self.config.min_chunk_size,
                    min(chunk_size, self.config.max_chunk_size),
                )

    def _create_chunks(
        self, aligned_pairs: list[tuple[np.ndarray, np.ndarray]], chunk_size: int
    ) -> list[list[tuple[int, tuple[np.ndarray, np.ndarray]]]]:
        """Create chunks of frame pairs with indices preserved.

        Args:
            aligned_pairs: List of frame pairs
            chunk_size: Size of each chunk

        Returns:
            List of chunks, each containing (index, pair) tuples
        """
        chunks = []
        for i in range(0, len(aligned_pairs), chunk_size):
            chunk = [
                (idx, pair)
                for idx, pair in enumerate(aligned_pairs[i : i + chunk_size], start=i)
            ]
            chunks.append(chunk)
        return chunks

    def _process_with_pool(
        self,
        chunks: list,
        metric_functions: dict[str, Callable | None],
        config: Any,
    ) -> list[dict]:
        """Process chunks using ProcessPoolExecutor.

        Args:
            chunks: List of chunks to process
            metric_functions: Metric calculation functions
            config: Metrics configuration

        Returns:
            List of results from each chunk
        """
        # Create partial function for processing
        process_func = partial(
            _process_chunk_worker, metric_functions=metric_functions, config=config
        )

        # We deliberately do NOT use ``with ProcessPoolExecutor(...) as executor``.
        # If a worker dies abruptly (segfault / OOM / killed) the pool enters a
        # broken state, and the context manager's ``__exit__`` calls
        # ``shutdown(wait=True)``, which DEADLOCKS joining the dead worker and
        # queue-feeder threads (CPython ``_wait_for_tstate_lock`` hang). That
        # turns a recoverable error into an infinite hang and starves the
        # sequential fallback in ``calculate_comprehensive_metrics_from_frames``
        # (it never gets to run because this call never returns). Managing the
        # executor lifecycle by hand lets us tear a broken pool down with
        # ``wait=False`` and re-raise so the caller can fall back cleanly.
        executor = ProcessPoolExecutor(max_workers=self.config.max_workers)
        # Register the live executor so the out-of-band reapers (the ``atexit``
        # hook for a normal interpreter exit, and the ``SIGTERM`` handler for a
        # plain ``kill``) can terminate its workers if the parent dies before
        # the ``finally`` below runs. The ``finally`` itself handles the two
        # in-band paths — a clean run and any exception, including the
        # ``KeyboardInterrupt`` Ctrl-C raises — and unregisters the executor.
        with _LIVE_EXECUTORS_LOCK:
            _LIVE_PROCESS_EXECUTORS.add(executor)
        broken = False
        try:
            # Submit all chunks
            futures = {
                executor.submit(process_func, chunk): i
                for i, chunk in enumerate(chunks)
            }

            # Collect results maintaining order
            results: list[dict | None] = [None] * len(chunks)
            for future in as_completed(futures):
                chunk_idx = futures[future]
                try:
                    results[chunk_idx] = future.result()
                except BrokenProcessPool:
                    # The pool is dead — no pending future can succeed, and
                    # swallowing this would emit empty per-chunk dicts that
                    # silently corrupt every downstream aggregate. Abort to the
                    # caller for an honest sequential recompute.
                    raise
                except Exception as e:
                    logger.error(f"Chunk {chunk_idx} processing failed: {e}")
                    # Return empty results for failed chunk
                    results[chunk_idx] = {}
        except BrokenProcessPool:
            broken = True
            logger.warning(
                "Process pool broke (a worker terminated abruptly); tearing it "
                "down without waiting and falling back to sequential metrics."
            )
            raise
        except BaseException:
            # Any other abnormal exit — KeyboardInterrupt, MemoryError,
            # SystemExit, or a non-BrokenProcessPool error from as_completed —
            # used to skip both the BrokenProcessPool branch and the clean
            # ``else`` shutdown, orphaning every spawned worker. Treat it like a
            # broken pool: tear down without waiting (a wait=True join could
            # itself deadlock) and re-raise.
            broken = True
            raise
        finally:
            # Single teardown point reached on EVERY exit path. A clean run
            # joins workers (wait=True). Any abnormal exit force-terminates the
            # workers by PID FIRST (snapshotting before shutdown nulls
            # ``_processes``), then shuts the executor down with wait=False to
            # dodge the broken-pool join deadlock.
            if broken:
                _terminate_pool_workers(executor)
                executor.shutdown(wait=False, cancel_futures=True)
            else:
                executor.shutdown(wait=True)
            with _LIVE_EXECUTORS_LOCK:
                _LIVE_PROCESS_EXECUTORS.discard(executor)

        return [r for r in results if r is not None]

    def _process_with_threads(
        self,
        chunks: list,
        metric_functions: dict[str, Callable | None],
        config: Any,
    ) -> list[dict]:
        """Process chunks using ThreadPoolExecutor (for I/O-bound operations).

        Args:
            chunks: List of chunks to process
            metric_functions: Metric calculation functions
            config: Metrics configuration

        Returns:
            List of results from each chunk
        """
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all chunks
            futures = {
                executor.submit(
                    _process_chunk_worker, chunk, metric_functions, config
                ): i
                for i, chunk in enumerate(chunks)
            }

            # Collect results maintaining order
            results: list[dict | None] = [None] * len(chunks)
            for future in as_completed(futures):
                chunk_idx = futures[future]
                try:
                    results[chunk_idx] = future.result()
                except Exception as e:
                    logger.error(f"Chunk {chunk_idx} processing failed: {e}")
                    results[chunk_idx] = {}

        return [r for r in results if r is not None]

    def _aggregate_results(
        self, chunk_results: list[dict], metric_names: list[str]
    ) -> dict[str, list[float]]:
        """Aggregate chunk results into final metric values.

        Args:
            chunk_results: List of results from each chunk
            metric_names: Names of metrics to aggregate

        Returns:
            Dictionary mapping metric names to ordered lists of values
        """
        # Find the maximum index to determine result size
        max_idx = -1
        for chunk_result in chunk_results:
            for metric_values in chunk_result.values():
                for idx, _ in metric_values:
                    max_idx = max(max_idx, idx)

        # Initialize result arrays with NaN ("not measured"). Any index left
        # unfilled (a frame whose metric produced no value) stays NaN rather
        # than a fabricated 0.0; the NaN-aware aggregator in
        # metrics._aggregate_metric then skips it honestly.
        aggregated = {name: [float("nan")] * (max_idx + 1) for name in metric_names}

        # Place values at their correct indices
        for chunk_result in chunk_results:
            for metric_name, indexed_values in chunk_result.items():
                if metric_name in aggregated:
                    for idx, value in indexed_values:
                        aggregated[metric_name][idx] = value

        return aggregated

    def get_profiling_data(self) -> dict:
        """Get profiling data from the last run.

        Returns:
            Dictionary with profiling metrics
        """
        return self._profiling_data.copy()


def _process_chunk_worker(
    chunk: list[tuple[int, tuple[np.ndarray, np.ndarray]]],
    metric_functions: dict[str, Callable | None],
    config: Any,
) -> dict[str, list[tuple[int, float]]]:
    # NOTE: only the KEYS of metric_functions are used (metric selection).
    # Values may be None -- workers resolve the real callables from their own
    # function_map below because functions must be importable in the
    # subprocess (pickling); the dict values are never invoked.
    """Worker function to process a chunk of frame pairs.

    This function is designed to be pickleable for multiprocessing.

    Args:
        chunk: List of (index, (orig_frame, comp_frame)) tuples
        metric_functions: Dictionary of metric calculation functions
        config: Metrics configuration object

    Returns:
        Dictionary mapping metric names to lists of (index, value) tuples
    """
    import cv2  # Import here for process safety

    from giflab.metrics import (
        calculate_ms_ssim,
        calculate_safe_psnr,
        chist,
        edge_similarity,
        fsim,
        gmsd,
        mse,
        rmse,
        sharpness_similarity,
        ssim,
        texture_similarity,
    )

    # Map function names to actual functions
    function_map: dict[str, Callable[..., Any]] = {
        "ssim": ssim,
        "ms_ssim": calculate_ms_ssim,
        "psnr": calculate_safe_psnr,
        "mse": mse,
        "rmse": rmse,
        "fsim": fsim,
        "gmsd": gmsd,
        "chist": chist,
        "edge_similarity": edge_similarity,
        "texture_similarity": texture_similarity,
        "sharpness_similarity": sharpness_similarity,
    }

    results: dict[str, list[tuple[int, float]]] = {
        name: [] for name in metric_functions
    }

    for idx, (orig_frame, comp_frame) in chunk:
        # Calculate each metric
        for metric_name in metric_functions:
            try:
                if metric_name == "ssim":
                    # SSIM needs grayscale
                    if len(orig_frame.shape) == 3:
                        orig_gray = cv2.cvtColor(orig_frame, cv2.COLOR_RGB2GRAY)
                        comp_gray = cv2.cvtColor(comp_frame, cv2.COLOR_RGB2GRAY)
                    else:
                        orig_gray = orig_frame
                        comp_gray = comp_frame
                    func = function_map.get(metric_name)
                    if func:
                        value = func(orig_gray, comp_gray, data_range=255.0)
                    else:
                        value = 0.0
                    value = max(0.0, min(1.0, value))

                elif metric_name == "psnr":
                    # PSNR needs normalization
                    func = function_map.get(metric_name)
                    if func:
                        value = func(orig_frame, comp_frame)
                        # Store raw value, normalization happens later
                    else:
                        value = 0.0

                elif metric_name == "edge_similarity":
                    # Edge similarity needs Canny thresholds
                    threshold1 = (
                        getattr(config, "EDGE_CANNY_THRESHOLD1", 100) if config else 100
                    )
                    threshold2 = (
                        getattr(config, "EDGE_CANNY_THRESHOLD2", 200) if config else 200
                    )
                    func = function_map.get(metric_name)
                    if func:
                        value = func(orig_frame, comp_frame, threshold1, threshold2)
                    else:
                        value = 0.0

                else:
                    # Standard metric calculation
                    func = function_map.get(metric_name)
                    if func:
                        value = func(orig_frame, comp_frame)
                    else:
                        value = 0.0

                results[metric_name].append((idx, float(value)))

            except Exception as e:
                # Log error and record NaN ("not measured") for this frame.
                # This is the DEFAULT live per-frame path (parallel metrics,
                # aligned_pairs > 1); a fabricated 0.0 here would drag the
                # NaN-aware aggregate in metrics._aggregate_metric toward zero
                # exactly as the sequential loop's old append(0.0) did. NaN
                # lets the surviving frames carry the score honestly.
                logger.warning(f"Metric {metric_name} failed for frame {idx}: {e}")
                results[metric_name].append((idx, float("nan")))

    return results


def create_parallel_calculator(enable: bool = True) -> ParallelMetricsCalculator | None:
    """Factory function to create a parallel calculator.

    Args:
        enable: Whether to enable parallel processing

    Returns:
        ParallelMetricsCalculator instance or None if disabled
    """
    # Check environment variable
    env_enabled = os.environ.get("GIFLAB_ENABLE_PARALLEL_METRICS", "true").lower()
    if not enable or env_enabled == "false":
        return None

    return ParallelMetricsCalculator()
