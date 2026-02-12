"""Parallel processing utilities for GIF metrics calculation.

This module provides parallelization infrastructure for frame-level metrics
to significantly reduce processing time for multi-frame GIFs.
"""

import logging
import multiprocessing as mp
import os
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from functools import partial
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ParallelConfig:
    """Configuration for parallel processing."""

    max_workers: int | None = None
    chunk_strategy: str = "adaptive"  # "adaptive", "fixed", "dynamic"
    min_chunk_size: int = 1
    max_chunk_size: int = 50
    use_process_pool: bool = (
        True  # Use ProcessPool for CPU-bound, ThreadPool for I/O-bound
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
        self, chunks: list, metric_functions: dict[str, Callable], config: Any
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

        with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
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
                except Exception as e:
                    logger.error(f"Chunk {chunk_idx} processing failed: {e}")
                    # Return empty results for failed chunk
                    results[chunk_idx] = {}

        return [r for r in results if r is not None]

    def _process_with_threads(
        self, chunks: list, metric_functions: dict[str, Callable], config: Any
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

        # Initialize result arrays
        aggregated = {name: [0.0] * (max_idx + 1) for name in metric_names}

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
    metric_functions: dict[str, Callable],
    config: Any,
) -> dict[str, list[tuple[int, float]]]:
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
    function_map = {
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
                # Log error and use default value
                logger.warning(f"Metric {metric_name} failed for frame {idx}: {e}")
                results[metric_name].append((idx, 0.0))

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
