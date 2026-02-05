"""Unified prediction pipeline runner.

This module runs the main GifLab pipeline for generating prediction training data.
It extracts features, runs compression sweeps, and stores results in SQLite.

Replaces the CSV-based output and elimination_cache with unified storage.
"""

from __future__ import annotations

import logging
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm import tqdm

from giflab.lossy import LossyEngine, apply_lossy_compression
from giflab.meta import extract_gif_metadata
from giflab.prediction.features import (
    compute_gif_sha,
    extract_features_for_storage,
)
from giflab.storage import (
    COLOR_COUNTS,
    ENGINES,
    FEATURE_EXTRACTION_VERSION,
    LOSSY_LEVELS,
    GifLabStorage,
)

logger = logging.getLogger(__name__)


class PredictionRunner:
    """Runner for generating prediction training data.

    This is the unified pipeline that:
    1. Extracts visual features from GIFs
    2. Runs compression sweeps at prediction-required granularity
    3. Stores results in SQLite for ML training
    """

    def __init__(
        self,
        db_path: Path,
        force: bool = False,
        upgrade: bool = False,
    ):
        """Initialize the prediction runner.

        Args:
            db_path: Path to SQLite database file
            force: Re-process all GIFs, overwriting existing data
            upgrade: Re-process only if feature version is older
        """
        self.storage = GifLabStorage(db_path)
        self.force = force
        self.upgrade = upgrade
        self.logger = logging.getLogger(__name__)

    def run(self, gif_paths: list[Path]) -> dict[str, Any]:
        """Run the prediction pipeline on a list of GIFs.

        Args:
            gif_paths: List of paths to GIF files

        Returns:
            Dict with statistics about the run
        """
        start_time = time.time()

        # Compute SHA for all GIFs
        self.logger.info(f"Computing SHA256 for {len(gif_paths)} GIFs...")
        gif_shas = {}
        for path in tqdm(gif_paths, desc="Hashing"):
            gif_shas[path] = compute_gif_sha(path)

        # Determine which GIFs need processing
        all_shas = list(gif_shas.values())
        new_gifs, incomplete = self.storage.get_pending_gifs(all_shas)

        if self.force:
            to_process = gif_paths
            self.logger.info(f"Force mode: processing all {len(to_process)} GIFs")
        elif self.upgrade:
            # Check version and only process outdated ones
            to_process = []
            for path, sha in gif_shas.items():
                status = self.storage.get_gif_status(sha)
                if status is None:
                    to_process.append(path)
                elif status["feature_version"] != FEATURE_EXTRACTION_VERSION:
                    to_process.append(path)
            self.logger.info(
                f"Upgrade mode: {len(to_process)} GIFs need version upgrade"
            )
        else:
            # Normal mode: process new and incomplete
            sha_to_path = {v: k for k, v in gif_shas.items()}
            to_process = [sha_to_path[sha] for sha in new_gifs]
            to_process += [sha_to_path[sha] for sha in incomplete]
            self.logger.info(
                f"Resume mode: {len(new_gifs)} new, {len(incomplete)} incomplete"
            )

        if not to_process:
            self.logger.info("No GIFs to process")
            return self._get_stats(start_time, 0, 0, 0)

        # Process each GIF
        processed = 0
        failed = 0
        skipped = 0

        for gif_path in tqdm(to_process, desc="Processing"):
            try:
                self._process_gif(gif_path)
                processed += 1
            except Exception as e:
                self.logger.error(f"Failed to process {gif_path}: {e}")
                failed += 1

        return self._get_stats(start_time, processed, failed, skipped)

    def _process_gif(self, gif_path: Path) -> None:
        """Process a single GIF: extract features and run compression sweeps."""
        gif_sha = compute_gif_sha(gif_path)

        # Check what work is needed
        status = self.storage.get_gif_status(gif_sha)

        # Extract features if needed
        if status is None or self.force:
            self.logger.debug(f"Extracting features for {gif_path.name}")
            features = extract_features_for_storage(gif_path)
            self.storage.save_gif_features(features)
        elif self.upgrade and status["feature_version"] != FEATURE_EXTRACTION_VERSION:
            self.logger.debug(f"Upgrading features for {gif_path.name}")
            features = extract_features_for_storage(gif_path)
            self.storage.save_gif_features(features)

        # Get missing compression runs
        if self.force:
            missing = self._all_compression_params()
        else:
            missing = self.storage.get_missing_compressions(gif_sha)

        if not missing:
            self.storage.mark_gif_complete(gif_sha)
            return

        # Get original metadata for compression ratio calculation
        metadata = extract_gif_metadata(gif_path)
        original_size_kb = metadata.orig_kilobytes

        # Run compression sweeps
        compression_runs = []
        for params in missing:
            try:
                result = self._run_compression(
                    gif_path,
                    gif_sha,
                    params["engine"],
                    params["lossy_level"],
                    params["color_count"],
                    original_size_kb,
                )
                compression_runs.append(result)
            except Exception as e:
                self.logger.debug(
                    f"Compression failed: {params['engine']} "
                    f"lossy={params['lossy_level']} colors={params['color_count']}: {e}"
                )
                self.storage.save_compression_failure({
                    "gif_sha": gif_sha,
                    "engine": params["engine"],
                    "lossy_level": params["lossy_level"],
                    "color_count": params["color_count"],
                    "frame_ratio": 1.0,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "error_traceback": None,
                    "giflab_version": "1.0.0",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                })

        # Save successful runs in batch
        if compression_runs:
            self.storage.save_compression_batch(compression_runs)

        # Mark complete if all compressions done
        remaining = self.storage.get_missing_compressions(gif_sha)
        if not remaining:
            self.storage.mark_gif_complete(gif_sha)

    def _run_compression(
        self,
        gif_path: Path,
        gif_sha: str,
        engine: str,
        lossy_level: int,
        color_count: int,
        original_size_kb: float,
    ) -> dict[str, Any]:
        """Run a single compression and return result dict."""
        start_ms = time.time() * 1000

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / f"compressed_{lossy_level}_{color_count}.gif"

            # Map engine string to enum
            lossy_engine = (
                LossyEngine.GIFSICLE if engine == "gifsicle" else LossyEngine.ANIMATELY
            )

            # Run compression
            apply_lossy_compression(
                gif_path,
                output_path,
                lossy_level=lossy_level,
                engine=lossy_engine,
            )

            if not output_path.exists():
                raise RuntimeError("Compression produced no output")

            size_kb = output_path.stat().st_size / 1024
            render_ms = int(time.time() * 1000 - start_ms)

        compression_ratio = original_size_kb / size_kb if size_kb > 0 else 0

        return {
            "gif_sha": gif_sha,
            "engine": engine,
            "lossy_level": lossy_level,
            "color_count": color_count,
            "frame_ratio": 1.0,
            "size_kb": size_kb,
            "compression_ratio": compression_ratio,
            "ssim_mean": None,  # Quality metrics computed separately if needed
            "ssim_std": None,
            "ssim_min": None,
            "ssim_max": None,
            "ms_ssim_mean": None,
            "psnr_mean": None,
            "temporal_consistency": None,
            "mse_mean": None,
            "fsim_mean": None,
            "gmsd_mean": None,
            "edge_similarity_mean": None,
            "composite_quality": None,
            "render_ms": render_ms,
            "giflab_version": "1.0.0",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

    def _all_compression_params(self) -> list[dict[str, Any]]:
        """Get all compression parameter combinations."""
        params = []
        for engine in ENGINES:
            for lossy in LOSSY_LEVELS:
                for colors in COLOR_COUNTS:
                    params.append({
                        "engine": engine,
                        "lossy_level": lossy,
                        "color_count": colors,
                    })
        return params

    def _get_stats(
        self,
        start_time: float,
        processed: int,
        failed: int,
        skipped: int,
    ) -> dict[str, Any]:
        """Get run statistics."""
        elapsed = time.time() - start_time
        db_stats = self.storage.get_statistics()

        return {
            "elapsed_seconds": elapsed,
            "processed": processed,
            "failed": failed,
            "skipped": skipped,
            **db_stats,
        }


def run_prediction_pipeline(
    input_dir: Path,
    output_db: Path,
    force: bool = False,
    upgrade: bool = False,
) -> dict[str, Any]:
    """Run the prediction pipeline on a directory of GIFs.

    Args:
        input_dir: Directory containing GIF files
        output_db: Path to output SQLite database
        force: Re-process all GIFs
        upgrade: Re-process only if feature version is older

    Returns:
        Dict with run statistics
    """
    # Find all GIF files
    gif_paths = list(input_dir.glob("**/*.gif"))
    if not gif_paths:
        logger.warning(f"No GIF files found in {input_dir}")
        return {"error": "No GIF files found"}

    logger.info(f"Found {len(gif_paths)} GIF files in {input_dir}")

    # Run pipeline
    runner = PredictionRunner(output_db, force=force, upgrade=upgrade)
    return runner.run(gif_paths)
