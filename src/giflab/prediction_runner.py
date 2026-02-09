"""Unified prediction pipeline runner.

This module runs the main GifLab pipeline for generating prediction training data.
It extracts features, runs compression sweeps with pipeline chaining support,
and stores results in SQLite using the normalized schema.

Supports two modes:
- Single-engine mode: Quick testing with just gifsicle/animately
- Full pipeline mode: All tool combinations (frame × color × lossy)

Replaces: CSV output, elimination_cache
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm import tqdm

import giflab
from giflab.meta import extract_gif_metadata
from giflab.metrics import calculate_comprehensive_metrics
from giflab.prediction.features import (
    compute_gif_sha,
    extract_features_for_storage,
)
from giflab.storage import (
    FEATURE_EXTRACTION_VERSION,
    GifLabStorage,
)

logger = logging.getLogger(__name__)


def _get_git_commit() -> str:
    """Get short git commit hash, or 'unknown' if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=Path(__file__).parent,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


class PredictionRunner:
    """Runner for generating prediction training data.

    Supports two modes:
    - single: Quick testing with single-tool pipelines (gifsicle-only, etc.)
    - full: All tool combinations via pipeline chaining

    Pipeline:
    1. Extract visual features from GIFs
    2. Run compression sweeps with full quality metrics
    3. Store results in normalized SQLite schema
    """

    def __init__(
        self,
        db_path: Path,
        mode: str = "single",
        force: bool = False,
        upgrade: bool = False,
    ):
        """Initialize the prediction runner.

        Args:
            db_path: Path to SQLite database file
            mode: "single" for quick runs, "full" for all pipelines
            force: Re-process all GIFs, overwriting existing data
            upgrade: Re-process only if feature version is older
        """
        self.storage = GifLabStorage(db_path)
        self.mode = mode
        self.force = force
        self.upgrade = upgrade
        self.logger = logging.getLogger(__name__)
        self._pipeline_ids: list[int] | None = None

        # Initialize tools and pipelines in database
        self._init_pipelines()

    def _init_pipelines(self) -> None:
        """Initialize tools and pipelines from capability registry."""
        n_tools = self.storage.populate_tools_from_registry()
        n_pipelines = self.storage.populate_pipelines_from_registry()
        self.logger.info(f"Initialized {n_tools} tools, {n_pipelines} pipelines")

        # Cache pipeline IDs for the selected mode
        if self.mode == "single":
            # Single-tool pipelines only (gifsicle-only, animately-only, etc.)
            self._pipeline_ids = self._get_single_tool_pipeline_ids()
        else:
            # All pipelines
            self._pipeline_ids = None  # None means all

    def _get_single_tool_pipeline_ids(self) -> list[int]:
        """Get pipeline IDs for single-tool pipelines."""
        # For now, create simple single-tool pipelines
        ids = []
        for lossy_tool in ["gifsicle", "animately"]:
            pipeline_id = self.storage.get_or_create_pipeline_id(
                frame_tool=None,
                color_tool=lossy_tool,  # Same tool for color
                lossy_tool=lossy_tool,
            )
            ids.append(pipeline_id)
        return ids

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
            self.logger.info(f"Force mode: processing {len(to_process)} GIFs")
        elif self.upgrade:
            to_process = []
            for path, sha in gif_shas.items():
                status = self.storage.get_gif_status(sha)
                if status is None:
                    to_process.append(path)
                elif status["feature_version"] != FEATURE_EXTRACTION_VERSION:
                    to_process.append(path)
            self.logger.info(f"Upgrade mode: {len(to_process)} need upgrade")
        else:
            sha_to_path = {v: k for k, v in gif_shas.items()}
            to_process = [sha_to_path[sha] for sha in new_gifs]
            to_process += [sha_to_path[sha] for sha in incomplete]
            self.logger.info(
                f"Resume: {len(new_gifs)} new, {len(incomplete)} incomplete"
            )

        if not to_process:
            self.logger.info("No GIFs to process")
            return self._get_stats(start_time, 0, 0, 0)

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
        """Process a single GIF: extract features and run compressions."""
        gif_sha = compute_gif_sha(gif_path)
        status = self.storage.get_gif_status(gif_sha)

        # Extract features if needed
        if status is None or self.force:
            self.logger.debug(f"Extracting features for {gif_path.name}")
            features = extract_features_for_storage(
                gif_path, precomputed_sha=gif_sha
            )
            self.storage.save_gif_features(features)
        elif self.upgrade:
            if status["feature_version"] != FEATURE_EXTRACTION_VERSION:
                self.logger.debug(f"Upgrading features for {gif_path.name}")
                features = extract_features_for_storage(
                    gif_path, precomputed_sha=gif_sha
                )
                self.storage.save_gif_features(features)

        # Get missing compression runs
        if self.force:
            missing = self._all_compression_params()
        else:
            missing = self.storage.get_missing_compressions(gif_sha, self._pipeline_ids)

        if not missing:
            self.storage.mark_gif_complete(gif_sha)
            return

        # Get original metadata
        metadata = extract_gif_metadata(gif_path)
        original_size_kb = metadata.orig_kilobytes

        # Run compression sweeps
        compression_runs = []
        for params in missing:
            try:
                result = self._run_compression(
                    gif_path,
                    gif_sha,
                    params["pipeline_id"],
                    params["param_preset_id"],
                    original_size_kb,
                )
                compression_runs.append(result)
            except Exception as e:
                self.logger.debug(f"Compression failed: {e}")
                self.storage.save_compression_failure(
                    {
                        "gif_sha": gif_sha,
                        "pipeline_id": params["pipeline_id"],
                        "param_preset_id": params["param_preset_id"],
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "error_traceback": None,
                        "giflab_version": giflab.__version__,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    }
                )

        if compression_runs:
            self.storage.save_compression_batch(compression_runs)

        # Mark complete if all compressions done
        remaining = self.storage.get_missing_compressions(gif_sha, self._pipeline_ids)
        if not remaining:
            self.storage.mark_gif_complete(gif_sha)

    def _run_compression(
        self,
        gif_path: Path,
        gif_sha: str,
        pipeline_id: int,
        param_preset_id: int,
        original_size_kb: float,
    ) -> dict[str, Any]:
        """Run a single compression with full quality metrics."""
        start_ms = time.time() * 1000

        # Get param values from preset
        with self.storage._connect() as conn:
            row = conn.execute(
                "SELECT lossy_level, color_count FROM param_presets WHERE id=?",
                (param_preset_id,),
            ).fetchone()
            if row is None:
                raise ValueError(f"Unknown param_preset_id: {param_preset_id}")
            lossy_level = row["lossy_level"]
            color_count = row["color_count"]

            # Get pipeline info
            prow = conn.execute(
                """SELECT lt.name as lossy_tool
                   FROM pipelines p
                   LEFT JOIN tools lt ON p.lossy_tool_id = lt.id
                   WHERE p.id = ?""",
                (pipeline_id,),
            ).fetchone()
            if prow is None:
                raise ValueError(f"Unknown pipeline_id: {pipeline_id}")
            lossy_tool = prow["lossy_tool"] or "gifsicle"

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "compressed.gif"

            # Run compression using the tool wrapper system
            self._execute_pipeline(
                gif_path, output_path, lossy_tool, lossy_level, color_count
            )

            if not output_path.exists():
                raise RuntimeError("Compression produced no output")

            size_kb = output_path.stat().st_size / 1024
            render_ms = int(time.time() * 1000 - start_ms)

            # Calculate full quality metrics
            try:
                metrics = calculate_comprehensive_metrics(
                    str(gif_path), str(output_path)
                )
            except Exception as e:
                self.logger.warning(f"Metrics calculation failed: {e}")
                metrics = {}

        compression_ratio = original_size_kb / size_kb if size_kb > 0 else 0

        return {
            "gif_sha": gif_sha,
            "pipeline_id": pipeline_id,
            "param_preset_id": param_preset_id,
            "size_kb": size_kb,
            "compression_ratio": compression_ratio,
            "ssim_mean": metrics.get("ssim_mean"),
            "ssim_std": metrics.get("ssim_std"),
            "ssim_min": metrics.get("ssim_min"),
            "ssim_max": metrics.get("ssim_max"),
            "ms_ssim_mean": metrics.get("ms_ssim_mean"),
            "psnr_mean": metrics.get("psnr_mean"),
            "temporal_consistency": metrics.get("temporal_consistency"),
            "mse_mean": metrics.get("mse_mean"),
            "fsim_mean": metrics.get("fsim_mean"),
            "gmsd_mean": metrics.get("gmsd_mean"),
            "edge_similarity_mean": metrics.get("edge_similarity_mean"),
            "composite_quality": metrics.get("composite_quality"),
            "render_ms": render_ms,
            "giflab_version": giflab.__version__,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

    def _execute_pipeline(
        self,
        input_path: Path,
        output_path: Path,
        lossy_tool: str,
        lossy_level: int,
        color_count: int | None,
    ) -> None:
        """Execute compression pipeline using tool wrappers."""
        from giflab.lossy import LossyEngine, apply_lossy_compression

        # Map tool name to engine enum
        if lossy_tool == "animately":
            engine = LossyEngine.ANIMATELY
        else:
            engine = LossyEngine.GIFSICLE

        apply_lossy_compression(
            input_path,
            output_path,
            lossy_level=lossy_level,
            color_keep_count=color_count,
            engine=engine,
        )

    def _all_compression_params(self) -> list[dict[str, Any]]:
        """Get all compression parameter combinations for selected pipelines."""
        params = []
        pipeline_ids = self._pipeline_ids or []

        # If no pipelines set, get all from database
        if not pipeline_ids:
            with self.storage._connect() as conn:
                rows = conn.execute("SELECT id FROM pipelines").fetchall()
                pipeline_ids = [r["id"] for r in rows]

        # Get all param presets
        with self.storage._connect() as conn:
            presets = conn.execute("SELECT id FROM param_presets").fetchall()
            preset_ids = [r["id"] for r in presets]

        for pipeline_id in pipeline_ids:
            for preset_id in preset_ids:
                params.append(
                    {
                        "pipeline_id": pipeline_id,
                        "param_preset_id": preset_id,
                    }
                )
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
    mode: str = "single",
    force: bool = False,
    upgrade: bool = False,
) -> dict[str, Any]:
    """Run the prediction pipeline on a directory of GIFs.

    Args:
        input_dir: Directory containing GIF files
        output_db: Path to output SQLite database
        mode: "single" for quick runs, "full" for all pipelines
        force: Re-process all GIFs
        upgrade: Re-process only if feature version is older

    Returns:
        Dict with run statistics
    """
    gif_paths = list(input_dir.glob("**/*.gif"))
    if not gif_paths:
        logger.warning(f"No GIF files found in {input_dir}")
        return {"error": "No GIF files found"}

    logger.info(f"Found {len(gif_paths)} GIF files in {input_dir}")

    runner = PredictionRunner(output_db, mode=mode, force=force, upgrade=upgrade)
    return runner.run(gif_paths)
