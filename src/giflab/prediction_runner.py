"""Unified prediction pipeline runner.

This module runs the main GifLab pipeline for generating prediction training data.
It extracts features, runs compression sweeps via the tool wrapper system,
and stores results in SQLite using the normalized schema.

Supports two modes:
- Single-engine mode: One pipeline per lossy engine (all 7 engines)
- Full pipeline mode: All tool combinations (frame x color x lossy)

All compression is dispatched through ``capability_registry.get_tool_class_by_name()``
so new engines are automatically supported when registered as tool wrappers.
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
import time
from datetime import UTC, datetime, timezone
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
        """Get pipeline IDs for single-tool pipelines (one per lossy engine).

        Dynamically discovers all available lossy engines from the capability
        registry and pairs each with a compatible color tool (same COMBINE_GROUP).
        """
        from giflab.capability_registry import tools_for

        ids = []
        lossy_tools = tools_for("lossy_compression")
        color_tools = tools_for("color_reduction")

        for lossy_cls in lossy_tools:
            # Find a compatible color tool (same COMBINE_GROUP)
            combine_group = getattr(lossy_cls, "COMBINE_GROUP", None)
            color_name = None
            if combine_group:
                for color_cls in color_tools:
                    if getattr(color_cls, "COMBINE_GROUP", None) == combine_group:
                        color_name = color_cls.NAME
                        break

            pipeline_id = self.storage.get_or_create_pipeline_id(
                frame_tool=None,
                color_tool=color_name,
                lossy_tool=lossy_cls.NAME,
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
            features = extract_features_for_storage(gif_path, precomputed_sha=gif_sha)
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
                        "created_at": datetime.now(UTC).isoformat(),
                    }
                )

        if compression_runs:
            self.storage.save_compression_batch(compression_runs)

        # Mark complete if all compressions done
        remaining = self.storage.get_missing_compressions(gif_sha, self._pipeline_ids)
        if not remaining:
            self.storage.mark_gif_complete(gif_sha)

    def _validate_gif_output(self, output_path: Path) -> None:
        """Validate that a compressed output file is a valid GIF.

        Checks magic bytes (GIF87a or GIF89a) and minimum viable size
        (header + logical screen descriptor = 13 bytes).

        Args:
            output_path: Path to the compressed GIF file.

        Raises:
            RuntimeError: If the file is not a valid GIF.
        """
        size = output_path.stat().st_size
        if size < 13:
            raise RuntimeError(
                f"Output GIF too small ({size} bytes); minimum viable size is 13 bytes"
            )
        with open(output_path, "rb") as f:
            magic = f.read(6)
        if magic not in (b"GIF87a", b"GIF89a"):
            raise RuntimeError(
                f"Output file is not a valid GIF (magic bytes: {magic!r})"
            )

    def _run_compression(
        self,
        gif_path: Path,
        gif_sha: str,
        pipeline_id: int,
        param_preset_id: int,
        original_size_kb: float,
    ) -> dict[str, Any]:
        """Run a single compression with full quality metrics."""
        from giflab.storage import QUALITY_METRIC_COLUMNS

        start_ms = time.time() * 1000

        # Get param values from preset
        with self.storage._connect() as conn:
            row = conn.execute(
                "SELECT lossy_level, color_count, frame_ratio FROM param_presets WHERE id=?",
                (param_preset_id,),
            ).fetchone()
            if row is None:
                raise ValueError(f"Unknown param_preset_id: {param_preset_id}")
            lossy_level = row["lossy_level"]
            color_count = row["color_count"]
            frame_ratio = row["frame_ratio"]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "compressed.gif"

            # Run compression using the tool wrapper system
            self._execute_pipeline(
                gif_path,
                output_path,
                pipeline_id,
                lossy_level,
                color_count,
                frame_ratio,
            )

            if not output_path.exists():
                raise RuntimeError("Compression produced no output")
            self._validate_gif_output(output_path)

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

        # Build result from QUALITY_METRIC_COLUMNS (single source of truth)
        result = {col: metrics.get(col) for col in QUALITY_METRIC_COLUMNS}
        result.update({
            "gif_sha": gif_sha,
            "pipeline_id": pipeline_id,
            "param_preset_id": param_preset_id,
            "size_kb": size_kb,
            "compression_ratio": compression_ratio,
            "render_ms": render_ms,
            "giflab_version": giflab.__version__,
            "created_at": datetime.now(UTC).isoformat(),
        })
        return result

    def _execute_pipeline(
        self,
        input_path: Path,
        output_path: Path,
        pipeline_id: int,
        lossy_level: int,
        color_count: int | None,
        frame_ratio: float = 1.0,
    ) -> None:
        """Execute compression pipeline using tool wrappers.

        Dispatches to the correct tool wrapper classes for each pipeline step
        (frame reduction → color reduction → lossy compression).

        Args:
            input_path: Source GIF file
            output_path: Compressed output file
            pipeline_id: Pipeline ID to look up tool names from DB
            lossy_level: Lossy compression level
            color_count: Color palette size (None to skip)
            frame_ratio: Frame keep ratio (1.0 = keep all)
        """
        from giflab.capability_registry import get_tool_class_by_name

        # Look up pipeline tool names from DB
        with self.storage._connect() as conn:
            prow = conn.execute(
                """SELECT ft.name as frame_tool, ct.name as color_tool,
                          lt.name as lossy_tool
                   FROM pipelines p
                   LEFT JOIN tools ft ON p.frame_tool_id = ft.id
                   LEFT JOIN tools ct ON p.color_tool_id = ct.id
                   LEFT JOIN tools lt ON p.lossy_tool_id = lt.id
                   WHERE p.id = ?""",
                (pipeline_id,),
            ).fetchone()
            if prow is None:
                raise ValueError(f"Unknown pipeline_id: {pipeline_id}")

        frame_tool_name = prow["frame_tool"]
        color_tool_name = prow["color_tool"]
        lossy_tool_name = prow["lossy_tool"]

        # Build ordered steps: frame → color → lossy
        steps: list[tuple[str, dict[str, Any]]] = []

        if frame_tool_name and frame_ratio < 1.0:
            steps.append((frame_tool_name, {"ratio": frame_ratio}))

        if color_tool_name and color_count is not None:
            steps.append((color_tool_name, {"colors": color_count}))

        if lossy_tool_name:
            params: dict[str, Any] = {"lossy_level": lossy_level}
            steps.append((lossy_tool_name, params))

        if not steps:
            raise ValueError(f"Pipeline {pipeline_id} has no executable steps")

        # Execute steps, chaining temp files
        current_input = input_path
        for i, (tool_name, params) in enumerate(steps):
            tool_cls = get_tool_class_by_name(tool_name)
            if tool_cls is None:
                raise ValueError(f"Unknown tool: {tool_name}")

            is_last = i == len(steps) - 1
            step_output = (
                output_path
                if is_last
                else (output_path.parent / f"_step{i}_{output_path.name}")
            )

            tool_instance = tool_cls()
            tool_instance.apply(current_input, step_output, params=params)

            # Clean up intermediate files
            if not is_last:
                current_input = step_output

        # Clean up intermediate temp files
        for i in range(len(steps) - 1):
            tmp = output_path.parent / f"_step{i}_{output_path.name}"
            if tmp.exists():
                tmp.unlink()

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
