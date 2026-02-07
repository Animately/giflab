"""Unified SQLite storage for GifLab prediction training data.

This module provides the single source of truth for all GifLab results:
- GIF features (extracted once per GIF)
- Compression runs (one per GIF × pipeline × params combination)
- Tools and pipelines (normalized lookup tables)

Schema design:
- Normalized: tools, pipelines, param_presets are lookup tables
- compression_runs uses foreign keys for efficiency
- Supports both single-engine and full pipeline chaining modes

Replaces: elimination_cache.py, CSV output
"""

from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator

# Feature extraction version - bump when feature extraction logic changes
FEATURE_EXTRACTION_VERSION = "1.0.0"

# Compression parameters for prediction training
LOSSY_LEVELS = [0, 20, 40, 60, 80, 100, 120]
COLOR_COUNTS = [256, 128, 64, 32, 16]
FRAME_RATIOS = [1.0]  # Can expand later


class GifLabStorage:
    """SQLite-based storage for GifLab prediction training data.

    Normalized schema with lookup tables for tools, pipelines, and param presets.
    Uses foreign keys for efficient storage and powerful queries.
    """

    def __init__(self, db_path: Path):
        """Initialize storage.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self._tool_cache: dict[str, int] = {}
        self._pipeline_cache: dict[str, int] = {}
        self._param_cache: dict[tuple[int, int, float], int] = {}
        self._init_database()

    def _init_database(self) -> None:
        """Initialize database schema with normalized tables."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        with self._connect() as conn:
            # Enable foreign keys
            conn.execute("PRAGMA foreign_keys = ON")

            # Tools lookup table (populated from capability_registry)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tools (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    variable TEXT NOT NULL,
                    version TEXT,
                    available BOOLEAN DEFAULT TRUE,
                    UNIQUE(name, variable)
                )
            """
            )

            # Pipelines lookup table (combinations of tools)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS pipelines (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL UNIQUE,
                    frame_tool_id INTEGER REFERENCES tools(id),
                    color_tool_id INTEGER REFERENCES tools(id),
                    lossy_tool_id INTEGER REFERENCES tools(id)
                )
            """
            )

            # Parameter presets lookup table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS param_presets (
                    id INTEGER PRIMARY KEY,
                    lossy_level INTEGER NOT NULL,
                    color_count INTEGER NOT NULL,
                    frame_ratio REAL NOT NULL DEFAULT 1.0,
                    UNIQUE(lossy_level, color_count, frame_ratio)
                )
            """
            )

            # GIF features table (one row per GIF)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS gif_features (
                    gif_sha TEXT PRIMARY KEY,
                    gif_name TEXT NOT NULL,
                    file_path TEXT,

                    -- Metadata
                    width INTEGER NOT NULL,
                    height INTEGER NOT NULL,
                    frame_count INTEGER NOT NULL,
                    duration_ms INTEGER NOT NULL,
                    file_size_bytes INTEGER NOT NULL,
                    unique_colors INTEGER NOT NULL,

                    -- Spatial features
                    entropy REAL NOT NULL,
                    edge_density REAL NOT NULL,
                    color_complexity REAL NOT NULL,
                    gradient_smoothness REAL NOT NULL,
                    contrast_score REAL NOT NULL,
                    text_density REAL NOT NULL,
                    dct_energy_ratio REAL NOT NULL,
                    color_histogram_entropy REAL NOT NULL,
                    dominant_color_ratio REAL NOT NULL,

                    -- Temporal features
                    motion_intensity REAL NOT NULL,
                    motion_smoothness REAL NOT NULL,
                    static_region_ratio REAL NOT NULL,
                    temporal_entropy REAL NOT NULL,
                    frame_similarity REAL NOT NULL,
                    inter_frame_mse_mean REAL NOT NULL,
                    inter_frame_mse_std REAL NOT NULL,

                    -- Compressibility
                    lossless_compression_ratio REAL NOT NULL,
                    transparency_ratio REAL NOT NULL,

                    -- CLIP content classification scores
                    clip_screen_capture REAL,
                    clip_vector_art REAL,
                    clip_photography REAL,
                    clip_hand_drawn REAL,
                    clip_3d_rendered REAL,
                    clip_pixel_art REAL,

                    -- Versioning and status
                    feature_extraction_version TEXT NOT NULL,
                    extracted_at TEXT NOT NULL,
                    compression_complete BOOLEAN DEFAULT FALSE
                )
            """
            )

            # Compression runs table (normalized with foreign keys)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS compression_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    gif_sha TEXT NOT NULL REFERENCES gif_features(gif_sha),
                    pipeline_id INTEGER NOT NULL REFERENCES pipelines(id),
                    param_preset_id INTEGER NOT NULL REFERENCES param_presets(id),

                    -- Outcomes
                    size_kb REAL NOT NULL,
                    compression_ratio REAL,

                    -- Quality metrics
                    ssim_mean REAL,
                    ssim_std REAL,
                    ssim_min REAL,
                    ssim_max REAL,
                    ms_ssim_mean REAL,
                    psnr_mean REAL,
                    temporal_consistency REAL,
                    mse_mean REAL,
                    fsim_mean REAL,
                    gmsd_mean REAL,
                    edge_similarity_mean REAL,
                    composite_quality REAL,

                    -- Performance
                    render_ms INTEGER,

                    -- Versioning
                    giflab_version TEXT,
                    created_at TEXT NOT NULL,

                    UNIQUE(gif_sha, pipeline_id, param_preset_id)
                )
            """
            )

            # Compression failures table (for debugging)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS compression_failures (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    gif_sha TEXT NOT NULL,
                    pipeline_id INTEGER REFERENCES pipelines(id),
                    param_preset_id INTEGER REFERENCES param_presets(id),

                    error_type TEXT NOT NULL,
                    error_message TEXT NOT NULL,
                    error_traceback TEXT,

                    giflab_version TEXT,
                    created_at TEXT NOT NULL
                )
            """
            )

            # Indexes for common queries
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_compression_gif
                ON compression_runs(gif_sha)
            """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_compression_pipeline
                ON compression_runs(pipeline_id)
            """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_compression_params
                ON compression_runs(param_preset_id)
            """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_features_incomplete
                ON gif_features(compression_complete) WHERE compression_complete = FALSE
            """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_tools_variable
                ON tools(variable)
            """
            )

            conn.commit()
            self.logger.debug(f"Initialized GifLab storage: {self.db_path}")

            # Populate param presets
            self._populate_param_presets(conn)

    def _populate_param_presets(self, conn: sqlite3.Connection) -> None:
        """Populate the param_presets table with standard grid."""
        for lossy in LOSSY_LEVELS:
            for colors in COLOR_COUNTS:
                for frame_ratio in FRAME_RATIOS:
                    conn.execute(
                        """INSERT OR IGNORE INTO param_presets
                           (lossy_level, color_count, frame_ratio)
                           VALUES (?, ?, ?)""",
                        (lossy, colors, frame_ratio),
                    )
        conn.commit()
        self._refresh_param_cache(conn)

    def _refresh_param_cache(self, conn: sqlite3.Connection) -> None:
        """Refresh the in-memory param preset cache."""
        rows = conn.execute(
            "SELECT id, lossy_level, color_count, frame_ratio FROM param_presets"
        ).fetchall()
        self._param_cache = {
            (r["lossy_level"], r["color_count"], r["frame_ratio"]): r["id"]
            for r in rows
        }

    def _refresh_tool_cache(self, conn: sqlite3.Connection) -> None:
        """Refresh the in-memory tool cache."""
        rows = conn.execute("SELECT id, name, variable FROM tools").fetchall()
        self._tool_cache = {f"{r['name']}_{r['variable']}": r["id"] for r in rows}

    def _refresh_pipeline_cache(self, conn: sqlite3.Connection) -> None:
        """Refresh the in-memory pipeline cache."""
        rows = conn.execute("SELECT id, name FROM pipelines").fetchall()
        self._pipeline_cache = {r["name"]: r["id"] for r in rows}

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
        finally:
            conn.close()

    # -------------------------------------------------------------------------
    # Tool and Pipeline Management
    # -------------------------------------------------------------------------

    def register_tool(
        self,
        name: str,
        variable: str,
        version: str | None = None,
        available: bool = True,
    ) -> int:
        """Register a tool in the database.

        Args:
            name: Tool name (e.g., "gifsicle", "ffmpeg")
            variable: Variable type ("frame_reduction", "color_reduction", "lossy_compression")
            version: Optional version string
            available: Whether tool is available on this system

        Returns:
            Tool ID
        """
        cache_key = f"{name}_{variable}"
        if cache_key in self._tool_cache:
            return self._tool_cache[cache_key]

        with self._connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO tools (name, variable, version, available)
                   VALUES (?, ?, ?, ?)""",
                (name, variable, version, available),
            )
            conn.commit()
            self._refresh_tool_cache(conn)
            return self._tool_cache[cache_key]

    def register_pipeline(
        self,
        name: str,
        frame_tool: str | None,
        color_tool: str | None,
        lossy_tool: str | None,
    ) -> int:
        """Register a pipeline in the database.

        Args:
            name: Pipeline identifier (e.g., "gifsicle-color__gifsicle-lossy__noop-frame")
            frame_tool: Frame reduction tool name (or None/noop)
            color_tool: Color reduction tool name
            lossy_tool: Lossy compression tool name

        Returns:
            Pipeline ID
        """
        if name in self._pipeline_cache:
            return self._pipeline_cache[name]

        with self._connect() as conn:
            # Get tool IDs
            frame_id = self._get_tool_id(conn, frame_tool, "frame_reduction")
            color_id = self._get_tool_id(conn, color_tool, "color_reduction")
            lossy_id = self._get_tool_id(conn, lossy_tool, "lossy_compression")

            conn.execute(
                """INSERT OR REPLACE INTO pipelines
                   (name, frame_tool_id, color_tool_id, lossy_tool_id)
                   VALUES (?, ?, ?, ?)""",
                (name, frame_id, color_id, lossy_id),
            )
            conn.commit()
            self._refresh_pipeline_cache(conn)
            return self._pipeline_cache[name]

    def _get_tool_id(
        self,
        conn: sqlite3.Connection,
        tool_name: str | None,
        variable: str,
    ) -> int | None:
        """Get tool ID, registering if needed."""
        if not tool_name or tool_name == "noop":
            return None

        cache_key = f"{tool_name}_{variable}"
        if cache_key in self._tool_cache:
            return self._tool_cache[cache_key]

        # Register the tool
        conn.execute(
            """INSERT OR IGNORE INTO tools (name, variable, available)
               VALUES (?, ?, TRUE)""",
            (tool_name, variable),
        )
        conn.commit()
        self._refresh_tool_cache(conn)
        return self._tool_cache.get(cache_key)

    def get_or_create_pipeline_id(
        self,
        frame_tool: str | None,
        color_tool: str | None,
        lossy_tool: str,
    ) -> int:
        """Get or create a pipeline ID for the given tool combination.

        Args:
            frame_tool: Frame tool name (None for noop)
            color_tool: Color tool name
            lossy_tool: Lossy tool name

        Returns:
            Pipeline ID
        """
        # Build canonical pipeline name
        frame_part = f"{frame_tool or 'noop'}-frame"
        color_part = f"{color_tool or 'noop'}-color"
        lossy_part = f"{lossy_tool}-lossy"
        name = f"{color_part}__{lossy_part}__{frame_part}"

        return self.register_pipeline(name, frame_tool, color_tool, lossy_tool)

    def get_param_preset_id(
        self,
        lossy_level: int,
        color_count: int,
        frame_ratio: float = 1.0,
    ) -> int:
        """Get param preset ID for the given parameters.

        Args:
            lossy_level: Lossy compression level
            color_count: Color count
            frame_ratio: Frame ratio

        Returns:
            Param preset ID
        """
        key = (lossy_level, color_count, frame_ratio)
        if key in self._param_cache:
            return self._param_cache[key]

        # Insert if not exists
        with self._connect() as conn:
            conn.execute(
                """INSERT OR IGNORE INTO param_presets
                   (lossy_level, color_count, frame_ratio)
                   VALUES (?, ?, ?)""",
                (lossy_level, color_count, frame_ratio),
            )
            conn.commit()
            self._refresh_param_cache(conn)
            return self._param_cache[key]

    def populate_tools_from_registry(self) -> int:
        """Populate tools table from capability_registry.

        Returns:
            Number of tools registered
        """
        try:
            from .capability_registry import tools_for
        except ImportError:
            self.logger.warning("Could not import capability_registry")
            return 0

        count = 0
        for variable in ["frame_reduction", "color_reduction", "lossy_compression"]:
            try:
                tools = tools_for(variable)
                for tool_cls in tools:
                    name = getattr(tool_cls, "NAME", tool_cls.__name__).split("-")[0]
                    version = (
                        tool_cls.version() if hasattr(tool_cls, "version") else None
                    )
                    self.register_tool(name, variable, version, tool_cls.available())
                    count += 1
            except Exception as e:
                self.logger.warning(f"Error registering tools for {variable}: {e}")

        return count

    def populate_pipelines_from_registry(self) -> int:
        """Populate pipelines table from dynamic_pipeline.

        Returns:
            Number of pipelines registered
        """
        try:
            from .dynamic_pipeline import generate_all_pipelines
        except ImportError:
            self.logger.warning("Could not import dynamic_pipeline")
            return 0

        count = 0
        for pipeline in generate_all_pipelines():
            try:
                # Extract tool names from pipeline steps
                frame_tool = None
                color_tool = None
                lossy_tool = None

                for step in pipeline.steps:
                    tool_name = getattr(step.tool_cls, "NAME", "").split("-")[0]
                    if step.variable == "frame_reduction":
                        frame_tool = tool_name if tool_name else None
                    elif step.variable == "color_reduction":
                        color_tool = tool_name
                    elif step.variable == "lossy_compression":
                        lossy_tool = tool_name

                self.register_pipeline(
                    pipeline.identifier(),
                    frame_tool,
                    color_tool,
                    lossy_tool,
                )
                count += 1
            except Exception as e:
                self.logger.warning(f"Error registering pipeline: {e}")

        return count

    # -------------------------------------------------------------------------
    # GIF Status and Queries
    # -------------------------------------------------------------------------

    def get_gif_status(self, gif_sha: str) -> dict[str, Any] | None:
        """Get processing status for a GIF.

        Returns:
            Dict with 'exists', 'compression_complete', 'feature_version' or None
        """
        with self._connect() as conn:
            row = conn.execute(
                """SELECT compression_complete, feature_extraction_version 
                   FROM gif_features WHERE gif_sha = ?""",
                (gif_sha,),
            ).fetchone()

            if row:
                return {
                    "exists": True,
                    "compression_complete": bool(row["compression_complete"]),
                    "feature_version": row["feature_extraction_version"],
                }
            return None

    def get_pending_gifs(self, gif_shas: list[str]) -> tuple[list[str], list[str]]:
        """Find which GIFs need processing.

        Args:
            gif_shas: List of GIF SHA256 hashes to check

        Returns:
            Tuple of (new_gifs, incomplete_gifs)
        """
        with self._connect() as conn:
            # Get all existing GIFs
            placeholders = ",".join("?" * len(gif_shas))
            existing = conn.execute(
                f"""SELECT gif_sha, compression_complete 
                    FROM gif_features 
                    WHERE gif_sha IN ({placeholders})""",
                gif_shas,
            ).fetchall()

            existing_map = {
                row["gif_sha"]: bool(row["compression_complete"]) for row in existing
            }

            new_gifs = [sha for sha in gif_shas if sha not in existing_map]
            incomplete = [sha for sha, complete in existing_map.items() if not complete]

            return new_gifs, incomplete

    def get_missing_compressions(
        self,
        gif_sha: str,
        pipeline_ids: list[int] | None = None,
    ) -> list[dict[str, Any]]:
        """Get list of compression runs not yet completed for a GIF.

        Args:
            gif_sha: GIF SHA256 hash
            pipeline_ids: Optional list of pipeline IDs to check (None = all)

        Returns:
            List of dicts with 'pipeline_id', 'param_preset_id'
        """
        with self._connect() as conn:
            # Get existing runs
            existing = conn.execute(
                """SELECT pipeline_id, param_preset_id
                   FROM compression_runs WHERE gif_sha = ?""",
                (gif_sha,),
            ).fetchall()

            existing_set = {
                (row["pipeline_id"], row["param_preset_id"]) for row in existing
            }

            # Get all pipelines if not specified
            if pipeline_ids is None:
                rows = conn.execute("SELECT id FROM pipelines").fetchall()
                pipeline_ids = [r["id"] for r in rows]

            # Get all param presets
            presets = conn.execute("SELECT id FROM param_presets").fetchall()
            preset_ids = [r["id"] for r in presets]

            missing = []
            for pipeline_id in pipeline_ids:
                for preset_id in preset_ids:
                    if (pipeline_id, preset_id) not in existing_set:
                        missing.append(
                            {
                                "pipeline_id": pipeline_id,
                                "param_preset_id": preset_id,
                            }
                        )

            return missing

    def save_gif_features(self, features: dict[str, Any]) -> None:
        """Save extracted features for a GIF.

        Args:
            features: Dict with all feature columns
        """
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO gif_features (
                    gif_sha, gif_name, file_path,
                    width, height, frame_count, duration_ms, file_size_bytes, unique_colors,
                    entropy, edge_density, color_complexity, gradient_smoothness,
                    contrast_score, text_density, dct_energy_ratio,
                    color_histogram_entropy, dominant_color_ratio,
                    motion_intensity, motion_smoothness, static_region_ratio,
                    temporal_entropy, frame_similarity,
                    inter_frame_mse_mean, inter_frame_mse_std,
                    lossless_compression_ratio, transparency_ratio,
                    clip_screen_capture, clip_vector_art, clip_photography,
                    clip_hand_drawn, clip_3d_rendered, clip_pixel_art,
                    feature_extraction_version, extracted_at, compression_complete
                ) VALUES (
                    :gif_sha, :gif_name, :file_path,
                    :width, :height, :frame_count, :duration_ms, :file_size_bytes, :unique_colors,
                    :entropy, :edge_density, :color_complexity, :gradient_smoothness,
                    :contrast_score, :text_density, :dct_energy_ratio,
                    :color_histogram_entropy, :dominant_color_ratio,
                    :motion_intensity, :motion_smoothness, :static_region_ratio,
                    :temporal_entropy, :frame_similarity,
                    :inter_frame_mse_mean, :inter_frame_mse_std,
                    :lossless_compression_ratio, :transparency_ratio,
                    :clip_screen_capture, :clip_vector_art, :clip_photography,
                    :clip_hand_drawn, :clip_3d_rendered, :clip_pixel_art,
                    :feature_extraction_version, :extracted_at, FALSE
                )
            """,
                features,
            )
            conn.commit()

    def save_compression_run(self, run: dict[str, Any]) -> None:
        """Save a compression run result using normalized schema.

        Args:
            run: Dict with 'gif_sha', 'pipeline_id', 'param_preset_id',
                 'size_kb', quality metrics, etc.
        """
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO compression_runs (
                    gif_sha, pipeline_id, param_preset_id,
                    size_kb, compression_ratio,
                    ssim_mean, ssim_std, ssim_min, ssim_max,
                    ms_ssim_mean, psnr_mean, temporal_consistency,
                    mse_mean, fsim_mean, gmsd_mean, edge_similarity_mean,
                    composite_quality, render_ms,
                    giflab_version, created_at
                ) VALUES (
                    :gif_sha, :pipeline_id, :param_preset_id,
                    :size_kb, :compression_ratio,
                    :ssim_mean, :ssim_std, :ssim_min, :ssim_max,
                    :ms_ssim_mean, :psnr_mean, :temporal_consistency,
                    :mse_mean, :fsim_mean, :gmsd_mean, :edge_similarity_mean,
                    :composite_quality, :render_ms,
                    :giflab_version, :created_at
                )
            """,
                run,
            )
            conn.commit()

    def save_compression_batch(self, runs: list[dict[str, Any]]) -> None:
        """Save multiple compression runs in a single transaction.

        Args:
            runs: List of compression run dicts with normalized keys
        """
        if not runs:
            return

        with self._connect() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO compression_runs (
                    gif_sha, pipeline_id, param_preset_id,
                    size_kb, compression_ratio,
                    ssim_mean, ssim_std, ssim_min, ssim_max,
                    ms_ssim_mean, psnr_mean, temporal_consistency,
                    mse_mean, fsim_mean, gmsd_mean, edge_similarity_mean,
                    composite_quality, render_ms,
                    giflab_version, created_at
                ) VALUES (
                    :gif_sha, :pipeline_id, :param_preset_id,
                    :size_kb, :compression_ratio,
                    :ssim_mean, :ssim_std, :ssim_min, :ssim_max,
                    :ms_ssim_mean, :psnr_mean, :temporal_consistency,
                    :mse_mean, :fsim_mean, :gmsd_mean, :edge_similarity_mean,
                    :composite_quality, :render_ms,
                    :giflab_version, :created_at
                )
            """,
                runs,
            )
            conn.commit()

    def mark_gif_complete(self, gif_sha: str) -> None:
        """Mark a GIF as fully processed."""
        with self._connect() as conn:
            conn.execute(
                "UPDATE gif_features SET compression_complete = TRUE WHERE gif_sha = ?",
                (gif_sha,),
            )
            conn.commit()

    def save_compression_failure(self, failure: dict[str, Any]) -> None:
        """Save a compression failure for debugging."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO compression_failures (
                    gif_sha, pipeline_id, param_preset_id,
                    error_type, error_message, error_traceback,
                    giflab_version, created_at
                ) VALUES (
                    :gif_sha, :pipeline_id, :param_preset_id,
                    :error_type, :error_message, :error_traceback,
                    :giflab_version, :created_at
                )
            """,
                failure,
            )
            conn.commit()

    def get_training_data(
        self,
        pipeline_id: int | None = None,
        lossy_tool: str | None = None,
        split_ratio: tuple[float, float, float] = (0.8, 0.1, 0.1),
    ) -> dict[str, Any]:
        """Export training data as flat DataFrame-ready format.

        Args:
            pipeline_id: Filter by pipeline ID (None for all)
            lossy_tool: Filter by lossy tool name (None for all)
            split_ratio: Train/val/test split ratios

        Returns:
            Dict with 'train', 'val', 'test' DataFrames
        """
        import pandas as pd

        with self._connect() as conn:
            query = """
                SELECT f.*,
                       p.name as pipeline_name,
                       lt.name as lossy_tool,
                       ct.name as color_tool,
                       ft.name as frame_tool,
                       pp.lossy_level, pp.color_count, pp.frame_ratio,
                       c.size_kb, c.compression_ratio,
                       c.ssim_mean, c.ms_ssim_mean, c.psnr_mean,
                       c.composite_quality
                FROM gif_features f
                JOIN compression_runs c ON f.gif_sha = c.gif_sha
                JOIN pipelines p ON c.pipeline_id = p.id
                JOIN param_presets pp ON c.param_preset_id = pp.id
                LEFT JOIN tools lt ON p.lossy_tool_id = lt.id
                LEFT JOIN tools ct ON p.color_tool_id = ct.id
                LEFT JOIN tools ft ON p.frame_tool_id = ft.id
                WHERE f.compression_complete = TRUE
            """
            params: list[int | str] = []
            if pipeline_id:
                query += " AND c.pipeline_id = ?"
                params.append(pipeline_id)
            if lossy_tool:
                query += " AND lt.name = ?"
                params.append(lossy_tool)

            df = pd.read_sql_query(query, conn, params=params or None)

        if df.empty:
            return {"train": df, "val": df, "test": df}

        # Split by GIF (not by row) to avoid data leakage
        # Shuffle with fixed seed for reproducible but unbiased splits
        import random
        gif_shas = list(df["gif_sha"].unique())
        rng = random.Random(42)
        rng.shuffle(gif_shas)
        n_gifs = len(gif_shas)

        train_end = int(n_gifs * split_ratio[0])
        val_end = train_end + int(n_gifs * split_ratio[1])

        train_gifs = set(gif_shas[:train_end])
        val_gifs = set(gif_shas[train_end:val_end])
        test_gifs = set(gif_shas[val_end:])

        return {
            "train": df[df["gif_sha"].isin(train_gifs)],
            "val": df[df["gif_sha"].isin(val_gifs)],
            "test": df[df["gif_sha"].isin(test_gifs)],
        }

    def get_compression_curves(
        self,
        gif_sha: str,
        pipeline_id: int,
    ) -> dict[str, dict[int, float]]:
        """Get compression curves for a GIF and pipeline.

        Returns:
            Dict with 'lossy' and 'color' curves as {param: size_kb}
        """
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT pp.lossy_level, pp.color_count, c.size_kb
                   FROM compression_runs c
                   JOIN param_presets pp ON c.param_preset_id = pp.id
                   WHERE c.gif_sha = ? AND c.pipeline_id = ?""",
                (gif_sha, pipeline_id),
            ).fetchall()

            lossy_curve = {}
            color_curve = {}

            for row in rows:
                # Lossy curve: varying lossy at fixed colors (256)
                if row["color_count"] == 256:
                    lossy_curve[row["lossy_level"]] = row["size_kb"]
                # Color curve: varying colors at fixed lossy (0)
                if row["lossy_level"] == 0:
                    color_curve[row["color_count"]] = row["size_kb"]

            return {"lossy": lossy_curve, "color": color_curve}

    def get_statistics(self) -> dict[str, Any]:
        """Get database statistics."""
        with self._connect() as conn:
            total_gifs = conn.execute("SELECT COUNT(*) FROM gif_features").fetchone()[0]
            complete_gifs = conn.execute(
                "SELECT COUNT(*) FROM gif_features WHERE compression_complete = TRUE"
            ).fetchone()[0]
            total_runs = conn.execute(
                "SELECT COUNT(*) FROM compression_runs"
            ).fetchone()[0]
            total_failures = conn.execute(
                "SELECT COUNT(*) FROM compression_failures"
            ).fetchone()[0]
            total_pipelines = conn.execute("SELECT COUNT(*) FROM pipelines").fetchone()[
                0
            ]
            total_presets = conn.execute(
                "SELECT COUNT(*) FROM param_presets"
            ).fetchone()[0]
            total_tools = conn.execute("SELECT COUNT(*) FROM tools").fetchone()[0]

            expected = total_pipelines * total_presets if total_pipelines else 0

            return {
                "total_gifs": total_gifs,
                "complete_gifs": complete_gifs,
                "incomplete_gifs": total_gifs - complete_gifs,
                "total_compression_runs": total_runs,
                "total_failures": total_failures,
                "total_pipelines": total_pipelines,
                "total_param_presets": total_presets,
                "total_tools": total_tools,
                "expected_runs_per_gif": expected,
            }

    def clear_incomplete(self) -> int:
        """Remove incomplete GIFs and their compression runs.

        Returns:
            Number of GIFs removed
        """
        with self._connect() as conn:
            incomplete = conn.execute(
                "SELECT gif_sha FROM gif_features WHERE compression_complete = FALSE"
            ).fetchall()

            shas = [row["gif_sha"] for row in incomplete]
            if not shas:
                return 0

            placeholders = ",".join("?" * len(shas))
            conn.execute(
                f"DELETE FROM compression_runs WHERE gif_sha IN ({placeholders})", shas
            )
            conn.execute(
                f"DELETE FROM gif_features WHERE gif_sha IN ({placeholders})", shas
            )
            conn.commit()

            return len(shas)
