"""Unified SQLite storage for GifLab prediction training data.

This module provides the single source of truth for all GifLab results:
- GIF features (extracted once per GIF)
- Compression runs (one per GIF × engine × params combination)
- Compression failures (for debugging)

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
ENGINES = ["gifsicle", "animately"]


class GifLabStorage:
    """SQLite-based storage for GifLab prediction training data.
    
    Schema:
        gif_features: One row per GIF with extracted visual features
        compression_runs: One row per compression attempt
        compression_failures: Failed compressions for debugging
    """

    def __init__(self, db_path: Path):
        """Initialize storage.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self._init_database()

    def _init_database(self) -> None:
        """Initialize database schema."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        with self._connect() as conn:
            # GIF features table (one row per GIF)
            conn.execute("""
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
            """)

            # Compression runs table (one row per compression)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS compression_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    gif_sha TEXT NOT NULL REFERENCES gif_features(gif_sha),
                    engine TEXT NOT NULL,
                    lossy_level INTEGER NOT NULL,
                    color_count INTEGER NOT NULL,
                    frame_ratio REAL NOT NULL DEFAULT 1.0,
                    
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
                    
                    UNIQUE(gif_sha, engine, lossy_level, color_count, frame_ratio)
                )
            """)

            # Compression failures table (for debugging)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS compression_failures (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    gif_sha TEXT NOT NULL,
                    engine TEXT NOT NULL,
                    lossy_level INTEGER NOT NULL,
                    color_count INTEGER NOT NULL,
                    frame_ratio REAL NOT NULL DEFAULT 1.0,
                    
                    error_type TEXT NOT NULL,
                    error_message TEXT NOT NULL,
                    error_traceback TEXT,
                    
                    giflab_version TEXT,
                    created_at TEXT NOT NULL
                )
            """)

            # Indexes for common queries
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_compression_gif 
                ON compression_runs(gif_sha)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_compression_params 
                ON compression_runs(engine, lossy_level, color_count)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_features_incomplete 
                ON gif_features(compression_complete) WHERE compression_complete = FALSE
            """)

            conn.commit()
            self.logger.debug(f"Initialized GifLab storage: {self.db_path}")

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def get_gif_status(self, gif_sha: str) -> dict[str, Any] | None:
        """Get processing status for a GIF.
        
        Returns:
            Dict with 'exists', 'compression_complete', 'feature_version' or None
        """
        with self._connect() as conn:
            row = conn.execute(
                """SELECT compression_complete, feature_extraction_version 
                   FROM gif_features WHERE gif_sha = ?""",
                (gif_sha,)
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
                gif_shas
            ).fetchall()
            
            existing_map = {row["gif_sha"]: bool(row["compression_complete"]) for row in existing}
            
            new_gifs = [sha for sha in gif_shas if sha not in existing_map]
            incomplete = [sha for sha, complete in existing_map.items() if not complete]
            
            return new_gifs, incomplete

    def get_missing_compressions(self, gif_sha: str) -> list[dict[str, Any]]:
        """Get list of compression runs not yet completed for a GIF.
        
        Returns:
            List of dicts with 'engine', 'lossy_level', 'color_count'
        """
        with self._connect() as conn:
            existing = conn.execute(
                """SELECT engine, lossy_level, color_count 
                   FROM compression_runs WHERE gif_sha = ?""",
                (gif_sha,)
            ).fetchall()
            
            existing_set = {
                (row["engine"], row["lossy_level"], row["color_count"]) 
                for row in existing
            }
            
            missing = []
            for engine in ENGINES:
                for lossy in LOSSY_LEVELS:
                    for colors in COLOR_COUNTS:
                        if (engine, lossy, colors) not in existing_set:
                            missing.append({
                                "engine": engine,
                                "lossy_level": lossy,
                                "color_count": colors,
                            })
            
            return missing

    def save_gif_features(self, features: dict[str, Any]) -> None:
        """Save extracted features for a GIF.
        
        Args:
            features: Dict with all feature columns
        """
        with self._connect() as conn:
            conn.execute("""
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
            """, features)
            conn.commit()

    def save_compression_run(self, run: dict[str, Any]) -> None:
        """Save a compression run result.
        
        Args:
            run: Dict with compression parameters and outcomes
        """
        with self._connect() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO compression_runs (
                    gif_sha, engine, lossy_level, color_count, frame_ratio,
                    size_kb, compression_ratio,
                    ssim_mean, ssim_std, ssim_min, ssim_max,
                    ms_ssim_mean, psnr_mean, temporal_consistency,
                    mse_mean, fsim_mean, gmsd_mean, edge_similarity_mean,
                    composite_quality, render_ms,
                    giflab_version, created_at
                ) VALUES (
                    :gif_sha, :engine, :lossy_level, :color_count, :frame_ratio,
                    :size_kb, :compression_ratio,
                    :ssim_mean, :ssim_std, :ssim_min, :ssim_max,
                    :ms_ssim_mean, :psnr_mean, :temporal_consistency,
                    :mse_mean, :fsim_mean, :gmsd_mean, :edge_similarity_mean,
                    :composite_quality, :render_ms,
                    :giflab_version, :created_at
                )
            """, run)
            conn.commit()

    def save_compression_batch(self, runs: list[dict[str, Any]]) -> None:
        """Save multiple compression runs in a single transaction.
        
        Args:
            runs: List of compression run dicts
        """
        if not runs:
            return
            
        with self._connect() as conn:
            conn.executemany("""
                INSERT OR REPLACE INTO compression_runs (
                    gif_sha, engine, lossy_level, color_count, frame_ratio,
                    size_kb, compression_ratio,
                    ssim_mean, ssim_std, ssim_min, ssim_max,
                    ms_ssim_mean, psnr_mean, temporal_consistency,
                    mse_mean, fsim_mean, gmsd_mean, edge_similarity_mean,
                    composite_quality, render_ms,
                    giflab_version, created_at
                ) VALUES (
                    :gif_sha, :engine, :lossy_level, :color_count, :frame_ratio,
                    :size_kb, :compression_ratio,
                    :ssim_mean, :ssim_std, :ssim_min, :ssim_max,
                    :ms_ssim_mean, :psnr_mean, :temporal_consistency,
                    :mse_mean, :fsim_mean, :gmsd_mean, :edge_similarity_mean,
                    :composite_quality, :render_ms,
                    :giflab_version, :created_at
                )
            """, runs)
            conn.commit()

    def mark_gif_complete(self, gif_sha: str) -> None:
        """Mark a GIF as fully processed."""
        with self._connect() as conn:
            conn.execute(
                "UPDATE gif_features SET compression_complete = TRUE WHERE gif_sha = ?",
                (gif_sha,)
            )
            conn.commit()

    def save_compression_failure(self, failure: dict[str, Any]) -> None:
        """Save a compression failure for debugging."""
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO compression_failures (
                    gif_sha, engine, lossy_level, color_count, frame_ratio,
                    error_type, error_message, error_traceback,
                    giflab_version, created_at
                ) VALUES (
                    :gif_sha, :engine, :lossy_level, :color_count, :frame_ratio,
                    :error_type, :error_message, :error_traceback,
                    :giflab_version, :created_at
                )
            """, failure)
            conn.commit()

    def get_training_data(
        self, 
        engine: str | None = None,
        split_ratio: tuple[float, float, float] = (0.8, 0.1, 0.1),
    ) -> dict[str, Any]:
        """Export training data as flat DataFrame-ready format.
        
        Args:
            engine: Filter by engine (None for all)
            split_ratio: Train/val/test split ratios
            
        Returns:
            Dict with 'train', 'val', 'test' DataFrames
        """
        import pandas as pd
        
        with self._connect() as conn:
            query = """
                SELECT f.*, 
                       c.engine, c.lossy_level, c.color_count, c.frame_ratio,
                       c.size_kb, c.compression_ratio,
                       c.ssim_mean, c.ms_ssim_mean, c.psnr_mean, c.composite_quality
                FROM gif_features f
                JOIN compression_runs c ON f.gif_sha = c.gif_sha
                WHERE f.compression_complete = TRUE
            """
            if engine:
                query += f" AND c.engine = '{engine}'"
            
            df = pd.read_sql_query(query, conn)
        
        # Split by GIF (not by row) to avoid data leakage
        gif_shas = df["gif_sha"].unique()
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

    def get_compression_curves(self, gif_sha: str, engine: str) -> dict[str, dict[int, float]]:
        """Get compression curves for a GIF.
        
        Returns:
            Dict with 'lossy' and 'color' curves as {param: size_kb}
        """
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT lossy_level, color_count, size_kb 
                   FROM compression_runs 
                   WHERE gif_sha = ? AND engine = ?""",
                (gif_sha, engine)
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
            total_runs = conn.execute("SELECT COUNT(*) FROM compression_runs").fetchone()[0]
            total_failures = conn.execute("SELECT COUNT(*) FROM compression_failures").fetchone()[0]
            
            return {
                "total_gifs": total_gifs,
                "complete_gifs": complete_gifs,
                "incomplete_gifs": total_gifs - complete_gifs,
                "total_compression_runs": total_runs,
                "total_failures": total_failures,
                "expected_runs_per_gif": len(LOSSY_LEVELS) * len(COLOR_COUNTS) * len(ENGINES),
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
                f"DELETE FROM compression_runs WHERE gif_sha IN ({placeholders})",
                shas
            )
            conn.execute(
                f"DELETE FROM gif_features WHERE gif_sha IN ({placeholders})",
                shas
            )
            conn.commit()
            
            return len(shas)
