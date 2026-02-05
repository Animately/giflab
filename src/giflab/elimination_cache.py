"""Pipeline results caching system for elimination framework.

DEPRECATED: This module is superseded by giflab.storage.GifLabStorage.
Use the new prediction_runner.py and storage.py for prediction training.

This module provides SQLite-based caching for pipeline test results
to avoid redundant testing and improve performance.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from .elimination_errors import ErrorTypes


class PipelineResultsCache:
    """SQLite-based cache for pipeline test results to avoid redundant testing."""

    def __init__(self, cache_db_path: Path, git_commit: str | None = None):
        """Initialize the results cache.

        Args:
            cache_db_path: Path to SQLite database file
            git_commit: Current git commit hash for cache invalidation
        """
        self.cache_db_path = cache_db_path
        self.git_commit = git_commit or "unknown"
        self.logger = logging.getLogger(__name__)
        self._pending_results: list[dict[str, Any]] = []  # Batch storage
        self._pending_failures: list[dict[str, Any]] = []  # Batch failure storage
        self._init_database()

    def _init_database(self) -> None:
        """Initialize the SQLite database schema."""
        try:
            with sqlite3.connect(self.cache_db_path) as conn:
                # Results table (successful tests)
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS pipeline_results (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        cache_key TEXT UNIQUE NOT NULL,
                        pipeline_id TEXT NOT NULL,
                        gif_name TEXT NOT NULL,
                        test_colors INTEGER NOT NULL,
                        test_lossy INTEGER NOT NULL,
                        test_frame_ratio REAL NOT NULL,
                        applied_colors INTEGER,
                        applied_lossy INTEGER,
                        applied_frame_ratio REAL,
                        actual_pipeline_steps INTEGER,
                        git_commit TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        result_json TEXT NOT NULL
                    )
                """
                )

                # Failures table (for debugging and analysis)
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS pipeline_failures (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        pipeline_id TEXT NOT NULL,
                        gif_name TEXT NOT NULL,
                        test_colors INTEGER NOT NULL,
                        test_lossy INTEGER NOT NULL,
                        test_frame_ratio REAL NOT NULL,
                        applied_colors INTEGER,
                        applied_lossy INTEGER,
                        applied_frame_ratio REAL,
                        actual_pipeline_steps INTEGER,
                        git_commit TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        error_type TEXT NOT NULL,
                        error_message TEXT NOT NULL,
                        error_traceback TEXT,
                        pipeline_steps TEXT,
                        tools_used TEXT
                    )
                """
                )

                # Create indexes for faster lookups
                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_cache_lookup
                    ON pipeline_results(pipeline_id, gif_name, test_colors, test_lossy, test_frame_ratio)
                """
                )

                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_failure_lookup
                    ON pipeline_failures(pipeline_id, error_type, created_at)
                """
                )

                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_failure_analysis
                    ON pipeline_failures(error_type, git_commit, created_at)
                """
                )

                conn.commit()
                self.logger.debug(f"📁 Initialized cache database: {self.cache_db_path}")

        except Exception as e:
            self.logger.warning(f"Failed to initialize cache database: {e}")

    def _generate_cache_key(self, pipeline_id: str, gif_name: str, params: dict) -> str:
        """Generate a unique cache key for a pipeline test combination."""
        import hashlib

        # Include PNG optimization potential in cache key to differentiate connection methods
        png_potential = self._determine_png_optimization_potential(pipeline_id)

        # Create a deterministic key from all parameters including PNG optimization
        key_data = f"{pipeline_id}|{gif_name}|{params['colors']}|{params['lossy']}|{params.get('frame_ratio', 1.0)}|{png_potential}|{self.git_commit}"
        return hashlib.sha256(key_data.encode()).hexdigest()

    def _determine_png_optimization_potential(self, pipeline_id: str) -> str:
        """Determine if pipeline has potential for PNG sequence optimization.

        This checks the pipeline structure to see if:
        1. The final step supports PNG input (animately-advanced, gifski)
        2. Previous steps support PNG export (imagemagick, ffmpeg, animately tools)

        Returns:
            'png_capable' if PNG optimization is possible, 'gif_only' otherwise
        """
        # Parse pipeline tools from pipeline_id
        parts = pipeline_id.split("__")
        if len(parts) < 2:
            return "gif_only"

        # Check if final step supports PNG input
        final_step = parts[-1].lower()
        supports_png_input = (
            "animately-advanced-lossy" in final_step or "gifski" in final_step
        )

        if not supports_png_input:
            return "gif_only"

        # Check if any preceding step can export PNG sequences
        for part in parts[:-1]:  # All steps except final
            part_lower = part.lower()
            can_export_png = (
                "imagemagick" in part_lower
                or "ffmpeg" in part_lower
                or "animately" in part_lower
            )
            if can_export_png:
                return "png_capable"

        return "gif_only"

    def get_cached_result(
        self, pipeline_id: str, gif_name: str, params: dict
    ) -> dict | None:
        """Retrieve cached result if it exists and is valid.

        Args:
            pipeline_id: Pipeline identifier
            gif_name: Test GIF name
            params: Test parameters dict

        Returns:
            Cached result dict or None if not found/invalid
        """
        cache_key = self._generate_cache_key(pipeline_id, gif_name, params)

        try:
            with sqlite3.connect(self.cache_db_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT result_json, git_commit, created_at
                    FROM pipeline_results
                    WHERE cache_key = ?
                """,
                    (cache_key,),
                )

                row = cursor.fetchone()
                if row:
                    result_json, cached_git_commit, created_at = row

                    # Check if cache is still valid (same git commit)
                    if cached_git_commit == self.git_commit:
                        self.logger.debug(
                            f"💾 Cache hit for {pipeline_id} on {gif_name}"
                        )
                        return cast(dict, json.loads(result_json))
                    else:
                        self.logger.debug(
                            f"💾 Cache invalidated for {pipeline_id} on {gif_name} (git commit changed)"
                        )
                        # Could optionally delete invalidated entries here
                        return None
                else:
                    self.logger.debug(f"💾 Cache miss for {pipeline_id} on {gif_name}")
                    return None

        except Exception as e:
            self.logger.warning(f"Failed to retrieve cached result: {e}")
            return None

    def queue_result(
        self, pipeline_id: str, gif_name: str, params: dict, result: dict
    ) -> None:
        """Queue a successful pipeline test result for batch storage.

        Args:
            pipeline_id: Pipeline identifier
            gif_name: Test GIF name
            params: Test parameters dict
            result: Test result dict to cache
        """
        cache_key = self._generate_cache_key(pipeline_id, gif_name, params)

        self._pending_results.append(
            {
                "cache_key": cache_key,
                "pipeline_id": pipeline_id,
                "gif_name": gif_name,
                "test_colors": params["colors"],
                "test_lossy": params["lossy"],
                "test_frame_ratio": params.get("frame_ratio", 1.0),
                "git_commit": self.git_commit,
                "created_at": datetime.now().isoformat(),
                "result_json": json.dumps(result),
            }
        )

        self.logger.debug(
            f"💾 Queued result for {pipeline_id} on {gif_name} (batch size: {len(self._pending_results)})"
        )

        # Auto-flush when batch size is reached
        self.flush_batch()

    def queue_failure(
        self, pipeline_id: str, gif_name: str, params: dict, error_info: dict
    ) -> None:
        """Queue a pipeline failure for batch storage and analysis.

        Args:
            pipeline_id: Pipeline identifier
            gif_name: Test GIF name
            params: Test parameters dict
            error_info: Error information dict with keys: error, error_traceback, pipeline_steps, tools_used
        """
        error_type = ErrorTypes.categorize_error(error_info.get("error", ""))

        self._pending_failures.append(
            {
                "pipeline_id": pipeline_id,
                "gif_name": gif_name,
                "test_colors": params["colors"],
                "test_lossy": params["lossy"],
                "test_frame_ratio": params.get("frame_ratio", 1.0),
                "git_commit": self.git_commit,
                "created_at": datetime.now().isoformat(),
                "error_type": error_type,
                "error_message": error_info.get("error", "Unknown error"),
                "error_traceback": error_info.get("error_traceback", ""),
                "pipeline_steps": json.dumps(error_info.get("pipeline_steps", [])),
                "tools_used": json.dumps(error_info.get("tools_used", [])),
            }
        )

        self.logger.debug(
            f"💾 Queued failure for {pipeline_id} on {gif_name} (batch size: {len(self._pending_failures)})"
        )

        # Auto-flush when batch size is reached
        self.flush_batch()

    def flush_batch(self, force: bool = False) -> None:
        """Flush pending results and failures to database.

        Args:
            force: If True, flush regardless of batch size
        """
        batch_size = 10  # Configurable batch size

        # Flush results if batch is ready or forced
        if force or len(self._pending_results) >= batch_size:
            self._flush_results_batch()

        # Flush failures if batch is ready or forced
        if force or len(self._pending_failures) >= batch_size:
            self._flush_failures_batch()

    def _flush_results_batch(self) -> None:
        """Flush pending successful results to database."""
        if not self._pending_results:
            return

        try:
            with sqlite3.connect(self.cache_db_path) as conn:
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO pipeline_results
                    (cache_key, pipeline_id, gif_name, test_colors, test_lossy, test_frame_ratio,
                     git_commit, created_at, result_json)
                    VALUES (:cache_key, :pipeline_id, :gif_name, :test_colors, :test_lossy,
                            :test_frame_ratio, :git_commit, :created_at, :result_json)
                """,
                    self._pending_results,
                )

                conn.commit()
                count = len(self._pending_results)
                self.logger.debug(f"💾 Flushed {count} cached results to database")
                self._pending_results.clear()

        except Exception as e:
            self.logger.warning(f"Failed to flush results batch: {e}")

    def _flush_failures_batch(self) -> None:
        """Flush pending failures to database."""
        if not self._pending_failures:
            return

        try:
            with sqlite3.connect(self.cache_db_path) as conn:
                conn.executemany(
                    """
                    INSERT INTO pipeline_failures
                    (pipeline_id, gif_name, test_colors, test_lossy, test_frame_ratio,
                     git_commit, created_at, error_type, error_message, error_traceback,
                     pipeline_steps, tools_used)
                    VALUES (:pipeline_id, :gif_name, :test_colors, :test_lossy, :test_frame_ratio,
                            :git_commit, :created_at, :error_type, :error_message, :error_traceback,
                            :pipeline_steps, :tools_used)
                """,
                    self._pending_failures,
                )

                conn.commit()
                count = len(self._pending_failures)
                self.logger.info(f"🔍 Stored {count} failures for debugging analysis")
                self._pending_failures.clear()

        except Exception as e:
            self.logger.warning(f"Failed to flush failures batch: {e}")

    def store_result(
        self, pipeline_id: str, gif_name: str, params: dict, result: dict
    ) -> None:
        """Store a pipeline test result (legacy method - now uses batching).

        Args:
            pipeline_id: Pipeline identifier
            gif_name: Test GIF name
            params: Test parameters dict
            result: Test result dict to cache
        """
        self.queue_result(pipeline_id, gif_name, params, result)
        self.flush_batch(
            force=True
        )  # Force flush for immediate storage (legacy behavior)

    def clear_cache(self) -> None:
        """Clear all cached results and failures."""
        try:
            with sqlite3.connect(self.cache_db_path) as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM pipeline_results")
                results_count = cursor.fetchone()[0]

                cursor = conn.execute("SELECT COUNT(*) FROM pipeline_failures")
                failures_count = cursor.fetchone()[0]

                conn.execute("DELETE FROM pipeline_results")
                conn.execute("DELETE FROM pipeline_failures")
                conn.commit()

                self.logger.info(
                    f"🗑️ Cleared {results_count} cached results and {failures_count} stored failures"
                )

        except Exception as e:
            self.logger.warning(f"Failed to clear cache: {e}")

    def get_cache_stats(self) -> dict:
        """Get statistics about the current cache."""
        try:
            with sqlite3.connect(self.cache_db_path) as conn:
                # Results statistics
                cursor = conn.execute("SELECT COUNT(*) FROM pipeline_results")
                total_results = cursor.fetchone()[0]

                cursor = conn.execute(
                    """
                    SELECT git_commit, COUNT(*)
                    FROM pipeline_results
                    GROUP BY git_commit
                    ORDER BY COUNT(*) DESC
                """
                )
                results_by_commit = dict(cursor.fetchall())

                cursor = conn.execute(
                    """
                    SELECT COUNT(*) FROM pipeline_results
                    WHERE git_commit = ?
                """,
                    (self.git_commit,),
                )
                current_commit_results = cursor.fetchone()[0]

                # Failures statistics
                cursor = conn.execute("SELECT COUNT(*) FROM pipeline_failures")
                total_failures = cursor.fetchone()[0]

                cursor = conn.execute(
                    """
                    SELECT error_type, COUNT(*)
                    FROM pipeline_failures
                    WHERE git_commit = ?
                    GROUP BY error_type
                    ORDER BY COUNT(*) DESC
                """,
                    (self.git_commit,),
                )
                current_failures_by_type = dict(cursor.fetchall())

                # Database file size
                db_size_bytes = (
                    self.cache_db_path.stat().st_size
                    if self.cache_db_path.exists()
                    else 0
                )
                db_size_mb = db_size_bytes / (1024 * 1024)

                return {
                    "total_results": total_results,
                    "current_commit_results": current_commit_results,
                    "results_by_commit": results_by_commit,
                    "total_failures": total_failures,
                    "current_failures_by_type": current_failures_by_type,
                    "database_size_mb": round(db_size_mb, 2),
                    "current_git_commit": self.git_commit,
                    "pending_batch_size": len(self._pending_results),
                    "pending_failures_size": len(self._pending_failures),
                }

        except Exception as e:
            self.logger.warning(f"Failed to get cache stats: {e}")
            return {
                "total_results": 0,
                "current_commit_results": 0,
                "results_by_commit": {},
                "total_failures": 0,
                "current_failures_by_type": {},
                "database_size_mb": 0,
                "current_git_commit": self.git_commit,
                "pending_batch_size": 0,
                "pending_failures_size": 0,
            }

    def query_failures(
        self,
        error_type: str | None = None,
        pipeline_id: str | None = None,
        recent_hours: int | None = None,
    ) -> list[dict]:
        """Query failures for debugging analysis.

        Args:
            error_type: Filter by specific error type (e.g., 'ffmpeg', 'timeout')
            pipeline_id: Filter by specific pipeline
            recent_hours: Only show failures from last N hours

        Returns:
            List of failure records
        """
        try:
            with sqlite3.connect(self.cache_db_path) as conn:
                # Build dynamic query
                where_conditions = ["git_commit = ?"]
                params = [self.git_commit]

                if error_type:
                    where_conditions.append("error_type = ?")
                    params.append(error_type)

                if pipeline_id:
                    where_conditions.append("pipeline_id = ?")
                    params.append(pipeline_id)

                if recent_hours:
                    from datetime import datetime, timedelta

                    cutoff_time = (
                        datetime.now() - timedelta(hours=recent_hours)
                    ).isoformat()
                    where_conditions.append("created_at >= ?")
                    params.append(cutoff_time)

                where_clause = " AND ".join(where_conditions)

                cursor = conn.execute(
                    f"""
                    SELECT pipeline_id, gif_name, test_colors, test_lossy, test_frame_ratio,
                           error_type, error_message, error_traceback, created_at,
                           pipeline_steps, tools_used
                    FROM pipeline_failures
                    WHERE {where_clause}
                    ORDER BY created_at DESC
                """,
                    params,
                )

                columns = [desc[0] for desc in cursor.description]
                failures = []

                for row in cursor.fetchall():
                    failure = dict(zip(columns, row, strict=True))
                    # Parse JSON fields
                    try:
                        failure["pipeline_steps"] = json.loads(
                            failure["pipeline_steps"]
                        )
                        failure["tools_used"] = json.loads(failure["tools_used"])
                    except (json.JSONDecodeError, TypeError):
                        failure["pipeline_steps"] = []
                        failure["tools_used"] = []

                    failures.append(failure)

                return failures

        except Exception as e:
            self.logger.warning(f"Failed to query failures: {e}")
            return []


def get_git_commit() -> str:
    """Get current git commit hash for cache invalidation.

    Returns:
        Git commit hash or 'unknown' if not available
    """
    try:
        import subprocess

        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent,
        )
        if result.returncode == 0:
            return result.stdout.strip()[:12]  # Short hash
    except Exception:
        pass
    return "unknown"
