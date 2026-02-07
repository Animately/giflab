"""Stub module for backward compatibility.

The elimination cache has been replaced by giflab.storage (SQLite-based).
This module provides stub functions to maintain backward compatibility
with core/runner.py until it is fully migrated.
"""

import subprocess
from pathlib import Path
from typing import Any


def get_git_commit() -> str:
    """Return short git commit hash if available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


class PipelineResultsCache:
    """Stub cache class for backward compatibility.

    NOTE: This is a stub. Real caching should use giflab.storage.
    """

    def __init__(
        self,
        cache_path: Path | None = None,
        content_type: str = "unknown",
    ):
        self.cache_path = cache_path
        self.content_type = content_type
        self._cache: dict[str, Any] = {}

    def get_result(self, pipeline_id: str, params: dict) -> dict | None:
        """Get cached result (stub - always returns None)."""
        return None

    def save_result(
        self,
        pipeline_id: str,
        params: dict,
        result: dict,
    ) -> None:
        """Save result to cache (stub - does nothing)."""
        pass

    def is_eliminated(self, pipeline_id: str) -> bool:
        """Check if pipeline is eliminated (stub - always False)."""
        return False

    def mark_eliminated(
        self,
        pipeline_id: str,
        reason: str,
        content_type: str | None = None,
    ) -> None:
        """Mark pipeline as eliminated (stub - does nothing)."""
        pass

    def get_elimination_reason(self, pipeline_id: str) -> str | None:
        """Get elimination reason (stub - always None)."""
        return None

    def clear(self) -> None:
        """Clear the cache (stub - does nothing)."""
        self._cache = {}
