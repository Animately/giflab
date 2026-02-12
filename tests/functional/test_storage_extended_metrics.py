"""Functional tests for extended metric storage in compression_runs."""

import tempfile
from datetime import UTC, datetime
from pathlib import Path

import pytest
from giflab.storage import QUALITY_METRIC_COLUMNS, GifLabStorage


@pytest.fixture()
def storage():
    """Create a GifLabStorage backed by a temporary database."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        s = GifLabStorage(db_path)
        yield s


def _seed_prerequisites(storage: GifLabStorage) -> tuple[int, int]:
    """Insert the minimum rows needed to satisfy foreign keys.

    Returns (pipeline_id, param_preset_id).
    """
    with storage._connect() as conn:
        # Insert a tool
        conn.execute(
            "INSERT OR IGNORE INTO tools (name, variable) VALUES ('test_tool', 'lossy')"
        )
        tool_id = conn.execute(
            "SELECT id FROM tools WHERE name='test_tool'"
        ).fetchone()["id"]

        # Insert a pipeline
        conn.execute(
            "INSERT OR IGNORE INTO pipelines (name, lossy_tool_id) VALUES ('test_pipe', ?)",
            (tool_id,),
        )
        pipeline_id = conn.execute(
            "SELECT id FROM pipelines WHERE name='test_pipe'"
        ).fetchone()["id"]

        # Insert a param preset
        conn.execute(
            "INSERT OR IGNORE INTO param_presets (lossy_level, color_count, frame_ratio) "
            "VALUES (60, 256, 1.0)"
        )
        preset_id = conn.execute(
            "SELECT id FROM param_presets WHERE lossy_level=60 AND color_count=256"
        ).fetchone()["id"]

        # Insert a gif_features row (required by foreign key)
        conn.execute(
            """INSERT OR IGNORE INTO gif_features (
                gif_sha, gif_name, width, height, frame_count, duration_ms,
                file_size_bytes, unique_colors, entropy, edge_density,
                color_complexity, gradient_smoothness, contrast_score,
                text_density, dct_energy_ratio, color_histogram_entropy,
                dominant_color_ratio, motion_intensity, motion_smoothness,
                static_region_ratio, temporal_entropy, frame_similarity,
                inter_frame_mse_mean, inter_frame_mse_std,
                lossless_compression_ratio, transparency_ratio,
                feature_extraction_version, extracted_at
            ) VALUES (
                'sha256_test', 'test.gif', 100, 100, 10, 1000,
                50000, 128, 7.5, 0.3, 0.5, 0.8, 0.6, 0.1, 0.4, 6.0,
                0.2, 0.5, 0.7, 0.3, 5.0, 0.9, 10.0, 2.0, 1.5, 0.0,
                '1.0.0', ?
            )""",
            (datetime.now(UTC).isoformat(),),
        )
        conn.commit()
    return pipeline_id, preset_id


class TestSaveCompressionBatchExtended:
    """Tests for save_compression_batch with extended metric columns."""

    def test_stores_and_retrieves_all_metric_columns(
        self, storage: GifLabStorage
    ) -> None:
        """All QUALITY_METRIC_COLUMNS round-trip through save and SELECT."""
        pipeline_id, preset_id = _seed_prerequisites(storage)

        # Build a run dict with a known value for every metric column
        run: dict = {
            "gif_sha": "sha256_test",
            "pipeline_id": pipeline_id,
            "param_preset_id": preset_id,
            "size_kb": 25.0,
            "compression_ratio": 2.0,
            "render_ms": 42,
            "giflab_version": "0.0.1-test",
            "created_at": datetime.now(UTC).isoformat(),
        }
        for i, col in enumerate(QUALITY_METRIC_COLUMNS):
            run[col] = float(i) / 100.0  # unique recognizable value per column

        storage.save_compression_batch([run])

        # Read it back and verify every metric column was stored
        with storage._connect() as conn:
            row = conn.execute(
                "SELECT * FROM compression_runs WHERE gif_sha='sha256_test'"
            ).fetchone()

        assert row is not None, "Compression run was not saved"
        for i, col in enumerate(QUALITY_METRIC_COLUMNS):
            expected = float(i) / 100.0
            actual = row[col]
            assert actual == pytest.approx(expected, abs=1e-9), (
                f"Column {col}: expected {expected}, got {actual}"
            )

    def test_stores_run_with_null_metrics(self, storage: GifLabStorage) -> None:
        """Metrics set to None are stored as NULL (no error)."""
        pipeline_id, preset_id = _seed_prerequisites(storage)

        run: dict = {
            "gif_sha": "sha256_test",
            "pipeline_id": pipeline_id,
            "param_preset_id": preset_id,
            "size_kb": 25.0,
            "compression_ratio": 2.0,
            "render_ms": 42,
            "giflab_version": "0.0.1-test",
            "created_at": datetime.now(UTC).isoformat(),
        }
        for col in QUALITY_METRIC_COLUMNS:
            run[col] = None

        storage.save_compression_batch([run])

        with storage._connect() as conn:
            row = conn.execute(
                "SELECT * FROM compression_runs WHERE gif_sha='sha256_test'"
            ).fetchone()

        assert row is not None
        for col in QUALITY_METRIC_COLUMNS:
            assert row[col] is None, f"Expected NULL for {col}, got {row[col]}"
