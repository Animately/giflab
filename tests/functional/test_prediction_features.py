"""Tests for prediction feature extraction.

Constitution Compliance:
- Principle IV (Test-Driven Quality): Tests for feature extraction
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from giflab.prediction.features import (
    _calculate_color_complexity,
    _calculate_dct_energy_ratio,
    _calculate_edge_density,
    _calculate_entropy,
    _calculate_transparency_ratio,
    _extract_frames_for_analysis,
    _extract_spatial_features,
    _extract_temporal_features,
    extract_gif_features,
)
from PIL import Image


def create_test_gif(
    path: Path,
    frames: int = 5,
    size: tuple[int, int] = (100, 100),
    colors: int = 256,
) -> None:
    """Create a simple test GIF for testing."""
    images = []
    for i in range(frames):
        # Create a simple gradient image with some variation per frame
        arr = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        arr[:, :, 0] = np.linspace(0, 255, size[0], dtype=np.uint8)
        arr[:, :, 1] = (i * 50) % 256
        arr[:, :, 2] = 128
        img = Image.fromarray(arr, mode="RGB")
        images.append(img)

    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=100,
        loop=0,
    )


class TestExtractGifFeatures:
    """Test the main extract_gif_features function."""

    def test_extract_features_returns_valid_schema(self) -> None:
        """Test that extraction returns a valid GifFeaturesV1."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gif_path = Path(tmpdir) / "test.gif"
            create_test_gif(gif_path, frames=5)

            features = extract_gif_features(gif_path)

            assert features.schema_version == "1.0.0"
            assert features.gif_name == "test.gif"
            assert len(features.gif_sha) == 64
            assert features.frame_count == 5
            assert features.width == 100
            assert features.height == 100

    def test_features_are_deterministic(self) -> None:
        """Test that same GIF produces identical features."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gif_path = Path(tmpdir) / "test.gif"
            create_test_gif(gif_path, frames=3)

            features1 = extract_gif_features(gif_path)
            features2 = extract_gif_features(gif_path)

            assert features1.gif_sha == features2.gif_sha
            assert features1.entropy == features2.entropy
            assert features1.edge_density == features2.edge_density
            assert features1.motion_intensity == features2.motion_intensity

    def test_file_not_found(self) -> None:
        """Test that FileNotFoundError is raised for missing file."""
        with pytest.raises(FileNotFoundError):
            extract_gif_features(Path("/nonexistent/path.gif"))

    def test_single_frame_gif(self) -> None:
        """Test extraction from single-frame GIF."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gif_path = Path(tmpdir) / "single.gif"
            create_test_gif(gif_path, frames=1)

            features = extract_gif_features(gif_path)

            assert features.frame_count == 1
            assert features.motion_intensity == 0.0
            assert features.static_region_ratio == 1.0


class TestFrameExtraction:
    """Test frame extraction utilities."""

    def test_extract_frames(self) -> None:
        """Test frame extraction from GIF."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gif_path = Path(tmpdir) / "test.gif"
            create_test_gif(gif_path, frames=10)

            frames = _extract_frames_for_analysis(gif_path, max_frames=5)

            assert len(frames) == 5
            assert all(isinstance(f, np.ndarray) for f in frames)
            assert all(f.shape == (100, 100, 3) for f in frames)

    def test_extract_all_frames_small_gif(self) -> None:
        """Test that small GIFs get all frames extracted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gif_path = Path(tmpdir) / "small.gif"
            create_test_gif(gif_path, frames=3)

            frames = _extract_frames_for_analysis(gif_path, max_frames=10)

            assert len(frames) == 3


class TestSpatialFeatures:
    """Test spatial feature extraction."""

    def test_extract_spatial_features(self) -> None:
        """Test spatial feature extraction from frame."""
        frame = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        features = _extract_spatial_features(frame)

        assert "entropy" in features
        assert "edge_density" in features
        assert "color_complexity" in features
        assert "gradient_smoothness" in features
        assert "contrast_score" in features
        assert "text_density" in features

        # All values should be in valid ranges
        assert 0 <= features["entropy"] <= 8
        assert 0 <= features["edge_density"] <= 1
        assert 0 <= features["color_complexity"] <= 1

    def test_entropy_range(self) -> None:
        """Test entropy calculation returns valid range."""
        # Uniform image = low entropy
        uniform = np.ones((100, 100), dtype=np.uint8) * 128
        assert _calculate_entropy(uniform) < 1.0

        # Random image = high entropy
        random_img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        assert _calculate_entropy(random_img) > 5.0

    def test_edge_density_range(self) -> None:
        """Test edge density calculation."""
        # Smooth image = low edge density
        smooth = np.ones((100, 100), dtype=np.uint8) * 128
        assert _calculate_edge_density(smooth) < 0.1

        # Edge image = higher edge density
        edges = np.zeros((100, 100), dtype=np.uint8)
        edges[::10, :] = 255  # Horizontal lines
        assert _calculate_edge_density(edges) > 0.05


class TestTemporalFeatures:
    """Test temporal feature extraction."""

    def test_temporal_features_single_frame(self) -> None:
        """Test temporal features with single frame."""
        frames = [np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)]
        features = _extract_temporal_features(frames)

        assert features["motion_intensity"] == 0.0
        assert features["static_region_ratio"] == 1.0
        assert features["frame_similarity"] == 1.0

    def test_temporal_features_identical_frames(self) -> None:
        """Test temporal features with identical frames."""
        frame = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        frames = [frame.copy() for _ in range(5)]
        features = _extract_temporal_features(frames)

        assert features["motion_intensity"] == 0.0
        assert features["inter_frame_mse_mean"] == 0.0

    def test_temporal_features_different_frames(self) -> None:
        """Test temporal features with different frames."""
        frames = [
            np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8) for _ in range(5)
        ]
        features = _extract_temporal_features(frames)

        assert features["motion_intensity"] > 0.0
        assert features["inter_frame_mse_mean"] > 0.0


class TestCompressibilityFeatures:
    """Test compressibility feature extraction."""

    def test_dct_energy_ratio_uniform(self) -> None:
        """Test DCT energy ratio for uniform image."""
        uniform = np.ones((100, 100, 3), dtype=np.uint8) * 128
        ratio = _calculate_dct_energy_ratio(uniform)

        # Uniform image should have low high-frequency energy
        assert 0 <= ratio <= 1
        assert ratio < 0.5

    def test_dct_energy_ratio_noisy(self) -> None:
        """Test DCT energy ratio for noisy image."""
        noisy = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        ratio = _calculate_dct_energy_ratio(noisy)

        # Noisy image should have higher high-frequency energy
        assert 0 <= ratio <= 1
        assert ratio > 0.3

    def test_color_complexity_gradient(self) -> None:
        """Test color complexity for gradient image."""
        gradient = np.zeros((100, 100, 3), dtype=np.uint8)
        gradient[:, :, 0] = np.linspace(0, 255, 100, dtype=np.uint8)
        complexity = _calculate_color_complexity(gradient)

        assert 0 <= complexity <= 1


# ──────────────────────────────────────────────────────────────────────────────
# Transparency audit regression tests (audit 2026-05-27)
# ──────────────────────────────────────────────────────────────────────────────


def _make_transparent_palette_gif(
    path: Path,
    transparent_idx: int,
    n_frames: int = 3,
    size: tuple[int, int] = (32, 32),
    visible_patch_size: int = 4,
) -> Path:
    """Build an animated palette GIF with a defined transparent index.

    The visible content (a patch in the top-left corner) is identical
    regardless of ``transparent_idx``. The *colour* that occupies the
    transparent palette slot varies, so naive ``convert('RGB')`` would resolve
    the transparent region differently — but the *alpha* is stable.
    """
    palette = [
        255,
        0,
        0,  # index 0 = red
        0,
        255,
        0,  # index 1 = green
        0,
        0,
        255,  # index 2 = blue
        255,
        255,
        0,  # index 3 = yellow
    ] + [0] * (256 * 3 - 12)

    # Visible (opaque) index: use one that is NOT the transparent index.
    visible_idx = (transparent_idx + 1) % 4

    frames = []
    for i in range(n_frames):
        img = Image.new("P", size, transparent_idx)
        img.putpalette(palette)
        # Stamp a unique pixel per frame to avoid PIL deduplicating frames.
        img.putpixel((size[0] - 1, i % size[1]), transparent_idx)
        for y in range(visible_patch_size):
            for x in range(visible_patch_size):
                img.putpixel((x, y), visible_idx)
        frames.append(img)

    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=100,
        loop=0,
        transparency=transparent_idx,
        disposal=2,
    )
    return path


def _make_asymmetric_transparency_gif(
    path: Path,
    size: tuple[int, int] = (32, 32),
) -> Path:
    """Build a 4-frame GIF where frame 0 is fully transparent and frames 1-3 are nearly opaque.

    To prevent PIL from deduplicating the nearly-identical opaque frames, each
    frame has a unique single-pixel marker. PIL's frame-deduplication only
    removes truly identical frames, so this ensures n_frames=4.

    Used to verify that transparency_ratio averages across ALL frames rather
    than only sampling frame 0.
    """
    palette = [255, 0, 0, 0, 255, 0] + [0] * (256 * 3 - 6)
    # index 0 = red (opaque), index 1 = green (transparent)

    frames = []

    # Frame 0: fully transparent (index 1 everywhere)
    f0 = Image.new("P", size, 1)
    f0.putpalette(palette)
    frames.append(f0)

    # Frames 1–3: fully opaque (index 0 everywhere), with unique pixel per frame
    # to prevent PIL from deduplicating them.
    for i in range(1, 4):
        fi = Image.new("P", size, 0)
        fi.putpalette(palette)
        # Unique marker: pixel at (i, 0) is transparent (but all others are opaque)
        fi.putpixel((i, 0), 1)
        frames.append(fi)

    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=100,
        loop=0,
        transparency=1,
        disposal=2,
    )
    return path


class TestTransparencyRatioAudit:
    """Audit regression tests for _calculate_transparency_ratio.

    Audit context (2026-05-27):
    - _extract_frames_for_analysis uses OpenCV, which composites transparent
      GIF pixels onto white natively. This is already stable across palette
      reordering — no bug, no fix needed there.
    - _calculate_transparency_ratio uses PIL convert('RGBA'), which reads the
      alpha channel rather than the colour under the transparent index. The
      alpha is stable across palette reordering (no colour-dependent bug). Two
      issues were found and fixed:
        1. Only frame 0 was sampled, unrepresentative for animated GIFs where
           transparency coverage varies per frame. Fixed: now averages all
           frames via img.n_frames iteration.
        2. On exception the function returned 0.0 instead of float('nan'),
           violating the CLAUDE.md 'NaN over fabricated values' rule. Fixed:
           now returns float('nan') so callers can detect measurement failure.
    """

    # ── Invariant 1: stability across palette reordering ─────────────────────

    def test_transparency_ratio_stable_across_palette_reorder(
        self, tmp_path: Path
    ) -> None:
        """Identical visible content with different transparent indices → same ratio.

        Locks the key invariant: PIL convert('RGBA') uses the declared
        transparency index to set alpha=0, not the colour under the index.
        The ratio must be identical regardless of which palette slot is
        transparent.
        """
        gif_idx1 = _make_transparent_palette_gif(
            tmp_path / "idx1.gif", transparent_idx=1
        )
        gif_idx2 = _make_transparent_palette_gif(
            tmp_path / "idx2.gif", transparent_idx=2
        )

        ratio_idx1 = _calculate_transparency_ratio(gif_idx1)
        ratio_idx2 = _calculate_transparency_ratio(gif_idx2)

        assert ratio_idx1 == ratio_idx2, (
            f"Transparency ratio differs across palette reordering: "
            f"{ratio_idx1} vs {ratio_idx2}. PIL must use alpha not colour."
        )

    def test_transparency_ratio_opaque_gif_returns_zero(self, tmp_path: Path) -> None:
        """Fully opaque GIF (no transparency declared) returns 0.0."""
        palette = [255, 0, 0] + [0] * (256 * 3 - 3)
        img = Image.new("P", (32, 32), 0)
        img.putpalette(palette)
        path = tmp_path / "opaque.gif"
        img.save(path)

        ratio = _calculate_transparency_ratio(path)

        assert ratio == 0.0, f"Opaque GIF should return 0.0, got {ratio}"

    def test_transparency_ratio_fully_transparent_gif_returns_one(
        self, tmp_path: Path
    ) -> None:
        """GIF where every pixel is transparent returns a ratio close to 1.0."""
        palette = [255, 0, 0, 0, 255, 0] + [0] * (256 * 3 - 6)
        img = Image.new("P", (8, 8), 1)  # index 1 = green = transparent
        img.putpalette(palette)
        path = tmp_path / "fully_transparent.gif"
        img.save(path, transparency=1)

        ratio = _calculate_transparency_ratio(path)

        assert ratio == pytest.approx(
            1.0, abs=0.01
        ), f"Fully transparent GIF should return ≈1.0, got {ratio}"

    # ── Invariant 2: multi-frame averaging ───────────────────────────────────

    def test_transparency_ratio_averages_all_frames_not_just_frame0(
        self, tmp_path: Path
    ) -> None:
        """Ratio averages across ALL frames, not just frame 0.

        Constructs a 4-frame GIF where:
          - frame 0: fully transparent → ratio = 1.0
          - frames 1–3: fully opaque (1 transparent pixel each) → ratio ≈ 0.001

        Frame-0-only sampling would return 1.0.
        All-frame averaging must return close to (1.0 + ~0 + ~0 + ~0) / 4 ≈ 0.25.

        The test asserts ratio < 0.5, which is satisfied by all-frame averaging
        and violated by frame-0-only sampling.
        """
        gif_path = _make_asymmetric_transparency_gif(tmp_path / "asymmetric.gif")

        # Sanity-check that PIL sees 4 frames (not deduplicated).
        with Image.open(gif_path) as img:
            n = getattr(img, "n_frames", 1)
        assert n == 4, (
            f"Test precondition failed: PIL should see 4 frames, got {n}. "
            "The fixture helper must produce non-identical frames."
        )

        ratio = _calculate_transparency_ratio(gif_path)

        assert ratio < 0.5, (
            f"All-frame average for 1-transparent + 3-opaque GIF should be ~0.25, "
            f"got {ratio}. Did _calculate_transparency_ratio only check frame 0?"
        )

    # ── Invariant 3: None on unreadable input ────────────────────────────────

    def test_transparency_ratio_returns_none_on_nonexistent_file(
        self, tmp_path: Path
    ) -> None:
        """Returns None when the file cannot be read.

        CLAUDE.md rule: 'NaN over fabricated values' — returning 0.0 silently
        corrupts composite_quality / ML tables. We use None here (not NaN)
        because the consumer is a Pydantic schema field with bounds
        ge=0.0, le=1.0 that rejects NaN (PR #19 reviewer-confirmed blocker).
        None is the schema-compatible 'missing measurement' sentinel; the
        ML pipeline coerces it to np.nan inside _features_to_matrix where
        bounds validation no longer applies.
        """
        result = _calculate_transparency_ratio(tmp_path / "does_not_exist.gif")

        assert result is None, (
            f"Expected None on unreadable file, got {result!r}. "
            "Returning 0.0 violates the 'NaN over fabricated values' rule; "
            "returning NaN crashes Pydantic schema validation."
        )

    def test_transparency_ratio_returns_none_on_corrupt_file(
        self, tmp_path: Path
    ) -> None:
        """Returns None when PIL cannot decode the file."""
        # A non-GIF file with .gif extension
        corrupt = tmp_path / "corrupt.gif"
        corrupt.write_bytes(b"not a real gif file")

        result = _calculate_transparency_ratio(corrupt)

        assert result is None, f"Expected None on corrupt file, got {result!r}."


class TestGifFeaturesSchemaAcceptsNoneTransparencyRatio:
    """PR #19 reviewer blocker: GifFeaturesV1 must accept None for transparency_ratio.

    The pre-fix schema declared transparency_ratio as `float` with
    ge=0.0, le=1.0. Pydantic V2 evaluates NaN against le=1.0 and raises
    ValidationError — so the previous 'return float(nan)' on error path
    hard-crashed extract_gif_features() instead of propagating the missing
    measurement. Schema now declares transparency_ratio: float | None.
    """

    def _valid_features_kwargs(self) -> dict:
        """Minimal valid kwargs for GifFeaturesV1 — only transparency_ratio varies."""
        from datetime import UTC, datetime

        return {
            "gif_sha": "a" * 64,
            "gif_name": "test.gif",
            "extraction_version": "1.0.0",
            "extracted_at": datetime.now(UTC),
            "width": 32,
            "height": 32,
            "frame_count": 1,
            "duration_ms": 100,
            "file_size_bytes": 100,
            "unique_colors": 4,
            "entropy": 1.0,
            "edge_density": 0.1,
            "color_complexity": 0.1,
            "gradient_smoothness": 0.5,
            "contrast_score": 0.5,
            "text_density": 0.1,
            "dct_energy_ratio": 0.3,
            "color_histogram_entropy": 2.0,
            "dominant_color_ratio": 0.5,
            "motion_intensity": 0.0,
            "motion_smoothness": 1.0,
            "static_region_ratio": 1.0,
            "temporal_entropy": 0.0,
            "frame_similarity": 1.0,
            "inter_frame_mse_mean": 0.0,
            "inter_frame_mse_std": 0.0,
            "lossless_compression_ratio": 0.5,
        }

    def test_schema_accepts_none_transparency_ratio(self) -> None:
        """None is a valid value for transparency_ratio (the 'unmeasurable' sentinel).

        Locks the schema fix: the field must be `float | None` so that callers
        can pass None when _calculate_transparency_ratio fails. Pre-fix this
        raised ValidationError because the field was declared `float`.
        """
        from giflab.prediction.schemas import GifFeaturesV1

        kwargs = self._valid_features_kwargs()
        kwargs["transparency_ratio"] = None

        features = GifFeaturesV1(**kwargs)

        assert features.transparency_ratio is None

    def test_schema_accepts_valid_float_transparency_ratio(self) -> None:
        """A valid float in [0.0, 1.0] is still accepted."""
        from giflab.prediction.schemas import GifFeaturesV1

        kwargs = self._valid_features_kwargs()
        kwargs["transparency_ratio"] = 0.42

        features = GifFeaturesV1(**kwargs)

        assert features.transparency_ratio == 0.42

    def test_schema_rejects_out_of_range_float(self) -> None:
        """Bounds validation still applies to non-None values."""
        from giflab.prediction.schemas import GifFeaturesV1
        from pydantic import ValidationError

        kwargs = self._valid_features_kwargs()
        kwargs["transparency_ratio"] = 1.5  # out of range

        with pytest.raises(ValidationError):
            GifFeaturesV1(**kwargs)


class TestExtractGifFeaturesNanCoercion:
    """Call-site coercion: extract_gif_features() must coerce a failed
    transparency measurement into a schema-compatible value (None) instead
    of letting NaN crash GifFeaturesV1 construction.
    """

    def test_extract_features_on_corrupt_gif_does_not_crash_schema(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When _calculate_transparency_ratio returns None, the schema accepts it.

        This is the end-to-end version of the schema test: we patch the
        transparency helper to return None (simulating a corrupted-transparent
        GIF) and verify that extract_gif_features() completes successfully
        with transparency_ratio=None on the resulting object.
        """
        # Make a valid simple GIF (no transparency complications)
        gif_path = tmp_path / "simple.gif"
        create_test_gif(gif_path, frames=3)

        # Force the transparency helper to report 'unmeasurable'
        import giflab.prediction.features as features_mod

        monkeypatch.setattr(
            features_mod,
            "_calculate_transparency_ratio",
            lambda _path: None,
        )

        # Should NOT raise ValidationError
        features = extract_gif_features(gif_path)

        assert features.transparency_ratio is None


class TestTransparencyRatioStorageNullable:
    """SQLite layer must accept NULL for transparency_ratio.

    The pre-fix gif_features schema declared transparency_ratio as REAL NOT NULL.
    With the Pydantic field now `float | None`, the storage column must be
    nullable too — otherwise None propagates from extract_gif_features() into
    save_gif_features() and crashes the INSERT with NOT NULL constraint failure.
    """

    def test_save_gif_features_accepts_null_transparency_ratio(
        self, tmp_path: Path
    ) -> None:
        """save_gif_features() round-trips None → NULL → None for transparency_ratio."""
        from datetime import UTC, datetime

        from giflab.storage import GifLabStorage

        db_path = tmp_path / "test.db"
        storage = GifLabStorage(db_path)

        row = {
            "gif_sha": "f" * 64,
            "gif_name": "no_transparency.gif",
            "file_path": "/tmp/no_transparency.gif",
            "width": 32,
            "height": 32,
            "frame_count": 1,
            "duration_ms": 100,
            "file_size_bytes": 100,
            "unique_colors": 4,
            "entropy": 1.0,
            "edge_density": 0.1,
            "color_complexity": 0.1,
            "gradient_smoothness": 0.5,
            "contrast_score": 0.5,
            "text_density": 0.1,
            "dct_energy_ratio": 0.3,
            "color_histogram_entropy": 2.0,
            "dominant_color_ratio": 0.5,
            "motion_intensity": 0.0,
            "motion_smoothness": 1.0,
            "static_region_ratio": 1.0,
            "temporal_entropy": 0.0,
            "frame_similarity": 1.0,
            "inter_frame_mse_mean": 0.0,
            "inter_frame_mse_std": 0.0,
            "lossless_compression_ratio": 0.5,
            # The bit under test: NULL transparency_ratio.
            "transparency_ratio": None,
            "clip_screen_capture": None,
            "clip_vector_art": None,
            "clip_photography": None,
            "clip_hand_drawn": None,
            "clip_3d_rendered": None,
            "clip_pixel_art": None,
            "feature_extraction_version": "1.0.0",
            "extracted_at": datetime.now(UTC).isoformat(),
        }

        # Must not raise IntegrityError
        storage.save_gif_features(row)

        # Round-trip: NULL persists as NULL (which SQLite returns as None)
        with storage._connect() as conn:
            stored = conn.execute(
                "SELECT transparency_ratio FROM gif_features WHERE gif_sha = ?",
                (row["gif_sha"],),
            ).fetchone()

        assert stored is not None, "row was not inserted"
        assert stored["transparency_ratio"] is None


class TestTransparencyRatioMigration:
    """Migration regression: existing DBs with the old NOT NULL constraint
    must be transparently upgraded to nullable when GifLabStorage opens them."""

    def test_migration_drops_not_null_and_preserves_data(self, tmp_path: Path) -> None:
        """Open a DB created with the OLD schema (NOT NULL); verify the
        migration kicks in, the column becomes nullable, AND the existing
        rows are preserved."""
        import sqlite3
        from datetime import UTC, datetime

        from giflab.storage import GifLabStorage

        db_path = tmp_path / "legacy.db"

        # Hand-build an OLD-schema gif_features table (transparency_ratio NOT NULL)
        # mirroring the schema before this PR.
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        conn.executescript(
            """
            CREATE TABLE gif_features (
                gif_sha TEXT PRIMARY KEY,
                gif_name TEXT NOT NULL,
                file_path TEXT,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL,
                frame_count INTEGER NOT NULL,
                duration_ms INTEGER NOT NULL,
                file_size_bytes INTEGER NOT NULL,
                unique_colors INTEGER NOT NULL,
                entropy REAL NOT NULL,
                edge_density REAL NOT NULL,
                color_complexity REAL NOT NULL,
                gradient_smoothness REAL NOT NULL,
                contrast_score REAL NOT NULL,
                text_density REAL NOT NULL,
                dct_energy_ratio REAL NOT NULL,
                color_histogram_entropy REAL NOT NULL,
                dominant_color_ratio REAL NOT NULL,
                motion_intensity REAL NOT NULL,
                motion_smoothness REAL NOT NULL,
                static_region_ratio REAL NOT NULL,
                temporal_entropy REAL NOT NULL,
                frame_similarity REAL NOT NULL,
                inter_frame_mse_mean REAL NOT NULL,
                inter_frame_mse_std REAL NOT NULL,
                lossless_compression_ratio REAL NOT NULL,
                transparency_ratio REAL NOT NULL,
                clip_screen_capture REAL,
                clip_vector_art REAL,
                clip_photography REAL,
                clip_hand_drawn REAL,
                clip_3d_rendered REAL,
                clip_pixel_art REAL,
                feature_extraction_version TEXT NOT NULL,
                extracted_at TEXT NOT NULL,
                compression_complete BOOLEAN DEFAULT FALSE
            );
            """
        )

        # Insert a sentinel row with a known transparency_ratio
        now = datetime.now(UTC).isoformat()
        conn.execute(
            """INSERT INTO gif_features (
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
                'sentinel_sha', 'sentinel.gif', 64, 64, 4, 400,
                1234, 16, 4.5, 0.2, 0.3, 0.6, 0.4, 0.05, 0.5, 5.0,
                0.4, 0.2, 0.8, 0.6, 3.0, 0.9, 5.0, 1.0, 0.7, 0.42,
                '1.0.0', ?
            )""",
            (now,),
        )
        conn.commit()
        conn.close()

        # Verify the legacy DB has NOT NULL on transparency_ratio
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cols = conn.execute("PRAGMA table_info(gif_features)").fetchall()
        t_col = next(c for c in cols if c["name"] == "transparency_ratio")
        assert t_col["notnull"] == 1, "Test setup: legacy DB should have NOT NULL"
        conn.close()

        # Open via GifLabStorage — this should trigger migration
        storage = GifLabStorage(db_path)

        # NOT NULL constraint should now be gone
        with storage._connect() as conn:
            cols = conn.execute("PRAGMA table_info(gif_features)").fetchall()
            t_col = next(c for c in cols if c["name"] == "transparency_ratio")
            assert (
                t_col["notnull"] == 0
            ), "Migration failed: transparency_ratio is still NOT NULL"

            # Sentinel row preserved with its value intact
            sentinel = conn.execute(
                "SELECT transparency_ratio, gif_name FROM gif_features "
                "WHERE gif_sha = 'sentinel_sha'"
            ).fetchone()
            assert sentinel is not None, "Migration dropped existing data"
            assert sentinel["transparency_ratio"] == pytest.approx(0.42)
            assert sentinel["gif_name"] == "sentinel.gif"

            # Inserting NULL now works
            storage.save_gif_features(
                {
                    "gif_sha": "post_migration_sha" + "0" * 46,
                    "gif_name": "after.gif",
                    "file_path": "/tmp/after.gif",
                    "width": 32,
                    "height": 32,
                    "frame_count": 1,
                    "duration_ms": 100,
                    "file_size_bytes": 100,
                    "unique_colors": 4,
                    "entropy": 1.0,
                    "edge_density": 0.1,
                    "color_complexity": 0.1,
                    "gradient_smoothness": 0.5,
                    "contrast_score": 0.5,
                    "text_density": 0.1,
                    "dct_energy_ratio": 0.3,
                    "color_histogram_entropy": 2.0,
                    "dominant_color_ratio": 0.5,
                    "motion_intensity": 0.0,
                    "motion_smoothness": 1.0,
                    "static_region_ratio": 1.0,
                    "temporal_entropy": 0.0,
                    "frame_similarity": 1.0,
                    "inter_frame_mse_mean": 0.0,
                    "inter_frame_mse_std": 0.0,
                    "lossless_compression_ratio": 0.5,
                    "transparency_ratio": None,
                    "clip_screen_capture": None,
                    "clip_vector_art": None,
                    "clip_photography": None,
                    "clip_hand_drawn": None,
                    "clip_3d_rendered": None,
                    "clip_pixel_art": None,
                    "feature_extraction_version": "1.0.0",
                    "extracted_at": now,
                }
            )

    def test_migration_is_idempotent(self, tmp_path: Path) -> None:
        """Opening an already-migrated DB doesn't re-run the rebuild."""
        from giflab.storage import GifLabStorage

        db_path = tmp_path / "already_new.db"

        # First open: creates the new (nullable) schema
        GifLabStorage(db_path)

        # Second open: should be a no-op for the migration
        GifLabStorage(db_path)  # must not crash, must not duplicate data


class TestFeaturesToMatrixNoneCoercion:
    """CurvePredictionModel._features_to_matrix must coerce None → np.nan
    so that downstream sklearn receives a numeric matrix and doesn't choke
    on Python None inside a float64 array."""

    def test_features_to_matrix_coerces_none_transparency_to_nan(self) -> None:
        """A GifFeaturesV1 with transparency_ratio=None becomes np.nan in the matrix."""
        from datetime import UTC, datetime

        from giflab.prediction.models import (
            FEATURE_COLUMNS,
            CurvePredictionModel,
        )
        from giflab.prediction.schemas import (
            CurveType,
            Engine,
            GifFeaturesV1,
        )

        features = GifFeaturesV1(
            gif_sha="a" * 64,
            gif_name="t.gif",
            extraction_version="1.0.0",
            extracted_at=datetime.now(UTC),
            width=32,
            height=32,
            frame_count=1,
            duration_ms=100,
            file_size_bytes=100,
            unique_colors=4,
            entropy=1.0,
            edge_density=0.1,
            color_complexity=0.1,
            gradient_smoothness=0.5,
            contrast_score=0.5,
            text_density=0.1,
            dct_energy_ratio=0.3,
            color_histogram_entropy=2.0,
            dominant_color_ratio=0.5,
            motion_intensity=0.0,
            motion_smoothness=1.0,
            static_region_ratio=1.0,
            temporal_entropy=0.0,
            frame_similarity=1.0,
            inter_frame_mse_mean=0.0,
            inter_frame_mse_std=0.0,
            lossless_compression_ratio=0.5,
            transparency_ratio=None,  # ← the case under test
        )

        model = CurvePredictionModel(engine=Engine.GIFSICLE, curve_type=CurveType.LOSSY)
        matrix = model._features_to_matrix([features])

        # Find the transparency_ratio column index
        idx = FEATURE_COLUMNS.index("transparency_ratio")

        # The value at that column should be NaN, not raise / not be None
        import math

        assert math.isnan(matrix[0, idx]), (
            f"Expected NaN at transparency_ratio column, got {matrix[0, idx]!r}. "
            "Did _features_to_matrix coerce None → np.nan?"
        )
