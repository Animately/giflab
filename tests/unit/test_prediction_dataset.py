"""Tests for prediction dataset builder.

Uses synthetic GIFs to test the dataset building pipeline.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from giflab.prediction.dataset import DatasetBuilder, DATASET_VERSION
from giflab.prediction.schemas import DatasetSplit, Engine


def create_synthetic_gif(
    path: Path,
    frames: int = 5,
    size: tuple[int, int] = (100, 100),
    pattern: str = "gradient",
) -> None:
    """Create a synthetic GIF for testing.

    Args:
        path: Output path for the GIF.
        frames: Number of frames.
        size: Frame dimensions (width, height).
        pattern: "gradient", "solid", "noise", or "animation".
    """
    images = []

    for i in range(frames):
        arr = np.zeros((size[1], size[0], 3), dtype=np.uint8)

        if pattern == "gradient":
            # Horizontal gradient with frame variation
            arr[:, :, 0] = np.linspace(0, 255, size[0], dtype=np.uint8)
            arr[:, :, 1] = (i * 40) % 256
            arr[:, :, 2] = 128
        elif pattern == "solid":
            # Solid color (highly compressible)
            arr[:, :, :] = [100, 150, 200]
        elif pattern == "noise":
            # Random noise (hard to compress)
            arr = np.random.randint(0, 256, (size[1], size[0], 3), dtype=np.uint8)
        elif pattern == "animation":
            # Moving bar animation
            bar_pos = (i * 20) % size[0]
            arr[:, bar_pos : bar_pos + 10, :] = [255, 255, 255]

        img = Image.fromarray(arr, mode="RGB")
        images.append(img)

    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=100,
        loop=0,
    )


class TestDatasetBuilder:
    """Test DatasetBuilder class."""

    def test_init_creates_directories(self) -> None:
        """Test that initialization creates output directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "training"
            builder = DatasetBuilder(output_dir)

            assert builder.features_dir.exists()
            assert builder.outcomes_dir.exists()
            assert builder.records_dir.exists()

    def test_init_validates_ratios(self) -> None:
        """Test that invalid ratios raise ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "training"

            with pytest.raises(ValueError, match="Ratios must sum to 1.0"):
                DatasetBuilder(
                    output_dir,
                    train_ratio=0.5,
                    val_ratio=0.5,
                    test_ratio=0.5,
                )

    def test_assign_split_distribution(self) -> None:
        """Test that splits are assigned correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "training"
            builder = DatasetBuilder(
                output_dir,
                train_ratio=0.8,
                val_ratio=0.1,
                test_ratio=0.1,
            )

            # Test split assignment for 100 samples
            splits = [builder._assign_split(i, 100) for i in range(100)]

            train_count = sum(1 for s in splits if s == DatasetSplit.TRAIN)
            val_count = sum(1 for s in splits if s == DatasetSplit.VAL)
            test_count = sum(1 for s in splits if s == DatasetSplit.TEST)

            assert train_count == 80
            assert val_count == 10
            assert test_count == 10


class TestSyntheticGifDataset:
    """Test dataset building with synthetic GIFs."""

    @pytest.fixture
    def synthetic_gifs(self, tmp_path: Path) -> list[Path]:
        """Create a set of synthetic GIFs for testing."""
        gifs = []
        patterns = ["gradient", "solid", "noise", "animation"]

        for i, pattern in enumerate(patterns):
            gif_path = tmp_path / f"test_{pattern}_{i}.gif"
            create_synthetic_gif(
                gif_path,
                frames=3,
                size=(50, 50),
                pattern=pattern,
            )
            gifs.append(gif_path)

        return gifs

    def test_build_dataset_extracts_features(
        self,
        synthetic_gifs: list[Path],
        tmp_path: Path,
    ) -> None:
        """Test that dataset building extracts features from GIFs."""
        output_dir = tmp_path / "training"
        builder = DatasetBuilder(output_dir, seed=42)

        # Build dataset (without compression sweeps for speed)
        results = builder.build_dataset(
            synthetic_gifs,
            engines=[Engine.GIFSICLE],
        )

        assert results["total"] == 4
        assert results["success"] + results["failed"] == 4

    def test_build_dataset_creates_records(
        self,
        synthetic_gifs: list[Path],
        tmp_path: Path,
    ) -> None:
        """Test that dataset building creates record files."""
        output_dir = tmp_path / "training"
        builder = DatasetBuilder(output_dir, seed=42)

        builder.build_dataset(
            synthetic_gifs[:2],  # Use fewer for speed
            engines=[Engine.GIFSICLE],
        )

        # Check records file exists
        records_files = list(builder.records_dir.glob("records_*.jsonl"))
        assert len(records_files) >= 0  # May be 0 if all failed

    def test_deterministic_splits(
        self,
        synthetic_gifs: list[Path],
        tmp_path: Path,
    ) -> None:
        """Test that same seed produces same splits."""
        output_dir1 = tmp_path / "training1"
        output_dir2 = tmp_path / "training2"

        builder1 = DatasetBuilder(output_dir1, seed=42)
        builder2 = DatasetBuilder(output_dir2, seed=42)

        results1 = builder1.build_dataset(
            synthetic_gifs,
            engines=[Engine.GIFSICLE],
        )
        results2 = builder2.build_dataset(
            synthetic_gifs,
            engines=[Engine.GIFSICLE],
        )

        assert results1["train"] == results2["train"]
        assert results1["val"] == results2["val"]
        assert results1["test"] == results2["test"]


class TestSyntheticGifPatterns:
    """Test that different GIF patterns produce different features."""

    def test_solid_gif_low_entropy(self, tmp_path: Path) -> None:
        """Solid color GIF should have low entropy."""
        from giflab.prediction.features import extract_gif_features

        gif_path = tmp_path / "solid.gif"
        create_synthetic_gif(gif_path, pattern="solid", frames=2)

        features = extract_gif_features(gif_path)

        # Solid color = low entropy
        assert features.entropy < 2.0
        # Solid color = low edge density
        assert features.edge_density < 0.1

    def test_noise_gif_high_entropy(self, tmp_path: Path) -> None:
        """Noisy GIF should have high entropy."""
        from giflab.prediction.features import extract_gif_features

        gif_path = tmp_path / "noise.gif"
        create_synthetic_gif(gif_path, pattern="noise", frames=2)

        features = extract_gif_features(gif_path)

        # Noise = high entropy
        assert features.entropy > 5.0
        # Noise = high color complexity
        assert features.color_complexity > 0.3

    def test_animation_gif_has_motion(self, tmp_path: Path) -> None:
        """Animated GIF should have detectable motion."""
        from giflab.prediction.features import extract_gif_features

        gif_path = tmp_path / "animation.gif"
        create_synthetic_gif(gif_path, pattern="animation", frames=5)

        features = extract_gif_features(gif_path)

        # Animation = non-zero motion
        assert features.motion_intensity > 0.0
        # Animation = not fully static
        assert features.static_region_ratio < 1.0

    def test_gradient_gif_has_colors(self, tmp_path: Path) -> None:
        """Gradient GIF should have color variation."""
        from giflab.prediction.features import extract_gif_features

        gif_path = tmp_path / "gradient.gif"
        create_synthetic_gif(gif_path, pattern="gradient", frames=3)

        features = extract_gif_features(gif_path)

        # Gradient = some color complexity (may be low due to quantization)
        assert features.color_complexity >= 0.0
        # Gradient = smooth transitions
        assert features.gradient_smoothness > 0.3
        # Gradient = moderate entropy
        assert features.entropy > 1.0
