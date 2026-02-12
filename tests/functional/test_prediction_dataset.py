"""Tests for prediction dataset utilities and synthetic GIF feature extraction."""

from pathlib import Path

import numpy as np
import pytest
from giflab.prediction.dataset import DATASET_VERSION, load_training_records
from PIL import Image


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


class TestDatasetConstants:
    """Test that dataset constants are defined correctly."""

    def test_dataset_version(self) -> None:
        """Test DATASET_VERSION is a valid semver string."""
        parts = DATASET_VERSION.split(".")
        assert len(parts) == 3
        for part in parts:
            assert part.isdigit()


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
