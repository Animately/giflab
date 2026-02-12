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
