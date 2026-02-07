"""Tests for giflab.meta module."""

import hashlib

import numpy as np
import pytest
from giflab.meta import (
    GifMetadata,
    calculate_entropy,
    compute_file_sha256,
    extract_gif_metadata,
)
from PIL import Image


class TestComputeFileSha256:
    """Tests for compute_file_sha256 function."""

    @pytest.mark.fast
    def test_sha256_of_known_content(self, tmp_path):
        """Test SHA256 computation with known content."""
        test_file = tmp_path / "test.txt"
        test_content = b"Hello, World!"
        test_file.write_bytes(test_content)

        # Calculate expected hash
        expected_hash = hashlib.sha256(test_content).hexdigest()

        # Test our function
        result = compute_file_sha256(test_file)
        assert result == expected_hash

    @pytest.mark.fast
    def test_sha256_empty_file(self, tmp_path):
        """Test SHA256 computation of empty file."""
        test_file = tmp_path / "empty.txt"
        test_file.write_bytes(b"")

        expected_hash = hashlib.sha256(b"").hexdigest()
        result = compute_file_sha256(test_file)
        assert result == expected_hash

    @pytest.mark.fast
    def test_sha256_large_file(self, tmp_path):
        """Test SHA256 computation with large file (chunk reading)."""
        test_file = tmp_path / "large.txt"
        # Create a file larger than 4KB to test chunked reading
        large_content = b"A" * 10000
        test_file.write_bytes(large_content)

        expected_hash = hashlib.sha256(large_content).hexdigest()
        result = compute_file_sha256(test_file)
        assert result == expected_hash


class TestCalculateEntropy:
    """Tests for calculate_entropy function."""

    @pytest.mark.fast
    def test_entropy_uniform_image(self):
        """Test entropy calculation for uniform (low entropy) image."""
        # Create a uniform gray image
        uniform_img = Image.new("L", (10, 10), color=128)
        entropy = calculate_entropy(uniform_img)

        # Uniform image should have very low entropy (close to 0)
        assert 0.0 <= entropy <= 1.0

    @pytest.mark.fast
    def test_entropy_random_image(self):
        """Test entropy calculation for random (high entropy) image."""
        # Create a random image
        random_array = np.random.randint(0, 256, (50, 50), dtype=np.uint8)
        random_img = Image.fromarray(random_array, mode="L")
        entropy = calculate_entropy(random_img)

        # Random image should have higher entropy
        assert entropy > 4.0  # Should be close to 8 for truly random

    @pytest.mark.fast
    def test_entropy_color_image(self):
        """Test entropy calculation converts color to grayscale."""
        # Create a color image
        color_img = Image.new("RGB", (10, 10), color=(255, 0, 0))
        entropy = calculate_entropy(color_img)

        # Should work without error and return reasonable value
        assert isinstance(entropy, float)
        assert entropy >= 0.0


class TestGifMetadata:
    """Tests for GifMetadata dataclass."""

    @pytest.mark.fast
    def test_gif_metadata_creation(self):
        """Test GifMetadata object creation."""
        metadata = GifMetadata(
            gif_sha="abc123",
            orig_filename="test.gif",
            orig_kilobytes=10.5,
            orig_width=100,
            orig_height=100,
            orig_frames=10,
            orig_fps=24.0,
            orig_n_colors=256,
            entropy=5.5,
        )

        assert metadata.gif_sha == "abc123"
        assert metadata.orig_filename == "test.gif"
        assert metadata.orig_kilobytes == 10.5
        assert metadata.orig_width == 100
        assert metadata.orig_height == 100
        assert metadata.orig_frames == 10
        assert metadata.orig_fps == 24.0
        assert metadata.orig_n_colors == 256
        assert metadata.entropy == 5.5
        assert metadata.source_platform == "unknown"
        assert metadata.source_metadata is None

    @pytest.mark.fast
    def test_gif_metadata_optional_entropy(self):
        """Test GifMetadata with optional entropy field."""
        metadata = GifMetadata(
            gif_sha="abc123",
            orig_filename="test.gif",
            orig_kilobytes=10.5,
            orig_width=100,
            orig_height=100,
            orig_frames=10,
            orig_fps=24.0,
            orig_n_colors=256,
        )

        assert metadata.entropy is None
        assert metadata.source_platform == "unknown"
        assert metadata.source_metadata is None


class TestExtractGifMetadata:
    """Tests for extract_gif_metadata function."""

    @pytest.fixture
    def simple_gif(self, tmp_path):
        """Create a simple test GIF file."""
        gif_path = tmp_path / "test.gif"

        # Create a simple animated GIF
        frames = []
        for i in range(3):
            # Create frames with different colors
            img = Image.new("P", (20, 20), color=i * 80)
            # Add a simple palette
            palette = []
            for j in range(256):
                palette.extend([j, j, j])  # Grayscale palette
            img.putpalette(palette)
            frames.append(img)

        # Save as animated GIF
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=100,  # 100ms per frame
            loop=0,
        )

        return gif_path

    @pytest.mark.fast
    def test_extract_gif_metadata_success(self, simple_gif):
        """Test successful GIF metadata extraction."""
        metadata = extract_gif_metadata(simple_gif)

        assert isinstance(metadata, GifMetadata)
        assert len(metadata.gif_sha) == 64  # SHA256 hex string length
        assert metadata.orig_filename == simple_gif.name
        assert metadata.orig_kilobytes > 0
        assert metadata.orig_width == 20
        assert metadata.orig_height == 20
        assert metadata.orig_frames == 3
        assert metadata.orig_fps > 0
        assert metadata.orig_n_colors > 0
        assert isinstance(metadata.entropy, float)

    @pytest.mark.fast
    def test_extract_gif_metadata_file_not_found(self, tmp_path):
        """Test error handling for non-existent file."""
        non_existent = tmp_path / "does_not_exist.gif"

        with pytest.raises(IOError, match="File not found"):
            extract_gif_metadata(non_existent)

    @pytest.mark.fast
    def test_extract_gif_metadata_not_gif(self, tmp_path):
        """Test error handling for non-GIF file."""
        # Create a PNG file instead of GIF
        png_path = tmp_path / "test.png"
        img = Image.new("RGB", (10, 10), color=(255, 0, 0))
        img.save(png_path, "PNG")

        with pytest.raises(ValueError, match="File is not a GIF"):
            extract_gif_metadata(png_path)

    @pytest.mark.fast
    def test_extract_gif_metadata_invalid_file(self, tmp_path):
        """Test error handling for corrupted file."""
        # Create a file with .gif extension but invalid content
        invalid_gif = tmp_path / "invalid.gif"
        invalid_gif.write_bytes(b"Not a GIF file")

        with pytest.raises(ValueError, match="Error processing GIF"):
            extract_gif_metadata(invalid_gif)

    @pytest.fixture
    def single_frame_gif(self, tmp_path):
        """Create a single-frame GIF for testing."""
        gif_path = tmp_path / "single.gif"

        # Create a single frame GIF
        img = Image.new("P", (50, 30), color=100)
        palette = []
        for i in range(256):
            palette.extend([i, 0, 0])  # Red gradient palette
        img.putpalette(palette)

        img.save(gif_path, "GIF")
        return gif_path

    @pytest.mark.fast
    def test_extract_single_frame_gif(self, single_frame_gif):
        """Test metadata extraction for single-frame GIF."""
        metadata = extract_gif_metadata(single_frame_gif)

        assert metadata.orig_width == 50
        assert metadata.orig_height == 30
        assert metadata.orig_frames == 1
        assert metadata.orig_fps > 0  # Should have reasonable default FPS
        assert metadata.orig_n_colors > 0

    @pytest.mark.fast
    def test_gif_metadata_consistency(self, simple_gif):
        """Test that repeated extractions give consistent results."""
        metadata1 = extract_gif_metadata(simple_gif)
        metadata2 = extract_gif_metadata(simple_gif)

        # All fields should be identical
        assert metadata1.gif_sha == metadata2.gif_sha
        assert metadata1.orig_filename == metadata2.orig_filename
        assert metadata1.orig_kilobytes == metadata2.orig_kilobytes
        assert metadata1.orig_width == metadata2.orig_width
        assert metadata1.orig_height == metadata2.orig_height
        assert metadata1.orig_frames == metadata2.orig_frames
        assert metadata1.orig_fps == metadata2.orig_fps
        assert metadata1.orig_n_colors == metadata2.orig_n_colors
        assert metadata1.entropy == metadata2.entropy


@pytest.mark.integration
class TestMetaIntegration:
    """Integration tests for the meta module."""

    def test_full_workflow(self, tmp_path):
        """Test the complete metadata extraction workflow."""
        # Create a test GIF with known properties
        gif_path = tmp_path / "workflow_test.gif"

        # Create frames with varying complexity
        frames = []
        for i in range(5):
            # Create progressively more complex frames
            if i == 0:
                # Simple solid color
                img = Image.new("P", (40, 40), color=50)
            else:
                # Add some noise/pattern
                arr = np.random.randint(0, 100, (40, 40), dtype=np.uint8)
                img = Image.fromarray(arr, mode="L")
                img = img.convert("P")

            # Add palette
            palette = list(range(256)) * 3
            img.putpalette(palette)
            frames.append(img)

        # Save as animated GIF with specific timing
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=50,  # 50ms per frame = 20 FPS
            loop=0,
        )

        # Extract metadata
        metadata = extract_gif_metadata(gif_path)

        # Verify all expected properties
        assert metadata.orig_width == 40
        assert metadata.orig_height == 40
        assert metadata.orig_frames == 5
        assert 15.0 <= metadata.orig_fps <= 25.0  # Should be close to 20 FPS
        assert metadata.orig_kilobytes > 0
        assert len(metadata.gif_sha) == 64
        assert metadata.orig_n_colors > 0
        assert isinstance(metadata.entropy, float)
        assert metadata.entropy >= 0.0  # Entropy can be 0 for uniform images
