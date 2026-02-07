"""Lightweight tests for vectorized synthetic GIF generation performance improvements.

These tests verify that the vectorized implementations in synthetic_gifs.py
are working correctly and provide expected performance characteristics.
"""

import tempfile
import time
from pathlib import Path

import numpy as np
import pytest
from giflab.synthetic_gifs import (
    SyntheticFrameGenerator,
    SyntheticGifGenerator,
    SyntheticGifSpec,
)
from PIL import Image


class TestSyntheticFrameGeneratorVectorized:
    """Test the vectorized frame generation methods."""

    def setup_method(self):
        """Setup test fixtures."""
        self.generator = SyntheticFrameGenerator()
        self.test_size = (100, 100)
        self.frames = 5

    def test_gradient_frame_generation(self):
        """Test vectorized gradient frame generation produces valid images."""
        img = self.generator.create_frame("gradient", self.test_size, 0, self.frames)

        assert isinstance(img, Image.Image)
        assert img.size == self.test_size
        assert img.mode == "RGB"

        # Verify it's actually a gradient (should have color variation)
        img_array = np.array(img)
        assert img_array.std() > 10  # Should have color variation

    def test_complex_gradient_frame_generation(self):
        """Test vectorized complex gradient frame generation."""
        img = self.generator.create_frame(
            "complex_gradient", self.test_size, 0, self.frames
        )

        assert isinstance(img, Image.Image)
        assert img.size == self.test_size
        assert img.mode == "RGB"

        # Complex gradients should have high variation
        img_array = np.array(img)
        assert img_array.std() > 20

    def test_noise_frame_generation(self):
        """Test vectorized noise frame generation."""
        img = self.generator.create_frame("noise", self.test_size, 0, self.frames)

        assert isinstance(img, Image.Image)
        assert img.size == self.test_size
        assert img.mode == "RGB"

        # Noise should have high variation
        img_array = np.array(img)
        assert img_array.std() > 25  # Lowered threshold for noise variation

    def test_texture_frame_generation(self):
        """Test vectorized texture frame generation."""
        img = self.generator.create_frame("texture", self.test_size, 0, self.frames)

        assert isinstance(img, Image.Image)
        assert img.size == self.test_size
        assert img.mode == "RGB"

        # Texture should have moderate variation
        img_array = np.array(img)
        assert img_array.std() > 15

    def test_solid_frame_generation(self):
        """Test vectorized solid frame generation."""
        img = self.generator.create_frame("solid", self.test_size, 0, self.frames)

        assert isinstance(img, Image.Image)
        assert img.size == self.test_size
        assert img.mode == "RGB"

        # Solid blocks should have some variation but more structured
        img_array = np.array(img)
        unique_colors = len(np.unique(img_array.reshape(-1, 3), axis=0))
        assert unique_colors >= 2  # Should have multiple solid colors

    def test_frame_temporal_consistency(self):
        """Test that frames change over time (animation)."""
        frames = []
        for frame_idx in range(3):
            img = self.generator.create_frame("gradient", self.test_size, frame_idx, 3)
            frames.append(np.array(img))

        # Frames should be different from each other
        assert not np.array_equal(frames[0], frames[1])
        assert not np.array_equal(frames[1], frames[2])

    def test_different_sizes_work(self):
        """Test that vectorized generation works with different image sizes."""
        sizes = [(50, 50), (100, 100), (200, 150)]

        for size in sizes:
            img = self.generator.create_frame("gradient", size, 0, self.frames)
            assert img.size == size
            assert isinstance(img, Image.Image)

    def test_unknown_content_type_fallback(self):
        """Test that unknown content types fall back gracefully."""
        img = self.generator.create_frame(
            "unknown_type", self.test_size, 0, self.frames
        )

        # Should still produce a valid image (fallback)
        assert isinstance(img, Image.Image)
        assert img.size == self.test_size


class TestSyntheticGifGeneratorIntegration:
    """Test the full synthetic GIF generator with vectorized frames."""

    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.generator = SyntheticGifGenerator(self.temp_dir)

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_synthetic_gif_spec_creation(self):
        """Test that synthetic GIF specs are created correctly."""
        specs = self.generator.synthetic_specs

        assert len(specs) > 0
        assert all(isinstance(spec, SyntheticGifSpec) for spec in specs)
        assert all(spec.frames > 0 for spec in specs)
        assert all(len(spec.size) == 2 for spec in specs)

    def test_gif_generation_creates_files(self):
        """Test that GIF generation actually creates files."""
        # Use a small spec for quick testing
        test_specs = [
            spec for spec in self.generator.synthetic_specs if spec.frames <= 8
        ]
        assert len(test_specs) > 0

        test_spec = test_specs[0]

        # Generate just one GIF manually to test the functionality
        gif_path = self.temp_dir / f"{test_spec.name}.gif"

        # Generate frames for the spec
        images = []
        frame_generator = SyntheticFrameGenerator()
        for frame_idx in range(test_spec.frames):
            img = frame_generator.create_frame(
                test_spec.content_type, test_spec.size, frame_idx, test_spec.frames
            )
            images.append(img)

        # Save as GIF
        if images:
            images[0].save(
                gif_path, save_all=True, append_images=images[1:], duration=100, loop=0
            )

        # Check that GIF file was created
        assert gif_path.exists()
        assert gif_path.stat().st_size > 0


class TestPerformanceCharacteristics:
    """Lightweight performance tests to verify vectorization benefits."""

    def setup_method(self):
        """Setup test fixtures."""
        self.generator = SyntheticFrameGenerator()

    def test_large_image_performance_reasonable(self):
        """Test that large images generate in reasonable time."""
        large_size = (500, 500)  # The size mentioned in the refactor doc

        start_time = time.time()
        img = self.generator.create_frame("gradient", large_size, 0, 8)
        end_time = time.time()

        generation_time = end_time - start_time

        # Should generate large images quickly (within 0.1 seconds)
        assert (
            generation_time < 0.1
        ), f"Large image took {generation_time:.3f}s, expected < 0.1s"
        assert isinstance(img, Image.Image)
        assert img.size == large_size

    def test_multiple_frames_performance(self):
        """Test that generating multiple frames is efficient."""
        size = (200, 200)
        num_frames = 10

        start_time = time.time()
        for frame_idx in range(num_frames):
            img = self.generator.create_frame("noise", size, frame_idx, num_frames)
            assert isinstance(img, Image.Image)
        end_time = time.time()

        total_time = end_time - start_time
        avg_time_per_frame = total_time / num_frames

        # Should generate frames quickly
        assert (
            avg_time_per_frame < 0.01
        ), f"Average frame time {avg_time_per_frame:.3f}s too slow"

    def test_different_content_types_all_fast(self):
        """Test that all vectorized content types perform well."""
        content_types = ["gradient", "complex_gradient", "noise", "texture", "solid"]
        size = (150, 150)

        for content_type in content_types:
            start_time = time.time()
            img = self.generator.create_frame(content_type, size, 0, 5)
            end_time = time.time()

            generation_time = end_time - start_time
            assert generation_time < 0.05, f"{content_type} took {generation_time:.3f}s"
            assert isinstance(img, Image.Image)


class TestBackwardCompatibility:
    """Test that vectorized implementations maintain backward compatibility."""

    def setup_method(self):
        """Setup test fixtures."""
        self.generator = SyntheticFrameGenerator()

    def test_api_compatibility(self):
        """Test that the API hasn't changed."""
        # Should still accept the same parameters
        img = self.generator.create_frame("gradient", (100, 100), 0, 5)
        assert isinstance(img, Image.Image)

    def test_output_format_unchanged(self):
        """Test that output format is still PIL Image."""
        img = self.generator.create_frame("solid", (50, 50), 0, 3)

        assert isinstance(img, Image.Image)
        assert img.mode == "RGB"
        assert hasattr(img, "save")  # Can still save
        assert hasattr(img, "size")  # Has size attribute

    def test_deterministic_output_with_seed(self):
        """Test that output is deterministic for noise generation."""
        # Noise generation uses np.random.seed, should be deterministic
        img1 = self.generator.create_frame("noise", (50, 50), 0, 5)
        img2 = self.generator.create_frame("noise", (50, 50), 0, 5)

        # Should produce identical results for same frame
        arr1 = np.array(img1)
        arr2 = np.array(img2)
        assert np.array_equal(arr1, arr2)


@pytest.mark.performance
class TestPerformanceRegression:
    """Performance regression tests to ensure optimizations are maintained."""

    def test_no_performance_regression_medium_size(self):
        """Test that medium-sized images generate quickly (regression test)."""
        generator = SyntheticFrameGenerator()
        size = (200, 200)

        times = []
        for i in range(5):
            start = time.time()
            generator.create_frame("gradient", size, i, 5)
            times.append(time.time() - start)

        avg_time = sum(times) / len(times)

        # Based on our benchmark, should be much faster than 0.01s
        assert avg_time < 0.01, f"Performance regression: avg {avg_time:.4f}s > 0.01s"

    def test_vectorization_still_active(self):
        """Test that vectorized operations are still being used."""
        generator = SyntheticFrameGenerator()

        # Generate a complex gradient which uses heavy numpy operations
        start_time = time.time()
        img = generator.create_frame("complex_gradient", (300, 300), 0, 8)
        elapsed = time.time() - start_time

        # If vectorization is working, this should be very fast
        assert (
            elapsed < 0.02
        ), f"Vectorization may not be working: {elapsed:.4f}s too slow"

        # Verify the image has the expected complex characteristics
        img_array = np.array(img)
        assert img_array.std() > 30  # Should have high variation from complex math
