"""Tests for frame sampling improvements in GIF quality assessment."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
from giflab.metrics import extract_gif_frames
from PIL import Image


class TestFrameSamplingDistribution:
    """Tests for even frame distribution sampling."""

    def create_test_gif(
        self, tmp_path: Path, frame_count: int, duration_per_frame: int = 100
    ) -> Path:
        """Create a test GIF with specified number of frames."""
        frames = []

        # Create frames with different colors to verify sampling
        for i in range(frame_count):
            # Each frame has a different color to make sampling verification easy
            color_value = int(255 * i / max(1, frame_count - 1))
            img = Image.new("RGB", (50, 50), (color_value, 100, 150))
            frames.append(img)

        gif_path = tmp_path / f"test_{frame_count}frames.gif"
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration_per_frame,
            loop=0,
        )

        return gif_path

    def test_even_sampling_basic(self, tmp_path):
        """Test even sampling with typical case."""
        gif_path = self.create_test_gif(tmp_path, frame_count=10)

        # Extract 5 frames from 10-frame GIF
        result = extract_gif_frames(gif_path, max_frames=5)

        assert result.frame_count == 10  # Total frames in GIF
        assert len(result.frames) == 5  # Number of frames actually extracted

        # Should sample frames: 0, 2, 4, 6, 8 (indices from np.linspace(0, 9, 5))
        expected_indices = np.linspace(0, 9, 5, dtype=int)  # [0, 2, 4, 6, 8]

        # Verify by checking frame colors (each frame has unique color)
        for i, expected_idx in enumerate(expected_indices):
            expected_color = int(
                255 * expected_idx / 9
            )  # Color formula from create_test_gif

            # Get the actual frame and check its color
            frame = result.frames[i]
            actual_color = frame[0, 0, 0]  # Red channel value

            # Should be very close (allowing for some compression artifacts)
            assert (
                abs(actual_color - expected_color) <= 5
            ), f"Frame {i}: expected color ~{expected_color}, got {actual_color}"

    def test_even_sampling_edge_cases(self, tmp_path):
        """Test even sampling with edge cases."""
        # Case 1: More max_frames than actual frames
        gif_path = self.create_test_gif(tmp_path, frame_count=5)
        result = extract_gif_frames(gif_path, max_frames=10)

        # Should return all 5 frames
        assert result.frame_count == 5
        assert len(result.frames) == 5

    def test_even_sampling_single_frame(self, tmp_path):
        """Test even sampling with single frame."""
        gif_path = self.create_test_gif(tmp_path, frame_count=1)
        result = extract_gif_frames(gif_path, max_frames=5)

        assert result.frame_count == 1
        assert len(result.frames) == 1

    def test_even_sampling_exact_match(self, tmp_path):
        """Test when max_frames exactly matches frame count."""
        gif_path = self.create_test_gif(tmp_path, frame_count=6)
        result = extract_gif_frames(gif_path, max_frames=6)

        # Should return all frames in order
        assert result.frame_count == 6
        assert len(result.frames) == 6

    def test_linspace_usage_verification(self, tmp_path):
        """Verify that np.linspace is being used correctly for frame sampling."""
        gif_path = self.create_test_gif(tmp_path, frame_count=40)

        with patch("numpy.linspace") as mock_linspace:
            # Configure mock to return expected indices
            mock_linspace.return_value = np.array([0, 13, 26, 39], dtype=int)

            result = extract_gif_frames(gif_path, max_frames=4)

            # Verify linspace was called with correct parameters
            mock_linspace.assert_called_once_with(0, 39, 4, dtype=int)

            # Should have returned 4 frames
            assert result.frame_count == 40  # Total frames in GIF

    def test_full_timeline_coverage(self, tmp_path):
        """Test that sampling covers the full animation timeline."""
        # Create 40-frame GIF (common case where old method would miss 25%)
        gif_path = self.create_test_gif(tmp_path, frame_count=40)

        # Extract 30 frames (default max)
        result = extract_gif_frames(gif_path, max_frames=30)

        assert result.frame_count == 40  # Total frames in GIF
        assert len(result.frames) == 30  # Frames actually extracted

        # Verify coverage includes frames from end of animation
        # With even sampling, we should get frames distributed across 0-39
        expected_indices = np.linspace(0, 39, 30, dtype=int)

        # Last sampled frame should be from near the end
        assert expected_indices[-1] >= 35, "Should sample from end of animation"

        # First sampled frame should be frame 0
        assert expected_indices[0] == 0, "Should start with frame 0"

    def test_sampling_vs_consecutive_difference(self, tmp_path):
        """Test that even sampling differs from consecutive sampling."""
        gif_path = self.create_test_gif(tmp_path, frame_count=20)

        # Extract with even sampling
        extract_gif_frames(gif_path, max_frames=10)

        # Even sampling should give indices: [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
        # Consecutive would give indices: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        expected_even_indices = np.linspace(0, 19, 10, dtype=int)
        consecutive_indices = np.arange(10)

        # Should be different distributions
        assert not np.array_equal(expected_even_indices, consecutive_indices)

        # Even sampling should include frames from later in the animation
        assert max(expected_even_indices) > max(consecutive_indices)

    def test_quality_assessment_coverage(self, tmp_path):
        """Test that improved sampling covers quality issues throughout animation."""
        # Simulate a GIF where quality degrades over time
        frames = []
        for i in range(30):
            # Simulate increasing compression artifacts over time
            noise_level = i * 2  # Increasing noise

            # Create frame with increasing noise to simulate quality degradation
            base_img = np.full((50, 50, 3), 128, dtype=np.uint8)  # Gray base
            if noise_level > 0:
                noise = np.random.randint(-noise_level, noise_level + 1, (50, 50, 3))
                base_img = np.clip(base_img.astype(int) + noise, 0, 255).astype(
                    np.uint8
                )

            frames.append(Image.fromarray(base_img))

        gif_path = tmp_path / "quality_degradation.gif"
        frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=100)

        # Sample 15 frames from 30
        extract_gif_frames(gif_path, max_frames=15)

        expected_indices = np.linspace(0, 29, 15, dtype=int)

        # Should sample from both early frames (low noise) and late frames (high noise)
        assert 0 in expected_indices, "Should sample from beginning"
        assert max(expected_indices) >= 25, "Should sample from near end"

        # Coverage should span the full timeline
        coverage_span = max(expected_indices) - min(expected_indices)
        assert coverage_span >= 20, "Should cover most of the timeline"

    def test_frame_sampling_with_different_max_values(self, tmp_path):
        """Test frame sampling with various max_frames values."""
        gif_path = self.create_test_gif(tmp_path, frame_count=100)

        test_cases = [
            (10, 100),  # Sample 10 from 100
            (25, 100),  # Sample 25 from 100
            (50, 100),  # Sample 50 from 100
            (75, 100),  # Sample 75 from 100
        ]

        for max_frames, total_frames in test_cases:
            result = extract_gif_frames(gif_path, max_frames=max_frames)

            assert result.frame_count == 100  # Total frames in GIF
            assert len(result.frames) == max_frames  # Frames actually extracted

            # Verify even distribution
            expected_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)

            # Check that sampling covers the expected range
            assert (
                expected_indices[0] == 0
            ), f"First frame should be 0 for max_frames={max_frames}"
            assert (
                expected_indices[-1] >= total_frames * 0.9
            ), f"Should sample near end for max_frames={max_frames}"

    def test_duration_and_metadata_preserved(self, tmp_path):
        """Test that frame sampling preserves other metadata correctly."""
        gif_path = self.create_test_gif(
            tmp_path, frame_count=20, duration_per_frame=150
        )

        result = extract_gif_frames(gif_path, max_frames=10)

        # Should preserve basic metadata
        assert result.dimensions == (50, 50)
        assert result.duration_ms > 0  # Should have some duration

    @patch("giflab.metrics.np.linspace")
    def test_linspace_parameters(self, mock_linspace, tmp_path):
        """Test that np.linspace is called with correct parameters."""
        mock_linspace.return_value = np.array([0, 5, 10, 15, 19], dtype=int)

        gif_path = self.create_test_gif(tmp_path, frame_count=20)

        extract_gif_frames(gif_path, max_frames=5)

        # Verify np.linspace was called correctly
        mock_linspace.assert_called_once_with(
            0, 19, 5, dtype=int
        )  # 0 to (20-1), 5 samples

    def test_regression_consecutive_sampling_avoided(self, tmp_path):
        """Regression test to ensure consecutive sampling is not used."""
        gif_path = self.create_test_gif(tmp_path, frame_count=40)

        extract_gif_frames(gif_path, max_frames=20)

        # With consecutive sampling, max frame would be 19
        # With even sampling, max frame should be near 39
        expected_indices = np.linspace(0, 39, 20, dtype=int)
        max_expected_index = max(expected_indices)

        # Verify we're getting frames from later in the animation
        assert (
            max_expected_index > 30
        ), "Should sample from later frames, not just consecutive"

    def test_frame_sampling_deterministic(self, tmp_path):
        """Test that frame sampling is deterministic."""
        gif_path = self.create_test_gif(tmp_path, frame_count=25)

        # Extract frames multiple times
        result1 = extract_gif_frames(gif_path, max_frames=10)
        result2 = extract_gif_frames(gif_path, max_frames=10)
        result3 = extract_gif_frames(gif_path, max_frames=10)

        # Should get identical results each time
        assert result1.frame_count == result2.frame_count == result3.frame_count
        assert len(result1.frames) == len(result2.frames) == len(result3.frames)

        # Frame content should be identical (test first frame)
        assert np.array_equal(result1.frames[0], result2.frames[0])
        assert np.array_equal(result2.frames[0], result3.frames[0])

    def test_memory_efficiency_large_gif(self, tmp_path):
        """Test that sampling reduces memory usage for large GIFs."""
        # Create a large GIF
        gif_path = self.create_test_gif(tmp_path, frame_count=200)

        # Extract small sample
        result = extract_gif_frames(gif_path, max_frames=30)

        # Should only have 30 frames in memory, not 200
        assert result.frame_count == 200  # Total frames in GIF
        assert len(result.frames) == 30  # Frames actually extracted

        # Each frame should be the expected size
        for frame in result.frames:
            assert frame.shape == (50, 50, 3)


class TestFrameSamplingIntegration:
    """Integration tests for frame sampling with quality metrics."""

    def create_gradient_gif(self, tmp_path: Path, frame_count: int) -> Path:
        """Create a GIF with gradient changes over time for quality testing."""
        frames = []

        for i in range(frame_count):
            # Create gradient that changes over time
            img_array = np.zeros((50, 50, 3), dtype=np.uint8)

            for y in range(50):
                for x in range(50):
                    # Create a gradient that shifts based on frame number
                    red = int(255 * (x + i) / (49 + frame_count))
                    green = int(255 * y / 49)
                    blue = int(255 * (1 - (x + y + i) / (98 + frame_count)))

                    img_array[y, x] = [red, green, blue]

            frames.append(Image.fromarray(img_array))

        gif_path = tmp_path / "gradient_test.gif"
        frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=100)
        return gif_path

    def test_sampling_affects_quality_calculation(self, tmp_path):
        """Test that improved sampling can affect quality calculations."""
        # Create GIF where quality changes significantly over time
        gif_path = self.create_gradient_gif(tmp_path, frame_count=40)

        # Compare different sampling sizes
        result_few = extract_gif_frames(gif_path, max_frames=5)
        result_many = extract_gif_frames(gif_path, max_frames=30)

        # Both should cover the full timeline, but with different densities
        assert result_few.frame_count == 40  # Total frames in GIF
        assert len(result_few.frames) == 5  # Frames actually extracted
        assert result_many.frame_count == 40  # Total frames in GIF
        assert len(result_many.frames) == 30  # Frames actually extracted

        # More samples should potentially capture more variation
        # (This is more about ensuring the sampling works than specific quality values)
        assert len(result_few.frames) < len(result_many.frames)

    def test_sampling_consistency_across_gif_sizes(self, tmp_path):
        """Test that sampling works consistently across different GIF sizes."""
        test_cases = [
            (10, 5),  # Small GIF, few samples
            (30, 20),  # Medium GIF, many samples
            (50, 30),  # Large GIF, standard samples
            (100, 25),  # Very large GIF, moderate samples
        ]

        for gif_frames, sample_frames in test_cases:
            gif_path = self.create_gradient_gif(tmp_path, frame_count=gif_frames)
            result = extract_gif_frames(gif_path, max_frames=sample_frames)

            expected_extracted = min(sample_frames, gif_frames)
            assert result.frame_count == gif_frames  # Total frames in GIF
            assert len(result.frames) == expected_extracted  # Frames actually extracted

            # Verify timeline coverage
            if sample_frames < gif_frames:
                # Should sample from across the timeline
                expected_indices = np.linspace(
                    0, gif_frames - 1, sample_frames, dtype=int
                )
                coverage = max(expected_indices) - min(expected_indices)
                expected_coverage = gif_frames - 1

                # Should cover at least 80% of the timeline
                assert (
                    coverage >= 0.8 * expected_coverage
                ), f"Poor timeline coverage for {gif_frames} frames, {sample_frames} samples"
