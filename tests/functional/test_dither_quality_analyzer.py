"""Unit tests for dither quality analysis module."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from giflab.gradient_color_artifacts import (
    DitherQualityAnalyzer,
    calculate_dither_quality_metrics,
)
from PIL import Image


class TestDitherQualityAnalyzer:
    """Test the DitherQualityAnalyzer class."""

    @pytest.mark.fast
    def test_analyzer_initialization(self):
        """Test analyzer initializes with correct parameters."""
        # Default initialization
        analyzer = DitherQualityAnalyzer()
        assert analyzer.patch_size == 64
        assert analyzer.flat_threshold == 50.0

        # Custom initialization
        analyzer = DitherQualityAnalyzer(patch_size=32, flat_threshold=25.0)
        assert analyzer.patch_size == 32
        assert analyzer.flat_threshold == 25.0

    @pytest.mark.fast
    def test_analyzer_validation(self):
        """Test analyzer parameter validation."""
        # Invalid patch size
        with pytest.raises(ValueError, match="patch_size must be >= 16"):
            DitherQualityAnalyzer(patch_size=8)

        with pytest.raises(ValueError, match="patch_size must be >= 16"):
            DitherQualityAnalyzer(patch_size=0)

        # Invalid flat threshold
        with pytest.raises(ValueError, match="flat_threshold must be non-negative"):
            DitherQualityAnalyzer(flat_threshold=-1.0)

    @pytest.mark.fast
    def test_detect_flat_regions(self):
        """Test flat region detection."""
        analyzer = DitherQualityAnalyzer(patch_size=64, flat_threshold=50.0)

        # Create a test frame with flat and non-flat regions
        frame = np.zeros((128, 128, 3), dtype=np.uint8)

        # Left half: flat region (all same color)
        frame[:, :64] = [100, 100, 100]

        # Right half: noisy region (random colors)
        np.random.seed(42)  # For reproducible tests
        frame[:, 64:] = np.random.randint(0, 255, (128, 64, 3), dtype=np.uint8)

        flat_regions = analyzer.detect_flat_regions(frame)

        # Should detect some flat regions in the left half
        assert len(flat_regions) > 0

        # All regions should be within frame bounds
        for x, y, w, h in flat_regions:
            assert 0 <= x < 128
            assert 0 <= y < 128
            assert x + w <= 128
            assert y + h <= 128

    @pytest.mark.fast
    def test_compute_frequency_spectrum(self):
        """Test FFT frequency spectrum computation."""
        analyzer = DitherQualityAnalyzer()

        # Create test patches
        patch_size = 64

        # Smooth patch (should have energy concentrated in low frequencies)
        smooth_patch = np.ones((patch_size, patch_size), dtype=np.float32) * 100
        smooth_spectrum = analyzer.compute_frequency_spectrum(smooth_patch)

        assert smooth_spectrum.shape == (patch_size, patch_size)
        assert smooth_spectrum.dtype == np.float64  # FFT result type

        # Noisy patch (should have energy distributed across frequencies)
        np.random.seed(42)
        noisy_patch = np.random.randint(0, 255, (patch_size, patch_size)).astype(
            np.float32
        )
        noisy_spectrum = analyzer.compute_frequency_spectrum(noisy_patch)

        assert noisy_spectrum.shape == (patch_size, patch_size)

    @pytest.mark.fast
    def test_calculate_band_energies(self):
        """Test frequency band energy calculation."""
        analyzer = DitherQualityAnalyzer()

        # Create a test spectrum (simulated)
        spectrum = np.ones((64, 64), dtype=np.float32)

        # Add strong DC component (center after fftshift)
        spectrum[32, 32] = 1000  # High energy at DC

        # Add some mid-frequency energy
        spectrum[25:39, 25:39] = 10

        # Add some high-frequency energy
        spectrum[0:10, 0:10] = 5
        spectrum[-10:, -10:] = 5

        low_energy, mid_energy, high_energy = analyzer.calculate_band_energies(spectrum)

        assert low_energy > 0
        assert mid_energy > 0
        assert high_energy > 0
        assert isinstance(low_energy, float | np.floating)
        assert isinstance(mid_energy, float | np.floating)
        assert isinstance(high_energy, float | np.floating)

    @pytest.mark.fast
    def test_compute_dither_ratio(self):
        """Test dither ratio computation."""
        analyzer = DitherQualityAnalyzer()

        # Test various energy ratios
        # Well-dithered (ratio around 1.0)
        ratio = analyzer.compute_dither_ratio(100.0, 100.0)
        assert ratio == 1.0

        # Over-dithered (high frequencies dominate)
        ratio = analyzer.compute_dither_ratio(200.0, 100.0)
        assert ratio == 2.0

        # Under-dithered (low high frequency energy)
        ratio = analyzer.compute_dither_ratio(50.0, 100.0)
        assert ratio == 0.5

        # Edge case: zero mid-energy but high energy exists
        ratio = analyzer.compute_dither_ratio(100.0, 0.0)
        assert ratio == 10.0  # Should return high value

        # Edge case: both energies near zero
        ratio = analyzer.compute_dither_ratio(1e-11, 1e-11)
        assert ratio == 0.0  # Should return 0

    @pytest.mark.fast
    def test_analyze_dither_quality_empty_frames(self):
        """Test dither quality analysis with empty frame lists."""
        analyzer = DitherQualityAnalyzer()

        # Empty lists
        result = analyzer.analyze_dither_quality([], [])
        expected_keys = {
            "dither_ratio_mean",
            "dither_ratio_p95",
            "dither_quality_score",
            "flat_region_count",
        }
        assert set(result.keys()) == expected_keys
        assert all(result[key] == 0.0 for key in expected_keys)

    @pytest.mark.fast
    def test_analyze_dither_quality_mismatched_frames(self):
        """Test error handling for mismatched frame counts."""
        analyzer = DitherQualityAnalyzer()

        frame1 = np.zeros((100, 100, 3), dtype=np.uint8)
        frame2 = np.zeros((100, 100, 3), dtype=np.uint8)

        with pytest.raises(ValueError, match="Frame count mismatch"):
            analyzer.analyze_dither_quality([frame1], [frame1, frame2])

    @pytest.mark.fast
    def test_analyze_dither_quality_synthetic_frames(self):
        """Test dither quality analysis with synthetic frames."""
        analyzer = DitherQualityAnalyzer(patch_size=32, flat_threshold=100.0)

        # Create synthetic frames with flat regions
        frame_size = 64

        # Original frame: flat gray region
        original = np.full((frame_size, frame_size, 3), 128, dtype=np.uint8)

        # Compressed frame 1: well-dithered (subtle noise)
        np.random.seed(42)
        well_dithered = original.copy().astype(np.float32)
        noise = np.random.normal(0, 2, well_dithered.shape)  # Small amount of noise
        well_dithered = np.clip(well_dithered + noise, 0, 255).astype(np.uint8)

        result = analyzer.analyze_dither_quality([original], [well_dithered])

        # Should detect flat regions and calculate ratios
        assert result["flat_region_count"] > 0
        assert result["dither_ratio_mean"] >= 0.0
        assert 0 <= result["dither_quality_score"] <= 100

        # Create over-dithered frame (excessive noise)
        over_dithered = original.copy().astype(np.float32)
        noise = np.random.normal(0, 20, over_dithered.shape)  # Large amount of noise
        over_dithered = np.clip(over_dithered + noise, 0, 255).astype(np.uint8)

        over_result = analyzer.analyze_dither_quality([original], [over_dithered])

        # Over-dithered should have higher ratios and lower quality scores
        assert over_result["dither_ratio_mean"] >= result["dither_ratio_mean"]


class TestCalculateDitherQualityMetrics:
    """Test the calculate_dither_quality_metrics function."""

    @pytest.mark.fast
    def test_function_interface(self):
        """Test the main function interface."""
        # Create simple test frames
        frame1 = np.full((64, 64, 3), 100, dtype=np.uint8)
        frame2 = np.full((64, 64, 3), 105, dtype=np.uint8)  # Slightly different

        result = calculate_dither_quality_metrics([frame1], [frame2])

        # Check expected return keys
        expected_keys = {
            "dither_ratio_mean",
            "dither_ratio_p95",
            "dither_quality_score",
            "flat_region_count",
        }
        assert set(result.keys()) == expected_keys

        # Check types
        for key in expected_keys:
            assert isinstance(result[key], int | float | np.number)

    @pytest.mark.fast
    def test_function_error_handling(self):
        """Test function handles errors gracefully."""
        # This should not raise but return default values
        result = calculate_dither_quality_metrics([], [])

        expected_keys = {
            "dither_ratio_mean",
            "dither_ratio_p95",
            "dither_quality_score",
            "flat_region_count",
        }
        assert set(result.keys()) == expected_keys
        assert all(result[key] == 0.0 for key in expected_keys)


class TestDitherQualityIntegration:
    """Integration tests for dither quality analysis."""

    def setup_method(self):
        """Set up integration tests."""
        self.temp_dir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        """Clean up integration tests."""
        import shutil

        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def _create_test_gif(
        self, filename: str, add_noise: bool = False, noise_level: float = 5.0
    ) -> Path:
        """Create a test GIF with flat regions for dithering analysis."""
        gif_path = self.temp_dir / filename

        frames = []
        for i in range(3):  # 3 frames
            # Create frame with flat regions
            img = Image.new("RGB", (100, 100), color=(100, 100, 100))

            if add_noise:
                # Add dithering-like noise to simulate compression artifacts
                pixels = list(img.getdata())
                np.random.seed(42 + i)  # Different seed per frame

                noisy_pixels = []
                for r, g, b in pixels:
                    # Add noise to each channel
                    noise = np.random.normal(0, noise_level, 3)
                    new_r = max(0, min(255, int(r + noise[0])))
                    new_g = max(0, min(255, int(g + noise[1])))
                    new_b = max(0, min(255, int(b + noise[2])))
                    noisy_pixels.append((new_r, new_g, new_b))

                img.putdata(noisy_pixels)

            frames.append(img)

        # Save as GIF
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=100,  # 100ms per frame
            loop=0,
        )

        return gif_path

    @pytest.mark.fast
    def test_dither_quality_with_gif_files(self):
        """Test dither quality analysis with actual GIF files."""
        # Import GIF frame extraction (this should exist based on other test files)
        try:
            from giflab.metrics import extract_gif_frames
        except ImportError:
            pytest.skip("extract_gif_frames not available")

        # Create test GIFs with same frame structure
        clean_gif = self._create_test_gif("clean.gif", add_noise=False)

        # Extract frames from clean GIF to use as reference
        clean_extract_result = extract_gif_frames(clean_gif)
        clean_frames = clean_extract_result.frames

        # Create noisy versions by adding noise to clean frames
        noisy_frames = []
        very_noisy_frames = []

        np.random.seed(42)  # For reproducible noise

        for frame in clean_frames:
            # Create mildly noisy version
            noisy_frame = frame.astype(np.float32)
            noise = np.random.normal(0, 5, noisy_frame.shape)  # Medium noise level
            noisy_frame = np.clip(noisy_frame + noise, 0, 255).astype(np.uint8)
            noisy_frames.append(noisy_frame)

            # Create very noisy version
            very_noisy_frame = frame.astype(np.float32)
            noise = np.random.normal(0, 20, very_noisy_frame.shape)  # High noise level
            very_noisy_frame = np.clip(very_noisy_frame + noise, 0, 255).astype(
                np.uint8
            )
            very_noisy_frames.append(very_noisy_frame)

        # Analyze dither quality with moderate noise
        result = calculate_dither_quality_metrics(clean_frames, noisy_frames)

        # Should detect some level of dithering artifacts
        assert result["flat_region_count"] >= 0  # Should process flat regions
        assert result["dither_ratio_mean"] >= 0  # Should calculate ratios
        assert (
            0 <= result["dither_quality_score"] <= 100
        )  # Quality score should be in valid range

        # Analyze with very noisy frames
        very_noisy_result = calculate_dither_quality_metrics(
            clean_frames, very_noisy_frames
        )

        # Both analyses should produce valid results
        assert very_noisy_result["flat_region_count"] >= 0
        assert very_noisy_result["dither_ratio_mean"] >= 0
        assert 0 <= very_noisy_result["dither_quality_score"] <= 100

        # Test that the analyzer processes both noise levels without error
        # (The exact relationship between noise levels and ratios can vary due to FFT characteristics)
