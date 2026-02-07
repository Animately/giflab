"""SSIM regression tests for gifski frame padding normalization.

These tests verify that frame dimension normalization (padding) doesn't
significantly degrade image quality as measured by SSIM.
"""
import tempfile

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageOps
from skimage.metrics import structural_similarity as ssim


def create_test_frame(size: tuple, pattern: str = "gradient") -> Image.Image:
    """Create a test frame with specified pattern."""
    img = Image.new("RGB", size)
    draw = ImageDraw.Draw(img)

    if pattern == "gradient":
        for x in range(size[0]):
            for y in range(size[1]):
                r = int((x / size[0]) * 255)
                g = int((y / size[1]) * 255)
                b = 128
                draw.point((x, y), (r, g, b))
    elif pattern == "checkerboard":
        for x in range(size[0]):
            for y in range(size[1]):
                if (x // 10) % 2 == (y // 10) % 2:
                    draw.point((x, y), (255, 255, 255))
                else:
                    draw.point((x, y), (0, 0, 0))
    elif pattern == "circles":
        draw.ellipse([10, 10, size[0] - 10, size[1] - 10], fill=(255, 0, 0))
        draw.ellipse(
            [size[0] // 4, size[1] // 4, 3 * size[0] // 4, 3 * size[1] // 4],
            fill=(0, 255, 0),
        )

    return img


def calculate_frame_ssim(img1: Image.Image, img2: Image.Image) -> float:
    """Calculate SSIM between two PIL Images."""
    # Convert to numpy arrays
    arr1 = np.array(img1.convert("RGB"))
    arr2 = np.array(img2.convert("RGB"))

    # Calculate SSIM for each channel and average
    ssim_scores = []
    for channel in range(3):  # RGB channels
        channel_ssim = ssim(arr1[:, :, channel], arr2[:, :, channel], data_range=255)
        ssim_scores.append(channel_ssim)

    return np.mean(ssim_scores)


class TestGifskiPaddingQuality:
    """Test frame padding quality preservation."""

    def test_center_padding_preserves_quality(self):
        """Test that center-aligned padding preserves image quality."""
        # Create original frame
        original = create_test_frame((100, 80), "gradient")

        # Pad to larger size using center alignment (same as gifski normalization)
        padded = ImageOps.pad(
            original, (120, 120), method=Image.LANCZOS, centering=(0.5, 0.5)
        )

        # Crop back to original size from center to simulate the same content area
        left = (padded.width - original.width) // 2
        top = (padded.height - original.height) // 2
        cropped = padded.crop((left, top, left + original.width, top + original.height))

        # Calculate SSIM
        ssim_score = calculate_frame_ssim(original, cropped)

        # Should be very high since we're just padding and cropping back
        assert (
            ssim_score > 0.95
        ), f"SSIM score {ssim_score} indicates quality degradation from padding"

    def test_different_patterns_maintain_quality(self):
        """Test padding quality with different image patterns."""
        patterns = ["gradient", "checkerboard", "circles"]
        min_ssim_scores = []

        for pattern in patterns:
            original = create_test_frame((90, 70), pattern)
            padded = ImageOps.pad(
                original, (100, 100), method=Image.LANCZOS, centering=(0.5, 0.5)
            )

            # Compare original content area
            left = (padded.width - original.width) // 2
            top = (padded.height - original.height) // 2
            cropped = padded.crop(
                (left, top, left + original.width, top + original.height)
            )

            ssim_score = calculate_frame_ssim(original, cropped)
            min_ssim_scores.append(ssim_score)

        # All patterns should maintain reasonable quality
        min_score = min(min_ssim_scores)
        assert (
            min_score > 0.25
        ), f"Minimum SSIM score {min_score} across patterns is too low"

    def test_asymmetric_padding_quality(self):
        """Test padding from asymmetric sizes (common in pipeline processing)."""
        # Test various asymmetric sizes that might occur in pipeline processing
        test_sizes = [
            ((98, 100), (100, 100)),  # Slight width difference
            ((100, 98), (100, 100)),  # Slight height difference
            ((85, 110), (110, 110)),  # More significant asymmetry
            ((120, 95), (120, 120)),  # Height needs padding
        ]

        for original_size, target_size in test_sizes:
            original = create_test_frame(original_size, "gradient")
            padded = ImageOps.pad(
                original, target_size, method=Image.LANCZOS, centering=(0.5, 0.5)
            )

            # Extract original content area from padded image
            left = (padded.width - original.width) // 2
            top = (padded.height - original.height) // 2
            content_area = padded.crop(
                (left, top, left + original.width, top + original.height)
            )

            ssim_score = calculate_frame_ssim(original, content_area)

            # Should maintain very high quality
            assert ssim_score > 0.995, (
                f"SSIM score {ssim_score} too low for padding "
                f"{original_size} -> {target_size}"
            )

    def test_no_padding_needed_unchanged(self):
        """Test that frames with consistent dimensions are unchanged."""
        original = create_test_frame((100, 100), "checkerboard")

        # "Pad" to same size should return identical result
        padded = ImageOps.pad(
            original, (100, 100), method=Image.LANCZOS, centering=(0.5, 0.5)
        )

        # Should be identical
        ssim_score = calculate_frame_ssim(original, padded)
        assert ssim_score == 1.0, "Padding to same size should preserve image exactly"

    def test_padding_preserves_transparency(self):
        """Test that padding works correctly with transparent/RGBA images."""
        # Create RGBA image with transparency
        original = Image.new("RGBA", (80, 60), (255, 0, 0, 128))  # Semi-transparent red
        draw = ImageDraw.Draw(original)
        draw.ellipse([10, 10, 70, 50], fill=(0, 255, 0, 255))  # Opaque green circle

        # Pad with transparent background
        padded = ImageOps.pad(
            original, (100, 100), method=Image.LANCZOS, centering=(0.5, 0.5)
        )

        # Extract content area
        left = (padded.width - original.width) // 2
        top = (padded.height - original.height) // 2
        content_area = padded.crop(
            (left, top, left + original.width, top + original.height)
        )

        # Convert to RGB for SSIM comparison (transparency handling)
        orig_rgb = Image.new("RGB", original.size, (255, 255, 255))
        orig_rgb.paste(original, mask=original)

        content_rgb = Image.new("RGB", content_area.size, (255, 255, 255))
        content_rgb.paste(
            content_area, mask=content_area if content_area.mode == "RGBA" else None
        )

        ssim_score = calculate_frame_ssim(orig_rgb, content_rgb)
        assert (
            ssim_score > 0.40
        ), f"SSIM score {ssim_score} indicates transparency handling issues"


def test_gifski_normalization_integration():
    """Integration test with actual gifski normalization function."""
    import sys
    from pathlib import Path

    # Add src to path for imports
    src_path = Path(__file__).parent.parent / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    try:
        from giflab.external_engines.gifski import _normalize_frame_dimensions
    except ImportError:
        pytest.skip("Could not import gifski module")

    # Create test frames with different dimensions
    with tempfile.TemporaryDirectory() as tmpdir:
        frame_files = []
        original_images = []

        # Create frames with different sizes
        sizes = [(98, 100), (100, 98), (99, 100), (100, 100)]
        for i, size in enumerate(sizes):
            img = create_test_frame(size, "gradient")
            original_images.append(img)

            frame_path = Path(tmpdir) / f"frame_{i:04d}.png"
            img.save(frame_path, format="PNG")
            frame_files.append(str(frame_path))

        # Normalize dimensions
        normalized_files = _normalize_frame_dimensions(frame_files)

        # Verify all normalized frames have same dimensions
        normalized_dimensions = []
        for norm_file in normalized_files:
            with Image.open(norm_file) as img:
                normalized_dimensions.append(img.size)

        # All should have same dimension now
        unique_dims = set(normalized_dimensions)
        assert len(unique_dims) == 1, f"Normalization failed: {unique_dims}"

        # Verify quality is preserved for the content areas
        target_size = normalized_dimensions[0]
        for i, (orig_img, norm_file) in enumerate(
            zip(original_images, normalized_files)
        ):
            with Image.open(norm_file) as norm_img:
                if orig_img.size == target_size:
                    # No padding was needed
                    ssim_score = calculate_frame_ssim(orig_img, norm_img)
                    assert ssim_score == 1.0, f"Frame {i} should be unchanged"
                else:
                    # Padding was applied - check content area
                    left = (target_size[0] - orig_img.size[0]) // 2
                    top = (target_size[1] - orig_img.size[1]) // 2
                    content_area = norm_img.crop(
                        (left, top, left + orig_img.size[0], top + orig_img.size[1])
                    )

                    ssim_score = calculate_frame_ssim(orig_img, content_area)
                    assert (
                        ssim_score > 0.95
                    ), f"Frame {i} quality degraded: SSIM {ssim_score}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
