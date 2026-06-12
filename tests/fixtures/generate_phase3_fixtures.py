#!/usr/bin/env python3
"""Generate test fixtures for Phase 3 conditional content-specific metrics.

This module creates synthetic test images and GIFs for validating:
1. Text/UI Content Validation (Task 3.1)
2. SSIMULACRA2 Modern Perceptual Metric (Task 3.2)

These fixtures provide controlled test cases for edge detection, component analysis,
OCR validation, edge acuity measurement, and SSIMULACRA2 score ranges.
"""

import math
import random
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class Phase3FixtureGenerator:
    """Generator for Phase 3 test fixtures."""

    def __init__(self, output_dir: Path | None = None):
        """Initialize fixture generator.

        Args:
            output_dir: Directory to save fixtures. Defaults to tests/fixtures/phase3
        """
        if output_dir is None:
            output_dir = Path(__file__).parent / "phase3"

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    # Text/UI Content Generation Methods

    def create_text_ui_image(
        self,
        image_type: str,
        size: tuple[int, int] = (200, 200),
        text_content: str = "Sample Text",
    ) -> Path:
        """Create synthetic text/UI content image.

        Args:
            image_type: Type of UI element ('clean_text', 'blurry_text', 'ui_buttons',
                       'terminal_text', 'mixed_content', 'no_text')
            size: Image dimensions as (width, height)
            text_content: Text to render for text-based types

        Returns:
            Path to created image file
        """
        width, height = size
        img = Image.new("RGB", size, (255, 255, 255))
        draw = ImageDraw.Draw(img)

        filename = f"text_ui_{image_type}_{width}x{height}.png"
        filepath = self.output_dir / filename

        if image_type == "clean_text":
            # Sharp, clear text - should have high OCR confidence
            try:
                # Try to use a standard font
                font = ImageFont.truetype("Arial", 16)
            except OSError:
                # Fall back to default font
                font = ImageFont.load_default()

            # Add multiple lines of text
            lines = [text_content, "Line 2: ABCD1234", "Line 3: !@#$%^&*"]
            y_offset = 20
            for line in lines:
                draw.text((10, y_offset), line, fill=(0, 0, 0), font=font)
                y_offset += 25

        elif image_type == "blurry_text":
            # Blurred text - should have lower OCR confidence
            try:
                font = ImageFont.truetype("Arial", 16)
            except OSError:
                font = ImageFont.load_default()

            # Draw text then apply blur
            draw.text((10, 20), text_content, fill=(0, 0, 0), font=font)
            draw.text((10, 50), "Blurred Text Test", fill=(0, 0, 0), font=font)

            # Convert to numpy for blur, then back to PIL
            img_array = np.array(img)
            blurred = cv2.GaussianBlur(img_array, (5, 5), 2.0)
            img = Image.fromarray(blurred)

        elif image_type == "ui_buttons":
            # UI elements with button-like rectangles
            colors = [(100, 150, 200), (200, 100, 100), (100, 200, 100)]

            for i, color in enumerate(colors):
                x = 20 + i * 60
                y = 20 + i * 30
                # Button rectangle
                draw.rectangle([x, y, x + 50, y + 25], fill=color, outline=(0, 0, 0))
                # Button text
                draw.text((x + 5, y + 5), f"BTN{i+1}", fill=(255, 255, 255))

        elif image_type == "terminal_text":
            # Terminal/console style text
            img = Image.new("RGB", size, (0, 0, 0))  # Black background
            draw = ImageDraw.Draw(img)

            # Use Pillow's bundled default font, never a system font: a system
            # lookup (e.g. Courier) succeeds on macOS but raises OSError on CI
            # runners, silently swapping in a differently-sized font and
            # changing detected component counts per platform.
            try:
                font = ImageFont.load_default(size=12)
            except TypeError:  # Pillow < 10.1 lacks the size parameter
                font = ImageFont.load_default()

            # Multi-word lines spaced >10px apart (the region-merge proximity
            # threshold in TextUIContentDetector) so each line survives as a
            # distinct detected text region rather than merging with its
            # neighbours; long single-word runs would instead be dropped by
            # the component aspect-ratio filter.
            terminal_lines = [
                "$ ls -la docs src tests",
                "total 64 files 12 dirs",
                "drwxr-x user staff 384",
                "$ make test ... all ok",
            ]

            y_offset = 4
            for line in terminal_lines:
                draw.text((5, y_offset), line, fill=(0, 255, 0), font=font)
                y_offset += 32

        elif image_type == "mixed_content":
            # Mix of text and graphics
            # Background gradient
            for y in range(height):
                color_val = int(255 * y / height)
                draw.line([(0, y), (width, y)], fill=(color_val, color_val, 255))

            # Add some text over the gradient
            draw.text((20, 30), "Mixed Content", fill=(0, 0, 0))
            draw.text((20, 50), "Text + Graphics", fill=(255, 255, 255))

            # Add some geometric shapes
            draw.ellipse([100, 80, 150, 130], fill=(255, 100, 100))
            draw.rectangle([20, 100, 80, 150], fill=(100, 255, 100))

        elif image_type == "no_text":
            # Pure graphics/patterns - no text content
            # Create a geometric pattern
            for x in range(0, width, 20):
                for y in range(0, height, 20):
                    color_r = int(255 * x / width)
                    color_g = int(255 * y / height)
                    color_b = int(255 * (x + y) / (width + height))
                    draw.rectangle(
                        [x, y, x + 18, y + 18], fill=(color_r, color_g, color_b)
                    )

        img.save(filepath)
        return filepath

    def create_edge_density_image(
        self, edge_density: str, size: tuple[int, int] = (100, 100)
    ) -> Path:
        """Create image with specific edge density characteristics.

        Args:
            edge_density: 'none', 'low', 'medium', 'high', 'extreme'
            size: Image dimensions

        Returns:
            Path to created image file
        """
        width, height = size
        img = Image.new("RGB", size, (128, 128, 128))
        draw = ImageDraw.Draw(img)

        filename = f"edge_density_{edge_density}_{width}x{height}.png"
        filepath = self.output_dir / filename

        if edge_density == "none":
            # Solid color - no edges
            pass  # Already created as solid gray

        elif edge_density == "low":
            # Few, simple edges
            draw.rectangle([20, 20, 80, 80], outline=(0, 0, 0), width=2)
            draw.line([(10, 50), (90, 50)], fill=(255, 255, 255), width=1)

        elif edge_density == "medium":
            # Moderate number of edges
            for i in range(5):
                x = 10 + i * 15
                draw.rectangle(
                    [x, 20, x + 10, 80], outline=(0, 0, 0), fill=(255, 255, 255)
                )

        elif edge_density == "high":
            # Many edges - text-like
            for i in range(8):
                for j in range(8):
                    x, y = 5 + i * 11, 5 + j * 11
                    if (i + j) % 2 == 0:
                        draw.rectangle(
                            [x, y, x + 8, y + 8],
                            fill=(0, 0, 0),
                            outline=(255, 255, 255),
                        )

        elif edge_density == "extreme":
            # Very high edge density - noise-like
            img_array = np.array(img)
            noise = np.random.randint(-50, 51, img_array.shape, dtype=np.int16)
            noisy = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            img = Image.fromarray(noisy)

        img.save(filepath)
        return filepath

    def create_component_test_image(
        self, component_type: str, size: tuple[int, int] = (150, 150)
    ) -> Path:
        """Create image for testing connected component analysis.

        Args:
            component_type: 'no_components', 'single_component', 'text_like',
                           'too_small', 'too_large', 'wrong_aspect'
            size: Image dimensions

        Returns:
            Path to created image file
        """
        width, height = size
        img = Image.new("RGB", size, (255, 255, 255))
        draw = ImageDraw.Draw(img)

        filename = f"components_{component_type}_{width}x{height}.png"
        filepath = self.output_dir / filename

        if component_type == "no_components":
            # Solid background - no components
            pass

        elif component_type == "single_component":
            # Single text-like rectangular component
            draw.rectangle([50, 60, 100, 80], fill=(0, 0, 0))

        elif component_type == "text_like":
            # Multiple text-like components with appropriate aspect ratios
            components = [
                (20, 30, 60, 40),  # Word 1
                (70, 30, 120, 40),  # Word 2
                (20, 50, 80, 60),  # Longer word
                (90, 50, 130, 60),  # Short word
            ]

            for x1, y1, x2, y2 in components:
                draw.rectangle([x1, y1, x2, y2], fill=(0, 0, 0))

        elif component_type == "too_small":
            # Components smaller than min_component_area (default 10 pixels)
            for i in range(10):
                x, y = 10 + i * 12, 50
                draw.rectangle([x, y, x + 2, y + 2], fill=(0, 0, 0))

        elif component_type == "too_large":
            # Component larger than max_component_area (default 500 pixels)
            # Create a large filled rectangle (75x75 = 5625 pixels > 500)
            draw.rectangle([20, 20, 95, 95], fill=(0, 0, 0))

        elif component_type == "wrong_aspect":
            # Components with wrong aspect ratios for text
            # Very tall, thin rectangles
            draw.rectangle([30, 20, 35, 100], fill=(0, 0, 0))
            draw.rectangle([50, 20, 55, 100], fill=(0, 0, 0))
            # Very wide, short rectangles
            draw.rectangle([80, 60, 140, 65], fill=(0, 0, 0))

        img.save(filepath)
        return filepath

    # SSIMULACRA2 Test Image Generation

    def create_ssimulacra2_test_pair(
        self, quality_level: str, size: tuple[int, int] = (128, 128)
    ) -> tuple[Path, Path]:
        """Create image pair with known SSIMULACRA2 quality characteristics.

        Args:
            quality_level: 'excellent', 'good', 'medium', 'poor', 'terrible'
            size: Image dimensions

        Returns:
            Tuple of (original_path, degraded_path)
        """
        width, height = size

        # Create original image with complex content
        original = self._create_complex_test_image(size)

        # Create degraded version based on quality level
        degraded = self._apply_degradation(original, quality_level)

        # Save both images
        orig_filename = f"ssim2_original_{quality_level}_{width}x{height}.png"
        deg_filename = f"ssim2_degraded_{quality_level}_{width}x{height}.png"

        orig_path = self.output_dir / orig_filename
        deg_path = self.output_dir / deg_filename

        original.save(orig_path)
        degraded.save(deg_path)

        return orig_path, deg_path

    def _create_complex_test_image(self, size: tuple[int, int]) -> Image.Image:
        """Create complex test image with various visual features."""
        width, height = size
        img = Image.new("RGB", size, (200, 200, 200))
        draw = ImageDraw.Draw(img)

        # Add gradient background
        for y in range(height):
            color_val = int(100 + 155 * y / height)
            draw.line([(0, y), (width, y)], fill=(color_val, color_val, 255))

        # Add geometric shapes with different textures
        draw.ellipse(
            [width // 4, height // 4, 3 * width // 4, 3 * height // 4],
            fill=(255, 100, 100),
            outline=(0, 0, 0),
        )

        # Add some fine details
        for i in range(0, width, 8):
            draw.line([(i, 0), (i, height // 6)], fill=(0, 0, 0))

        # Add text if possible
        try:
            font = ImageFont.load_default()
            draw.text((10, 10), "Test Image", fill=(0, 0, 0), font=font)
        except Exception:
            pass

        return img

    def _apply_degradation(self, img: Image.Image, quality_level: str) -> Image.Image:
        """Apply different levels of degradation to create test pairs."""
        img_array = np.array(img)

        if quality_level == "excellent":
            # Minimal degradation - barely noticeable
            noise = np.random.normal(0, 2, img_array.shape).astype(np.int16)

        elif quality_level == "good":
            # Light degradation
            noise = np.random.normal(0, 5, img_array.shape).astype(np.int16)
            # Light blur
            img_array = cv2.GaussianBlur(img_array, (3, 3), 0.5)

        elif quality_level == "medium":
            # Moderate degradation
            noise = np.random.normal(0, 10, img_array.shape).astype(np.int16)
            img_array = cv2.GaussianBlur(img_array, (3, 3), 1.0)

        elif quality_level == "poor":
            # Heavy degradation
            noise = np.random.normal(0, 20, img_array.shape).astype(np.int16)
            img_array = cv2.GaussianBlur(img_array, (5, 5), 1.5)

        elif quality_level == "terrible":
            # Severe degradation
            noise = np.random.normal(0, 40, img_array.shape).astype(np.int16)
            img_array = cv2.GaussianBlur(img_array, (7, 7), 2.0)
            # Add compression artifacts simulation
            img_array = img_array[::2, ::2]  # Downsample
            img_array = np.repeat(
                np.repeat(img_array, 2, axis=0), 2, axis=1
            )  # Upsample
        else:
            noise = np.zeros_like(img_array, dtype=np.int16)

        # Apply noise
        degraded = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        return Image.fromarray(degraded)

    def create_sharpness_test_image(
        self, sharpness_level: str, size: tuple[int, int] = (100, 100)
    ) -> Path:
        """Create image for testing MTF50 edge acuity measurement.

        Args:
            sharpness_level: 'sharp', 'moderate', 'soft', 'blurry'
            size: Image dimensions

        Returns:
            Path to created image file
        """
        width, height = size
        img = Image.new("RGB", size, (128, 128, 128))
        draw = ImageDraw.Draw(img)

        # Create sharp edges
        draw.rectangle([20, 20, 80, 80], fill=(255, 255, 255), outline=(0, 0, 0))
        draw.line([(10, height // 2), (90, height // 2)], fill=(0, 0, 0), width=2)

        # Apply different levels of blur
        img_array = np.array(img)

        if sharpness_level == "sharp":
            # No blur
            pass
        elif sharpness_level == "moderate":
            img_array = cv2.GaussianBlur(img_array, (3, 3), 0.5)
        elif sharpness_level == "soft":
            img_array = cv2.GaussianBlur(img_array, (5, 5), 1.0)
        elif sharpness_level == "blurry":
            img_array = cv2.GaussianBlur(img_array, (9, 9), 2.0)

        img = Image.fromarray(img_array)

        filename = f"sharpness_{sharpness_level}_{width}x{height}.png"
        filepath = self.output_dir / filename
        img.save(filepath)

        return filepath

    def generate_all_fixtures(self) -> dict:
        """Generate complete set of Phase 3 test fixtures.

        Returns:
            Dictionary mapping fixture categories to lists of file paths
        """
        fixtures = {
            "text_ui_images": [],
            "edge_density_images": [],
            "component_images": [],
            "ssimulacra2_pairs": [],
            "sharpness_images": [],
        }

        # Text/UI content images
        text_types = [
            "clean_text",
            "blurry_text",
            "ui_buttons",
            "terminal_text",
            "mixed_content",
            "no_text",
        ]
        for text_type in text_types:
            path = self.create_text_ui_image(text_type)
            fixtures["text_ui_images"].append(path)

        # Edge density images
        edge_densities = ["none", "low", "medium", "high", "extreme"]
        for density in edge_densities:
            path = self.create_edge_density_image(density)
            fixtures["edge_density_images"].append(path)

        # Component test images
        component_types = [
            "no_components",
            "single_component",
            "text_like",
            "too_small",
            "too_large",
            "wrong_aspect",
        ]
        for comp_type in component_types:
            path = self.create_component_test_image(comp_type)
            fixtures["component_images"].append(path)

        # SSIMULACRA2 test pairs
        quality_levels = ["excellent", "good", "medium", "poor", "terrible"]
        for quality in quality_levels:
            orig_path, deg_path = self.create_ssimulacra2_test_pair(quality)
            fixtures["ssimulacra2_pairs"].append((orig_path, deg_path))

        # Sharpness test images
        sharpness_levels = ["sharp", "moderate", "soft", "blurry"]
        for sharpness in sharpness_levels:
            path = self.create_sharpness_test_image(sharpness)
            fixtures["sharpness_images"].append(path)

        return fixtures


def main():
    """Generate all Phase 3 test fixtures."""
    generator = Phase3FixtureGenerator()
    fixtures = generator.generate_all_fixtures()

    print("Generated Phase 3 test fixtures:")
    for category, files in fixtures.items():
        print(f"  {category}: {len(files)} files")

    print(f"\nAll fixtures saved to: {generator.output_dir}")


if __name__ == "__main__":
    main()
