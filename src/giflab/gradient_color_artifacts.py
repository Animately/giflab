"""
Gradient and color artifact detection for GIF compression validation.

This module implements advanced gradient and color artifact detection metrics designed
specifically for debugging compression failures, including:
- Banding detection for gradient posterization using histogram analysis
- Perceptual color validation using CIEDE2000 color differences
- Smart patch sampling for efficient analysis
"""

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

try:
    from skimage import color as skcolor

    SKIMAGE_AVAILABLE = True
except ImportError:
    logger.warning(
        "scikit-image not available. Color space conversion will use fallback methods."
    )
    SKIMAGE_AVAILABLE = False


class GradientBandingDetector:
    """Detects banding artifacts in gradients caused by posterization."""

    def __init__(self, patch_size: int = 64, variance_threshold: float = 100.0):
        """Initialize gradient banding detector.

        Args:
            patch_size: Size of analysis patches in pixels
            variance_threshold: Variance threshold for identifying smooth regions
        """
        # Validate and set patch size
        if patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {patch_size}")
        self.patch_size = max(8, patch_size)  # Minimum size for meaningful analysis

        # Validate variance threshold
        if variance_threshold < 0:
            raise ValueError(
                f"variance_threshold must be non-negative, got {variance_threshold}"
            )
        self.variance_threshold = variance_threshold

    def detect_gradient_regions(
        self, frame: np.ndarray, step_size: int | None = None
    ) -> list[tuple[int, int, int, int]]:
        """Identify smooth gradient regions in a frame.

        Args:
            frame: RGB frame as numpy array [H, W, 3]
            step_size: Step size for sliding window (defaults to patch_size // 2)

        Returns:
            List of (x, y, width, height) tuples for gradient regions
        """
        if step_size is None:
            step_size = self.patch_size // 2

        # Ensure proper data type for OpenCV
        if frame.dtype not in [np.uint8, np.uint16, np.float32]:
            if frame.dtype in [np.int32, np.int64]:
                # Convert integer types to uint8, clipping to valid range
                frame = np.clip(frame, 0, 255).astype(np.uint8)
            else:
                # Convert other types to float32
                frame = frame.astype(np.float32)

        # Convert to grayscale for gradient analysis
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32)
        h, w = gray.shape

        gradient_regions = []

        for y in range(0, h - self.patch_size, step_size):
            for x in range(0, w - self.patch_size, step_size):
                patch = gray[y : y + self.patch_size, x : x + self.patch_size]

                # Check if patch has gradient characteristics
                if self._is_gradient_patch(patch):
                    gradient_regions.append((x, y, self.patch_size, self.patch_size))

        return self._merge_overlapping_regions(gradient_regions)

    def _is_gradient_patch(self, patch: np.ndarray) -> bool:
        """Check if a patch contains gradient characteristics."""
        # Calculate gradients using Sobel operators
        grad_x = cv2.Sobel(patch, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(patch, cv2.CV_64F, 0, 1, ksize=3)

        # Calculate gradient magnitude
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

        # Check for smooth gradient characteristics:
        # 1. Low variance (smooth region)
        # 2. Consistent gradient direction
        # 3. Non-zero gradient (not completely flat)

        variance = float(np.var(patch))
        grad_mean = float(np.mean(gradient_magnitude))
        grad_std = float(np.std(gradient_magnitude))

        # Smooth region with consistent gradients
        is_smooth = variance < self.variance_threshold
        has_gradient = grad_mean > 1.0  # Minimum gradient strength
        is_consistent = grad_std / max(grad_mean, 1.0) < 0.8  # Consistent gradient

        return is_smooth and has_gradient and is_consistent

    def calculate_gradient_magnitude_histogram(
        self, patch: np.ndarray, bins: int = 32
    ) -> np.ndarray:
        """Calculate gradient magnitude histogram for posterization analysis.

        Args:
            patch: Grayscale patch as numpy array
            bins: Number of histogram bins

        Returns:
            Normalized histogram of gradient magnitudes (sums to 1.0)
        """
        # Calculate gradients
        grad_x = cv2.Sobel(patch.astype(np.float32), cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(patch.astype(np.float32), cv2.CV_64F, 0, 1, ksize=3)

        # Calculate gradient magnitude
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

        # Create histogram and manually normalize to sum to 1.0
        hist, _ = np.histogram(gradient_magnitude.flatten(), bins=bins)
        hist = hist.astype(np.float32)
        hist_sum = np.sum(hist)
        if hist_sum > 0:
            hist = hist / hist_sum
        else:
            # Handle case where all gradients are zero (uniform patch)
            hist = np.zeros(bins, dtype=np.float32)

        return hist

    def detect_contours_in_patches(self, patch: np.ndarray) -> int:
        """Detect false contours in patch caused by posterization.

        Args:
            patch: Grayscale patch as numpy array

        Returns:
            Number of detected false contours
        """
        # Use edge detection to find contours
        edges = cv2.Canny(patch.astype(np.uint8), 30, 100)

        # Find contours
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Count significant contours (filter out noise)
        significant_contours = 0
        for contour in contours:
            if cv2.contourArea(contour) > 10:  # Minimum area threshold
                significant_contours += 1

        return significant_contours

    def calculate_banding_severity(
        self, original_patch: np.ndarray, compressed_patch: np.ndarray
    ) -> float:
        """Calculate banding severity score for a patch pair.

        Args:
            original_patch: Original grayscale patch
            compressed_patch: Compressed grayscale patch

        Returns:
            Banding severity score (0-100)
        """
        # Calculate gradient histograms
        original_hist = self.calculate_gradient_magnitude_histogram(original_patch)
        compressed_hist = self.calculate_gradient_magnitude_histogram(compressed_patch)

        # Compare histograms using chi-squared distance
        # Posterization creates spikes in gradient histogram
        hist_diff = np.sum(
            (original_hist - compressed_hist) ** 2
            / (original_hist + compressed_hist + 1e-10)
        )

        # Count false contours in compressed version
        original_contours = self.detect_contours_in_patches(original_patch)
        compressed_contours = self.detect_contours_in_patches(compressed_patch)

        # More contours in compressed = more false edges = more banding
        contour_excess = max(0, compressed_contours - original_contours)

        # Combine metrics into 0-100 severity score
        # Weight histogram difference and contour excess
        severity = (hist_diff * 30.0) + (contour_excess * 10.0)

        # Clamp to 0-100 range
        return float(min(100.0, max(0.0, severity)))

    def detect_banding_artifacts(
        self, original_frames: list[np.ndarray], compressed_frames: list[np.ndarray]
    ) -> dict[str, float]:
        """Detect banding artifacts across frame pairs.

        Args:
            original_frames: List of original RGB frames
            compressed_frames: List of compressed RGB frames

        Returns:
            Dictionary with banding detection metrics
        """
        if not original_frames or not compressed_frames:
            return {
                "banding_score_mean": 0.0,
                "banding_score_p95": 0.0,
                "banding_patch_count": 0,
                "gradient_region_count": 0,
            }

        min_frame_count = min(len(original_frames), len(compressed_frames))
        banding_scores = []
        total_patches = 0
        total_gradient_regions = 0

        for i in range(min_frame_count):
            original_frame = original_frames[i]
            compressed_frame = compressed_frames[i]

            # Ensure frames have the same shape
            h, w = min(original_frame.shape[:2]), min(compressed_frame.shape[:2])
            original_frame = original_frame[:h, :w]
            compressed_frame = compressed_frame[:h, :w]

            # Find gradient regions in original frame
            gradient_regions = self.detect_gradient_regions(original_frame)
            total_gradient_regions += len(gradient_regions)

            # Analyze each gradient region
            for x, y, patch_w, patch_h in gradient_regions:
                # Extract patches
                orig_patch = cv2.cvtColor(
                    original_frame[y : y + patch_h, x : x + patch_w], cv2.COLOR_RGB2GRAY
                )
                comp_patch = cv2.cvtColor(
                    compressed_frame[y : y + patch_h, x : x + patch_w],
                    cv2.COLOR_RGB2GRAY,
                )

                # Calculate banding severity
                severity = self.calculate_banding_severity(orig_patch, comp_patch)
                banding_scores.append(severity)
                total_patches += 1

        if not banding_scores:
            return {
                "banding_score_mean": 0.0,
                "banding_score_p95": 0.0,
                "banding_patch_count": 0,
                "gradient_region_count": total_gradient_regions,
            }

        return {
            "banding_score_mean": float(np.mean(banding_scores)),
            "banding_score_p95": float(np.percentile(banding_scores, 95)),
            "banding_patch_count": total_patches,
            "gradient_region_count": total_gradient_regions,
        }

    def _merge_overlapping_regions(
        self, regions: list[tuple[int, int, int, int]]
    ) -> list[tuple[int, int, int, int]]:
        """Merge overlapping rectangular regions."""
        if not regions:
            return []

        # Simple merge: combine regions that overlap significantly
        merged = [regions[0]]

        for region in regions[1:]:
            x, y, w, h = region
            merged_with_existing = False

            for i, existing in enumerate(merged):
                ex, ey, ew, eh = existing

                # Check for overlap
                overlap_x = max(0, min(x + w, ex + ew) - max(x, ex))
                overlap_y = max(0, min(y + h, ey + eh) - max(y, ey))
                overlap_area = overlap_x * overlap_y

                region_area = w * h
                existing_area = ew * eh

                # Merge if overlap is significant
                if overlap_area > 0.3 * min(region_area, existing_area):
                    new_x = min(x, ex)
                    new_y = min(y, ey)
                    new_w = max(x + w, ex + ew) - new_x
                    new_h = max(y + h, ey + eh) - new_y
                    merged[i] = (new_x, new_y, new_w, new_h)
                    merged_with_existing = True
                    break

            if not merged_with_existing:
                merged.append(region)

        return merged


class PerceptualColorValidator:
    """Validates color fidelity using perceptual color difference metrics."""

    def __init__(self, patch_size: int = 64, jnd_thresholds: list[float] | None = None):
        """Initialize perceptual color validator.

        Args:
            patch_size: Size of analysis patches in pixels
            jnd_thresholds: Just Noticeable Difference thresholds for ΔE00
        """
        self.patch_size = patch_size
        self.jnd_thresholds = jnd_thresholds or [1.0, 2.0, 3.0, 5.0]

    def rgb_to_lab(self, rgb_image: np.ndarray) -> np.ndarray:
        """Convert RGB image to CIELAB color space.

        Args:
            rgb_image: RGB image as numpy array [H, W, 3] with values 0-255

        Returns:
            Lab image as numpy array [H, W, 3]
        """
        if not SKIMAGE_AVAILABLE:
            # Fallback: simple RGB approximation (not accurate but functional)
            logger.warning(
                "Using RGB approximation for Lab conversion - results may be inaccurate"
            )
            return rgb_image.astype(np.float32)

        # Ensure proper data type and range
        if rgb_image.dtype == np.uint8:
            rgb_normalized = rgb_image.astype(np.float32) / 255.0
        else:
            rgb_normalized = rgb_image.astype(np.float32)
            if rgb_normalized.max() > 1.0:
                rgb_normalized /= 255.0

        # Convert to Lab using scikit-image
        try:
            lab_image = skcolor.rgb2lab(rgb_normalized)
            return np.asarray(lab_image, dtype=np.float32)
        except Exception as e:
            logger.error(f"Failed to convert RGB to Lab: {e}")
            # Return RGB as fallback
            return rgb_image.astype(np.float32)

    def calculate_deltae2000(self, lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
        """Calculate CIEDE2000 color difference.

        Args:
            lab1: First Lab color array [..., 3]
            lab2: Second Lab color array [..., 3]

        Returns:
            ΔE00 values with same shape as input (minus last dimension)
        """
        # This is a simplified implementation of CIEDE2000
        # For production use, consider using colormath library or full implementation

        # Extract L*, a*, b* components
        L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
        L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]

        # Calculate differences
        delta_L = L2 - L1
        delta_a = a2 - a1
        delta_b = b2 - b1

        # Calculate chroma and hue
        C1 = np.sqrt(a1**2 + b1**2)
        C2 = np.sqrt(a2**2 + b2**2)
        delta_C = C2 - C1

        delta_H_squared = delta_a**2 + delta_b**2 - delta_C**2
        delta_H_squared = np.maximum(0, delta_H_squared)  # Avoid negative values
        delta_H = np.sqrt(delta_H_squared)

        # Simplified CIEDE2000 approximation
        # This omits the complex weighting functions for simplicity
        # For high accuracy, implement full CIEDE2000 formula

        # Average chroma
        C_avg = (C1 + C2) / 2.0

        # Lightness weighting
        L_avg = (L1 + L2) / 2.0
        SL = 1.0 + (0.015 * (L_avg - 50) ** 2) / np.sqrt(20 + (L_avg - 50) ** 2)

        # Chroma weighting
        SC = 1.0 + 0.045 * C_avg

        # Hue weighting (simplified)
        SH = 1.0 + 0.015 * C_avg

        # Calculate ΔE00 (simplified)
        delta_E00 = np.sqrt(
            (delta_L / SL) ** 2 + (delta_C / SC) ** 2 + (delta_H / SH) ** 2
        )

        return np.asarray(delta_E00, dtype=np.float32)

    def sample_color_patches(
        self, frame: np.ndarray, num_samples: int = 16
    ) -> list[tuple[int, int, int, int]]:
        """Smart sampling of color patches from frame.

        Args:
            frame: RGB frame as numpy array
            num_samples: Target number of patches to sample

        Returns:
            List of (x, y, width, height) tuples for sampled patches
        """
        h, w = frame.shape[:2]

        # Ensure we can fit the patches
        max_patches_x = max(1, w // self.patch_size)
        max_patches_y = max(1, h // self.patch_size)
        max_patches = max_patches_x * max_patches_y

        actual_samples = min(num_samples, max_patches)

        patches: list[tuple[int, int, int, int]] = []

        if actual_samples <= max_patches_x * max_patches_y:
            # Grid sampling with some randomness
            step_x = max(1, max_patches_x // int(np.sqrt(actual_samples)))
            step_y = max(1, max_patches_y // int(np.sqrt(actual_samples)))

            for y in range(0, max_patches_y, step_y):
                for x in range(0, max_patches_x, step_x):
                    if len(patches) >= actual_samples:
                        break

                    # Convert grid position to pixel coordinates
                    px = x * self.patch_size
                    py = y * self.patch_size

                    # Ensure patch fits within frame
                    if px + self.patch_size <= w and py + self.patch_size <= h:
                        patches.append((px, py, self.patch_size, self.patch_size))

                if len(patches) >= actual_samples:
                    break

        return patches[:actual_samples]

    def calculate_color_difference_metrics(
        self, original_frames: list[np.ndarray], compressed_frames: list[np.ndarray]
    ) -> dict[str, float]:
        """Calculate perceptual color difference metrics.

        Args:
            original_frames: List of original RGB frames
            compressed_frames: List of compressed RGB frames

        Returns:
            Dictionary with color difference metrics
        """
        if not original_frames or not compressed_frames:
            return {
                "deltae_mean": 0.0,
                "deltae_p95": 0.0,
                "deltae_max": 0.0,
                "deltae_pct_gt1": 0.0,
                "deltae_pct_gt2": 0.0,
                "deltae_pct_gt3": 0.0,
                "deltae_pct_gt5": 0.0,
                "color_patch_count": 0,
            }

        min_frame_count = min(len(original_frames), len(compressed_frames))
        deltae_values = []
        total_patches = 0

        for i in range(min_frame_count):
            original_frame = original_frames[i]
            compressed_frame = compressed_frames[i]

            # Ensure frames have the same shape
            h, w = min(original_frame.shape[:2]), min(compressed_frame.shape[:2])
            original_frame = original_frame[:h, :w]
            compressed_frame = compressed_frame[:h, :w]

            # Sample color patches
            patches = self.sample_color_patches(original_frame)

            for x, y, patch_w, patch_h in patches:
                # Extract patches
                orig_patch = original_frame[y : y + patch_h, x : x + patch_w]
                comp_patch = compressed_frame[y : y + patch_h, x : x + patch_w]

                # Convert to Lab color space
                orig_lab = self.rgb_to_lab(orig_patch)
                comp_lab = self.rgb_to_lab(comp_patch)

                # Calculate ΔE00 for patch
                patch_deltae = self.calculate_deltae2000(orig_lab, comp_lab)

                # Use mean ΔE for this patch
                mean_deltae = float(np.mean(patch_deltae))
                deltae_values.append(mean_deltae)
                total_patches += 1

        if not deltae_values:
            return {
                "deltae_mean": 0.0,
                "deltae_p95": 0.0,
                "deltae_max": 0.0,
                "deltae_pct_gt1": 0.0,
                "deltae_pct_gt2": 0.0,
                "deltae_pct_gt3": 0.0,
                "deltae_pct_gt5": 0.0,
                "color_patch_count": total_patches,
            }

        deltae_array = np.array(deltae_values)

        # Calculate threshold percentages
        pct_gt1 = float(np.mean(deltae_array > 1.0) * 100)
        pct_gt2 = float(np.mean(deltae_array > 2.0) * 100)
        pct_gt3 = float(np.mean(deltae_array > 3.0) * 100)
        pct_gt5 = float(np.mean(deltae_array > 5.0) * 100)

        return {
            "deltae_mean": float(np.mean(deltae_array)),
            "deltae_p95": float(np.percentile(deltae_array, 95)),
            "deltae_max": float(np.max(deltae_array)),
            "deltae_pct_gt1": pct_gt1,
            "deltae_pct_gt2": pct_gt2,
            "deltae_pct_gt3": pct_gt3,
            "deltae_pct_gt5": pct_gt5,
            "color_patch_count": total_patches,
        }


class DitherQualityAnalyzer:
    """Analyzes dithering quality using FFT-based frequency analysis.

    Detects over-dithering (excessive noise) and under-dithering (banding) by analyzing
    the frequency spectrum of flat/gradient regions.
    """

    def __init__(self, patch_size: int = 64, flat_threshold: float = 50.0):
        """Initialize dither quality analyzer.

        Args:
            patch_size: Size of analysis patches in pixels
            flat_threshold: Variance threshold for identifying smooth regions suitable for dithering analysis
        """
        if patch_size <= 0 or patch_size < 16:
            raise ValueError(f"patch_size must be >= 16, got {patch_size}")
        self.patch_size = patch_size

        if flat_threshold < 0:
            raise ValueError(
                f"flat_threshold must be non-negative, got {flat_threshold}"
            )
        self.flat_threshold = flat_threshold

    def detect_flat_regions(
        self, frame: np.ndarray, step_size: int | None = None
    ) -> list[tuple[int, int, int, int]]:
        """Identify flat/smooth regions suitable for dithering analysis.

        Args:
            frame: RGB frame as numpy array [H, W, 3]
            step_size: Step size for sliding window (defaults to patch_size // 2)

        Returns:
            List of (x, y, width, height) tuples for flat regions
        """
        if step_size is None:
            step_size = self.patch_size // 2

        # Convert to grayscale for analysis
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)

        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32)
        h, w = gray.shape

        flat_regions = []

        for y in range(0, h - self.patch_size, step_size):
            for x in range(0, w - self.patch_size, step_size):
                patch = gray[y : y + self.patch_size, x : x + self.patch_size]

                # Check if patch is flat/smooth (low variance)
                variance = np.var(patch)
                if variance < self.flat_threshold:
                    flat_regions.append((x, y, self.patch_size, self.patch_size))

        return flat_regions

    def compute_frequency_spectrum(self, patch: np.ndarray) -> np.ndarray:
        """Apply 2D FFT to analyze frequency content of a patch.

        Args:
            patch: Grayscale patch as numpy array [H, W]

        Returns:
            2D magnitude spectrum (frequencies)
        """
        # Apply 2D FFT and compute magnitude spectrum
        fft_result = np.fft.fft2(patch)
        magnitude_spectrum = np.abs(fft_result)

        # Shift zero frequency to center for easier analysis
        magnitude_spectrum = np.fft.fftshift(magnitude_spectrum)

        return magnitude_spectrum

    def calculate_band_energies(
        self, spectrum: np.ndarray
    ) -> tuple[float, float, float]:
        """Divide frequency spectrum into bands and calculate energies.

        Args:
            spectrum: 2D magnitude spectrum from FFT

        Returns:
            Tuple of (low_energy, mid_energy, high_energy)
        """
        h, w = spectrum.shape
        center_h, center_w = h // 2, w // 2

        # Create distance matrix from center (DC component)
        y, x = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((x - center_w) ** 2 + (y - center_h) ** 2)

        # Normalize distance to [0, 1] where 1 is the maximum possible distance
        max_dist = np.sqrt(center_h**2 + center_w**2)
        normalized_dist = dist_from_center / max_dist

        # Define frequency bands (normalized distances)
        # Low: 0-0.1 (DC + very low frequencies)
        # Mid: 0.1-0.5 (ideal dithering frequencies)
        # High: 0.5-1.0 (high frequencies, potential noise)
        low_mask = normalized_dist <= 0.1
        mid_mask = (normalized_dist > 0.1) & (normalized_dist <= 0.5)
        high_mask = normalized_dist > 0.5

        # Calculate energy in each band (sum of squared magnitudes)
        low_energy = np.sum(spectrum[low_mask] ** 2)
        mid_energy = np.sum(spectrum[mid_mask] ** 2)
        high_energy = np.sum(spectrum[high_mask] ** 2)

        return low_energy, mid_energy, high_energy

    def compute_dither_ratio(self, high_energy: float, mid_energy: float) -> float:
        """Calculate dithering quality ratio from frequency band energies.

        Args:
            high_energy: Energy in high frequency band
            mid_energy: Energy in mid frequency band

        Returns:
            Dither ratio where:
            - < 0.8: Under-dithered (too smooth, potential banding)
            - 0.8-1.3: Well-dithered
            - > 1.3: Over-dithered (excessive noise)
        """
        # Avoid division by zero
        if mid_energy < 1e-10:
            # If mid-energy is essentially zero, check high energy
            return 10.0 if high_energy > 1e-10 else 0.0

        return high_energy / mid_energy

    def analyze_dither_quality(
        self, original_frames: list[np.ndarray], compressed_frames: list[np.ndarray]
    ) -> dict[str, float]:
        """Main entry point for dither quality analysis.

        Args:
            original_frames: List of original RGB frames
            compressed_frames: List of compressed RGB frames

        Returns:
            Dictionary with dither quality metrics
        """
        if len(original_frames) != len(compressed_frames):
            raise ValueError(
                f"Frame count mismatch: {len(original_frames)} vs {len(compressed_frames)}"
            )

        all_ratios = []
        total_flat_regions = 0

        try:
            for orig_frame, comp_frame in zip(
                original_frames, compressed_frames, strict=True
            ):
                # Detect flat regions in original (these should benefit from good dithering)
                flat_regions = self.detect_flat_regions(orig_frame)
                total_flat_regions += len(flat_regions)

                # Convert compressed frame to grayscale for analysis
                if comp_frame.dtype != np.uint8:
                    comp_frame = np.clip(comp_frame, 0, 255).astype(np.uint8)
                comp_gray = cv2.cvtColor(comp_frame, cv2.COLOR_RGB2GRAY).astype(
                    np.float32
                )

                # Analyze each flat region in the compressed frame
                for x, y, w, h in flat_regions:
                    patch = comp_gray[y : y + h, x : x + w]

                    # Compute frequency spectrum
                    spectrum = self.compute_frequency_spectrum(patch)

                    # Calculate band energies
                    low_energy, mid_energy, high_energy = self.calculate_band_energies(
                        spectrum
                    )

                    # Compute dither ratio
                    ratio = self.compute_dither_ratio(high_energy, mid_energy)
                    all_ratios.append(ratio)

            # Calculate summary statistics
            if not all_ratios:
                return {
                    "dither_ratio_mean": 0.0,
                    "dither_ratio_p95": 0.0,
                    "dither_quality_score": 0.0,
                    "flat_region_count": 0,
                }

            ratios_array = np.array(all_ratios)
            mean_ratio = float(np.mean(ratios_array))
            p95_ratio = float(np.percentile(ratios_array, 95))

            # Calculate quality score (0-100, higher is better)
            # Ideal ratio range is 0.8-1.3, score decreases as we move away from this range
            quality_scores = []
            for ratio in all_ratios:
                if 0.8 <= ratio <= 1.3:
                    # Well-dithered: score based on how close to 1.0 (perfect balance)
                    distance_from_ideal = abs(ratio - 1.0)
                    score = max(80, 100 - (distance_from_ideal * 40))
                elif ratio < 0.8:
                    # Under-dithered: score decreases with how far below 0.8
                    score = max(0, 80 * (ratio / 0.8))
                else:  # ratio > 1.3
                    # Over-dithered: score decreases with how far above 1.3
                    score = max(0, 80 * (2.0 - min(ratio, 2.0)) / 0.7)
                quality_scores.append(score)

            mean_quality = float(np.mean(quality_scores))

            return {
                "dither_ratio_mean": mean_ratio,
                "dither_ratio_p95": p95_ratio,
                "dither_quality_score": mean_quality,
                "flat_region_count": total_flat_regions,
            }

        except Exception as e:
            logger.error(f"Failed to analyze dither quality: {e}")
            return {
                "dither_ratio_mean": 0.0,
                "dither_ratio_p95": 0.0,
                "dither_quality_score": 0.0,
                "flat_region_count": 0,
            }


# Main entry points for external use


def calculate_banding_metrics(
    original_frames: list[np.ndarray], compressed_frames: list[np.ndarray]
) -> dict[str, float]:
    """Main entry point for banding detection.

    Args:
        original_frames: List of original RGB frames
        compressed_frames: List of compressed RGB frames

    Returns:
        Dictionary with banding detection metrics
    """
    detector = GradientBandingDetector()
    return detector.detect_banding_artifacts(original_frames, compressed_frames)


def calculate_perceptual_color_metrics(
    original_frames: list[np.ndarray], compressed_frames: list[np.ndarray]
) -> dict[str, float]:
    """Main entry point for perceptual color validation.

    Args:
        original_frames: List of original RGB frames
        compressed_frames: List of compressed RGB frames

    Returns:
        Dictionary with perceptual color difference metrics
    """
    validator = PerceptualColorValidator()
    return validator.calculate_color_difference_metrics(
        original_frames, compressed_frames
    )


def calculate_dither_quality_metrics(
    original_frames: list[np.ndarray], compressed_frames: list[np.ndarray]
) -> dict[str, float]:
    """Main entry point for dither quality analysis.

    Args:
        original_frames: List of original RGB frames
        compressed_frames: List of compressed RGB frames

    Returns:
        Dictionary with dither quality metrics
    """
    analyzer = DitherQualityAnalyzer()
    return analyzer.analyze_dither_quality(original_frames, compressed_frames)


def calculate_color_quantization_metrics(
    original_frames: list[np.ndarray], compressed_frames: list[np.ndarray]
) -> dict[str, float]:
    """Calculate color quantization error metrics.

    Measures how well the compressed GIF preserves the original color space:
    - Unique color count in original vs compressed frames
    - Palette distance (mean minimum Euclidean distance between color tables)

    Args:
        original_frames: List of original RGB frames [H, W, 3]
        compressed_frames: List of compressed RGB frames [H, W, 3]

    Returns:
        Dictionary with color quantization metrics
    """
    from scipy.spatial.distance import cdist

    try:
        original_colors_total = 0
        compressed_colors_total = 0
        palette_distances = []

        num_frames = min(len(original_frames), len(compressed_frames))
        # Sample up to 10 frames for efficiency
        sample_indices = np.linspace(0, num_frames - 1, min(10, num_frames), dtype=int)

        for idx in sample_indices:
            orig = original_frames[idx]
            comp = compressed_frames[idx]

            # Reshape to (N, 3) pixel arrays
            orig_pixels = orig.reshape(-1, 3).astype(np.float64)
            comp_pixels = comp.reshape(-1, 3).astype(np.float64)

            # Get unique colors
            orig_unique = np.unique(orig_pixels, axis=0)
            comp_unique = np.unique(comp_pixels, axis=0)

            original_colors_total += len(orig_unique)
            compressed_colors_total += len(comp_unique)

            # Calculate palette distance: mean of minimum distances from
            # original colors to their nearest compressed color
            if len(orig_unique) > 0 and len(comp_unique) > 0:
                # Cap palette size for computational efficiency
                # Use deterministic uniform sampling for reproducibility
                max_palette = 256
                if len(orig_unique) > max_palette:
                    indices = np.linspace(
                        0, len(orig_unique) - 1, max_palette, dtype=int
                    )
                    orig_unique = orig_unique[indices]
                if len(comp_unique) > max_palette:
                    indices = np.linspace(
                        0, len(comp_unique) - 1, max_palette, dtype=int
                    )
                    comp_unique = comp_unique[indices]

                dist_matrix = cdist(orig_unique, comp_unique, metric="euclidean")
                min_distances = dist_matrix.min(axis=1)
                palette_distances.append(float(np.mean(min_distances)))

        num_sampled = len(sample_indices)
        avg_original = original_colors_total / num_sampled if num_sampled > 0 else 0
        avg_compressed = compressed_colors_total / num_sampled if num_sampled > 0 else 0
        avg_palette_distance = (
            float(np.mean(palette_distances)) if palette_distances else 0.0
        )

        return {
            "color_count_original": float(avg_original),
            "color_count_compressed": float(avg_compressed),
            "palette_distance": avg_palette_distance,
        }

    except Exception as e:
        logger.error(f"Failed to calculate color quantization metrics: {e}")
        return {
            "color_count_original": 0.0,
            "color_count_compressed": 0.0,
            "palette_distance": 0.0,
        }


def calculate_posterization_score(
    original_frames: list[np.ndarray], compressed_frames: list[np.ndarray]
) -> dict[str, float]:
    """Calculate posterization score measuring flat bands in smooth gradients.

    Detects where compression has reduced smooth gradients to discrete color steps.
    Uses the existing GradientBandingDetector to find gradient regions, then counts
    distinct intensity levels in those regions comparing original vs compressed.

    Args:
        original_frames: List of original RGB frames [H, W, 3]
        compressed_frames: List of compressed RGB frames [H, W, 3]

    Returns:
        Dictionary with posterization score (0 = no posterization, 1 = severe)
    """
    try:
        detector = GradientBandingDetector(patch_size=32, variance_threshold=100.0)
        posterization_scores = []

        num_frames = min(len(original_frames), len(compressed_frames))
        sample_indices = np.linspace(0, num_frames - 1, min(10, num_frames), dtype=int)

        for idx in sample_indices:
            orig = original_frames[idx]
            comp = compressed_frames[idx]

            # Find gradient regions in the original frame
            gradient_regions = detector.detect_gradient_regions(orig)

            if not gradient_regions:
                continue

            frame_ratios = []
            for x, y, w, h in gradient_regions:
                # Extract patches from both frames
                orig_patch = cv2.cvtColor(
                    orig[y : y + h, x : x + w], cv2.COLOR_RGB2GRAY
                )
                comp_patch = cv2.cvtColor(
                    comp[y : y + h, x : x + w], cv2.COLOR_RGB2GRAY
                )

                # Count unique intensity levels
                orig_levels = len(np.unique(orig_patch))
                comp_levels = len(np.unique(comp_patch))

                if orig_levels > 1:
                    # Ratio of level reduction: 1.0 means all levels preserved
                    preservation_ratio = comp_levels / orig_levels
                    # Invert: 0 = perfect, 1 = severe posterization
                    frame_ratios.append(1.0 - min(1.0, preservation_ratio))

            if frame_ratios:
                posterization_scores.append(float(np.mean(frame_ratios)))

        score = float(np.mean(posterization_scores)) if posterization_scores else 0.0
        return {"posterization_score": max(0.0, min(1.0, score))}

    except Exception as e:
        logger.error(f"Failed to calculate posterization score: {e}")
        return {"posterization_score": 0.0}


def calculate_transparency_artifact_score(
    original_frames: list[np.ndarray], compressed_frames: list[np.ndarray]
) -> dict[str, float]:
    """Calculate transparency artifact score for GIFs with alpha channels.

    Measures corruption in transparent regions after compression:
    - Alpha channel preservation (how well transparency is maintained)
    - Color fidelity in semi-transparent regions

    Only meaningful when frames have an alpha channel (4-channel RGBA).
    Returns 0.0 (no artifacts) when no transparency is present.

    Args:
        original_frames: List of original frames (RGB or RGBA)
        compressed_frames: List of compressed frames (RGB or RGBA)

    Returns:
        Dictionary with transparency artifact score (0 = no artifacts, 1 = severe)
    """
    try:
        num_frames = min(len(original_frames), len(compressed_frames))
        if num_frames == 0:
            return {"transparency_artifact_score": 0.0}

        # Check if frames have alpha channel
        has_alpha = (
            original_frames[0].ndim == 3
            and original_frames[0].shape[2] == 4
            and compressed_frames[0].ndim == 3
            and compressed_frames[0].shape[2] == 4
        )

        if not has_alpha:
            return {"transparency_artifact_score": 0.0}

        artifact_scores = []
        sample_indices = np.linspace(0, num_frames - 1, min(10, num_frames), dtype=int)

        for idx in sample_indices:
            orig = original_frames[idx]
            comp = compressed_frames[idx]

            orig_alpha = orig[:, :, 3].astype(np.float64)
            comp_alpha = comp[:, :, 3].astype(np.float64)

            # Check if there are any transparent pixels
            transparency_ratio = np.mean(orig_alpha < 255)
            if transparency_ratio < 0.01:
                continue  # Skip frames with negligible transparency

            # Alpha preservation error (normalized to 0-1)
            alpha_error = np.mean(np.abs(orig_alpha - comp_alpha)) / 255.0

            # Color fidelity in semi-transparent regions (alpha between 1-254)
            semi_transparent = (orig_alpha > 0) & (orig_alpha < 255)
            if np.any(semi_transparent):
                orig_rgb = orig[:, :, :3].astype(np.float64)
                comp_rgb = comp[:, :, :3].astype(np.float64)
                color_diff = (
                    np.mean(
                        np.abs(orig_rgb[semi_transparent] - comp_rgb[semi_transparent])
                    )
                    / 255.0
                )
            else:
                color_diff = 0.0

            # Combined score: 60% alpha preservation, 40% color fidelity
            frame_score = 0.6 * alpha_error + 0.4 * color_diff
            artifact_scores.append(frame_score)

        score = float(np.mean(artifact_scores)) if artifact_scores else 0.0
        return {"transparency_artifact_score": max(0.0, min(1.0, score))}

    except Exception as e:
        logger.error(f"Failed to calculate transparency artifact score: {e}")
        return {"transparency_artifact_score": 0.0}


def calculate_gradient_color_metrics(
    original_frames: list[np.ndarray], compressed_frames: list[np.ndarray]
) -> dict[str, float]:
    """Combined entry point for all gradient and color metrics.

    Args:
        original_frames: List of original RGB frames
        compressed_frames: List of compressed RGB frames

    Returns:
        Dictionary with all gradient and color artifact metrics
    """
    try:
        banding_metrics = calculate_banding_metrics(original_frames, compressed_frames)
        color_metrics = calculate_perceptual_color_metrics(
            original_frames, compressed_frames
        )
        dither_metrics = calculate_dither_quality_metrics(
            original_frames, compressed_frames
        )
        quantization_metrics = calculate_color_quantization_metrics(
            original_frames, compressed_frames
        )
        posterization_metrics = calculate_posterization_score(
            original_frames, compressed_frames
        )
        transparency_metrics = calculate_transparency_artifact_score(
            original_frames, compressed_frames
        )

        # Combine all metrics
        combined_metrics = {
            **banding_metrics,
            **color_metrics,
            **dither_metrics,
            **quantization_metrics,
            **posterization_metrics,
            **transparency_metrics,
        }

        logger.debug(
            f"Calculated gradient and color metrics: {len(combined_metrics)} metrics"
        )
        return combined_metrics

    except Exception as e:
        logger.error(f"Failed to calculate gradient and color metrics: {e}")
        # Return empty metrics on failure
        return {
            "banding_score_mean": 0.0,
            "banding_score_p95": 0.0,
            "banding_patch_count": 0,
            "gradient_region_count": 0,
            "deltae_mean": 0.0,
            "deltae_p95": 0.0,
            "deltae_max": 0.0,
            "deltae_pct_gt1": 0.0,
            "deltae_pct_gt2": 0.0,
            "deltae_pct_gt3": 0.0,
            "deltae_pct_gt5": 0.0,
            "color_patch_count": 0,
            "dither_ratio_mean": 0.0,
            "dither_ratio_p95": 0.0,
            "dither_quality_score": 0.0,
            "flat_region_count": 0,
            "color_count_original": 0.0,
            "color_count_compressed": 0.0,
            "palette_distance": 0.0,
            "posterization_score": 0.0,
            "transparency_artifact_score": 0.0,
        }
