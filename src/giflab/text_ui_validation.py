"""Text and UI content validation for GIF compression debugging.

This module provides targeted validation metrics for text and UI content,
detecting readability degradation that can occur during compression.
Only triggered when text/UI content is detected to control computational cost.
"""

import logging
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# OCR library availability check
try:
    import pytesseract

    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    logger.debug("pytesseract not available - OCR functionality will be limited")

try:
    import easyocr

    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    logger.debug("easyocr not available - using tesseract only")


class TextUIContentDetector:
    """Detects text and UI elements in frames using edge detection and component analysis."""

    def __init__(
        self,
        edge_threshold: float = 30.0,  # Lowered from 50.0 for better text detection
        min_component_area: int = 10,
        max_component_area: int = 500,
        edge_density_threshold: float = 0.03,  # Lowered from 0.10 to detect actual text/UI
    ):
        """Initialize text/UI content detector.

        Args:
            edge_threshold: Canny edge detection threshold
            min_component_area: Minimum area for text-like components
            max_component_area: Maximum area for text-like components
            edge_density_threshold: Minimum edge density to consider text/UI content
        """
        self.edge_threshold = edge_threshold
        self.min_component_area = min_component_area
        self.max_component_area = max_component_area
        self.edge_density_threshold = edge_density_threshold

    def detect_text_ui_regions(
        self, frame: np.ndarray
    ) -> list[tuple[int, int, int, int]]:
        """Detect text and UI regions in a frame.

        Args:
            frame: Input frame as numpy array (H, W, 3)

        Returns:
            List of bounding boxes as (x, y, w, h) tuples
        """
        # Validate input
        if frame is None:
            raise ValueError("Frame cannot be None")

        # Quick edge density check
        edge_density = self._detect_edge_density(frame)
        if edge_density < self.edge_density_threshold:
            logger.debug(
                f"Edge density {edge_density:.3f} below threshold {self.edge_density_threshold}"
            )
            return []

        # Convert to grayscale for processing
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        else:
            gray = frame.copy()

        # Apply edge detection
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(blurred, self.edge_threshold, self.edge_threshold * 2)

        # Morphological operations to connect nearby text elements
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        # Find text-like connected components
        text_components = self._find_text_like_components(edges_closed)

        if not text_components:
            logger.debug("No text-like components found")
            return []

        # Filter to keep only UI-like components
        ui_components = self._filter_ui_components(text_components)

        if not ui_components:
            logger.debug("No UI-like components found after filtering")
            return []

        # Convert component bounding boxes to region list
        regions = []
        for comp in ui_components:
            x, y, w, h = comp["bbox"]
            # Add small padding around detected regions
            padding = 2
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(frame.shape[1] - x, w + 2 * padding)
            h = min(frame.shape[0] - y, h + 2 * padding)
            regions.append((x, y, w, h))

        # Merge overlapping or nearby regions
        regions = self._merge_nearby_regions(regions)

        logger.debug(
            f"Detected {len(regions)} text/UI regions with edge density {edge_density:.3f}"
        )
        return regions

    def _merge_nearby_regions(
        self, regions: list[tuple[int, int, int, int]]
    ) -> list[tuple[int, int, int, int]]:
        """Merge overlapping or nearby text regions.

        Args:
            regions: List of bounding boxes as (x, y, w, h) tuples

        Returns:
            List of merged bounding boxes
        """
        if len(regions) <= 1:
            return regions

        # Convert to (x1, y1, x2, y2) format for easier overlap calculation
        boxes = [(x, y, x + w, y + h) for x, y, w, h in regions]

        merged = []
        used = [False] * len(boxes)

        for i, box1 in enumerate(boxes):
            if used[i]:
                continue

            x1, y1, x2, y2 = box1

            # Find all boxes that overlap or are nearby
            for j, box2 in enumerate(boxes[i + 1 :], i + 1):
                if used[j]:
                    continue

                bx1, by1, bx2, by2 = box2

                # Check for overlap or proximity (within 10 pixels)
                proximity_threshold = 10

                if (
                    x1 <= bx2 + proximity_threshold
                    and bx1 <= x2 + proximity_threshold
                    and y1 <= by2 + proximity_threshold
                    and by1 <= y2 + proximity_threshold
                ):
                    # Merge boxes
                    x1 = min(x1, bx1)
                    y1 = min(y1, by1)
                    x2 = max(x2, bx2)
                    y2 = max(y2, by2)
                    used[j] = True

            merged.append((x1, y1, x2 - x1, y2 - y1))
            used[i] = True

        return merged

    def _detect_edge_density(self, frame: np.ndarray) -> float:
        """Calculate edge density for text/UI content detection.

        Args:
            frame: Input frame as numpy array

        Returns:
            Edge density as percentage of pixels that are edges
        """
        # Validate input
        if frame.size == 0:
            raise ValueError("Frame is empty")
        if len(frame.shape) < 2:
            raise ValueError("Frame must be at least 2-dimensional")

        try:
            # Convert to grayscale if needed
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            else:
                gray = frame.copy()

            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)

            # Detect edges using Canny
            edges = cv2.Canny(blurred, self.edge_threshold, self.edge_threshold * 2)

            # Calculate edge density
            total_pixels = edges.shape[0] * edges.shape[1]
            edge_pixels = np.sum(edges > 0)
            edge_density = edge_pixels / total_pixels

            return float(edge_density)
        except cv2.error as e:
            logger.debug(f"Edge detection failed: {e}")
            return 0.0  # Return safe default on CV2 errors

    def _find_text_like_components(
        self, binary_image: np.ndarray
    ) -> list[dict[str, Any]]:
        """Find connected components that look like text.

        Args:
            binary_image: Binary edge image

        Returns:
            List of text-like connected components with stats
        """
        # Find connected components with statistics
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary_image, connectivity=8, ltype=cv2.CV_32S
        )

        text_components = []

        # Skip background label (0)
        for label_id in range(1, num_labels):
            # Extract component statistics
            x, y, w, h, area = stats[label_id]

            # Filter by area - text components are typically small to medium
            if area < self.min_component_area or area > self.max_component_area:
                continue

            # Filter by aspect ratio - text components have reasonable width/height ratios
            aspect_ratio = w / max(h, 1)
            if (
                aspect_ratio < 0.1 or aspect_ratio > 10.0
            ):  # Very thin lines or very wide shapes
                continue

            # Filter by dimensions - text components have minimum size
            if w < 3 or h < 3:
                continue

            # Filter by solidity (filled area / bounding box area)
            bounding_area = w * h
            solidity = area / max(bounding_area, 1)
            if solidity < 0.1:  # Too sparse, likely not text
                continue

            text_components.append(
                {
                    "label": label_id,
                    "bbox": (x, y, w, h),
                    "area": area,
                    "aspect_ratio": aspect_ratio,
                    "solidity": solidity,
                    "centroid": centroids[label_id],
                }
            )

        return text_components

    def _filter_ui_components(
        self, components: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Filter components to keep only UI-like elements.

        Args:
            components: List of connected components with stats

        Returns:
            Filtered list of UI-like components
        """
        if not components:
            return []

        ui_components = []

        # Group components by proximity to find text lines/blocks
        height_groups: dict[int, list[dict[str, Any]]] = {}
        for comp in components:
            x, y, w, h = comp["bbox"]
            # Group by similar y-coordinate (text lines)
            y_group = round(y / 10) * 10  # Group within 10 pixels vertically
            if y_group not in height_groups:
                height_groups[y_group] = []
            height_groups[y_group].append(comp)

        # Process each height group
        for _, group_components in height_groups.items():
            if (
                len(group_components) < 2
            ):  # Single components are less likely to be UI text
                # But still include if they meet certain criteria
                comp = group_components[0]
                x, y, w, h = comp["bbox"]

                # Keep larger single components that look like buttons/labels
                if (
                    comp["area"] > 50
                    and 0.3 <= comp["aspect_ratio"] <= 3.0
                    and comp["solidity"] > 0.3
                ):
                    ui_components.append(comp)
                continue

            # Sort by x-coordinate for horizontal text alignment
            group_components.sort(key=lambda c: c["bbox"][0])

            # Check for horizontal alignment (text lines)
            y_coords = [c["bbox"][1] for c in group_components]
            y_variance = np.var(y_coords) if len(y_coords) > 1 else 0

            # Check for similar heights (consistent text size)
            heights = [c["bbox"][3] for c in group_components]
            height_variance = np.var(heights) if len(heights) > 1 else 0

            # Keep groups with good alignment and similar heights
            if y_variance < 25 and height_variance < 100:  # Allow some variation
                ui_components.extend(group_components)
            else:
                # Keep individual components that meet size criteria
                for comp in group_components:
                    if (
                        comp["area"] > 30
                        and 0.2 <= comp["aspect_ratio"] <= 5.0
                        and comp["solidity"] > 0.2
                    ):
                        ui_components.append(comp)

        # Sort by position for consistent processing
        ui_components.sort(key=lambda c: (c["bbox"][1], c["bbox"][0]))

        return ui_components


class OCRValidator:
    """Measures OCR confidence delta between original and compressed frames."""

    def __init__(self, use_tesseract: bool = True, fallback_to_easyocr: bool = False):
        """Initialize OCR validator.

        Args:
            use_tesseract: Use pytesseract as primary OCR engine
            fallback_to_easyocr: Use easyocr as fallback when tesseract fails
        """
        self.use_tesseract = use_tesseract and TESSERACT_AVAILABLE
        self.fallback_to_easyocr = fallback_to_easyocr and EASYOCR_AVAILABLE

        if not self.use_tesseract and not self.fallback_to_easyocr:
            logger.warning(
                "No OCR libraries available - OCR validation will be skipped"
            )

    def calculate_ocr_confidence_delta(
        self,
        original: np.ndarray,
        compressed: np.ndarray,
        regions: list[tuple[int, int, int, int]],
    ) -> dict[str, float]:
        """Calculate OCR confidence delta between original and compressed frames.

        Args:
            original: Original frame
            compressed: Compressed frame
            regions: List of text regions as (x, y, w, h) tuples

        Returns:
            Dictionary with OCR confidence metrics
        """
        if not regions:
            return {
                "ocr_conf_delta_mean": 0.0,
                "ocr_conf_delta_min": 0.0,
                "ocr_regions_analyzed": 0,
            }

        confidence_deltas = []
        analyzed_regions = 0

        for region in regions:
            x, y, w, h = region

            # Validate region bounds
            if (
                x < 0
                or y < 0
                or x + w > original.shape[1]
                or y + h > original.shape[0]
                or x + w > compressed.shape[1]
                or y + h > compressed.shape[0]
            ):
                logger.debug(f"Skipping invalid region bounds: {region}")
                continue

            # Extract confidence for original and compressed regions
            orig_confidence = self._extract_text_confidence(original, region)
            comp_confidence = self._extract_text_confidence(compressed, region)

            # Skip regions with very low original confidence (likely not text)
            if orig_confidence < 0.2:
                logger.debug(f"Skipping low-confidence region: {orig_confidence:.3f}")
                continue

            # Calculate delta
            delta = self._calculate_confidence_delta(orig_confidence, comp_confidence)
            confidence_deltas.append(delta)
            analyzed_regions += 1

            logger.debug(
                f"Region {region}: orig={orig_confidence:.3f}, comp={comp_confidence:.3f}, delta={delta:.3f}"
            )

        if not confidence_deltas:
            logger.debug("No valid OCR confidence measurements obtained")
            return {
                "ocr_conf_delta_mean": 0.0,
                "ocr_conf_delta_min": 0.0,
                "ocr_regions_analyzed": 0,
            }

        # Calculate aggregate statistics
        mean_delta = float(np.mean(confidence_deltas))
        min_delta = float(
            np.min(confidence_deltas)
        )  # Most negative (worst degradation)

        logger.debug(
            f"OCR analysis: {analyzed_regions} regions, mean_delta={mean_delta:.3f}, min_delta={min_delta:.3f}"
        )

        return {
            "ocr_conf_delta_mean": mean_delta,
            "ocr_conf_delta_min": min_delta,
            "ocr_regions_analyzed": analyzed_regions,
        }

    def _extract_text_confidence(
        self, frame: np.ndarray, region: tuple[int, int, int, int]
    ) -> float:
        """Extract OCR confidence for a specific region.

        Args:
            frame: Input frame
            region: Bounding box (x, y, w, h)

        Returns:
            OCR confidence score (0-1)
        """
        x, y, w, h = region

        # Extract region of interest
        if len(frame.shape) == 3:
            roi = frame[y : y + h, x : x + w]
        else:
            roi = frame[y : y + h, x : x + w]

        # Skip very small regions
        if w < 8 or h < 8:
            return 0.0

        # Preprocess region for better OCR
        processed_roi = self._preprocess_for_ocr(roi)

        if self.use_tesseract and TESSERACT_AVAILABLE:
            try:
                # Use pytesseract with confidence data
                data = pytesseract.image_to_data(
                    processed_roi,
                    output_type=pytesseract.Output.DICT,
                    config="--psm 8 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz ",
                )

                # Extract confidence scores
                confidences = [conf for conf in data["conf"] if conf > 0]
                if confidences:
                    # Return average confidence normalized to 0-1
                    avg_confidence = float(np.mean(confidences) / 100.0)
                    return float(max(0.0, min(1.0, avg_confidence)))

            except Exception as e:
                logger.debug(f"Tesseract OCR failed: {e}")

        if self.fallback_to_easyocr and EASYOCR_AVAILABLE:
            try:
                # Use EasyOCR as fallback
                reader = easyocr.Reader(["en"], gpu=False, verbose=False)
                results = reader.readtext(processed_roi)

                if results:
                    # EasyOCR returns confidence directly
                    confidences = [
                        conf for _, text, conf in results if len(text.strip()) > 0
                    ]
                    if confidences:
                        return float(np.mean(confidences))

            except Exception as e:
                logger.debug(f"EasyOCR failed: {e}")

        # Fallback: Use simple template matching confidence
        return self._template_matching_confidence(processed_roi)

    def _preprocess_for_ocr(self, roi: np.ndarray) -> np.ndarray:
        """Preprocess region of interest for better OCR performance.

        Args:
            roi: Region of interest

        Returns:
            Preprocessed image optimized for OCR
        """
        # Convert to grayscale if needed
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        else:
            gray = roi.copy()

        # Resize if too small (OCR works better on larger text)
        height, width = gray.shape
        if height < 20 or width < 20:
            scale_factor = max(2.0, 20.0 / min(height, width))
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            gray = cv2.resize(
                gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC
            )

        # Apply denoising
        denoised = cv2.fastNlMeansDenoising(gray)

        # Enhance contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)

        # Binarize using adaptive threshold
        binary = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

        return binary

    def _template_matching_confidence(self, roi: np.ndarray) -> float:
        """Fallback confidence estimation using simple image properties.

        Args:
            roi: Preprocessed region of interest

        Returns:
            Estimated confidence based on image properties (0-1)
        """
        # Calculate image sharpness using Laplacian variance
        laplacian_var = float(cv2.Laplacian(roi, cv2.CV_64F).var())

        # Calculate contrast using standard deviation
        contrast = float(roi.std())

        # Calculate edge density
        edges = cv2.Canny(roi, 50, 150)
        edge_density = float(np.sum(edges > 0)) / (roi.shape[0] * roi.shape[1])

        # Combine metrics for confidence estimate
        # Higher sharpness, contrast, and edge density = higher confidence
        sharpness_score = min(1.0, laplacian_var / 100.0)
        contrast_score = min(1.0, contrast / 80.0)
        edge_score = min(1.0, edge_density * 10.0)

        # Weighted combination
        confidence = float(
            0.4 * sharpness_score + 0.3 * contrast_score + 0.3 * edge_score
        )
        return float(max(0.1, min(1.0, confidence)))  # Clamp between 0.1 and 1.0

    def _calculate_confidence_delta(self, orig_conf: float, comp_conf: float) -> float:
        """Calculate confidence delta between original and compressed.

        Args:
            orig_conf: Original confidence score
            comp_conf: Compressed confidence score

        Returns:
            Confidence delta (negative values indicate degradation)
        """
        return comp_conf - orig_conf


class EdgeAcuityAnalyzer:
    """Measures edge sharpness using MTF50 (Modulation Transfer Function)."""

    def __init__(self, mtf_threshold: float = 0.5):
        """Initialize edge acuity analyzer.

        Args:
            mtf_threshold: MTF threshold for MTF50 calculation (default 0.5)
        """
        self.mtf_threshold = mtf_threshold

    def calculate_mtf50(
        self, frame: np.ndarray, regions: list[tuple[int, int, int, int]]
    ) -> dict[str, float]:
        """Calculate MTF50 edge acuity for detected text/UI regions.

        Args:
            frame: Input frame
            regions: List of text/UI regions as (x, y, w, h) tuples

        Returns:
            Dictionary with MTF50 metrics
        """
        if not regions:
            return {
                "mtf50_ratio_mean": 1.0,
                "mtf50_ratio_min": 1.0,
                "edge_sharpness_score": 100.0,
            }

        mtf50_values = []
        analyzed_edges = 0

        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        else:
            gray = frame.copy()

        for region in regions:
            x, y, w, h = region

            # Validate region bounds
            if x < 0 or y < 0 or x + w > gray.shape[1] or y + h > gray.shape[0]:
                logger.debug(f"Skipping invalid region bounds: {region}")
                continue

            # Extract region of interest
            roi = gray[y : y + h, x : x + w]

            # Skip very small regions
            if w < 16 or h < 16:
                continue

            # Find suitable edges in the region
            edge_profiles = self._extract_edge_profiles_from_region(roi)

            for edge_profile in edge_profiles:
                if len(edge_profile) < 10:  # Need sufficient samples
                    continue

                try:
                    # Calculate MTF from edge profile
                    mtf_curve = self._calculate_mtf(edge_profile)
                    mtf50 = self._find_mtf50_frequency(mtf_curve)

                    if mtf50 > 0:  # Valid MTF50 measurement
                        mtf50_values.append(mtf50)
                        analyzed_edges += 1

                except Exception as e:
                    logger.debug(f"MTF50 calculation failed for edge: {e}")
                    continue

        if not mtf50_values:
            logger.debug("No valid MTF50 measurements obtained")
            return {
                "mtf50_ratio_mean": 1.0,
                "mtf50_ratio_min": 1.0,
                "edge_sharpness_score": 50.0,  # Neutral score when no measurements
            }

        # Calculate statistics
        mean_mtf50 = float(np.mean(mtf50_values))
        min_mtf50 = float(np.min(mtf50_values))

        # Convert to sharpness score (higher MTF50 = sharper edges)
        # Normalize to 0-100 scale where 50+ is good sharpness
        sharpness_score = min(100.0, mean_mtf50 * 100.0)

        logger.debug(
            f"MTF50 analysis: {analyzed_edges} edges, mean={mean_mtf50:.3f}, min={min_mtf50:.3f}"
        )

        return {
            "mtf50_ratio_mean": mean_mtf50,
            "mtf50_ratio_min": min_mtf50,
            "edge_sharpness_score": sharpness_score,
        }

    def _extract_edge_profiles_from_region(self, roi: np.ndarray) -> list[np.ndarray]:
        """Extract edge profiles from a region of interest.

        Args:
            roi: Region of interest

        Returns:
            List of 1D edge profiles suitable for MTF analysis
        """
        edge_profiles = []

        # Use Canny edge detection to find strong edges
        edges = cv2.Canny(roi, 50, 150)

        # Find contours to identify edge locations
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        for contour in contours:
            if len(contour) < 20:  # Skip very short contours
                continue

            # Extract edge profiles along the contour
            for i in range(0, len(contour), 5):  # Sample every 5th point
                point = contour[i][0]
                x, y = point

                # Extract horizontal and vertical profiles through this edge point
                profiles = self._extract_profiles_at_point(roi, x, y)
                edge_profiles.extend(profiles)

                # Limit total number of profiles to avoid excessive computation
                if len(edge_profiles) > 20:
                    break

            if len(edge_profiles) > 20:
                break

        return edge_profiles[:20]  # Return up to 20 best profiles

    def _extract_profiles_at_point(
        self, roi: np.ndarray, x: int, y: int
    ) -> list[np.ndarray]:
        """Extract horizontal and vertical intensity profiles at a point.

        Args:
            roi: Region of interest
            x: X coordinate
            y: Y coordinate

        Returns:
            List of intensity profiles (horizontal and vertical)
        """
        profiles = []
        height, width = roi.shape

        # Extract horizontal profile (around y)
        profile_half_width = 8
        if (
            y >= profile_half_width
            and y < height - profile_half_width
            and x >= profile_half_width
            and x < width - profile_half_width
        ):
            # Horizontal profile
            h_profile = roi[
                y,
                max(0, x - profile_half_width) : min(width, x + profile_half_width + 1),
            ]
            if len(h_profile) > 10:
                profiles.append(h_profile.astype(np.float64))

            # Vertical profile
            v_profile = roi[
                max(0, y - profile_half_width) : min(
                    height, y + profile_half_width + 1
                ),
                x,
            ]
            if len(v_profile) > 10:
                profiles.append(v_profile.astype(np.float64))

        return profiles

    def _calculate_mtf(self, edge_profile: np.ndarray) -> np.ndarray:
        """Calculate Modulation Transfer Function from edge profile.

        Args:
            edge_profile: 1D edge profile

        Returns:
            MTF curve as frequency response
        """
        if len(edge_profile) < 10:
            return np.array([])

        try:
            # Smooth the profile to reduce noise
            smoothed = cv2.GaussianBlur(
                edge_profile.reshape(-1, 1), (3, 1), 0
            ).flatten()

            # Calculate the Line Spread Function (LSF) by taking the derivative
            lsf = np.gradient(smoothed)

            # Window the LSF to reduce FFT artifacts
            window = np.hanning(len(lsf))
            windowed_lsf = lsf * window

            # Zero-pad to improve frequency resolution
            padded_length = max(64, 2 ** int(np.ceil(np.log2(len(windowed_lsf)))))
            padded_lsf = np.zeros(padded_length)
            start_idx = (padded_length - len(windowed_lsf)) // 2
            padded_lsf[start_idx : start_idx + len(windowed_lsf)] = windowed_lsf

            # Take FFT to get MTF
            fft_result = np.fft.fft(padded_lsf)
            mtf = np.abs(fft_result)

            # Normalize MTF (peak at 1.0)
            if mtf[0] > 0:
                mtf = mtf / mtf[0]

            # Return only the positive frequencies (first half)
            return mtf[: len(mtf) // 2]

        except Exception as e:
            logger.debug(f"MTF calculation failed: {e}")
            return np.array([])

    def _find_mtf50_frequency(self, mtf_curve: np.ndarray) -> float:
        """Find frequency where MTF drops to 50%.

        Args:
            mtf_curve: MTF frequency response

        Returns:
            MTF50 frequency value (normalized 0-1)
        """
        if len(mtf_curve) < 5:
            return 0.0

        try:
            # Find where MTF drops below the threshold (default 0.5)
            below_threshold = np.where(mtf_curve < self.mtf_threshold)[0]

            if len(below_threshold) == 0:
                # MTF never drops below threshold - very sharp edge
                return 1.0

            # Get the first index where MTF drops below threshold
            mtf50_index = below_threshold[0]

            if mtf50_index == 0:
                # MTF starts below threshold - very blurry edge
                return 0.0

            # Interpolate for more precise MTF50 frequency
            if mtf50_index < len(mtf_curve) - 1:
                # Linear interpolation between the points above and below threshold
                y1 = mtf_curve[mtf50_index - 1]
                y2 = mtf_curve[mtf50_index]
                x1 = mtf50_index - 1
                x2 = mtf50_index

                # Find exact crossing point
                if y1 != y2:
                    x_interp = x1 + (self.mtf_threshold - y1) / (y2 - y1) * (x2 - x1)
                else:
                    x_interp = mtf50_index
            else:
                x_interp = mtf50_index

            # Normalize to 0-1 range based on curve length
            normalized_freq = x_interp / len(mtf_curve)

            return float(min(1.0, max(0.0, normalized_freq)))

        except Exception as e:
            logger.debug(f"MTF50 frequency calculation failed: {e}")
            return 0.0


def should_validate_text_ui(
    frames: list[np.ndarray], quick_check: bool = True
) -> tuple[bool, dict[str, Any]]:
    """Determine if frames contain text/UI content worth validating.

    Args:
        frames: List of frames to analyze
        quick_check: If True, only check first frame for speed

    Returns:
        Tuple of (should_validate, content_hints)

    Criteria:
    - Edge density in the 3-10% range (text/UI typically has moderate edge density)
    - Presence of small, regular components (10-500 pixel area)
    - High contrast regions with sharp boundaries
    - Text-like aspect ratios (0.5-2.0 for individual components)
    """
    if not frames:
        return False, {"edge_density": 0.0, "component_count": 0}

    # Determine frames to analyze
    if quick_check:
        analyze_frames = frames[:1]  # Just first frame
    else:
        # Sample up to 3 frames from different positions
        frame_count = len(frames)
        if frame_count <= 3:
            analyze_frames = frames
        else:
            indices = [0, frame_count // 2, frame_count - 1]
            analyze_frames = [frames[i] for i in indices]

    # Initialize detector for content analysis with lower thresholds
    detector = TextUIContentDetector(
        edge_threshold=30.0,  # Lower threshold for better text detection
        min_component_area=8,  # Smaller minimum for detection
        max_component_area=1000,  # Larger maximum for detection
        edge_density_threshold=0.03,  # 3% threshold for detection (lowered from 8%)
    )

    total_edge_density = 0.0
    total_components = 0
    max_edge_density = 0.0

    for frame in analyze_frames:
        # Calculate edge density
        edge_density = detector._detect_edge_density(frame)
        total_edge_density += edge_density
        max_edge_density = max(max_edge_density, edge_density)

        # Quick component analysis
        regions = detector.detect_text_ui_regions(frame)
        total_components += len(regions)

    # Calculate averages
    avg_edge_density = total_edge_density / len(analyze_frames)
    avg_components = total_components / len(analyze_frames)

    # Content analysis hints
    content_hints = {
        "edge_density": float(avg_edge_density),
        "max_edge_density": float(max_edge_density),
        "component_count": int(total_components),
        "avg_components_per_frame": float(avg_components),
        "frames_analyzed": len(analyze_frames),
    }

    # Decision criteria - text/UI typically has moderate edge density (3-10%)
    # Subtitles may have lower edge density (2-3%)
    # Too high edge density (>15%) usually indicates patterns/noise, not text

    # Check for subtitle patterns (lower edge density but specific location/contrast)
    # Subtitles often have very clean text that produces lower edge density
    is_subtitle_pattern = (
        0.015
        <= avg_edge_density
        <= 0.035  # Subtitle edge density range (1.5-3.5%)
        # Don't require components for subtitles - they may be too clean to detect as components
    )

    # Standard text/UI detection
    is_standard_text = (
        (0.03 <= avg_edge_density <= 0.10)  # Text/UI typical range
        or (0.03 <= max_edge_density <= 0.12)  # Peak edge density in range
        or avg_components >= 2.0  # Average 2+ components per frame
        or total_components >= 5  # At least 5 total components found
    )

    should_validate = (
        is_subtitle_pattern or is_standard_text
    ) and avg_edge_density < 0.15  # Exclude high-noise patterns

    logger.debug(
        f"Text/UI content detection: should_validate={should_validate}, "
        f"avg_edge_density={avg_edge_density:.3f}, max_edge_density={max_edge_density:.3f}, "
        f"total_components={total_components}"
    )

    return should_validate, content_hints


def calculate_text_ui_metrics(
    original_frames: list[np.ndarray],
    compressed_frames: list[np.ndarray],
    max_frames: int = 5,
) -> dict[str, Any]:
    """Calculate text/UI validation metrics between original and compressed frames.

    Main entry point for text/UI content validation. Only called when
    text/UI content is detected to control computational cost.

    Args:
        original_frames: List of original frames
        compressed_frames: List of compressed frames
        max_frames: Maximum number of frames to analyze

    Returns:
        Dictionary containing text/UI validation metrics
    """
    if len(original_frames) != len(compressed_frames):
        raise ValueError("Original and compressed frame counts must match")

    # Sample frames for analysis
    frame_count = min(len(original_frames), max_frames)
    sample_indices = np.linspace(0, len(original_frames) - 1, frame_count, dtype=int)

    # Initialize detectors
    detector = TextUIContentDetector()
    ocr_validator = OCRValidator()
    edge_analyzer = EdgeAcuityAnalyzer()

    # Check if frames actually contain text/UI content
    sample_frames = [
        original_frames[i] for i in sample_indices[:3]
    ]  # Quick check on 3 frames
    should_validate, content_hints = should_validate_text_ui(sample_frames)

    if not should_validate:
        logger.debug("No text/UI content detected - skipping validation")
        return {
            "has_text_ui_content": False,
            "text_ui_edge_density": content_hints.get("edge_density", 0.0),
            "text_ui_component_count": content_hints.get("component_count", 0),
        }

    logger.info(f"Analyzing text/UI content in {frame_count} frames")

    # Collect metrics across sampled frames
    all_ocr_deltas = []
    all_mtf50_ratios = []
    total_regions = 0

    for i in sample_indices:
        orig_frame = original_frames[i]
        comp_frame = compressed_frames[i]

        # Detect text/UI regions in original frame
        text_regions = detector.detect_text_ui_regions(orig_frame)
        total_regions += len(text_regions)

        if len(text_regions) == 0:
            continue

        # Calculate OCR confidence delta
        ocr_metrics = ocr_validator.calculate_ocr_confidence_delta(
            orig_frame, comp_frame, text_regions
        )
        all_ocr_deltas.extend([ocr_metrics["ocr_conf_delta_mean"]])

        # Calculate MTF50 for original and compressed
        orig_mtf = edge_analyzer.calculate_mtf50(orig_frame, text_regions)
        comp_mtf = edge_analyzer.calculate_mtf50(comp_frame, text_regions)

        if orig_mtf["mtf50_ratio_mean"] > 0:
            mtf50_ratio = comp_mtf["mtf50_ratio_mean"] / orig_mtf["mtf50_ratio_mean"]
            all_mtf50_ratios.append(mtf50_ratio)

    # Aggregate results
    result_metrics = {
        "has_text_ui_content": True,
        "text_ui_edge_density": content_hints.get("edge_density", 0.0),
        "text_ui_component_count": content_hints.get("component_count", 0),
        "ocr_regions_analyzed": total_regions,
    }

    if all_ocr_deltas:
        result_metrics.update(
            {
                "ocr_conf_delta_mean": float(np.mean(all_ocr_deltas)),
                "ocr_conf_delta_min": float(np.min(all_ocr_deltas)),
            }
        )

    if all_mtf50_ratios:
        result_metrics.update(
            {
                "mtf50_ratio_mean": float(np.mean(all_mtf50_ratios)),
                "mtf50_ratio_min": float(np.min(all_mtf50_ratios)),
                "edge_sharpness_score": min(
                    100.0, float(np.mean(all_mtf50_ratios)) * 100.0
                ),
            }
        )

    logger.info(f"Text/UI validation complete: {total_regions} regions analyzed")
    return result_metrics
