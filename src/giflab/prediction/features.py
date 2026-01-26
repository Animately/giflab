"""Feature extraction for compression curve prediction.

This module extracts visual features from GIFs for training compression
prediction models. It combines features from the existing tagger with
new compressibility-specific features.

Constitution Compliance:
- Principle II (ML-Ready Data): Deterministic, schema-validated outputs
- Principle VI (LLM-Optimized): Explicit patterns, type hints, docstrings
"""

import hashlib
import logging
import zlib
from datetime import datetime, timezone
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

from giflab.meta import extract_gif_metadata
from giflab.prediction import FEATURE_EXTRACTOR_VERSION
from giflab.prediction.schemas import GifFeaturesV1

logger = logging.getLogger(__name__)

# Maximum frames to process for temporal features
MAX_FRAMES_FOR_FEATURES = 50

# Frame sampling for very large GIFs
LARGE_GIF_FRAME_THRESHOLD = 500


def extract_gif_features(gif_path: Path) -> GifFeaturesV1:
    """Extract all visual features from a GIF for compression prediction.

    This is the main entry point for feature extraction. It combines:
    - Basic metadata (dimensions, frame count, colors)
    - Spatial features (entropy, edges, gradients, colors)
    - Temporal features (motion, static regions, inter-frame differences)
    - Compressibility features (lossless ratio, DCT energy)

    Args:
        gif_path: Path to the GIF file to analyze.

    Returns:
        GifFeaturesV1 schema with all extracted features.

    Raises:
        FileNotFoundError: If GIF file does not exist.
        ValueError: If GIF cannot be processed.
    """
    if not gif_path.exists():
        raise FileNotFoundError(f"GIF not found: {gif_path}")

    logger.debug(f"Extracting features from: {gif_path}")

    # Get basic metadata
    metadata = extract_gif_metadata(gif_path)
    gif_sha = metadata.gif_sha

    # Extract frames for analysis
    frames = _extract_frames_for_analysis(gif_path)
    if not frames:
        raise ValueError(f"Could not extract frames from GIF: {gif_path}")

    # Get representative frame for spatial analysis
    representative_frame = frames[len(frames) // 2]  # Middle frame

    # Calculate all features
    spatial_features = _extract_spatial_features(representative_frame)
    temporal_features = _extract_temporal_features(frames)
    compressibility_features = _extract_compressibility_features(
        gif_path, frames, representative_frame
    )
    color_features = _extract_color_features(frames)
    transparency_ratio = _calculate_transparency_ratio(gif_path)

    # Calculate duration from fps and frame count
    duration_ms = int((metadata.orig_frames / max(metadata.orig_fps, 0.1)) * 1000)
    file_size_bytes = int(metadata.orig_kilobytes * 1024)

    # Build the feature object
    return GifFeaturesV1(
        gif_sha=gif_sha,
        gif_name=gif_path.name,
        extraction_version=FEATURE_EXTRACTOR_VERSION,
        extracted_at=datetime.now(timezone.utc),
        # Metadata
        width=metadata.orig_width,
        height=metadata.orig_height,
        frame_count=metadata.orig_frames,
        duration_ms=duration_ms,
        file_size_bytes=file_size_bytes,
        unique_colors=min(metadata.orig_n_colors, 256),
        # Spatial features
        entropy=spatial_features["entropy"],
        edge_density=spatial_features["edge_density"],
        color_complexity=spatial_features["color_complexity"],
        gradient_smoothness=spatial_features["gradient_smoothness"],
        contrast_score=spatial_features["contrast_score"],
        text_density=spatial_features["text_density"],
        dct_energy_ratio=compressibility_features["dct_energy_ratio"],
        color_histogram_entropy=color_features["color_histogram_entropy"],
        dominant_color_ratio=color_features["dominant_color_ratio"],
        # Temporal features
        motion_intensity=temporal_features["motion_intensity"],
        motion_smoothness=temporal_features["motion_smoothness"],
        static_region_ratio=temporal_features["static_region_ratio"],
        temporal_entropy=temporal_features["temporal_entropy"],
        frame_similarity=temporal_features["frame_similarity"],
        inter_frame_mse_mean=temporal_features["inter_frame_mse_mean"],
        inter_frame_mse_std=temporal_features["inter_frame_mse_std"],
        # Compressibility
        lossless_compression_ratio=compressibility_features[
            "lossless_compression_ratio"
        ],
        transparency_ratio=transparency_ratio,
    )


def _extract_frames_for_analysis(
    gif_path: Path,
    max_frames: int = MAX_FRAMES_FOR_FEATURES,
) -> list[np.ndarray]:
    """Extract frames from GIF for feature analysis.

    For large GIFs, samples frames evenly to stay within limits.

    Args:
        gif_path: Path to the GIF file.
        max_frames: Maximum number of frames to extract.

    Returns:
        List of RGB numpy arrays (H, W, 3).
    """
    cap = None
    try:
        cap = cv2.VideoCapture(str(gif_path))
        if not cap.isOpened():
            logger.error(f"Failed to open GIF: {gif_path}")
            return []

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            logger.error(f"Invalid frame count for {gif_path}")
            return []

        # Determine which frames to extract
        if frame_count <= max_frames:
            frame_indices = list(range(frame_count))
        else:
            # Sample evenly across the GIF
            frame_indices = np.linspace(
                0, frame_count - 1, max_frames, dtype=int
            ).tolist()

        frames = []
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)

        return frames

    except Exception as e:
        logger.error(f"Frame extraction failed for {gif_path}: {e}")
        return []
    finally:
        if cap is not None:
            cap.release()


def _extract_spatial_features(frame: np.ndarray) -> dict[str, float]:
    """Extract spatial features from a single frame.

    Args:
        frame: RGB numpy array (H, W, 3).

    Returns:
        Dictionary of spatial feature values.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    return {
        "entropy": _calculate_entropy(gray),
        "edge_density": _calculate_edge_density(gray),
        "color_complexity": _calculate_color_complexity(frame),
        "gradient_smoothness": _calculate_gradient_smoothness(gray),
        "contrast_score": _calculate_contrast_score(gray),
        "text_density": _calculate_text_density(gray),
    }


def _extract_temporal_features(frames: list[np.ndarray]) -> dict[str, float]:
    """Extract temporal features from frame sequence.

    Args:
        frames: List of RGB numpy arrays.

    Returns:
        Dictionary of temporal feature values.
    """
    if len(frames) < 2:
        return {
            "motion_intensity": 0.0,
            "motion_smoothness": 1.0,
            "static_region_ratio": 1.0,
            "temporal_entropy": 0.0,
            "frame_similarity": 1.0,
            "inter_frame_mse_mean": 0.0,
            "inter_frame_mse_std": 0.0,
        }

    # Calculate inter-frame differences
    frame_diffs = []
    mse_values = []

    for i in range(len(frames) - 1):
        diff = np.abs(
            frames[i].astype(np.float32) - frames[i + 1].astype(np.float32)
        )
        frame_diffs.append(diff)

        mse = np.mean(diff**2)
        mse_values.append(mse)

    mse_array = np.array(mse_values)

    # Motion intensity: normalized mean difference
    motion_intensity = float(np.mean(mse_array) / (255**2))
    motion_intensity = min(motion_intensity * 10, 1.0)  # Scale to 0-1

    # Motion smoothness: inverse of MSE variance (smooth = consistent motion)
    if len(mse_values) > 1:
        mse_std = float(np.std(mse_array))
        motion_smoothness = 1.0 / (1.0 + mse_std / 1000)
    else:
        motion_smoothness = 1.0

    # Static region ratio: percentage of pixels that don't change much
    static_threshold = 10  # Pixel difference threshold
    static_ratios = []
    for diff in frame_diffs:
        static_pixels = np.mean(diff < static_threshold)
        static_ratios.append(static_pixels)
    static_region_ratio = float(np.mean(static_ratios))

    # Temporal entropy: complexity of motion patterns
    if len(mse_values) > 1:
        # Normalize MSE values to probability distribution
        mse_norm = mse_array / (np.sum(mse_array) + 1e-10)
        temporal_entropy = float(-np.sum(mse_norm * np.log2(mse_norm + 1e-10)))
        temporal_entropy = min(temporal_entropy, 8.0)  # Cap at 8 bits
    else:
        temporal_entropy = 0.0

    # Frame similarity: average structural similarity
    frame_similarity = 1.0 - motion_intensity

    return {
        "motion_intensity": motion_intensity,
        "motion_smoothness": motion_smoothness,
        "static_region_ratio": static_region_ratio,
        "temporal_entropy": temporal_entropy,
        "frame_similarity": frame_similarity,
        "inter_frame_mse_mean": float(np.mean(mse_array)),
        "inter_frame_mse_std": float(np.std(mse_array)) if len(mse_values) > 1 else 0.0,
    }


def _extract_compressibility_features(
    gif_path: Path,
    frames: list[np.ndarray],
    representative_frame: np.ndarray,
) -> dict[str, float]:
    """Extract features that predict compressibility.

    Args:
        gif_path: Path to the GIF file.
        frames: List of extracted frames.
        representative_frame: Frame for DCT analysis.

    Returns:
        Dictionary of compressibility features.
    """
    # Lossless compression ratio: how well does zlib compress the raw data
    raw_data = b"".join(frame.tobytes() for frame in frames[:10])
    compressed_size = len(zlib.compress(raw_data, level=9))
    lossless_ratio = compressed_size / (len(raw_data) + 1)
    lossless_ratio = min(lossless_ratio, 2.0)  # Cap at 2.0

    # DCT energy ratio: high-freq vs low-freq energy
    dct_energy_ratio = _calculate_dct_energy_ratio(representative_frame)

    return {
        "lossless_compression_ratio": lossless_ratio,
        "dct_energy_ratio": dct_energy_ratio,
    }


def _extract_color_features(frames: list[np.ndarray]) -> dict[str, float]:
    """Extract color distribution features.

    Args:
        frames: List of RGB numpy arrays.

    Returns:
        Dictionary of color features.
    """
    # Use first few frames for color analysis
    sample_frames = frames[:5]

    # Combine pixels from sample frames
    all_pixels = np.vstack([f.reshape(-1, 3) for f in sample_frames])

    # Color histogram entropy
    hist_entropy = _calculate_color_histogram_entropy(all_pixels)

    # Dominant color ratio: top 10 colors as percentage of all pixels
    dominant_ratio = _calculate_dominant_color_ratio(all_pixels)

    return {
        "color_histogram_entropy": hist_entropy,
        "dominant_color_ratio": dominant_ratio,
    }


def _calculate_entropy(gray: np.ndarray) -> float:
    """Calculate image entropy (information content)."""
    hist, _ = np.histogram(gray.flatten(), bins=256, range=(0, 256))
    hist = hist / hist.sum()
    hist = hist[hist > 0]
    entropy = -np.sum(hist * np.log2(hist))
    return min(float(entropy), 8.0)


def _calculate_edge_density(gray: np.ndarray) -> float:
    """Calculate ratio of edge pixels to total pixels."""
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.mean(edges > 0)
    return float(edge_density)


def _calculate_color_complexity(frame: np.ndarray) -> float:
    """Calculate color distribution complexity."""
    # Quantize to reduce computation
    quantized = (frame // 32) * 32
    unique_colors = len(np.unique(quantized.reshape(-1, 3), axis=0))
    # Normalize to 0-1 (max 512 quantized colors)
    return min(unique_colors / 512, 1.0)


def _calculate_gradient_smoothness(gray: np.ndarray) -> float:
    """Calculate gradient transition smoothness."""
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

    # Smoothness is inverse of gradient variance
    grad_std = np.std(gradient_magnitude)
    smoothness = 1.0 / (1.0 + grad_std / 50)
    return float(smoothness)


def _calculate_contrast_score(gray: np.ndarray) -> float:
    """Calculate image contrast level."""
    contrast = np.std(gray) / 128.0
    return min(float(contrast), 1.0)


def _calculate_text_density(gray: np.ndarray) -> float:
    """Estimate density of text/UI elements using edge patterns."""
    # High-frequency content suggests text/UI
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    text_density = np.mean(np.abs(laplacian)) / 50
    return min(float(text_density), 1.0)


def _calculate_dct_energy_ratio(frame: np.ndarray) -> float:
    """Calculate ratio of high-frequency to low-frequency DCT energy.

    High ratio = more detail = harder to compress with lossy.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32)

    # Resize to standard size for consistent DCT
    gray = cv2.resize(gray, (64, 64))

    # Apply DCT
    dct = cv2.dct(gray)

    # Split into low and high frequency regions
    low_freq = dct[:16, :16]
    high_freq_mask = np.ones_like(dct)
    high_freq_mask[:16, :16] = 0
    high_freq = dct * high_freq_mask

    low_energy = np.sum(np.abs(low_freq))
    high_energy = np.sum(np.abs(high_freq))

    if low_energy + high_energy < 1e-10:
        return 0.5

    ratio = high_energy / (low_energy + high_energy)
    return float(ratio)


def _calculate_color_histogram_entropy(pixels: np.ndarray) -> float:
    """Calculate entropy of color distribution."""
    # Quantize colors to 8 levels per channel
    quantized = pixels // 32

    # Create color codes
    color_codes = (
        quantized[:, 0] * 64 + quantized[:, 1] * 8 + quantized[:, 2]
    )

    # Calculate histogram
    hist, _ = np.histogram(color_codes, bins=512, range=(0, 512))
    hist = hist / hist.sum()
    hist = hist[hist > 0]

    entropy = -np.sum(hist * np.log2(hist))
    return min(float(entropy), 8.0)


def _calculate_dominant_color_ratio(pixels: np.ndarray) -> float:
    """Calculate ratio of pixels using top-10 colors."""
    # Quantize to reduce unique colors
    quantized = (pixels // 16) * 16

    # Count unique colors
    unique, counts = np.unique(quantized, axis=0, return_counts=True)

    # Get top 10 colors
    top_indices = np.argsort(counts)[-10:]
    top_counts = counts[top_indices]

    dominant_ratio = top_counts.sum() / len(pixels)
    return float(dominant_ratio)


def _calculate_transparency_ratio(gif_path: Path) -> float:
    """Calculate ratio of transparent pixels in the GIF."""
    try:
        with Image.open(gif_path) as img:
            if img.mode != "RGBA" and "transparency" not in img.info:
                return 0.0

            # Check first frame
            img_rgba = img.convert("RGBA")
            alpha = np.array(img_rgba)[:, :, 3]
            transparent_pixels = np.sum(alpha < 128)
            total_pixels = alpha.size

            return float(transparent_pixels / total_pixels)

    except Exception as e:
        logger.debug(f"Could not calculate transparency for {gif_path}: {e}")
        return 0.0


def compute_gif_sha(gif_path: Path) -> str:
    """Compute SHA256 hash of a GIF file.

    Args:
        gif_path: Path to the GIF file.

    Returns:
        Lowercase hex SHA256 hash (64 characters).
    """
    sha256 = hashlib.sha256()
    with open(gif_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest().lower()
