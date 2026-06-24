"""Metadata extraction and hashing for GIF files."""

# Version of the GIF metadata-extraction contract. The official value is a human
# decision supplied out-of-band and resolved at review time; "TBD" is a placeholder.
__GIFLAB_META_VERSION__ = "TBD"

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


@dataclass
class GifMetadata:
    """Metadata extracted from a GIF file."""

    gif_sha: str
    orig_filename: str
    orig_kilobytes: float
    orig_width: int
    orig_height: int
    orig_frames: int
    orig_fps: float
    orig_n_colors: int
    entropy: float | None = None

    # Source tracking fields
    source_platform: str = "unknown"
    source_metadata: dict[str, Any] | None = None


def compute_file_sha256(file_path: Path) -> str:
    """Compute SHA256 hash of a file.

    Args:
        file_path: Path to the file to hash

    Returns:
        Hexadecimal SHA256 hash string
    """
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        # Read file in chunks to handle large files
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def extract_gif_metadata(
    file_path: Path,
    source_platform: str = "unknown",
    source_metadata: dict[str, Any] | None = None,
) -> GifMetadata:
    """Extract metadata from a GIF file.

    Args:
        file_path: Path to the GIF file
        source_platform: Platform where this GIF was sourced from
        source_metadata: Additional metadata about the source/collection context

    Returns:
        GifMetadata object with extracted information

    Raises:
        ValueError: If file is not a valid GIF
        IOError: If file cannot be read
    """
    if not file_path.exists():
        raise OSError(f"File not found: {file_path}")

    # Compute file hash and size
    gif_sha = compute_file_sha256(file_path)
    file_size_bytes = file_path.stat().st_size
    orig_kilobytes = file_size_bytes / 1024.0

    try:
        # Open GIF with PIL
        with Image.open(file_path) as img:
            if img.format != "GIF":
                raise ValueError(f"File is not a GIF: {file_path}")

            # Basic dimensions
            orig_width, orig_height = img.size

            # Count frames using safer method
            frame_count = 0
            durations = []

            try:
                # Use PIL's built-in frame counting if available
                if hasattr(img, "n_frames"):
                    frame_count = img.n_frames
                    # Still need to collect durations for FPS calculation
                    for i in range(frame_count):
                        try:
                            img.seek(i)
                            duration = img.info.get("duration", 100)
                            durations.append(duration)
                        except Exception:
                            durations.append(100)  # Default fallback
                else:
                    # Fallback to manual counting with safety limits
                    current_frame = 0
                    while True:
                        try:
                            img.seek(current_frame)
                            frame_count = current_frame + 1
                            duration = img.info.get("duration", 100)
                            durations.append(duration)
                            current_frame += 1
                        except EOFError:
                            break
                        except Exception:
                            break

                        # Safety limit to prevent infinite loops
                        if current_frame > 10000:
                            raise ValueError(
                                f"GIF appears to have excessive frames (>{current_frame}), possibly corrupted"
                            )

            except EOFError:
                pass  # Normal end of frames
            except Exception as e:
                raise ValueError(f"Error counting frames in GIF: {e}") from e

            # Validate frame count
            if frame_count <= 0:
                frame_count = 1  # At least one frame for valid GIF
                durations = [100]  # Default duration

            # Calculate average FPS from frame durations
            if durations:
                avg_duration_ms = sum(durations) / len(durations)
                orig_fps = 1000.0 / avg_duration_ms if avg_duration_ms > 0 else 10.0
            else:
                orig_fps = 10.0  # Default fallback

            # Count unique colors by examining palette
            img.seek(0)  # Go back to first frame

            if img.mode == "P":  # Palette mode
                # Get palette and count unique colors
                palette = img.getpalette()
                if palette:
                    # Palette is in RGB format, so divide by 3 to get color count
                    orig_n_colors = len(
                        {tuple(palette[i : i + 3]) for i in range(0, len(palette), 3)}
                    )
                else:
                    orig_n_colors = 256  # Default max for palette mode
            else:
                # Convert to palette mode to count colors
                quantized = img.quantize(colors=256)
                palette = quantized.getpalette()
                if palette:
                    orig_n_colors = len(
                        {tuple(palette[i : i + 3]) for i in range(0, len(palette), 3)}
                    )
                else:
                    orig_n_colors = 256

            # Calculate entropy for the first frame
            img.seek(0)
            entropy = calculate_entropy(img)

    except Exception as e:
        raise ValueError(f"Error processing GIF {file_path}: {str(e)}") from e

    return GifMetadata(
        gif_sha=gif_sha,
        orig_filename=file_path.name,
        orig_kilobytes=orig_kilobytes,
        orig_width=orig_width,
        orig_height=orig_height,
        orig_frames=frame_count,
        orig_fps=round(orig_fps, 2),
        orig_n_colors=orig_n_colors,
        entropy=entropy,
        source_platform=source_platform,
        source_metadata=source_metadata,
    )


def calculate_entropy(image: Image.Image) -> float:
    """Calculate entropy of an image for complexity measurement.

    Args:
        image: PIL Image object

    Returns:
        Entropy value as float
    """
    # Convert to grayscale if needed
    if image.mode != "L":
        gray_image = image.convert("L")
    else:
        gray_image = image

    # Convert to numpy array
    img_array = np.array(gray_image)

    # Calculate histogram
    histogram, _ = np.histogram(img_array, bins=256, range=(0, 256))

    # Normalize histogram to get probabilities
    histogram_sum = histogram.sum()
    if histogram_sum == 0:
        # Handle edge case of empty or invalid image
        return 0.0

    histogram = histogram / histogram_sum

    # Remove zero probabilities to avoid log(0)
    histogram = histogram[histogram > 0]

    # Handle edge case where image is completely uniform
    if len(histogram) <= 1:
        return 0.0

    # Calculate Shannon entropy: -sum(p * log2(p))
    entropy = -np.sum(histogram * np.log2(histogram))

    # Ensure non-negative result (handle floating point precision issues)
    entropy = max(0.0, entropy)

    return float(round(entropy, 3))
