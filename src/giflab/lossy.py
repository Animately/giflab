"""Lossy compression functionality for GIF optimization.

This module provides unified interfaces for both gifsicle and animately compression engines.
Each engine has different command-line syntax and capabilities, but this module provides
a consistent Python API for both.

Engine Documentation & Best Practices:

Gifsicle (https://www.lcdf.org/gifsicle/):
- Powerful command-line GIF manipulation tool
- Frame selection syntax: #0 #2 #4 (specify frames to keep)
- Optimization levels: --optimize, -O1, -O2, -O3
- Lossy compression: --lossy=LEVEL (0-200, higher = more compression)
- Color reduction: --colors N (reduce palette to N colors)
- Command structure: gifsicle [OPTIONS] INPUT [FRAME_SELECTION] --output OUTPUT

Animately Engine:
- Internal compression engine with streamlined CLI
- Frame reduction: --reduce RATIO (0.0-1.0, decimal format)
- Lossy compression: --lossy LEVEL (compression level)
- Color reduction: --colors N (reduce palette to N colors)
- Command structure: animately --input INPUT [OPTIONS] --output OUTPUT

Key Differences:
- Gifsicle: Frame selection by index (#0 #2 #4), input file before frame args
- Animately: Frame reduction by ratio (--reduce 0.5), input/output via flags
- Gifsicle: More complex but powerful optimization options
- Animately: Simpler syntax, consistent flag-based interface

Usage Examples:
    # Gifsicle frame reduction (keep every other frame)
    gifsicle --optimize input.gif #0 #2 #4 #6 --output output.gif

    # Animately frame reduction (keep 50% of frames)
    animately --input input.gif --reduce 0.50 --output output.gif

    # Both engines with lossy compression
    gifsicle --optimize --lossy=40 input.gif --output output.gif
    animately --input input.gif --lossy 40 --output output.gif
"""

import json
import logging
import os
import subprocess
import tempfile
import time
from collections.abc import Generator
from contextlib import contextmanager
from enum import Enum
from pathlib import Path
from shutil import rmtree, which
from typing import Any

RUN_TIMEOUT = int(os.getenv("GIFLAB_RUN_TIMEOUT", "10"))

# Constants for advanced lossy compression
MIN_FRAME_DELAY_MS = 20  # Minimum frame delay in milliseconds (GIF standard)
ADVANCED_TIMEOUT_MULTIPLIER = 2  # Timeout multiplier for PNG sequence processing

from .color_keep import (
    build_animately_color_args,
    build_gifsicle_color_args,
    validate_color_keep_count,
)
from .config import DEFAULT_ENGINE_CONFIG
from .external_engines.imagemagick import export_png_sequence
from .frame_keep import (
    build_animately_frame_args,
    build_gifsicle_frame_args,
    validate_frame_keep_ratio,
)
from .input_validation import validate_path_security
from .meta import extract_gif_metadata
from .system_tools import discover_tool


class LossyEngine(Enum):
    """Supported lossy compression engines."""

    GIFSICLE = "gifsicle"
    ANIMATELY_STANDARD = "animately-standard"
    ANIMATELY = "animately-standard"  # Alias for ANIMATELY_STANDARD (legacy name)
    ANIMATELY_ADVANCED = "animately-advanced"
    ANIMATELY_HARD = "animately-hard"
    IMAGEMAGICK = "imagemagick"
    FFMPEG = "ffmpeg"
    GIFSKI = "gifski"


def get_gifsicle_version() -> str:
    """Get the version of gifsicle.

    Returns:
        Version string of gifsicle (e.g., "1.94")

    Raises:
        RuntimeError: If gifsicle is not available or version cannot be determined
    """
    gifsicle_path = DEFAULT_ENGINE_CONFIG.GIFSICLE_PATH

    if not gifsicle_path or not Path(gifsicle_path).exists():
        raise RuntimeError(f"Gifsicle not found at {gifsicle_path}")

    try:
        result = subprocess.run(
            [gifsicle_path, "--version"], capture_output=True, text=True, timeout=10
        )

        if result.returncode != 0:
            raise RuntimeError(f"Gifsicle version check failed: {result.stderr}")

        # Parse version from output (e.g., "LCDF Gifsicle 1.94")
        output_lines = result.stdout.strip().split("\n")
        for line in output_lines:
            if "gifsicle" in line.lower() and any(char.isdigit() for char in line):
                # Extract version number (e.g., "1.94" from "LCDF Gifsicle 1.94")
                words = line.split()
                for word in words:
                    if "." in word and any(char.isdigit() for char in word):
                        return word

        # Fallback - return first line if version parsing fails
        return output_lines[0] if output_lines else "unknown"

    except subprocess.TimeoutExpired:
        raise RuntimeError("Gifsicle version check timed out") from None
    except Exception as e:
        raise RuntimeError(f"Failed to get gifsicle version: {str(e)}") from e


def get_animately_version() -> str:
    """Get the version of animately.

    Returns:
        Version string of animately (e.g., "1.0.0" or "animately-engine" if version not detectable)

    Raises:
        RuntimeError: If animately is not available
    """
    animately_path = DEFAULT_ENGINE_CONFIG.ANIMATELY_PATH

    if not animately_path or not Path(animately_path).exists():
        raise RuntimeError(f"Animately not found at {animately_path}")

    try:
        # Try --version first
        result = subprocess.run(
            [animately_path, "--version"], capture_output=True, text=True, timeout=10
        )

        if result.returncode == 0 and result.stdout.strip():
            output_lines = result.stdout.strip().split("\n")
            for line in output_lines:
                if any(char.isdigit() for char in line):
                    # Extract version number if found
                    words = line.split()
                    for word in words:
                        if "." in word and any(char.isdigit() for char in word):
                            return word
            return output_lines[0] if output_lines else "animately-engine"

        # If --version fails, try --help and look for version info
        result = subprocess.run(
            [animately_path, "--help"], capture_output=True, text=True, timeout=10
        )

        if result.returncode == 0 or result.stderr:
            # Check both stdout and stderr for version information
            output_text = (result.stdout + result.stderr).lower()
            if "version" in output_text:
                # Try to extract version from help text
                lines = output_text.split("\n")
                for line in lines:
                    if "version" in line and any(char.isdigit() for char in line):
                        words = line.split()
                        for word in words:
                            if "." in word and any(char.isdigit() for char in word):
                                return word

        # Fallback - return a generic identifier
        return "animately-engine"

    except subprocess.TimeoutExpired:
        raise RuntimeError("Animately version check timed out") from None
    except Exception as e:
        raise RuntimeError(f"Failed to get animately version: {str(e)}") from e


def get_engine_version(engine: LossyEngine) -> str:
    """Get the version of the specified engine.

    Args:
        engine: Engine to get version for

    Returns:
        Version string of the engine

    Raises:
        RuntimeError: If engine is not available or version cannot be determined
    """
    if engine == LossyEngine.GIFSICLE:
        return get_gifsicle_version()
    elif engine == LossyEngine.ANIMATELY:
        return get_animately_version()
    else:
        raise ValueError(f"Unsupported engine: {engine}")


def _analyze_gif_disposal_complexity(input_path: Path) -> dict[str, Any]:
    """Analyze GIF structure to determine disposal method complexity.

    This function examines the GIF frame structure to detect complex optimized GIFs
    that rely heavily on disposal methods and may be corrupted by frame reduction.

    Args:
        input_path: Path to the input GIF file

    Returns:
        Dictionary with complexity analysis:
        - complexity_score: Float 0.0-1.0 (higher = more complex disposal dependencies)
        - has_variable_frames: Boolean indicating frames have different sizes
        - has_offsets: Boolean indicating frames use non-zero offsets
        - frame_size_variance: Standard deviation of frame sizes (normalized)
        - disposal_risk: Assessment of disposal corruption risk

    Note:
        Uses ImageMagick identify to analyze frame structure without loading pixel data.
    """
    import re
    import statistics
    import subprocess

    try:
        # Check if ImageMagick is available
        imagemagick_tool = discover_tool("imagemagick")
        if not imagemagick_tool.available:
            # Fallback to low complexity if ImageMagick is unavailable
            return {
                "complexity_score": 0.0,
                "has_variable_frames": False,
                "has_offsets": False,
                "frame_size_variance": 0.0,
                "disposal_risk": "low",
            }

        # Use ImageMagick identify to get frame geometry information
        result = subprocess.run(
            ["identify", str(input_path)], capture_output=True, text=True, timeout=30
        )

        if result.returncode != 0:
            # Fallback to low complexity if analysis fails
            return {
                "complexity_score": 0.0,
                "has_variable_frames": False,
                "has_offsets": False,
                "frame_size_variance": 0.0,
                "disposal_risk": "low",
            }

        # Parse frame geometry information
        lines = result.stdout.strip().split("\n")
        frame_data = []

        for line in lines:
            # Extract geometry info: widthxheight canvasxcanvas+xoffset+yoffset
            # Example: "100x100 100x100+0+0" or "90x41 100x100+3+59"
            geometry_match = re.search(r"(\d+)x(\d+)\s+\d+x\d+\+(\d+)\+(\d+)", line)
            if geometry_match:
                width = int(geometry_match.group(1))
                height = int(geometry_match.group(2))
                x_offset = int(geometry_match.group(3))
                y_offset = int(geometry_match.group(4))
                frame_data.append(
                    {
                        "width": width,
                        "height": height,
                        "x_offset": x_offset,
                        "y_offset": y_offset,
                        "area": width * height,
                    }
                )

        if len(frame_data) <= 1:
            # Single frame or no data - low complexity
            return {
                "complexity_score": 0.0,
                "has_variable_frames": False,
                "has_offsets": False,
                "frame_size_variance": 0.0,
                "disposal_risk": "low",
            }

        # Calculate complexity metrics
        areas = [frame["area"] for frame in frame_data]
        x_offsets = [frame["x_offset"] for frame in frame_data]
        y_offsets = [frame["y_offset"] for frame in frame_data]

        # Check for variable frame sizes
        has_variable_frames = len(set(areas)) > 1

        # Check for non-zero offsets
        has_offsets = any(
            x != 0 or y != 0 for x, y in zip(x_offsets, y_offsets, strict=True)
        )

        # Calculate frame size variance (normalized by max area)
        if len(areas) > 1:
            area_variance = statistics.stdev(areas)
            max_area = max(areas)
            normalized_variance = area_variance / max_area if max_area > 0 else 0.0
        else:
            normalized_variance = 0.0

        # Calculate overall complexity score
        complexity_score = 0.0

        # Factor 1: Variable frame sizes (0.4 weight)
        if has_variable_frames:
            size_variation_ratio = len(set(areas)) / len(areas)
            complexity_score += 0.4 * size_variation_ratio

        # Factor 2: Non-zero offsets (0.3 weight)
        if has_offsets:
            unique_positions = len(
                {(x, y) for x, y in zip(x_offsets, y_offsets, strict=True)}
            )
            position_variation_ratio = unique_positions / len(frame_data)
            complexity_score += 0.3 * position_variation_ratio

        # Factor 3: Frame size variance (0.3 weight)
        complexity_score += 0.3 * min(1.0, normalized_variance * 2.0)  # Cap at 1.0

        # Determine disposal risk level
        if complexity_score >= 0.7:
            disposal_risk = "high"
        elif complexity_score >= 0.4:
            disposal_risk = "medium"
        else:
            disposal_risk = "low"

        return {
            "complexity_score": round(complexity_score, 3),
            "has_variable_frames": has_variable_frames,
            "has_offsets": has_offsets,
            "frame_size_variance": round(normalized_variance, 3),
            "disposal_risk": disposal_risk,
        }

    except Exception:
        # Fallback to low complexity on any error
        return {
            "complexity_score": 0.0,
            "has_variable_frames": False,
            "has_offsets": False,
            "frame_size_variance": 0.0,
            "disposal_risk": "low",
        }


def _select_optimal_disposal_method(
    input_path: Path, frame_keep_ratio: float, total_frames: int
) -> str | None:
    """Select optimal disposal method for gifsicle based on GIF content analysis.

    This function analyzes the GIF structure and content to determine the best disposal method
    to use when reducing frames. It now includes intelligent detection of complex optimized GIFs
    that rely heavily on disposal methods.

    Different disposal methods work better for different types of animations:

    - none: Don't dispose of frame, leave it in place (good for solid backgrounds)
    - background: Clear frame to background before next frame (good for complex animations)
    - previous: Restore to previous undisposed frame (rarely optimal for frame reduction)

    Args:
        input_path: Path to the input GIF file
        frame_keep_ratio: Ratio of frames to keep (0.0 to 1.0)
        total_frames: Total number of frames in the source GIF

    Returns:
        Disposal method string ("none", "background", or None for default)

    Note:
        Now uses GIF structural analysis to detect disposal complexity. For complex optimized
        GIFs with variable frame sizes and offsets, forces safer disposal methods regardless
        of frame reduction ratio to prevent disposal artifact corruption.
    """
    try:
        # Analyze GIF structural complexity to detect disposal dependencies
        complexity_analysis = _analyze_gif_disposal_complexity(input_path)

        disposal_risk = complexity_analysis.get("disposal_risk", "low")
        complexity_score = complexity_analysis.get("complexity_score", 0.0)
        has_variable_frames = complexity_analysis.get("has_variable_frames", False)
        has_offsets = complexity_analysis.get("has_offsets", False)

        # For high-complexity GIFs with disposal dependencies, use safer methods
        if disposal_risk == "high" or complexity_score >= 0.7:
            # High complexity: Force background disposal to prevent corruption
            # This handles cases like animation_heavy with complex frame interdependencies
            return "background"

        elif disposal_risk == "medium" or (has_variable_frames and has_offsets):
            # Medium complexity: Use background disposal for any frame reduction
            if frame_keep_ratio < 1.0:
                return "background"
            else:
                return None  # No frame reduction, preserve original disposal

        # For low complexity GIFs, use the original ratio-based logic
        else:
            # For high frame reduction ratios (keeping < 50% of frames), disposal artifacts are more likely
            if frame_keep_ratio <= 0.5:
                # For aggressive frame reduction on simple GIFs, let gifsicle preserve original disposal
                return None

            # For moderate frame reduction (keeping 50-80% of frames), background disposal often works well
            elif frame_keep_ratio <= 0.8:
                return "background"

            # For light frame reduction (keeping > 80% of frames), none disposal may preserve more detail
            else:
                return "none"

    except Exception:
        # If analysis fails, fall back to the original conservative logic
        if frame_keep_ratio <= 0.5:
            return None
        elif frame_keep_ratio <= 0.8:
            return "background"
        else:
            return "none"


def compress_with_gifsicle(
    input_path: Path,
    output_path: Path,
    lossy_level: int,
    frame_keep_ratio: float = 1.0,
    color_keep_count: int | None = None,
) -> dict[str, Any]:
    """Compress GIF using gifsicle with lossy, frame reduction, and color reduction options.

    Constructs and executes a gifsicle command following best practices from:
    https://www.lcdf.org/gifsicle/

    Command Construction:
    1. Base: gifsicle --optimize
    2. Lossy: --lossy=LEVEL (if level > 0)
    3. Colors: --colors N (if color_keep_count specified)
    4. Input: INPUT_FILE (must come before frame selection)
    5. Frames: #0 #2 #4 (frame selection, if frame_keep_ratio < 1.0)
    6. Output: --output OUTPUT_FILE

    Example Commands:
        # Lossless with frame reduction
        gifsicle --optimize input.gif #0 #2 #4 #6 --output output.gif

        # Lossy with color reduction
        gifsicle --optimize --lossy=40 --colors 64 input.gif --output output.gif

        # Full optimization
        gifsicle --optimize --lossy=40 --colors 64 input.gif #0 #2 #4 --output output.gif

    Args:
        input_path: Path to input GIF file
        output_path: Path to save compressed GIF
        lossy_level: Lossy compression level for gifsicle (0=lossless, higher=more lossy)
        frame_keep_ratio: Ratio of frames to keep (0.0 to 1.0)
        color_keep_count: Number of colors to keep (optional)

    Returns:
        Dictionary with compression metadata including:
        - render_ms: Processing time in milliseconds
        - engine: "gifsicle"
        - command: Full command that was executed
        - original_frames: Number of frames in input
        - original_colors: Number of colors in input

    Raises:
        RuntimeError: If gifsicle command fails or binary not found
        ValueError: If paths contain dangerous characters

    Note:
        Frame selection (#0 #2 #4) must come AFTER input file for gifsicle.
        This is different from --delete which conflicts with --optimize.
    """
    # Get GIF metadata for frame and color information
    try:
        metadata = extract_gif_metadata(input_path)
        total_frames = metadata.orig_frames
        original_colors = metadata.orig_n_colors
    except Exception as e:
        raise RuntimeError(
            f"Failed to extract metadata from {input_path}: {str(e)}"
        ) from e

    # Build gifsicle command
    gifsicle_path = DEFAULT_ENGINE_CONFIG.GIFSICLE_PATH
    if not gifsicle_path or not _is_executable(gifsicle_path):
        raise RuntimeError(
            "gifsicle not found. Please install it or set GIFSICLE_PATH in EngineConfig."
        )
    cmd = [gifsicle_path, "--optimize"]

    # Add disposal method handling for frame reduction to prevent stacking artifacts
    if frame_keep_ratio < 1.0:
        disposal_method = _select_optimal_disposal_method(
            input_path, frame_keep_ratio, total_frames
        )
        if disposal_method:
            cmd.extend([f"--disposal={disposal_method}"])

    # Add lossy compression if level > 0
    if lossy_level > 0:
        cmd.extend([f"--lossy={lossy_level}"])

    # Add color reduction arguments
    if color_keep_count is not None:
        validate_color_keep_count(color_keep_count)
        color_args = build_gifsicle_color_args(
            color_keep_count, original_colors, dithering=False
        )
        cmd.extend(color_args)

    # Add input and output with path validation
    validated_input = validate_path_security(input_path)
    validated_output = validate_path_security(output_path)

    input_str = str(validated_input.resolve())  # Resolve to absolute path
    output_str = str(validated_output.resolve())  # Resolve to absolute path

    # Add input file first (required for frame operations)
    cmd.append(input_str)

    # Add frame reduction arguments AFTER input file with timing preservation
    if frame_keep_ratio < 1.0:
        # Extract timing information for frame reduction
        try:
            from .frame_keep import (
                build_gifsicle_timing_args,
                calculate_frame_indices,
                extract_gif_timing_info,
            )

            timing_info = extract_gif_timing_info(input_path)
            original_delays = timing_info["frame_delays"]
            loop_count = timing_info["loop_count"]

            # Calculate which frames to keep
            frame_indices = calculate_frame_indices(total_frames, frame_keep_ratio)

            # Add frame selection arguments
            frame_args = build_gifsicle_frame_args(frame_keep_ratio, total_frames)
            cmd.extend(frame_args)

            # Add timing preservation arguments with adjusted delays
            timing_args = build_gifsicle_timing_args(
                original_delays, frame_indices, loop_count
            )
            cmd.extend(timing_args)

        except Exception:
            # Fallback to old behavior if timing extraction fails
            frame_args = build_gifsicle_frame_args(frame_keep_ratio, total_frames)
            cmd.extend(frame_args)
            # Add basic loop preservation
            cmd.extend(["--loopcount=0"])
    else:
        # Even when not reducing frames, preserve timing and loop count
        try:
            from .frame_keep import build_gifsicle_timing_args, extract_gif_timing_info

            timing_info = extract_gif_timing_info(input_path)
            original_delays = timing_info["frame_delays"]
            loop_count = timing_info["loop_count"]

            # No frame reduction, so use all frame indices
            frame_indices = list(range(len(original_delays)))
            timing_args = build_gifsicle_timing_args(
                original_delays, frame_indices, loop_count
            )
            cmd.extend(timing_args)
        except Exception:
            # Default to infinite loop
            cmd.extend(["--loopcount=0"])

    # Add output
    cmd.extend(["--output", output_str])

    # Command ready for execution

    # Execute command and measure time
    start_time = time.time()

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True, timeout=RUN_TIMEOUT
        )

        end_time = time.time()
        elapsed_seconds = end_time - start_time
        # Cap at reasonable maximum to prevent overflow (24 hours = 86400000 ms)
        render_ms = min(int(elapsed_seconds * 1000), 86400000)

        # Verify output file was created
        if not output_path.exists():
            raise RuntimeError(f"Gifsicle failed to create output file: {output_path}")

        # Get engine version
        try:
            engine_version = get_gifsicle_version()
        except RuntimeError:
            engine_version = "unknown"

        return {
            "render_ms": render_ms,
            "engine": "gifsicle",
            "engine_version": engine_version,
            "lossy_level": lossy_level,
            "frame_keep_ratio": frame_keep_ratio,
            "color_keep_count": color_keep_count,
            "original_frames": total_frames,
            "original_colors": original_colors,
            "command": " ".join(cmd),
            "stderr": result.stderr if result.stderr else None,
        }

    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Gifsicle failed with exit code {e.returncode}: {e.stderr}"
        ) from e
    except subprocess.TimeoutExpired as e:
        # subprocess.run already cleans up the child process on timeout.
        # The previous kill() attempt used the wrong object (cmd string) and was ineffective.
        raise RuntimeError(f"Gifsicle timed out after {RUN_TIMEOUT} seconds") from e
    except Exception as e:
        raise RuntimeError(f"Gifsicle execution failed: {str(e)}") from e


def compress_with_animately(
    input_path: Path,
    output_path: Path,
    lossy_level: int,
    frame_keep_ratio: float = 1.0,
    color_keep_count: int | None = None,
) -> dict[str, Any]:
    """Compress GIF using animately CLI with lossy, frame reduction, and color reduction options.

    Constructs and executes an animately command using its flag-based interface.

    Animately CLI Reference:
    Usage: animately.exe [OPTION...]
      -i, --input arg        Path to input gif file
      -o, --output arg       Path to output gif file
      -l, --lossy arg        Lossy compression level
      -f, --reduce arg       Reduce frames (ratio 0.0-1.0)
      -p, --colors arg       Reduce palette colors
      -d, --delay arg        Delay between frames
      -t, --trim-frames arg  Trim frames
      -m, --trim-ms arg      Trim in milliseconds
      -s, --scale arg        Scale factor
      -c, --crop arg         Crop dimensions
      -z, --zoom arg         Zoom factor
      -u, --tone arg         Duotone effect
      -r, --repeated-frame   Repeated frame optimization
      -e, --meta arg         GIF metadata
      -y, --loops arg        Loop count
      -h, --help             Show help

    Command Construction:
    1. Base: animately
    2. Input: --input INPUT_FILE
    3. Output: --output OUTPUT_FILE
    4. Lossy: --lossy LEVEL (if level > 0)
    5. Frames: --reduce RATIO (if frame_keep_ratio < 1.0)
    6. Colors: --colors N (if color_keep_count specified)

    Example Commands:
        # Lossless with frame reduction
        animately --input input.gif --reduce 0.50 --output output.gif

        # Lossy with color reduction
        animately --input input.gif --lossy 40 --colors 64 --output output.gif

        # Full optimization
        animately --input input.gif --lossy 40 --reduce 0.50 --colors 64 --output output.gif

    Args:
        input_path: Path to input GIF file
        output_path: Path to save compressed GIF
        lossy_level: Lossy compression level for animately
        frame_keep_ratio: Ratio of frames to keep (0.0 to 1.0)
        color_keep_count: Number of colors to keep (optional)

    Returns:
        Dictionary with compression metadata including:
        - render_ms: Processing time in milliseconds
        - engine: "animately"
        - command: Full command that was executed
        - original_frames: Number of frames in input
        - original_colors: Number of colors in input

    Raises:
        RuntimeError: If animately command fails or binary not found
        ValueError: If paths contain dangerous characters

    Note:
        Animately uses decimal ratios (0.50 for 50%) and consistent flag syntax.
        All parameters are specified via flags, unlike gifsicle's mixed approach.
    """
    animately_path = DEFAULT_ENGINE_CONFIG.ANIMATELY_PATH

    # Check if animately is available
    if not animately_path or not _is_executable(animately_path):
        raise RuntimeError(
            "Animately launcher not found. "
            "Please install animately or set ANIMATELY_PATH in EngineConfig."
        )

    # Get GIF metadata for frame and color information
    try:
        metadata = extract_gif_metadata(input_path)
        total_frames = metadata.orig_frames
        original_colors = metadata.orig_n_colors
    except Exception as e:
        raise RuntimeError(
            f"Failed to extract metadata from {input_path}: {str(e)}"
        ) from e

    # Build animately command
    cmd = [animately_path]

    # Add input and output with path validation
    validated_input = validate_path_security(input_path)
    validated_output = validate_path_security(output_path)

    input_str = str(validated_input.resolve())  # Resolve to absolute path
    output_str = str(validated_output.resolve())  # Resolve to absolute path

    # Add input and output paths
    cmd.extend(["--input", input_str, "--output", output_str])

    # Add lossy compression if level > 0
    if lossy_level > 0:
        cmd.extend(["--lossy", str(lossy_level)])

    # Add frame reduction arguments with timing preservation
    if frame_keep_ratio < 1.0:
        frame_args = build_animately_frame_args(frame_keep_ratio, total_frames)
        cmd.extend(frame_args)

        # Extract and preserve original timing information
        try:
            from .frame_keep import (
                calculate_adjusted_delays,
                calculate_frame_indices,
                extract_gif_timing_info,
            )

            timing_info = extract_gif_timing_info(input_path)
            original_delays = timing_info["frame_delays"]
            loop_count = timing_info["loop_count"]

            # Calculate adjusted delays for remaining frames
            frame_indices = calculate_frame_indices(total_frames, frame_keep_ratio)
            adjusted_delays = calculate_adjusted_delays(original_delays, frame_indices)

            # Use average delay to maintain timing consistency
            if adjusted_delays:
                avg_delay = sum(adjusted_delays) / len(adjusted_delays)
                cmd.extend(["--delay", str(int(avg_delay))])

            # Preserve loop count
            if loop_count is not None:
                cmd.extend(["--loops", str(loop_count)])
            else:
                cmd.extend(["--loops", "0"])  # Default to infinite loop

        except Exception:
            # Fallback: just preserve basic looping
            cmd.extend(["--loops", "0"])
    else:
        # Even when not reducing frames, preserve original loop count
        try:
            from .frame_keep import extract_gif_timing_info

            timing_info = extract_gif_timing_info(input_path)
            loop_count = timing_info["loop_count"]
            if loop_count is not None:
                cmd.extend(["--loops", str(loop_count)])
            else:
                cmd.extend(["--loops", "0"])
        except Exception:
            cmd.extend(["--loops", "0"])

    # Add color reduction arguments
    if color_keep_count is not None:
        validate_color_keep_count(color_keep_count)
        color_args = build_animately_color_args(color_keep_count, original_colors)
        cmd.extend(color_args)

    # Execute command and measure time
    start_time = time.time()

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True, timeout=RUN_TIMEOUT
        )

        end_time = time.time()
        elapsed_seconds = end_time - start_time
        # Cap at reasonable maximum to prevent overflow (24 hours = 86400000 ms)
        render_ms = min(int(elapsed_seconds * 1000), 86400000)

        # Verify output file was created
        if not output_path.exists():
            raise RuntimeError(f"Animately failed to create output file: {output_path}")

        # Get engine version
        try:
            engine_version = get_animately_version()
        except RuntimeError:
            engine_version = "unknown"

        return {
            "render_ms": render_ms,
            "engine": "animately-standard",
            "engine_version": engine_version,
            "lossy_level": lossy_level,
            "frame_keep_ratio": frame_keep_ratio,
            "color_keep_count": color_keep_count,
            "original_frames": total_frames,
            "original_colors": original_colors,
            "command": " ".join(cmd),
            "stderr": result.stderr if result.stderr else None,
        }

    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Animately failed with exit code {e.returncode}: {e.stderr}"
        ) from e
    except subprocess.TimeoutExpired as e:
        # subprocess.run already terminates the child process on timeout.
        raise RuntimeError(f"Animately timed out after {RUN_TIMEOUT} seconds") from e
    except Exception as e:
        raise RuntimeError(f"Animately execution failed: {str(e)}") from e


def compress_with_animately_hard(
    input_path: Path,
    output_path: Path,
    lossy_level: int,
) -> dict[str, Any]:
    """Compress GIF using animately CLI with --hard --lossy flags.

    This uses Animately's hard compression mode which applies more aggressive
    optimization. The --hard flag combines with --lossy for maximum size reduction.

    Command: animately --input <in> --output <out> --hard --lossy <level>

    Args:
        input_path: Path to input GIF file
        output_path: Path to save compressed GIF
        lossy_level: Lossy compression level for animately

    Returns:
        Dictionary with compression metadata.

    Raises:
        RuntimeError: If animately command fails or binary not found
        ValueError: If paths contain dangerous characters
    """
    animately_path = DEFAULT_ENGINE_CONFIG.ANIMATELY_PATH

    if not animately_path or not _is_executable(animately_path):
        raise RuntimeError(
            "Animately launcher not found. "
            "Please install animately or set ANIMATELY_PATH in EngineConfig."
        )

    # Build command with --hard --lossy flags
    validated_input = validate_path_security(input_path)
    validated_output = validate_path_security(output_path)

    input_str = str(validated_input.resolve())
    output_str = str(validated_output.resolve())

    cmd = [
        animately_path,
        "--input",
        input_str,
        "--output",
        output_str,
        "--hard",
        "--lossy",
        str(lossy_level),
    ]

    start_time = time.time()

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True, timeout=RUN_TIMEOUT
        )

        end_time = time.time()
        elapsed_seconds = end_time - start_time
        render_ms = min(int(elapsed_seconds * 1000), 86400000)

        if not output_path.exists():
            raise RuntimeError(f"Animately failed to create output file: {output_path}")

        try:
            engine_version = get_animately_version()
        except RuntimeError:
            engine_version = "unknown"

        return {
            "render_ms": render_ms,
            "engine": "animately-hard",
            "engine_version": engine_version,
            "lossy_level": lossy_level,
            "command": " ".join(cmd),
            "stderr": result.stderr if result.stderr else None,
        }

    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Animately hard mode failed with exit code {e.returncode}: {e.stderr}"
        ) from e
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(
            f"Animately hard mode timed out after {RUN_TIMEOUT} seconds"
        ) from e
    except Exception as e:
        raise RuntimeError(f"Animately hard mode execution failed: {str(e)}") from e


def _is_executable(path: str) -> bool:
    """Check if a path is an executable file.

    Args:
        path: The file path to check.

    Returns:
        True if the path is an executable file, False otherwise.
    """
    # First check if it's an absolute path
    if Path(path).is_absolute():
        return Path(path).is_file() and os.access(path, os.X_OK)

    # Otherwise, check if it's available in PATH
    return which(path) is not None


def validate_lossy_level(lossy_level: int, engine: LossyEngine) -> None:
    """Validate that the lossy level is appropriate for the given engine.

    Args:
        lossy_level: Lossy compression level to validate
        engine: Engine to validate against

    Raises:
        ValueError: If lossy level is not supported by the engine
    """
    if lossy_level < 0:
        raise ValueError(f"Lossy level must be non-negative, got {lossy_level}")

    # Both engines support the same lossy levels in this project
    # but we could add engine-specific validation here if needed
    valid_levels = [0, 40, 120]
    if lossy_level not in valid_levels:
        raise ValueError(
            f"Lossy level {lossy_level} not in supported levels: {valid_levels}"
        )


def apply_compression_with_all_params(
    input_path: Path,
    output_path: Path,
    lossy_level: int,
    frame_keep_ratio: float,
    color_keep_count: int,
    engine: LossyEngine = LossyEngine.GIFSICLE,
) -> dict[str, Any]:
    """Apply compression with all parameters (lossy, frame, color) in a single pass.

    This is the main function for single-pass compression with full parameter support.

    Args:
        input_path: Path to input GIF file
        output_path: Path to save compressed GIF
        lossy_level: Lossy compression level
        frame_keep_ratio: Ratio of frames to keep
        color_keep_count: Number of colors to keep
        engine: Compression engine to use

    Returns:
        Dictionary with compression metadata

    Raises:
        ValueError: If any parameter is invalid
        IOError: If input file cannot be read or output cannot be written
        RuntimeError: If compression engine fails
    """
    # Validate all parameters
    if lossy_level < 0:
        raise ValueError(f"lossy_level must be non-negative, got {lossy_level}")

    validate_frame_keep_ratio(frame_keep_ratio)
    validate_color_keep_count(color_keep_count)

    if not input_path.exists():
        raise OSError(f"Input file not found: {input_path}")

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Dispatch to appropriate engine with all parameters
    if engine == LossyEngine.GIFSICLE:
        return compress_with_gifsicle(
            input_path, output_path, lossy_level, frame_keep_ratio, color_keep_count
        )
    elif engine == LossyEngine.ANIMATELY:
        return compress_with_animately(
            input_path, output_path, lossy_level, frame_keep_ratio, color_keep_count
        )
    else:
        raise ValueError(f"Unsupported engine: {engine}")


def get_compression_estimate(
    input_path: Path, lossy_level: int, frame_keep_ratio: float, color_keep_count: int
) -> dict[str, Any]:
    """Estimate compression effects without actually compressing the file.

    Args:
        input_path: Path to input GIF file
        lossy_level: Lossy compression level
        frame_keep_ratio: Ratio of frames to keep
        color_keep_count: Number of colors to keep

    Returns:
        Dictionary with compression estimates

    Raises:
        IOError: If input file cannot be read
        ValueError: If parameters are invalid
    """
    validate_frame_keep_ratio(frame_keep_ratio)
    validate_color_keep_count(color_keep_count)

    if not input_path.exists():
        raise OSError(f"Input file not found: {input_path}")

    try:
        # Get original metadata
        metadata = extract_gif_metadata(input_path)

        # Calculate frame reduction estimate
        target_frames = max(1, int(metadata.orig_frames * frame_keep_ratio))
        frame_reduction = (metadata.orig_frames - target_frames) / metadata.orig_frames

        # Calculate color reduction estimate
        target_colors = min(color_keep_count, metadata.orig_n_colors)
        color_reduction = (
            (metadata.orig_n_colors - target_colors) / metadata.orig_n_colors
            if metadata.orig_n_colors > 0
            else 0.0
        )

        # Rough size estimate (very approximate)
        size_reduction_estimate = (
            frame_reduction * 0.6 + color_reduction * 0.3 + (lossy_level / 120) * 0.1
        )
        estimated_size_kb = metadata.orig_kilobytes * (1.0 - size_reduction_estimate)

        return {
            "original_size_kb": metadata.orig_kilobytes,
            "estimated_size_kb": max(0.1, estimated_size_kb),  # Minimum 0.1KB
            "estimated_compression_ratio": metadata.orig_kilobytes
            / max(0.1, estimated_size_kb),
            "frame_reduction_percent": frame_reduction * 100.0,
            "color_reduction_percent": color_reduction * 100.0,
            "target_frames": target_frames,
            "target_colors": target_colors,
            "lossy_level": lossy_level,
            "quality_loss_estimate": min(
                100.0,
                lossy_level / 120 * 50 + frame_reduction * 30 + color_reduction * 20,
            ),
        }

    except Exception as e:
        raise OSError(f"Error estimating compression for {input_path}: {str(e)}") from e


@contextmanager
def _managed_temp_directory(
    prefix: str = "animately_png_",
) -> Generator[Path, None, None]:
    """Context manager for temporary directory creation and cleanup.

    Args:
        prefix: Prefix for the temporary directory name

    Yields:
        Path to the temporary directory

    Note:
        Automatically cleans up the directory when exiting the context,
        even if an exception occurs.
    """
    temp_dir = tempfile.mkdtemp(prefix=prefix)
    temp_path = Path(temp_dir)
    try:
        yield temp_path
    finally:
        try:
            rmtree(temp_path)
        except Exception:
            # Best effort cleanup - don't fail if cleanup fails
            pass


def _validate_animately_availability() -> str:
    """Validate that Animately is available and return its path.

    Returns:
        Path to the Animately executable

    Raises:
        RuntimeError: If Animately is not available
    """
    animately_path = DEFAULT_ENGINE_CONFIG.ANIMATELY_PATH
    if not animately_path or not _is_executable(animately_path):
        raise RuntimeError(
            "Animately launcher not found. "
            "Please install animately or set ANIMATELY_PATH in EngineConfig."
        )
    return animately_path


def _extract_gif_metadata(input_path: Path) -> tuple[int, int]:
    """Extract metadata from GIF file.

    Args:
        input_path: Path to the input GIF file

    Returns:
        Tuple of (total_frames, original_colors)

    Raises:
        RuntimeError: If metadata extraction fails
    """
    logger = logging.getLogger(__name__)
    try:
        metadata = extract_gif_metadata(input_path)
        total_frames = metadata.orig_frames
        original_colors = metadata.orig_n_colors
        logger.debug(f"GIF metadata: {total_frames} frames, {original_colors} colors")
        return total_frames, original_colors
    except Exception as e:
        logger.error(f"Failed to extract metadata from {input_path}: {str(e)}")
        raise RuntimeError(
            f"Failed to extract metadata from {input_path}: {str(e)}"
        ) from e


def _setup_png_sequence_directory(
    png_sequence_dir: Path | None, input_path: Path, total_frames: int
) -> tuple[Path, dict[str, Any], bool]:
    """Set up PNG sequence directory, either using provided or creating new.

    Args:
        png_sequence_dir: Optional existing PNG sequence directory
        input_path: Path to input GIF for PNG export if needed
        total_frames: Number of frames expected

    Returns:
        Tuple of (png_sequence_dir, png_export_result, was_provided)

    Raises:
        RuntimeError: If PNG sequence setup fails
    """
    logger = logging.getLogger(__name__)

    if png_sequence_dir is not None:
        # PNG sequence was provided by previous pipeline step - use it directly
        png_sequence_dir.mkdir(parents=True, exist_ok=True)

        # Create basic metadata for provided sequence
        png_files = list(png_sequence_dir.glob("frame_*.png"))
        png_export_result = {
            "png_sequence_dir": str(png_sequence_dir),
            "frame_count": len(png_files),
            "frame_pattern": "frame_%04d.png",
            "render_ms": 0,  # No export time since it was pre-provided
            "engine": "provided_by_previous_step",
        }

        if len(png_files) < 1:
            raise RuntimeError("Provided PNG sequence directory contains no frames")

        return png_sequence_dir, png_export_result, True

    else:
        # No PNG sequence provided - create our own from GIF using context manager
        # Note: This will be used within the main function's context manager
        temp_dir = tempfile.mkdtemp(prefix="animately_png_")
        png_sequence_dir = Path(temp_dir)

        # Export GIF to PNG sequence using ImageMagick
        logger.info(f"Exporting {total_frames} frames to PNG sequence")
        png_export_result = export_png_sequence(input_path, png_sequence_dir)

        frame_count = png_export_result.get("frame_count", 0)
        if not isinstance(frame_count, int | float) or int(frame_count) < 1:
            logger.error("PNG sequence export failed: no frames generated")
            raise RuntimeError("Failed to export PNG sequence: no frames generated")

        logger.debug(
            f"PNG export completed: {png_export_result.get('frame_count', 0)} frames "
            f"in {png_export_result.get('render_ms', 0)}ms"
        )

        return png_sequence_dir, png_export_result, False


def _extract_frame_timing(input_path: Path, total_frames: int) -> list[int]:
    """Extract frame timing information from GIF file.

    Args:
        input_path: Path to the input GIF file
        total_frames: Expected number of frames

    Returns:
        List of frame delays in milliseconds
    """
    logger = logging.getLogger(__name__)
    frame_delays = []

    try:
        from PIL import Image

        logger.debug("Extracting frame timing information from GIF")
        with Image.open(input_path) as img:
            # Extract original FPS for debugging
            original_fps = (
                1000.0 / img.info.get("duration", 100)
                if img.info.get("duration")
                else 10.0
            )
            logger.info(f"Original GIF FPS: {original_fps:.2f}")

            for i in range(total_frames):
                try:
                    img.seek(i)
                    duration = img.info.get("duration", 100)  # Default 100ms
                    frame_delays.append(max(MIN_FRAME_DELAY_MS, duration))
                    if i < 3:  # Log first few frames for debugging
                        logger.debug(f"Frame {i}: duration={duration}ms")
                except (EOFError, Exception):
                    frame_delays.append(100)  # Default fallback

        avg_delay = sum(frame_delays) / len(frame_delays) if frame_delays else 100
        calculated_fps = 1000.0 / avg_delay if avg_delay > 0 else 0.0
        logger.info(
            f"Extracted timing for {len(frame_delays)} frames (avg: {avg_delay:.1f}ms, calculated FPS: {calculated_fps:.2f})"
        )

    except Exception as e:
        logger.warning(f"Failed to extract frame timing, using defaults: {e}")
        # Fallback to default timing if extraction fails
        frame_delays = [100] * total_frames

    return frame_delays


def _generate_frame_list(
    png_sequence_dir: Path, frame_delays: list[int]
) -> list[dict[str, Any]]:
    """Generate frame list with timing for JSON configuration.

    Args:
        png_sequence_dir: Directory containing PNG files
        frame_delays: List of frame delays in milliseconds

    Returns:
        List of frame configuration dictionaries
    """
    png_files = sorted(png_sequence_dir.glob("frame_*.png"))
    frame_files = []

    for i, png_file in enumerate(png_files):
        delay = frame_delays[i] if i < len(frame_delays) else 100
        frame_files.append({"png": str(png_file.absolute()), "delay": delay})

    return frame_files


def _generate_json_config(
    png_sequence_dir: Path,
    lossy_level: int,
    color_keep_count: int | None,
    frame_files: list[dict[str, Any]],
) -> Path:
    """Generate JSON configuration file for Animately advanced lossy mode.

    Args:
        png_sequence_dir: Directory to write JSON config to
        lossy_level: Lossy compression level (0-100)
        color_keep_count: Optional color count limit
        frame_files: List of frame configuration dictionaries

    Returns:
        Path to the generated JSON configuration file

    Raises:
        RuntimeError: If JSON configuration generation fails
    """
    logger = logging.getLogger(__name__)

    try:
        json_config = {
            "lossy": max(0, min(100, lossy_level)),  # Clamp to valid range
            "frames": frame_files,
        }

        # Add color reduction if specified
        if color_keep_count is not None:
            validate_color_keep_count(color_keep_count)
            json_config["colors"] = max(2, min(256, color_keep_count))

        # Debug the frame delays in the JSON config
        sample_delays = [frame["delay"] for frame in frame_files[:5]]  # First 5 frames
        avg_json_delay = (
            sum(frame["delay"] for frame in frame_files) / len(frame_files)
            if frame_files
            else 0
        )
        json_fps = 1000.0 / avg_json_delay if avg_json_delay > 0 else 0.0

        logger.info(
            f"Generated JSON config with {len(frame_files)} frames, "
            f"lossy={json_config['lossy']}, "
            f"colors={json_config.get('colors', 'default')}"
        )
        logger.info(
            f"JSON config delays - sample: {sample_delays}, avg: {avg_json_delay:.1f}ms, FPS: {json_fps:.2f}"
        )
        logger.debug(
            f"Generated JSON config with {len(frame_files)} frames, "
            f"lossy={json_config['lossy']}, "
            f"colors={json_config.get('colors', 'default')}"
        )

        # Write JSON config to file
        json_config_path = png_sequence_dir / "animately_config.json"
        with open(json_config_path, "w") as f:
            json.dump(json_config, f, indent=2)

        logger.debug(f"JSON configuration written to {json_config_path}")
        return json_config_path

    except (ValueError, TypeError, OSError) as e:
        logger.error(f"Failed to generate JSON configuration: {e}")
        raise RuntimeError(f"JSON configuration generation failed: {str(e)}") from e
    except Exception as e:
        logger.error(f"Unexpected error during JSON config generation: {e}")
        raise RuntimeError(f"Unexpected error generating JSON config: {str(e)}") from e


def _execute_animately_advanced(
    animately_path: str, json_config_path: Path, output_path: Path
) -> tuple[int, str | None]:
    """Execute Animately with advanced lossy configuration.

    Args:
        animately_path: Path to Animately executable
        json_config_path: Path to JSON configuration file
        output_path: Path for output file

    Returns:
        Tuple of (render_ms, stderr_output)

    Raises:
        RuntimeError: If Animately execution fails
    """
    logger = logging.getLogger(__name__)

    validated_output = validate_path_security(output_path)
    output_str = str(validated_output.resolve())

    cmd = [
        animately_path,
        "--advanced-lossy",
        str(json_config_path),
        "--output",
        output_str,
    ]

    logger.info(f"Executing Animately advanced lossy: {' '.join(cmd)}")

    # Execute command and measure time
    start_time = time.time()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=RUN_TIMEOUT * ADVANCED_TIMEOUT_MULTIPLIER,
        )

        end_time = time.time()
        elapsed_seconds = end_time - start_time
        render_ms = min(int(elapsed_seconds * 1000), 86400000)

        # Verify output file was created
        if not output_path.exists():
            logger.error(f"Output file not created: {output_path}")
            raise RuntimeError(
                f"Animately advanced lossy failed to create output file: {output_path}"
            )

        output_size = output_path.stat().st_size
        logger.info(
            f"Compression completed in {render_ms}ms, output size: {output_size} bytes"
        )

        # Validate that Animately preserved the timing correctly
        try:
            from .meta import extract_gif_metadata

            output_metadata = extract_gif_metadata(output_path)
            output_fps = output_metadata.orig_fps

            # Calculate expected FPS from the JSON config we provided
            with open(json_config_path) as f:
                config_data = json.load(f)

            if "frames" in config_data and config_data["frames"]:
                frame_delays = [
                    frame.get("delay", 100) for frame in config_data["frames"]
                ]
                expected_avg_delay = sum(frame_delays) / len(frame_delays)
                expected_fps = (
                    1000.0 / expected_avg_delay if expected_avg_delay > 0 else 10.0
                )

                fps_deviation = (
                    abs(output_fps - expected_fps) / expected_fps
                    if expected_fps > 0
                    else 0
                )

                logger.info(
                    f"FPS validation: expected {expected_fps:.2f}, got {output_fps:.2f}, deviation {fps_deviation*100:.1f}%"
                )

                if fps_deviation > 0.5:  # More than 50% FPS deviation
                    logger.error(
                        f"❌ Animately advanced-lossy timing corruption detected! "
                        f"Expected FPS: {expected_fps:.2f}, Got: {output_fps:.2f} "
                        f"(deviation: {fps_deviation*100:.1f}%)"
                    )
                    logger.error(
                        "Known Animately bug: advanced-lossy mode does not preserve JSON delay values"
                    )

                    # Add timing corruption to stderr for tracking in results
                    timing_error = f"FPS corruption: expected {expected_fps:.2f}fps, got {output_fps:.2f}fps"
                    stderr_output = (
                        f"{timing_error}. {result.stderr}"
                        if result.stderr
                        else timing_error
                    )
                    return render_ms, stderr_output

        except Exception as e:
            logger.warning(f"Could not validate output timing: {e}")

        return render_ms, result.stderr if result.stderr else None

    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Animately advanced lossy failed with exit code {e.returncode}: {e.stderr}"
        ) from e
    except subprocess.TimeoutExpired as e:
        timeout_duration = RUN_TIMEOUT * ADVANCED_TIMEOUT_MULTIPLIER
        # subprocess.run already cleans up; simply propagate a clear error.
        raise RuntimeError(
            f"Animately advanced lossy timed out after {timeout_duration} seconds"
        ) from e
    except Exception as e:
        raise RuntimeError(
            f"Animately advanced lossy execution failed: {str(e)}"
        ) from e


def compress_with_animately_advanced_lossy(
    input_path: Path,
    output_path: Path,
    lossy_level: int,
    color_keep_count: int | None = None,
    png_sequence_dir: Path | None = None,
) -> dict[str, Any]:
    """Compress GIF using Animately's advanced lossy mode with PNG sequence input.

    This function implements the advanced PNG sequence pipeline:
    1. Export GIF frames to PNG sequence (preserving timing)
    2. Generate JSON configuration for advanced lossy mode
    3. Use Animately's --advanced-lossy flag with the JSON config

    This approach provides superior compression for gradients and mixed content
    compared to traditional GIF-to-GIF compression pipelines.

    Args:
        input_path: Path to input GIF file
        output_path: Path to save compressed GIF
        lossy_level: Lossy compression level (0-100)
        color_keep_count: Number of colors to keep (1-256, optional)
        png_sequence_dir: Directory for PNG sequence (optional, auto-generated if None)

    Returns:
        Dictionary with compression metadata including:
        - render_ms: Processing time in milliseconds
        - engine: "animately-advanced"
        - command: Full command that was executed
        - png_sequence_metadata: PNG export information
        - json_config_path: Path to generated JSON config

    Raises:
        RuntimeError: If animately command fails or binary not found
        ValueError: If paths contain dangerous characters
        OSError: If PNG sequence export fails

    Note:
        Uses ImageMagick to export PNG sequence and generates JSON config
        with frame timing information preserved from the original GIF.
    """
    logger = logging.getLogger(__name__)

    logger.info(
        f"Starting Animately advanced lossy compression: {input_path.name} "
        f"(lossy={lossy_level}, colors={color_keep_count})"
    )

    # Step 1: Validate Animately availability
    animately_path = _validate_animately_availability()

    # Step 2: Extract GIF metadata
    total_frames, original_colors = _extract_gif_metadata(input_path)

    # Step 3: Set up PNG sequence directory
    png_dir, png_export_result, was_provided = _setup_png_sequence_directory(
        png_sequence_dir, input_path, total_frames
    )

    # Use context manager for cleanup only if we created the directory
    if was_provided:
        # Use the provided directory directly
        return _process_advanced_lossy(
            input_path,
            output_path,
            lossy_level,
            color_keep_count,
            png_dir,
            png_export_result,
            total_frames,
            original_colors,
            animately_path,
        )
    else:
        # Use context manager for automatic cleanup
        try:
            return _process_advanced_lossy(
                input_path,
                output_path,
                lossy_level,
                color_keep_count,
                png_dir,
                png_export_result,
                total_frames,
                original_colors,
                animately_path,
            )
        finally:
            # Clean up temporary directory
            if png_dir.name.startswith("animately_png_"):
                try:
                    rmtree(png_dir)
                except Exception:
                    pass  # Best effort cleanup


def _process_advanced_lossy(
    input_path: Path,
    output_path: Path,
    lossy_level: int,
    color_keep_count: int | None,
    png_sequence_dir: Path,
    png_export_result: dict[str, Any],
    total_frames: int,
    original_colors: int,
    animately_path: str,
) -> dict[str, Any]:
    """Process the advanced lossy compression pipeline.

    This is the core processing function separated for better testability.

    Args:
        input_path: Path to input GIF file
        output_path: Path to save compressed GIF
        lossy_level: Lossy compression level (0-100)
        color_keep_count: Number of colors to keep (1-256, optional)
        png_sequence_dir: Directory containing PNG sequence
        png_export_result: Metadata from PNG export
        total_frames: Number of frames in GIF
        original_colors: Number of colors in original GIF
        animately_path: Path to Animately executable

    Returns:
        Dictionary with compression metadata
    """
    # Step 4: Extract frame timing from original GIF
    frame_delays = _extract_frame_timing(input_path, total_frames)

    # Step 5: Generate frame list with timing
    frame_files = _generate_frame_list(png_sequence_dir, frame_delays)

    # Step 6: Generate JSON configuration
    json_config_path = _generate_json_config(
        png_sequence_dir, lossy_level, color_keep_count, frame_files
    )

    # Step 7: Execute Animately
    render_ms, stderr_output = _execute_animately_advanced(
        animately_path, json_config_path, output_path
    )

    # Step 8: Get engine version and return results
    try:
        engine_version = get_animately_version()
    except RuntimeError:
        engine_version = "unknown"

    return {
        "render_ms": render_ms,
        "engine": "animately-advanced",
        "engine_version": engine_version,
        "lossy_level": lossy_level,
        "color_keep_count": color_keep_count,
        "original_frames": total_frames,
        "original_colors": original_colors,
        "command": f"{animately_path} --advanced-lossy {json_config_path} --output {output_path}",
        "png_sequence_metadata": png_export_result,
        "json_config_path": str(json_config_path),
        "frames_processed": len(frame_files),
        "stderr": stderr_output,
    }
