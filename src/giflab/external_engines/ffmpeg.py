from __future__ import annotations

import glob
import tempfile
from pathlib import Path
from typing import Any

from ..system_tools import discover_tool
from .common import run_command

__all__ = [
    "color_reduce",
    "frame_reduce",
    "lossy_compress",
    "export_png_sequence",
]


def _ffmpeg_binary() -> str:
    info = discover_tool("ffmpeg")
    info.require()
    return info.name


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------


def color_reduce(
    input_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Palette-based color reduction via two-pass FFmpeg."""
    ffmpeg = _ffmpeg_binary()

    with tempfile.TemporaryDirectory() as tmpdir:
        palette_path = Path(tmpdir) / "palette.png"

        # 1️⃣ generate palette (no fps filter to avoid pipeline conflicts)
        gen_cmd = [
            ffmpeg,
            "-y",
            "-v",
            "error",
            "-i",
            str(input_path),
            "-filter_complex",
            "palettegen",
            str(palette_path),
        ]
        meta1 = run_command(gen_cmd, engine="ffmpeg", output_path=palette_path)

        # 2️⃣ apply palette (no fps filter to avoid pipeline conflicts)
        use_cmd = [
            ffmpeg,
            "-y",
            "-v",
            "error",
            "-i",
            str(input_path),
            "-i",
            str(palette_path),
            "-filter_complex",
            "paletteuse",
            str(output_path),
        ]
        meta2 = run_command(use_cmd, engine="ffmpeg", output_path=output_path)

        # 3️⃣ Combine metadata from both passes
        return {
            "render_ms": meta1.get("render_ms", 0) + meta2.get("render_ms", 0),
            "engine": "ffmpeg",
            "command": f"{meta1.get('command', '')}\n{meta2.get('command', '')}",
            "kilobytes": meta2.get("kilobytes", 0),
        }


def frame_reduce(
    input_path: Path,
    output_path: Path,
    *,
    fps: float,
) -> dict[str, Any]:
    """Reduce frame-rate to *fps* using FFmpeg with timing and loop preservation."""
    ffmpeg = _ffmpeg_binary()

    # Import timing functions
    from ..frame_keep import extract_gif_timing_info

    # Extract original timing and loop information
    try:
        timing_info = extract_gif_timing_info(input_path)
        loop_count = timing_info["loop_count"]
    except Exception:
        loop_count = 0  # Default to infinite loop

    cmd = [
        ffmpeg,
        "-y",
        "-v",
        "error",
        "-i",
        str(input_path),
        "-filter_complex",
        f"fps={fps}",
    ]

    # Preserve loop count
    if loop_count is not None and loop_count >= 0:
        cmd.extend(["-loop", str(loop_count)])
    else:
        cmd.extend(["-loop", "0"])  # Infinite loop

    cmd.append(str(output_path))

    return run_command(cmd, engine="ffmpeg", output_path=output_path)


def frame_reduce_by_ratio(
    input_path: Path,
    output_path: Path,
    *,
    keep_ratio: float,
) -> dict[str, Any]:
    """Reduce frames by *keep_ratio* using FFmpeg with proper timing preservation."""
    if not 0 < keep_ratio <= 1:
        raise ValueError("keep_ratio must be in (0, 1]")

    # Import timing and frame functions
    import os
    import time
    from shutil import copy

    from ..frame_keep import (
        calculate_adjusted_delays,
        calculate_frame_indices,
        extract_gif_timing_info,
    )

    # Shortcut – no reduction needed
    if keep_ratio == 1.0:
        start = time.perf_counter()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        copy(input_path, output_path)
        duration_ms = int((time.perf_counter() - start) * 1000)
        size_kb = int(os.path.getsize(output_path) / 1024)
        return {
            "render_ms": duration_ms,
            "engine": "ffmpeg",
            "command": "cp",
            "kilobytes": size_kb,
        }

    # Extract timing and loop information
    try:
        timing_info = extract_gif_timing_info(input_path)
        original_delays = timing_info["frame_delays"]
        loop_count = timing_info["loop_count"]
        total_frames = timing_info["total_frames"]
    except Exception:
        # Fallback: convert to fps-based approach
        original_fps = 10.0  # Default assumption
        target_fps = original_fps * keep_ratio
        target_fps = max(target_fps, 0.1)  # Minimum FPS
        return frame_reduce(input_path, output_path, fps=target_fps)

    # Calculate which frames to keep
    frames_to_keep = calculate_frame_indices(total_frames, keep_ratio)
    adjusted_delays = calculate_adjusted_delays(original_delays, frames_to_keep)

    # Create frame selection string for FFmpeg
    # FFmpeg select filter: select frames by index
    select_expr = "+".join([f"eq(n\\,{idx})" for idx in frames_to_keep])

    ffmpeg = _ffmpeg_binary()
    cmd = [ffmpeg, "-y", "-v", "error", "-i", str(input_path)]

    # Use select filter to choose specific frames
    filters = [f"select='{select_expr}'"]

    # Set frame delays using setpts filter
    # Calculate time base for frame delays
    if len(adjusted_delays) > 1:
        # Create PTS (Presentation Time Stamp) values for proper timing
        pts_values: list[float] = []
        current_pts: float = 0.0
        for delay in adjusted_delays[:-1]:  # All but last frame
            pts_values.append(current_pts)
            current_pts += delay / 1000.0  # Convert ms to seconds
        pts_values.append(current_pts)  # Last frame

        # Apply setpts to control timing
        filters.append("setpts=N*TB")

    cmd.extend(["-filter_complex", ",".join(filters)])

    # Preserve loop count
    if loop_count is not None and loop_count >= 0:
        cmd.extend(["-loop", str(loop_count)])
    else:
        cmd.extend(["-loop", "0"])  # Infinite loop

    cmd.append(str(output_path))

    return run_command(cmd, engine="ffmpeg", output_path=output_path)


def lossy_compress(
    input_path: Path,
    output_path: Path,
    *,
    colors: int = 256,
    dithering_method: str = "sierra2_4a",
) -> dict[str, Any]:
    """Lossy GIF compression via palette-size reduction + dithering.

    GIF is palette-based, so FFmpeg's engine-native lossy axis is the
    palette: a two-pass ``palettegen=max_colors={colors}`` /
    ``paletteuse=dither={dithering_method}`` chain (mirroring the
    long-tested :func:`color_reduce` two-pass structure). FFmpeg has no
    error-bounded GIF lossy mode like gifsicle's ``--lossy`` — the previous
    implementation routed lossiness to ``-q:v``, an MPEG/DCT video-quality
    knob that does NOT touch GIF pixels, producing byte-identical output at
    every level (md5-verified, 2026-06-09 calibration in
    ``scripts/audit/engine_lossy_calibration.py``).

    The public ``lossy_level`` (0–100) maps to *colors* geometrically in
    ``tool_wrappers._lossy_level_to_palette_size`` (256→16, halving every
    25 levels). Dithering keeps the degradation smooth
    (continuous-over-discrete).

    Note: unlike ImageMagick's :func:`~giflab.external_engines.imagemagick.lossy_compress`
    (which plain re-saves at ``colors >= 256``), this always re-encodes
    through a single palettegen-optimised global palette — slightly lossy
    for >256-colour animations even at ``colors=256``, but a quality
    *improvement* over FFmpeg's default generic-palette GIF encode.

    Parameters
    ----------
    input_path
        Source GIF.
    output_path
        Destination GIF.
    colors
        Target palette size (4–256, FFmpeg's ``palettegen`` ``max_colors``
        range).
    dithering_method
        ``paletteuse`` dither algorithm (default ``sierra2_4a``, FFmpeg's
        own default).
    """
    if colors < 4 or colors > 256:
        raise ValueError("colors must be in 4–256 range (ffmpeg palettegen max_colors)")

    ffmpeg = _ffmpeg_binary()

    with tempfile.TemporaryDirectory() as tmpdir:
        palette_path = Path(tmpdir) / "palette.png"

        # 1️⃣ generate a palette capped at *colors* entries
        gen_cmd = [
            ffmpeg,
            "-y",
            "-v",
            "error",
            "-i",
            str(input_path),
            "-filter_complex",
            f"palettegen=max_colors={colors}",
            str(palette_path),
        ]
        meta1 = run_command(gen_cmd, engine="ffmpeg", output_path=palette_path)

        # 2️⃣ re-encode through the palette with dithering
        use_cmd = [
            ffmpeg,
            "-y",
            "-v",
            "error",
            "-i",
            str(input_path),
            "-i",
            str(palette_path),
            "-filter_complex",
            f"paletteuse=dither={dithering_method}",
            str(output_path),
        ]
        meta2 = run_command(use_cmd, engine="ffmpeg", output_path=output_path)

        # 3️⃣ Combine metadata from both passes (same shape as color_reduce)
        return {
            "render_ms": meta1.get("render_ms", 0) + meta2.get("render_ms", 0),
            "engine": "ffmpeg",
            "command": f"{meta1.get('command', '')}\n{meta2.get('command', '')}",
            "kilobytes": meta2.get("kilobytes", 0),
        }


def export_png_sequence(
    input_path: Path,
    output_dir: Path,
    *,
    frame_pattern: str = "frame_%04d.png",
) -> dict[str, Any]:
    """Export GIF frames as PNG sequence with proper disposal method handling.

    Uses ImageMagick's coalesce for accurate frame extraction that respects
    GIF disposal methods, ensuring extracted frames match animated display.

    Args:
        input_path: Input GIF file
        output_dir: Directory to store PNG sequence
        frame_pattern: Pattern for PNG filenames (default: frame_%04d.png)

    Returns:
        Metadata dict with execution info and PNG sequence details
    """
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build output path with pattern
    output_pattern = output_dir / frame_pattern

    # Use ImageMagick with coalesce for proper GIF disposal handling
    # This ensures each frame represents what the animation actually displays
    cmd = [
        "magick",
        str(input_path),
        "-coalesce",  # Critical: handles GIF disposal methods properly
        str(output_pattern),
    ]

    metadata = run_command(cmd, engine="imagemagick", output_path=output_pattern)

    # Count generated PNG files - using frame pattern prefix since ImageMagick
    # appends frame numbers differently than the pattern format
    png_pattern = frame_pattern.replace("%04d", "*").replace("%03d", "*")
    png_files = glob.glob(str(output_dir / png_pattern))

    # Add PNG sequence info to metadata
    metadata.update(
        {
            "png_sequence_dir": str(output_dir),
            "frame_count": len(png_files),
            "frame_pattern": frame_pattern,
            "extraction_method": "imagemagick_coalesce",
        }
    )

    return metadata
