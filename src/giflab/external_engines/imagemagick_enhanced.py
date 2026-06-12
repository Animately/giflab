"""Enhanced ImageMagick External Engine

Supports comprehensive dithering methods for pipeline elimination testing.
Based on research findings identifying 13 different dithering methods.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from shutil import copy
from typing import Any, Literal

from ..system_tools import discover_tool
from .common import run_command

__all__ = [
    "color_reduce_with_dithering",
    "frame_reduce",
    "IMAGEMAGICK_DITHERING_METHODS",
]

# All 13 ImageMagick dithering methods from research
IMAGEMAGICK_DITHERING_METHODS = [
    "None",
    "FloydSteinberg",
    "Riemersma",  # ⭐ Best performer from research
    "Threshold",
    "Random",
    "Ordered",
    # Ordered variants (research shows these are redundant)
    "O2x2",
    "O3x3",
    "O4x4",
    "O8x8",
    # Halftone variants (research shows these are redundant)
    "H4x4a",
    "H6x6a",
    "H8x8a",
]

ImageMagickDitheringMethod = Literal[
    "None",
    "FloydSteinberg",
    "Riemersma",
    "Threshold",
    "Random",
    "Ordered",
    "O2x2",
    "O3x3",
    "O4x4",
    "O8x8",
    "H4x4a",
    "H6x6a",
    "H8x8a",
]


def _magick_binary() -> str:
    """Return the preferred ImageMagick binary (``magick`` or ``convert``)."""
    info = discover_tool("imagemagick")
    info.require()
    return info.name


def color_reduce_with_dithering(
    input_path: Path,
    output_path: Path,
    *,
    colors: int = 32,
    dithering_method: ImageMagickDitheringMethod = "None",
) -> dict[str, Any]:
    """Enhanced color reduction with specific dithering method support.

    Parameters
    ----------
    input_path
        Source GIF.
    output_path
        Destination GIF.
    colors
        Target palette size (1–256).
    dithering_method
        Specific dithering method to use from research findings.
    """
    if colors < 1 or colors > 256:
        raise ValueError("colors must be between 1 and 256 inclusive")

    if dithering_method not in IMAGEMAGICK_DITHERING_METHODS:
        raise ValueError(
            f"dithering_method must be one of {IMAGEMAGICK_DITHERING_METHODS}"
        )

    cmd = [
        _magick_binary(),
        str(input_path),
    ]

    # Apply dithering method
    if dithering_method == "None":
        cmd.append("+dither")
    else:
        cmd.extend(["-dither", dithering_method])

    cmd += ["-colors", str(colors), str(output_path)]

    # Add metadata about the dithering method used
    result = run_command(cmd, engine="imagemagick", output_path=output_path)
    result["dithering_method"] = dithering_method
    result["pipeline_variant"] = f"imagemagick_dither_{dithering_method.lower()}"

    return result


def frame_reduce(
    input_path: Path,
    output_path: Path,
    *,
    keep_ratio: float,
) -> dict[str, Any]:
    """Drop frames to achieve *keep_ratio* using a simple "delete every Nth" rule."""
    if not 0 < keep_ratio <= 1:
        raise ValueError("keep_ratio must be in (0, 1]")

    # Shortcut – no reduction needed.
    if keep_ratio == 1.0:
        start = time.perf_counter()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        copy(input_path, output_path)
        duration_ms = int((time.perf_counter() - start) * 1000)
        size_kb = int(os.path.getsize(output_path) / 1024)
        return {
            "render_ms": duration_ms,
            "engine": "imagemagick",
            "command": "cp",
            "kilobytes": size_kb,
        }

    # Very naive deletion strategy: drop every *step* frame where
    #   step = round(1/keep_ratio).
    # For example          keep_ratio=0.5 ⇒ step=2 ⇒ delete 1--2
    step = max(2, round(1 / keep_ratio))
    delete_pattern = f"1--{step}"  # "delete every <step> frames starting at 1"

    cmd = [
        _magick_binary(),
        str(input_path),
        "-coalesce",
        "-delete",
        delete_pattern,
        "-layers",
        "optimize",
        str(output_path),
    ]

    return run_command(cmd, engine="imagemagick", output_path=output_path)


def test_redundant_methods(input_path: Path, colors: int = 16) -> dict[str, bytes]:
    """Test if supposedly redundant methods produce identical outputs.

    Returns a dict mapping method names to output file hashes for comparison.
    This validates the research findings about redundant ImageMagick methods.
    """
    import hashlib
    import tempfile

    results = {}

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        for method in IMAGEMAGICK_DITHERING_METHODS:
            output_path = tmpdir_path / f"test_{method.lower()}.gif"

            try:
                color_reduce_with_dithering(
                    input_path, output_path, colors=colors, dithering_method=method  # type: ignore
                )

                # Calculate hash of output file
                with open(output_path, "rb") as f:
                    file_hash = hashlib.sha256(f.read()).digest()
                    results[method] = file_hash

            except Exception as e:
                results[method] = f"ERROR: {e}".encode()

    return results


def identify_redundant_methods(input_paths: list[Path]) -> dict[str, set[str]]:
    """Identify which dithering methods produce identical results.

    Returns a dict mapping representative methods to sets of equivalent methods.
    This helps validate research findings about redundant methods.
    """
    from collections import defaultdict

    # Group methods by their output hashes across multiple test images
    hash_to_methods = defaultdict(set)

    for input_path in input_paths:
        method_hashes = test_redundant_methods(input_path)

        for method, file_hash in method_hashes.items():
            if isinstance(file_hash, bytes):  # Skip errors
                hash_to_methods[file_hash].add(method)

    # Find groups of equivalent methods (methods that always produce same hash)
    equivalence_groups = {}
    processed_methods = set()

    for methods in hash_to_methods.values():
        if len(methods) > 1:  # Only care about groups with multiple methods
            # Use first method alphabetically as representative
            representative = min(methods)
            if representative not in processed_methods:
                equivalence_groups[representative] = methods
                processed_methods.update(methods)

    return equivalence_groups
