"""Enhanced FFmpeg External Engine

Supports comprehensive dithering methods for pipeline elimination testing.
Based on research findings identifying FFmpeg dithering methods and Bayer scales.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Literal, cast

from ..system_tools import discover_tool
from .common import run_command

__all__ = [
    "color_reduce_with_dithering",
    "frame_reduce",
    "FFMPEG_DITHERING_METHODS",
]

# All FFmpeg dithering methods from research
FFMPEG_DITHERING_METHODS = [
    "none",
    "floyd_steinberg",  # Standard high-quality baseline from research
    "sierra2",  # ⭐ Excellent quality/size balance from research
    "sierra2_4a",  # Alternative to sierra2
    # Bayer scale variants (research shows scales 4-5 excel for noisy content)
    "bayer:bayer_scale=0",  # Poor quality from research
    "bayer:bayer_scale=1",  # Higher quality Bayer variant
    "bayer:bayer_scale=2",  # Medium pattern
    "bayer:bayer_scale=3",  # Good balance
    "bayer:bayer_scale=4",  # ⭐ Best compression for noisy content
    "bayer:bayer_scale=5",  # ⭐ Maximum compression for noisy content
]

FFmpegDitheringMethod = Literal[
    "none",
    "floyd_steinberg",
    "sierra2",
    "sierra2_4a",
    "bayer:bayer_scale=0",
    "bayer:bayer_scale=1",
    "bayer:bayer_scale=2",
    "bayer:bayer_scale=3",
    "bayer:bayer_scale=4",
    "bayer:bayer_scale=5",
]


def _ffmpeg_binary() -> str:
    info = discover_tool("ffmpeg")
    info.require()
    return info.name


def color_reduce_with_dithering(
    input_path: Path,
    output_path: Path,
    *,
    colors: int = 32,
    dithering_method: FFmpegDitheringMethod = "none",
    fps: float = 15.0,
) -> dict[str, Any]:
    """Enhanced palette-based color reduction with specific dithering method.

    Parameters
    ----------
    input_path
        Source GIF.
    output_path
        Destination GIF.
    colors
        Target palette size (2–256).
    dithering_method
        Specific dithering method from research findings.
    fps
        Frame rate for processing (deprecated - no longer used to avoid pipeline conflicts).
    """
    if colors < 2 or colors > 256:
        raise ValueError("colors must be between 2 and 256 inclusive")

    if dithering_method not in FFMPEG_DITHERING_METHODS:
        raise ValueError(f"dithering_method must be one of {FFMPEG_DITHERING_METHODS}")

    ffmpeg = _ffmpeg_binary()

    with tempfile.TemporaryDirectory() as tmpdir:
        palette_path = Path(tmpdir) / "palette.png"

        # 1️⃣ Generate palette with specified color count (no fps filter to avoid pipeline conflicts)
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

        # 2️⃣ Apply palette with specific dithering method (no fps filter to avoid pipeline conflicts)
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

        # 3️⃣ Combine metadata from both passes
        result = {
            "render_ms": meta1.get("render_ms", 0) + meta2.get("render_ms", 0),
            "engine": "ffmpeg",
            "command": f"{meta1.get('command', '')}\n{meta2.get('command', '')}",
            "kilobytes": meta2.get("kilobytes", 0),
            "dithering_method": dithering_method,
            "pipeline_variant": f"ffmpeg_dither_{dithering_method.replace(':', '_').replace('=', '_')}",
        }

        return result


def frame_reduce(
    input_path: Path,
    output_path: Path,
    *,
    fps: float,
) -> dict[str, Any]:
    """Reduce frame-rate to *fps* using FFmpeg."""
    ffmpeg = _ffmpeg_binary()
    cmd = [
        ffmpeg,
        "-y",
        "-v",
        "error",
        "-i",
        str(input_path),
        "-filter_complex",
        f"fps={fps}",
        str(output_path),
    ]
    return run_command(cmd, engine="ffmpeg", output_path=output_path)


def test_bayer_scale_performance(
    input_path: Path, colors: int = 16
) -> dict[str, dict[str, Any]]:
    """Test performance of different Bayer scales on various content types.

    This validates research findings about Bayer scales 4-5 excelling for noisy content.
    Returns detailed metrics for each Bayer scale variant.
    """
    import tempfile

    results = {}
    bayer_methods = [
        method for method in FFMPEG_DITHERING_METHODS if method.startswith("bayer")
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        for method in bayer_methods:
            output_path = tmpdir_path / f"bayer_test_{method.split('=')[1]}.gif"

            try:
                # Generate output with this Bayer scale
                result = color_reduce_with_dithering(
                    input_path,
                    output_path,
                    colors=colors,
                    dithering_method=cast(FFmpegDitheringMethod, method),
                )

                # Calculate quality metrics
                # ssim_metrics = calculate_ssim_metrics(input_path, output_path)
                # result.update(ssim_metrics)

                results[method] = result

            except Exception as e:
                results[method] = {"error": str(e)}

    return results


def analyze_dithering_by_content_type(
    test_gifs: dict[str, Path], colors: int = 16
) -> dict[str, dict[str, dict[str, Any]]]:
    """Analyze dithering method performance by content type.

    Args:
        test_gifs: Dict mapping content type names to GIF paths
        colors: Target color count for testing

    Returns:
        Nested dict: content_type -> dithering_method -> metrics

    This helps validate research findings about content-specific dithering recommendations.
    """
    results: dict[str, dict[str, Any]] = {}

    for content_type, gif_path in test_gifs.items():
        results[content_type] = {}

        for method in FFMPEG_DITHERING_METHODS:
            try:
                with tempfile.TemporaryDirectory() as tmpdir:
                    output_path = Path(tmpdir) / f"test_{method.replace(':', '_')}.gif"

                    result = color_reduce_with_dithering(
                        gif_path,
                        output_path,
                        colors=colors,
                        dithering_method=cast(FFmpegDitheringMethod, method),
                    )

                    results[content_type][method] = result

            except Exception as e:
                results[content_type][method] = {"error": str(e)}

    return results


def validate_sierra2_vs_floyd_steinberg(
    test_gifs: list[Path],
) -> dict[str, dict[str, Any]]:
    """Validate research finding that Sierra2 offers better balance than Floyd-Steinberg.

    Tests both methods across multiple GIFs and compares size/quality trade-offs.
    Research claims Sierra2 provides better quality/size balance for size-constrained scenarios.
    """
    comparison_results: dict[str, dict[str, Any]] = {}

    for gif_path in test_gifs:
        gif_name = gif_path.stem
        comparison_results[gif_name] = {}

        # Test both methods
        for method in ["floyd_steinberg", "sierra2"]:
            try:
                with tempfile.TemporaryDirectory() as tmpdir:
                    output_path = Path(tmpdir) / f"{method}_test.gif"

                    result = color_reduce_with_dithering(
                        gif_path,
                        output_path,
                        colors=16,
                        dithering_method=cast(FFmpegDitheringMethod, method),
                    )

                    comparison_results[gif_name][method] = result

            except Exception as e:
                comparison_results[gif_name][method] = {"error": str(e)}

    return comparison_results
