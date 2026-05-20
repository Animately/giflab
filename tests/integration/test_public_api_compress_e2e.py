"""End-to-end tests for ``giflab.compress`` using real engine binaries.

Skipped per-engine when the binary is unavailable on PATH.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from giflab import CompressResult, compress
from giflab.tool_wrappers import (
    AnimatelyLossyCompressor,
    GifsicleLossyCompressor,
)


def _build_real_gif(
    path: Path, frames: int = 5, size: tuple[int, int] = (64, 64)
) -> None:
    """Create a small but real multi-frame GIF that engines can actually compress."""
    images = []
    for i in range(frames):
        img = Image.new("RGB", size, color=(i * 50 % 255, 100, 150))
        draw = ImageDraw.Draw(img)
        draw.rectangle([i * 5, i * 5, i * 5 + 20, i * 5 + 20], fill=(255, 255, 255))
        images.append(img)
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=200,
        loop=0,
    )


@pytest.fixture
def real_gif(tmp_path: Path) -> Path:
    path = tmp_path / "real.gif"
    _build_real_gif(path)
    return path


@pytest.mark.skipif(
    not GifsicleLossyCompressor.available(), reason="gifsicle binary not on PATH"
)
def test_compress_gifsicle_e2e(tmp_path: Path, real_gif: Path) -> None:
    out_path = tmp_path / "out_gifsicle.gif"
    result = compress(
        input_path=real_gif,
        output_path=out_path,
        engine="gifsicle",
        params={"lossy_level": 40},
    )

    assert isinstance(result, CompressResult)
    assert out_path.exists()
    assert result.output_bytes > 0
    assert result.output_bytes == out_path.stat().st_size
    assert result.render_ms >= 0
    assert result.engine == "gifsicle"
    assert result.engine_version  # non-empty
    assert result.params == {"lossy_level": 40}


@pytest.mark.skipif(
    not AnimatelyLossyCompressor.available(), reason="animately binary not on PATH"
)
def test_compress_animately_e2e(tmp_path: Path, real_gif: Path) -> None:
    out_path = tmp_path / "out_animately.gif"
    result = compress(
        input_path=real_gif,
        output_path=out_path,
        engine="animately",
        params={"lossy_level": 40},
    )

    assert isinstance(result, CompressResult)
    assert out_path.exists()
    assert result.output_bytes > 0
    assert result.engine == "animately"
    assert result.engine_version  # non-empty


@pytest.mark.skipif(
    not GifsicleLossyCompressor.available(), reason="gifsicle binary not on PATH"
)
def test_compress_overwrites_existing_output(tmp_path: Path, real_gif: Path) -> None:
    """Per contract: compress() overwrites output_path if it exists."""
    out_path = tmp_path / "exists.gif"
    out_path.write_bytes(b"stale data")
    stale_size = out_path.stat().st_size

    result = compress(real_gif, out_path, engine="gifsicle", params={"lossy_level": 40})

    # New file replaces the stale bytes; size will differ.
    assert result.output_bytes != stale_size
    assert result.output_path == out_path
