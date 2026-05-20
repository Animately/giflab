"""End-to-end tests for ``giflab.measure`` using real metric computation.

These tests run actual SSIM/MS-SSIM/PSNR/CHIST calculations against a real
reference + candidate pair. LPIPS is excluded — model download is too
brittle for CI. LPIPS coverage stays in functional (mocked).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from giflab import MeasureResult, compress, measure
from giflab.tool_wrappers import GifsicleLossyCompressor


def _build_real_gif(
    path: Path, frames: int = 5, size: tuple[int, int] = (64, 64)
) -> None:
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
def reference_and_candidate(tmp_path: Path) -> tuple[Path, Path]:
    """A real reference GIF plus a compressed candidate produced by gifsicle."""
    ref = tmp_path / "ref.gif"
    _build_real_gif(ref)

    if not GifsicleLossyCompressor.available():
        pytest.skip("gifsicle binary not on PATH — cannot build candidate")

    cand = tmp_path / "cand.gif"
    compress(ref, cand, engine="gifsicle", params={"lossy_level": 60})
    return ref, cand


def test_measure_ssim_only(reference_and_candidate: tuple[Path, Path]) -> None:
    ref, cand = reference_and_candidate
    result = measure(ref, cand, metrics=["ssim"])

    assert isinstance(result, MeasureResult)
    assert result.ssim is not None
    assert 0.0 <= result.ssim <= 1.0
    # All others must be unpopulated.
    for other in ("ms_ssim", "psnr", "lpips", "gmsd", "fsim", "chist"):
        assert getattr(result, other) is None


def test_measure_ssim_and_psnr(reference_and_candidate: tuple[Path, Path]) -> None:
    ref, cand = reference_and_candidate
    result = measure(ref, cand, metrics=["ssim", "psnr"])

    assert result.ssim is not None
    assert result.psnr is not None
    assert result.ms_ssim is None


def test_measure_chist(reference_and_candidate: tuple[Path, Path]) -> None:
    ref, cand = reference_and_candidate
    result = measure(ref, cand, metrics=["chist"])

    assert result.chist is not None
    assert result.ssim is None


def test_measure_ms_ssim(reference_and_candidate: tuple[Path, Path]) -> None:
    ref, cand = reference_and_candidate
    result = measure(ref, cand, metrics=["ms_ssim"])

    assert result.ms_ssim is not None
    assert result.ssim is None
