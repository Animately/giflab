"""End-to-end tests for ``giflab.measure`` using real metric computation.

These tests run actual SSIM/MS-SSIM/PSNR/CHIST calculations against a real
reference + candidate pair. The LPIPS test patches the torch-loading layer
specifically (``calculate_deep_perceptual_quality_metrics``) so the real
giflab metrics pipeline still runs end-to-end — only the model forward
pass is mocked. This exercises the public→internal key mapping for
``lpips_quality_mean`` without requiring a torch model download in CI.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

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
    # PSNR is reported in dB. Real-world values are well above 1.0 (a 5dB
    # PSNR would already be a destroyed image). Guards against silent
    # regressions to the old normalized-[0,1] behaviour.
    assert result.psnr > 1.0
    assert result.psnr <= 50.0  # PSNR_MAX_DB cap


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


# Fake return from the deep-perceptual layer. Mirrors the real shape: the
# branch at src/giflab/metrics.py:1750-1756 calls
# calculate_deep_perceptual_quality_metrics() and merges its return dict
# verbatim into the result, so any key the real function produces becomes
# a top-level result key. The 3 lpips_quality_* keys below are what the
# real function returns; we add the same shape here.
_FAKE_DEEP_PERCEPTUAL_RESULT = {
    "lpips_quality_mean": 0.073,
    "lpips_quality_p95": 0.18,
    "lpips_quality_max": 0.22,
}


def test_measure_lpips_e2e_real_pipeline_mocked_model(
    reference_and_candidate: tuple[Path, Path],
) -> None:
    """Real comprehensive-metrics path with LPIPS; only the model is mocked.

    Regression guard for bug_001 (the original silent-None for LPIPS).
    Exercises:
      - the real ``calculate_comprehensive_metrics`` code path
      - the real merge of ``deep_perceptual_metrics`` into the result dict
        at ``src/giflab/metrics.py:2925-2930``
      - the real ``measure()`` projection through
        ``_PUBLIC_TO_INTERNAL_METRIC_KEY["lpips"] == "lpips_quality_mean"``

    Without the torch model load (which is flaky in CI).
    """
    ref, cand = reference_and_candidate
    with patch(
        "giflab.deep_perceptual_metrics.calculate_deep_perceptual_quality_metrics",
        return_value=_FAKE_DEEP_PERCEPTUAL_RESULT,
    ) as mock_deep:
        result = measure(ref, cand, metrics=["lpips"])

    # The metrics layer must have invoked the deep-perceptual computation.
    mock_deep.assert_called_once()
    # And the projection must have pulled the mean through to the public field.
    assert result.lpips == pytest.approx(0.073)
    # No other public metric should be populated.
    for other in ("ssim", "ms_ssim", "psnr", "gmsd", "fsim", "chist"):
        assert getattr(result, other) is None


def test_measure_propagates_lpips_gate_to_deep_perceptual_config(
    reference_and_candidate: tuple[Path, Path],
) -> None:
    """FR-009 end-to-end: ENABLE_DEEP_PERCEPTUAL must reach the model layer.

    Regression guard for the combo of two settings that both have to be right
    for cost-avoidance: ``force_all_metrics=True`` (bypasses the conditional
    optimizer) AND ``ENABLE_DEEP_PERCEPTUAL=False``.

    Behaviour since v0.3.2 (FR-009 fix): ``ENABLE_DEEP_PERCEPTUAL=False`` short
    -circuits at the call site in ``calculate_comprehensive_metrics_from_frames``,
    so ``calculate_deep_perceptual_quality_metrics`` is not invoked at all —
    stronger than the previous "called with ``disable_deep_perceptual=True``".
    The LPIPS path stays cost-free.
    """
    ref, cand = reference_and_candidate

    # Case A: LPIPS not requested — function must not be called at all.
    with patch(
        "giflab.deep_perceptual_metrics.calculate_deep_perceptual_quality_metrics",
        return_value=_FAKE_DEEP_PERCEPTUAL_RESULT,
    ) as mock_deep:
        measure(ref, cand, metrics=["ssim"])

    mock_deep.assert_not_called()

    # Case B: LPIPS requested — function must be called, and the disable flag
    # in the config dict must be False (LPIPS is allowed to run).
    with patch(
        "giflab.deep_perceptual_metrics.calculate_deep_perceptual_quality_metrics",
        return_value=_FAKE_DEEP_PERCEPTUAL_RESULT,
    ) as mock_deep:
        measure(ref, cand, metrics=["lpips"])

    mock_deep.assert_called_once()
    args, kwargs = mock_deep.call_args
    deep_config = args[2] if len(args) > 2 else kwargs.get("config", {})
    assert (
        deep_config.get("disable_deep_perceptual") is False
    ), f"LPIPS requested → disable flag must be False; got config={deep_config}"
