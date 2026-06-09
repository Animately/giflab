"""End-to-end tests for ``giflab.measure`` using real metric computation.

These tests run actual SSIM/MS-SSIM/PSNR/CHIST calculations against a real
reference + candidate pair. The LPIPS test patches the torch-loading layer
specifically (``calculate_deep_perceptual_quality_metrics``) so the real
giflab metrics pipeline still runs end-to-end — only the model forward
pass is mocked. This exercises the public→internal key mapping for
``lpips_quality_mean`` without requiring a torch model download in CI.
"""

from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from giflab import MeasureResult, compress, measure
from giflab.tool_wrappers import GifsicleLossyCompressor
from PIL import Image, ImageDraw


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


def _build_noisy_gif(
    path: Path, frames: int = 5, size: tuple[int, int] = (96, 96), seed: int = 42
) -> None:
    """High-frequency-noise reference: heavy gifsicle lossy genuinely degrades it.

    The flat synthetic GIF in ``_build_real_gif`` compresses near-losslessly even
    at high lossy levels — every quality metric pins to ~1.0, so removing the
    LPIPS contributor and redistributing its 4% weight across other ~1.0
    contributors does not move the composite at all (it stays clamped at 1.0).
    That makes the determinism-lock test below a no-op. A noisy reference forces
    the metrics to *spread* under heavy lossy, so the weight-redistribution delta
    is real and exceeds ``pytest.approx`` tolerance — see
    ``test_measure_composite_quality_deterministic_across_request_sets`` which
    asserts that spread explicitly via the non-degeneracy check.
    """
    rng = np.random.default_rng(seed)
    images = [
        Image.fromarray(
            rng.integers(0, 256, size=(size[1], size[0], 3), dtype=np.uint8), "RGB"
        )
        for _ in range(frames)
    ]
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


@pytest.fixture
def degraded_reference_and_candidate(tmp_path: Path) -> tuple[Path, Path]:
    """A noisy reference + heavily-degraded gifsicle candidate (lossy 200).

    Distinct from ``reference_and_candidate``: the noise + heavy lossy spreads the
    per-metric quality scores apart so that the LPIPS contributor's
    weight-redistribution actually shifts the composite. Required by the
    determinism-lock test — a benign pair makes that test vacuous.
    """
    ref = tmp_path / "noisy_ref.gif"
    _build_noisy_gif(ref)

    if not GifsicleLossyCompressor.available():
        pytest.skip("gifsicle binary not on PATH — cannot build candidate")

    cand = tmp_path / "noisy_cand.gif"
    compress(ref, cand, engine="gifsicle", params={"lossy_level": 200})
    return ref, cand


def test_measure_ssim_only(reference_and_candidate: tuple[Path, Path]) -> None:
    ref, cand = reference_and_candidate
    result = measure(ref, cand, metrics=["ssim"])

    assert isinstance(result, MeasureResult)
    assert result.ssim is not None
    assert 0.0 <= result.ssim <= 1.0
    # All others must be unpopulated.
    for other in (
        "ms_ssim",
        "psnr",
        "lpips",
        "gmsd",
        "fsim",
        "chist",
        "composite_quality",
    ):
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


def test_measure_composite_quality_e2e(
    reference_and_candidate: tuple[Path, Path],
) -> None:
    """Real-pipeline composite_quality is a finite value in [0, 1].

    Exercises the full ``calculate_comprehensive_metrics`` → composite path and
    the public projection of the bare ``composite_quality`` key. The composite is
    NaN-tolerant only in the pathological case where half or more of the present
    contributor weight is unmeasurable — for a real reference/candidate pair with
    the cheap structural metrics all computable, it must be finite.
    """
    ref, cand = reference_and_candidate
    result = measure(ref, cand, metrics=["composite_quality"])

    assert isinstance(result, MeasureResult)
    assert result.composite_quality is not None
    # SSIM/PSNR/etc. are all computable on a real pair, so well under half the
    # weight is unmeasurable here — assert finite and in range.
    assert math.isfinite(result.composite_quality)
    assert 0.0 <= result.composite_quality <= 1.0
    # Requesting composite alone leaves the other public fields unset.
    for other in ("ssim", "ms_ssim", "psnr", "lpips", "gmsd", "fsim", "chist"):
        assert getattr(result, other) is None


def test_measure_composite_quality_deterministic_across_request_sets(
    degraded_reference_and_candidate: tuple[Path, Path],
) -> None:
    """Regression lock for the load-bearing finding: composite is request-set-invariant.

    Because measure() forces ENABLE_DEEP_PERCEPTUAL on whenever composite_quality
    is requested, the LPIPS contributor is always present in the weighted
    aggregate — so co-requesting "lpips" cannot change the composite value.
    Without the gate fix, ``measure(["composite_quality"])`` would gate LPIPS off,
    redistribute its 4% weight, and land on a *different* number than
    ``measure(["composite_quality", "lpips"])`` — the first assertion below would
    then fail.

    The earlier version of this test used a benign flat-colour pair on which
    every metric pinned to ~1.0; redistributing the LPIPS weight across other
    ~1.0 contributors did not move the composite, so the test passed even with
    the gate reverted (a no-op lock). This version uses a deliberately degraded
    pair (noisy reference + gifsicle lossy 200) AND adds an explicit
    *non-degeneracy* assertion: it computes, via the internal metrics path, the
    composite that the bug *would* produce (LPIPS gated off) and asserts it
    differs from the gate-forced public value. That guards both the gate fix and
    the fixture's spread — if a future change makes the pair benign again, the
    non-degeneracy assertion fails loudly rather than letting the lock rot back
    into a no-op.
    """
    ref, cand = degraded_reference_and_candidate

    composite_alone = measure(
        ref, cand, metrics=["composite_quality"]
    ).composite_quality
    composite_with_lpips = measure(
        ref, cand, metrics=["composite_quality", "lpips"]
    ).composite_quality

    assert composite_alone is not None
    assert composite_with_lpips is not None
    assert math.isfinite(composite_alone)
    # THE LOCK: request-set invariance. Fails if the gate is reverted, because
    # then composite_alone (LPIPS off → 4% redistributed) ≠ composite_with_lpips.
    assert composite_alone == pytest.approx(composite_with_lpips)

    # Non-degeneracy guard: prove the pair is degraded enough that the gate is
    # actually doing work — i.e. the LPIPS-off composite (what the bug emits)
    # genuinely differs from the gate-forced value. This is what the gate fix
    # converts from "request-set-dependent" to "invariant"; if it were ~0 the
    # lock above would be vacuous (the benign-fixture failure mode this test was
    # rewritten to close). The internal path is used only to synthesise the
    # would-be-buggy value; production composite_quality never gates LPIPS off.
    from giflab.config import MetricsConfig
    from giflab.metrics import calculate_comprehensive_metrics

    lpips_off = MetricsConfig()
    lpips_off.ENABLE_DEEP_PERCEPTUAL = False
    lpips_off.ENABLE_TEMPORAL_ARTIFACTS = False
    buggy_composite = calculate_comprehensive_metrics(
        ref, cand, config=lpips_off, force_all_metrics=True
    )["composite_quality"]
    assert math.isfinite(buggy_composite)
    # The would-be-buggy (LPIPS-off) composite must be measurably apart from the
    # gate-forced one — otherwise the invariance assertion above proves nothing.
    # Empirically ~0.004 on this fixture; require a margin comfortably above
    # pytest.approx's default relative tolerance (~1e-6) yet below the observed
    # delta so the test is not flaky.
    assert abs(composite_with_lpips - buggy_composite) > 1e-4
