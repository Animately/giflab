"""Functional tests for the public ``giflab.measure`` API.

Mocks ``calculate_comprehensive_metrics`` — these tests do not load LPIPS
models or do real metric computation. Real-metric coverage lives in
``tests/integration/test_public_api_measure_e2e.py``.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from giflab import (
    SUPPORTED_METRICS,
    MeasureResult,
    UnknownMetricError,
    measure,
)


def _fake_full_result() -> dict[str, float]:
    """A pretend ``calculate_comprehensive_metrics`` result dict.

    Mirrors the real internal-key shape so projection bugs (LPIPS keyed as
    ``lpips_quality_mean`` not ``lpips``, PSNR stored normalized to [0,1]
    not in dB) cannot be hidden by mock convenience.

    Expected projection through ``measure()``:
      - ssim/ms_ssim/gmsd/fsim/chist pass through unchanged
      - psnr is denormalized: 0.5 * PSNR_MAX_DB(50.0) = 25.0 dB
      - lpips reads lpips_quality_mean = 0.08
    """
    return {
        "ssim": 0.91,
        "ms_ssim": 0.92,
        # PSNR is normalized internally; public surface denormalizes to dB.
        # 0.5 is chosen because it is exactly representable in IEEE 754, so
        # 0.5 * 50.0 == 25.0 holds under any float implementation — no
        # rounding-accident dependency in the test assertion.
        "psnr": 0.5,
        "lpips_quality_mean": 0.08,  # internal key — not bare "lpips"
        "gmsd": 0.05,
        "fsim": 0.93,
        "chist": 0.97,
    }


# Expected public-API scalars after projection of _fake_full_result().
_EXPECTED_PUBLIC = {
    "ssim": 0.91,
    "ms_ssim": 0.92,
    "psnr": 25.0,  # 0.5 * PSNR_MAX_DB (50.0) — exact in IEEE 754
    "lpips": 0.08,
    "gmsd": 0.05,
    "fsim": 0.93,
    "chist": 0.97,
}


@pytest.fixture
def two_gifs(tiny_gif: Path, tmp_path: Path) -> tuple[Path, Path]:
    """Reference + candidate path pair (both exist, both readable)."""
    candidate = tmp_path / "candidate.gif"
    candidate.write_bytes(tiny_gif.read_bytes())
    return tiny_gif, candidate


@pytest.mark.parametrize("metric", list(SUPPORTED_METRICS))
def test_measure_happy_path_per_metric(
    two_gifs: tuple[Path, Path], metric: str
) -> None:
    """For each supported metric, requesting it alone populates only that field."""
    ref, cand = two_gifs

    with patch(
        "giflab.public_api.calculate_comprehensive_metrics",
        return_value=_fake_full_result(),
    ):
        result = measure(ref, cand, metrics=[metric])

    assert isinstance(result, MeasureResult)
    assert getattr(result, metric) is not None
    # All other fields must be None.
    for other in SUPPORTED_METRICS:
        if other != metric:
            assert getattr(result, other) is None


def test_measure_multi_metric_subset(two_gifs: tuple[Path, Path]) -> None:
    ref, cand = two_gifs

    with patch(
        "giflab.public_api.calculate_comprehensive_metrics",
        return_value=_fake_full_result(),
    ):
        result = measure(ref, cand, metrics=["ssim", "ms_ssim", "psnr"])

    assert result.ssim == _EXPECTED_PUBLIC["ssim"]
    assert result.ms_ssim == _EXPECTED_PUBLIC["ms_ssim"]
    assert result.psnr == pytest.approx(
        _EXPECTED_PUBLIC["psnr"]
    )  # in dB, not normalized
    assert result.lpips is None
    assert result.gmsd is None
    assert result.fsim is None
    assert result.chist is None


def test_measure_all_metrics(two_gifs: tuple[Path, Path]) -> None:
    ref, cand = two_gifs

    with patch(
        "giflab.public_api.calculate_comprehensive_metrics",
        return_value=_fake_full_result(),
    ):
        result = measure(ref, cand, metrics=list(SUPPORTED_METRICS))

    for metric, expected in _EXPECTED_PUBLIC.items():
        actual = getattr(result, metric)
        # pytest.approx handles the PSNR denormalization (caller × 50.0)
        # and is a no-op for already-exact values.
        assert actual == pytest.approx(
            expected
        ), f"public field {metric!r} expected {expected}, got {actual}"


def test_measure_unknown_metric_raises_before_computation(
    two_gifs: tuple[Path, Path]
) -> None:
    ref, cand = two_gifs
    mock_metrics = patch("giflab.public_api.calculate_comprehensive_metrics")

    with mock_metrics as m:
        with pytest.raises(UnknownMetricError) as exc_info:
            measure(ref, cand, metrics=["ssim", "not_a_real_metric"])  # type: ignore[list-item]

        # Computation must not have started.
        m.assert_not_called()

    msg = str(exc_info.value)
    assert "not_a_real_metric" in msg
    for m_name in SUPPORTED_METRICS:
        assert m_name in msg


def test_measure_empty_metrics_raises_valueerror(two_gifs: tuple[Path, Path]) -> None:
    ref, cand = two_gifs
    with pytest.raises(ValueError):
        measure(ref, cand, metrics=[])


def test_measure_duplicate_metric_names_tolerated(two_gifs: tuple[Path, Path]) -> None:
    ref, cand = two_gifs
    with patch(
        "giflab.public_api.calculate_comprehensive_metrics",
        return_value=_fake_full_result(),
    ):
        result = measure(ref, cand, metrics=["ssim", "ssim", "psnr"])

    assert result.ssim == _EXPECTED_PUBLIC["ssim"]
    assert result.psnr == pytest.approx(_EXPECTED_PUBLIC["psnr"])
    assert result.ms_ssim is None


def test_measure_missing_input_raises_filenotfounderror(tmp_path: Path) -> None:
    missing_ref = tmp_path / "no_ref.gif"
    missing_cand = tmp_path / "no_cand.gif"
    with pytest.raises(FileNotFoundError):
        measure(missing_ref, missing_cand, metrics=["ssim"])


def test_measure_all_or_nothing_on_metric_failure(two_gifs: tuple[Path, Path]) -> None:
    """If the underlying call raises, measure() does not return partial results."""
    ref, cand = two_gifs

    with patch(
        "giflab.public_api.calculate_comprehensive_metrics",
        side_effect=RuntimeError("metric computation blew up"),
    ):
        with pytest.raises(Exception) as exc_info:
            measure(ref, cand, metrics=["ssim", "psnr"])

    # The raised exception preserves the underlying cause.
    assert "metric computation blew up" in str(exc_info.value) or isinstance(
        exc_info.value.__cause__, RuntimeError
    )


def test_measure_does_not_enable_lpips_when_not_requested(
    two_gifs: tuple[Path, Path],
) -> None:
    """FR-009: requesting only cheap metrics must not enable LPIPS computation.

    LPIPS is the only expensive metric (loads a torch model). The other 6 are
    computed in a shared pass over frames and are not individually gated;
    requesting any of them implicitly computes the cheap-metric batch.
    """
    ref, cand = two_gifs

    with patch(
        "giflab.public_api.calculate_comprehensive_metrics",
        return_value=_fake_full_result(),
    ) as mock_calc:
        measure(ref, cand, metrics=["ssim"])

    # The config kwarg must have ENABLE_DEEP_PERCEPTUAL=False.
    _, kwargs = mock_calc.call_args
    config = kwargs.get("config")
    assert config is not None
    assert config.ENABLE_DEEP_PERCEPTUAL is False


def test_measure_enables_lpips_when_requested(two_gifs: tuple[Path, Path]) -> None:
    ref, cand = two_gifs
    with patch(
        "giflab.public_api.calculate_comprehensive_metrics",
        return_value=_fake_full_result(),
    ) as mock_calc:
        measure(ref, cand, metrics=["ssim", "lpips"])

    _, kwargs = mock_calc.call_args
    config = kwargs.get("config")
    assert config is not None
    assert config.ENABLE_DEEP_PERCEPTUAL is True


def test_measure_disables_temporal_artifacts_for_cheap_metrics(
    two_gifs: tuple[Path, Path],
) -> None:
    """FR-009: cheap metrics must not load LPIPS via the temporal_artifacts path.

    temporal_artifacts.calculate_enhanced_temporal_metrics internally loads
    LPIPS (for lpips_t_*), so its config gate must be False when no requested
    metric needs it. v0.3.0 surface contains no temporal metrics, so this
    should hold for every individual SUPPORTED_METRIC and any combination.
    """
    ref, cand = two_gifs

    with patch(
        "giflab.public_api.calculate_comprehensive_metrics",
        return_value=_fake_full_result(),
    ) as mock_calc:
        measure(ref, cand, metrics=["ssim"])

    _, kwargs = mock_calc.call_args
    config = kwargs.get("config")
    assert config is not None
    assert config.ENABLE_TEMPORAL_ARTIFACTS is False


@pytest.mark.parametrize("metric", list(SUPPORTED_METRICS))
def test_measure_disables_temporal_artifacts_for_every_public_metric(
    two_gifs: tuple[Path, Path], metric: str
) -> None:
    """No v0.3.0 public metric should trigger the temporal_artifacts pipeline."""
    ref, cand = two_gifs
    with patch(
        "giflab.public_api.calculate_comprehensive_metrics",
        return_value=_fake_full_result(),
    ) as mock_calc:
        measure(ref, cand, metrics=[metric])

    _, kwargs = mock_calc.call_args
    config = kwargs.get("config")
    assert config is not None
    assert config.ENABLE_TEMPORAL_ARTIFACTS is False, (
        f"Requesting {metric!r} unexpectedly enabled temporal_artifacts"
    )


def test_measure_lpips_reads_quality_mean_key(two_gifs: tuple[Path, Path]) -> None:
    """Regression: LPIPS must read the internal ``lpips_quality_mean`` key.

    The bare ``lpips`` key is never set by ``calculate_comprehensive_metrics``
    — the deep-perceptual branch surfaces ``lpips_quality_{mean,p95,max}``.
    Without the public→internal mapping, ``measure(metrics=["lpips"])`` would
    silently return ``MeasureResult(lpips=None, ...)``.
    """
    ref, cand = two_gifs
    # Internal-shape result that has NO bare "lpips" key — only the real ones.
    internal_result = {
        "ssim": 0.91,
        "lpips_quality_mean": 0.073,
        "lpips_quality_p95": 0.18,
        "lpips_quality_max": 0.22,
    }
    with patch(
        "giflab.public_api.calculate_comprehensive_metrics",
        return_value=internal_result,
    ):
        result = measure(ref, cand, metrics=["lpips"])

    assert result.lpips == 0.073


def test_measure_raises_when_internal_key_missing(two_gifs: tuple[Path, Path]) -> None:
    """If a requested metric resolves to None after projection, raise.

    Surfaces silent key-drift (e.g. internal metrics layer renames a key)
    rather than returning a None field the caller can't distinguish from
    "not requested".
    """
    ref, cand = two_gifs
    # Result dict missing the gmsd key entirely.
    incomplete = {"ssim": 0.91}
    with patch(
        "giflab.public_api.calculate_comprehensive_metrics",
        return_value=incomplete,
    ):
        with pytest.raises(Exception) as exc_info:
            measure(ref, cand, metrics=["gmsd"])

    assert "gmsd" in str(exc_info.value)


def test_measure_psnr_returned_in_decibels(two_gifs: tuple[Path, Path]) -> None:
    """Regression: PSNR must be returned in dB, not normalized [0,1].

    Internally PSNR is divided by PSNR_MAX_DB (default 50.0). The public
    surface promises dB units (industry convention), so the projection
    multiplies it back.
    """
    ref, cand = two_gifs
    # Internal normalized value 0.7 → 35.0 dB on the public surface.
    with patch(
        "giflab.public_api.calculate_comprehensive_metrics",
        return_value={"psnr": 0.7},
    ):
        result = measure(ref, cand, metrics=["psnr"])

    assert result.psnr == pytest.approx(35.0)  # 0.7 * 50.0
    # And a real consumer's sanity check — PSNR in dB is usually > 1.
    assert result.psnr > 1.0
