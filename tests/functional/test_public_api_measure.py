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

    All 7 public metrics present with distinguishable values.
    """
    return {
        "ssim": 0.91,
        "ms_ssim": 0.92,
        "psnr": 30.5,
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

    assert result.ssim == 0.91
    assert result.ms_ssim == 0.92
    assert result.psnr == 30.5
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

    for metric, expected in _fake_full_result().items():
        assert getattr(result, metric) == expected


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

    assert result.ssim == 0.91
    assert result.psnr == 30.5
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
