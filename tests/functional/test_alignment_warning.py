"""Tests for the frame-drop alignment warning surfaced by the metrics pipeline.

Background ([[giflab-alignment-warning-threshold]], Outlier 2): imperfect
frame-drop alignment (``alignment_accuracy`` well below 1.0) was computed by the
timing validator but never surfaced — a silent ``alignment_accuracy=0.976`` could
slip through with no signal to any downstream consumer.

``calculate_comprehensive_metrics_from_frames`` now sets an ``alignment_warning``
float key (1.0 = warn, 0.0 = no warn) whenever a REAL alignment value lands below
``MetricsConfig.ALIGNMENT_WARNING_THRESHOLD`` (default 0.98). The warning must be
NaN-honest: a missing (NaN), failure-sentinel (-1.0), or absent alignment value
must NEVER fire the warning.

The alignment value is driven by ``extract_timing_metrics_for_csv``, imported at
call time inside ``_from_frames`` from
``giflab.wrapper_validation.timing_validation`` — so the controlled-value tests
monkeypatch that attribute on the source module (the lookup site).

Layer: functional — synthetic frames + monkeypatched timing extraction, no real
GIF I/O for the controlled-value cases.
"""

import math
from pathlib import Path

import numpy as np
import pytest

import giflab.wrapper_validation.timing_validation as timing_validation_module
from giflab.config import MetricsConfig
from giflab.metrics import calculate_comprehensive_metrics_from_frames


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _frames(n: int = 3) -> list[np.ndarray]:
    """Identical synthetic RGB frames (alignment is file-derived, so the pixel
    content is irrelevant to the warning). 64x64 is large enough for every
    downstream detector (LPIPS et al.) to run cleanly."""
    return [np.full((64, 64, 3), 128, dtype=np.uint8) for _ in range(n)]


def _file_metadata(tmp_path: Path) -> dict:
    """file_metadata carrying both paths so the timing-validation branch runs.

    The paths themselves are never read because we monkeypatch the timing
    validator and its extractor; they only need to be present so the
    ``original_path``/``compressed_path`` guard at the timing branch is True.
    """
    orig = tmp_path / "orig.gif"
    comp = tmp_path / "comp.gif"
    orig.write_bytes(b"GIF89a")
    comp.write_bytes(b"GIF89a")
    return {"original_path": orig, "compressed_path": comp}


def _patch_alignment(monkeypatch, value) -> None:
    """Force the timing validation to yield a controlled ``alignment_accuracy``.

    Two patches on the source module (the lookup site, since both names are
    imported at call time inside ``_from_frames``):
      - ``TimingGridValidator.validate_timing_integrity`` -> a no-op stub so the
        real validator never chokes on the placeholder GIF bytes;
      - ``extract_timing_metrics_for_csv`` -> returns the controlled alignment.
    """

    def _fake_validate(_self, _orig, _comp):
        return object()  # opaque placeholder; the extractor ignores it

    def _fake_extract(_validation_result):
        return {
            "timing_grid_ms": 10,
            "grid_length": 3,
            "duration_diff_ms": 0,
            "timing_drift_score": 0.0,
            "max_timing_drift_ms": 0,
            "alignment_accuracy": value,
        }

    monkeypatch.setattr(
        timing_validation_module.TimingGridValidator,
        "validate_timing_integrity",
        _fake_validate,
    )
    monkeypatch.setattr(
        timing_validation_module,
        "extract_timing_metrics_for_csv",
        _fake_extract,
    )


def _run(monkeypatch, tmp_path, alignment_value) -> dict:
    _patch_alignment(monkeypatch, alignment_value)
    # force_all_metrics=True forces the standard (non-conditional) path, which
    # is the path that runs timing validation and sets alignment_warning.
    # Identical high-quality frames would otherwise take the conditional
    # high-tier fast path, which never computes alignment at all.
    return calculate_comprehensive_metrics_from_frames(
        _frames(),
        _frames(),
        file_metadata=_file_metadata(tmp_path),
        force_all_metrics=True,
    )


# ---------------------------------------------------------------------------
# Config: threshold default + range guard
# ---------------------------------------------------------------------------


class TestAlignmentWarningThresholdConfig:
    def test_default_threshold_is_098(self):
        assert MetricsConfig().ALIGNMENT_WARNING_THRESHOLD == 0.98

    def test_in_range_value_accepted(self):
        assert MetricsConfig(ALIGNMENT_WARNING_THRESHOLD=0.5).ALIGNMENT_WARNING_THRESHOLD == 0.5

    def test_above_one_rejected(self):
        with pytest.raises(ValueError):
            MetricsConfig(ALIGNMENT_WARNING_THRESHOLD=1.5)

    def test_below_zero_rejected(self):
        with pytest.raises(ValueError):
            MetricsConfig(ALIGNMENT_WARNING_THRESHOLD=-0.1)


# ---------------------------------------------------------------------------
# The four alignment states + the absent (no file_metadata) production case
# ---------------------------------------------------------------------------


class TestAlignmentWarningKey:
    def test_sub_threshold_real_value_warns(self, monkeypatch, tmp_path):
        """A genuine 0.847 (< 0.98) sets alignment_warning == 1.0."""
        result = _run(monkeypatch, tmp_path, 0.847)
        assert result["alignment_accuracy"] == 0.847
        assert result["alignment_warning"] == 1.0

    def test_perfect_value_does_not_warn(self, monkeypatch, tmp_path):
        """A perfect 1.0 (>= 0.98) sets alignment_warning == 0.0."""
        result = _run(monkeypatch, tmp_path, 1.0)
        assert result["alignment_warning"] == 0.0

    def test_at_threshold_does_not_warn(self, monkeypatch, tmp_path):
        """Exactly at threshold (0.98) does NOT warn (strict ``<``)."""
        result = _run(monkeypatch, tmp_path, 0.98)
        assert result["alignment_warning"] == 0.0

    def test_nan_alignment_does_not_warn(self, monkeypatch, tmp_path, caplog):
        """A NaN (missing) alignment never warns and logs nothing."""
        with caplog.at_level("WARNING"):
            result = _run(monkeypatch, tmp_path, float("nan"))
        assert result["alignment_warning"] == 0.0
        assert "alignment" not in caplog.text.lower()

    def test_failure_sentinel_does_not_warn(self, monkeypatch, tmp_path):
        """The documented -1.0 failure sentinel must NOT trip the warning."""
        result = _run(monkeypatch, tmp_path, -1.0)
        assert result["alignment_warning"] == 0.0

    def test_absent_alignment_does_not_warn(self):
        """No file_metadata paths -> timing_metrics == {} -> alignment never set.

        This is the dominant production case (frame-based metrics with no file
        paths). alignment_warning must be a falsey 0.0 / absent, never 1.0, and
        no warning is logged. force_all_metrics keeps us on the standard path so
        the warning block runs (and resolves to 0.0 with alignment absent);
        without it, identical frames take the conditional high-tier fast path,
        which likewise never sets alignment_warning to 1.0.
        """
        result = calculate_comprehensive_metrics_from_frames(
            _frames(),
            _frames(),
            force_all_metrics=True,
        )
        assert "alignment_accuracy" not in result
        assert result.get("alignment_warning", 0.0) == 0.0

    def test_warning_value_is_float(self, monkeypatch, tmp_path):
        """alignment_warning is a float (0.0/1.0), not a Python bool — matches
        the storage REAL columns and the ssimulacra2_triggered precedent."""
        result = _run(monkeypatch, tmp_path, 0.5)
        assert isinstance(result["alignment_warning"], float)
        assert not isinstance(result["alignment_warning"], bool)
