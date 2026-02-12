"""Smoke tests for storage module constants."""

from giflab.storage import QUALITY_METRIC_COLUMNS


class TestQualityMetricColumns:
    """Tests for QUALITY_METRIC_COLUMNS constant."""

    def test_importable_and_non_empty(self) -> None:
        """QUALITY_METRIC_COLUMNS is importable and contains entries."""
        assert isinstance(QUALITY_METRIC_COLUMNS, list)
        assert len(QUALITY_METRIC_COLUMNS) > 0

    def test_all_strings(self) -> None:
        """Every entry is a string."""
        for col in QUALITY_METRIC_COLUMNS:
            assert isinstance(col, str), f"Expected str, got {type(col)} for {col!r}"

    def test_no_duplicates(self) -> None:
        """No duplicate column names."""
        assert len(QUALITY_METRIC_COLUMNS) == len(set(QUALITY_METRIC_COLUMNS))

    def test_contains_core_metrics(self) -> None:
        """Contains expected core metric names."""
        expected = {"ssim_mean", "psnr_mean", "composite_quality", "mse_mean"}
        actual = set(QUALITY_METRIC_COLUMNS)
        assert expected.issubset(actual), f"Missing: {expected - actual}"
