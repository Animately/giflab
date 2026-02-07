import numpy as np
import pytest
from giflab.data_prep import (
    apply_confidence_weights,
    clip_outliers,
    minmax_scale,
    normalise_metrics,
    zscore_scale,
)


class TestScalingHelpers:
    def test_minmax_scale_basic(self):
        data = [0, 5, 10]
        scaled = minmax_scale(data)
        assert np.allclose(scaled, [0.0, 0.5, 1.0])

    def test_minmax_scale_constant(self):
        data = [3, 3, 3]
        scaled = minmax_scale(data, feature_range=(0, 1))
        assert np.allclose(scaled, [0.5, 0.5, 0.5])

    def test_zscore_scale_basic(self):
        data = [0, 5, 10]
        scaled = zscore_scale(data)
        assert np.isclose(np.mean(scaled), 0.0)
        assert np.isclose(np.std(scaled), 1.0)

    def test_zscore_scale_constant(self):
        data = [7, 7, 7]
        scaled = zscore_scale(data)
        assert np.allclose(scaled, [0.0, 0.0, 0.0])


class TestMetricNormalisation:
    def test_normalise_metrics_zscore(self):
        metrics = {"a": 1.0, "b": 2.0, "c": 3.0}
        norm = normalise_metrics(metrics, method="zscore")
        assert np.isclose(np.mean(list(norm.values())), 0.0)

    def test_normalise_metrics_minmax(self):
        metrics = {"x": 0.0, "y": 10.0}
        norm = normalise_metrics(metrics, method="minmax")
        assert np.isclose(norm["x"], 0.0)
        assert np.isclose(norm["y"], 1.0)


class TestConfidenceWeighting:
    def test_apply_confidence_weights(self):
        metrics = {"m": 10.0, "n": 5.0}
        conf = {"m": 0.5, "n": 2.0}
        weighted = apply_confidence_weights(metrics, conf)
        assert weighted["m"] == 5.0
        assert weighted["n"] == 10.0

    def test_apply_confidence_weights_negative_error(self):
        metrics = {"v": 1.0}
        conf = {"v": -1.0}
        with pytest.raises(ValueError):
            apply_confidence_weights(metrics, conf)


class TestOutlierClipping:
    def test_clip_outliers_iqr(self):
        data = [1, 2, 2, 2, 100]  # 100 is an outlier
        clipped = clip_outliers(data, method="iqr", factor=1.5)
        assert max(clipped) < 100  # Outlier should be clipped down

    def test_clip_outliers_sigma(self):
        data = [0, 0, 0, 0, 50]
        clipped = clip_outliers(data, method="sigma", factor=1.5)
        assert max(clipped) < 50
