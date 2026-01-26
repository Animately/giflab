"""Tests for prediction module Pydantic schemas.

Constitution Compliance:
- Principle IV (Test-Driven Quality): Tests for schema validation
"""

from datetime import datetime, timezone

import pytest

from giflab.prediction.schemas import (
    CompressionCurveV1,
    CurveType,
    DatasetSplit,
    Engine,
    GifFeaturesV1,
)


class TestGifFeaturesV1:
    """Test GifFeaturesV1 schema validation."""

    def test_valid_features(self) -> None:
        """Test creating valid GifFeaturesV1."""
        features = GifFeaturesV1(
            gif_sha="a" * 64,
            gif_name="test.gif",
            extraction_version="1.0.0",
            extracted_at=datetime.now(timezone.utc),
            width=100,
            height=100,
            frame_count=10,
            duration_ms=1000,
            file_size_bytes=50000,
            unique_colors=256,
            entropy=5.5,
            edge_density=0.3,
            color_complexity=0.5,
            gradient_smoothness=0.7,
            contrast_score=0.4,
            text_density=0.1,
            dct_energy_ratio=0.6,
            color_histogram_entropy=4.0,
            dominant_color_ratio=0.8,
            motion_intensity=0.2,
            motion_smoothness=0.9,
            static_region_ratio=0.7,
            temporal_entropy=2.0,
            frame_similarity=0.85,
            inter_frame_mse_mean=100.0,
            inter_frame_mse_std=20.0,
            lossless_compression_ratio=0.5,
        )
        assert features.gif_sha == "a" * 64
        assert features.schema_version == "1.0.0"

    def test_sha_must_be_64_chars(self) -> None:
        """Test that gif_sha must be exactly 64 characters."""
        with pytest.raises(ValueError):
            GifFeaturesV1(
                gif_sha="abc",  # Too short
                gif_name="test.gif",
                extraction_version="1.0.0",
                extracted_at=datetime.now(timezone.utc),
                width=100,
                height=100,
                frame_count=10,
                duration_ms=1000,
                file_size_bytes=50000,
                unique_colors=256,
                entropy=5.5,
                edge_density=0.3,
                color_complexity=0.5,
                gradient_smoothness=0.7,
                contrast_score=0.4,
                text_density=0.1,
                dct_energy_ratio=0.6,
                color_histogram_entropy=4.0,
                dominant_color_ratio=0.8,
                motion_intensity=0.2,
                motion_smoothness=0.9,
                static_region_ratio=0.7,
                temporal_entropy=2.0,
                frame_similarity=0.85,
                inter_frame_mse_mean=100.0,
                inter_frame_mse_std=20.0,
                lossless_compression_ratio=0.5,
            )

    def test_sha_must_be_hex(self) -> None:
        """Test that gif_sha must be hexadecimal."""
        with pytest.raises(ValueError):
            GifFeaturesV1(
                gif_sha="g" * 64,  # 'g' is not hex
                gif_name="test.gif",
                extraction_version="1.0.0",
                extracted_at=datetime.now(timezone.utc),
                width=100,
                height=100,
                frame_count=10,
                duration_ms=1000,
                file_size_bytes=50000,
                unique_colors=256,
                entropy=5.5,
                edge_density=0.3,
                color_complexity=0.5,
                gradient_smoothness=0.7,
                contrast_score=0.4,
                text_density=0.1,
                dct_energy_ratio=0.6,
                color_histogram_entropy=4.0,
                dominant_color_ratio=0.8,
                motion_intensity=0.2,
                motion_smoothness=0.9,
                static_region_ratio=0.7,
                temporal_entropy=2.0,
                frame_similarity=0.85,
                inter_frame_mse_mean=100.0,
                inter_frame_mse_std=20.0,
                lossless_compression_ratio=0.5,
            )

    def test_entropy_range(self) -> None:
        """Test that entropy must be 0-8."""
        with pytest.raises(ValueError):
            GifFeaturesV1(
                gif_sha="a" * 64,
                gif_name="test.gif",
                extraction_version="1.0.0",
                extracted_at=datetime.now(timezone.utc),
                width=100,
                height=100,
                frame_count=10,
                duration_ms=1000,
                file_size_bytes=50000,
                unique_colors=256,
                entropy=10.0,  # Too high
                edge_density=0.3,
                color_complexity=0.5,
                gradient_smoothness=0.7,
                contrast_score=0.4,
                text_density=0.1,
                dct_energy_ratio=0.6,
                color_histogram_entropy=4.0,
                dominant_color_ratio=0.8,
                motion_intensity=0.2,
                motion_smoothness=0.9,
                static_region_ratio=0.7,
                temporal_entropy=2.0,
                frame_similarity=0.85,
                inter_frame_mse_mean=100.0,
                inter_frame_mse_std=20.0,
                lossless_compression_ratio=0.5,
            )


class TestCompressionCurveV1:
    """Test CompressionCurveV1 schema validation."""

    def test_valid_lossy_curve(self) -> None:
        """Test creating valid lossy compression curve."""
        curve = CompressionCurveV1(
            gif_sha="b" * 64,
            engine=Engine.GIFSICLE,
            curve_type=CurveType.LOSSY,
            is_predicted=False,
            created_at=datetime.now(timezone.utc),
            size_at_lossy_0=100.0,
            size_at_lossy_20=90.0,
            size_at_lossy_40=80.0,
            size_at_lossy_60=70.0,
            size_at_lossy_80=60.0,
            size_at_lossy_100=50.0,
            size_at_lossy_120=45.0,
        )
        assert curve.engine == Engine.GIFSICLE
        assert curve.curve_type == CurveType.LOSSY

    def test_valid_color_curve(self) -> None:
        """Test creating valid color reduction curve."""
        curve = CompressionCurveV1(
            gif_sha="c" * 64,
            engine=Engine.ANIMATELY,
            curve_type=CurveType.COLORS,
            is_predicted=True,
            model_version="1.0.0",
            confidence_scores=[0.9, 0.85, 0.8, 0.75, 0.7],
            created_at=datetime.now(timezone.utc),
            size_at_colors_256=100.0,
            size_at_colors_128=85.0,
            size_at_colors_64=70.0,
            size_at_colors_32=55.0,
            size_at_colors_16=40.0,
        )
        assert curve.is_predicted is True
        assert curve.model_version == "1.0.0"

    def test_get_lossy_curve_points(self) -> None:
        """Test lossy curve point extraction."""
        curve = CompressionCurveV1(
            gif_sha="d" * 64,
            engine=Engine.GIFSICLE,
            curve_type=CurveType.LOSSY,
            is_predicted=False,
            created_at=datetime.now(timezone.utc),
            size_at_lossy_0=100.0,
            size_at_lossy_40=80.0,
        )
        points = curve.get_lossy_curve_points()
        assert points[0] == 100.0
        assert points[40] == 80.0
        assert points[20] is None

    def test_confidence_scores_range(self) -> None:
        """Test that confidence scores must be 0-1."""
        with pytest.raises(ValueError):
            CompressionCurveV1(
                gif_sha="e" * 64,
                engine=Engine.GIFSICLE,
                curve_type=CurveType.LOSSY,
                is_predicted=True,
                confidence_scores=[1.5],  # Too high
                created_at=datetime.now(timezone.utc),
            )


class TestEnums:
    """Test enum values."""

    def test_engine_values(self) -> None:
        """Test Engine enum values."""
        assert Engine.GIFSICLE.value == "gifsicle"
        assert Engine.ANIMATELY.value == "animately"

    def test_curve_type_values(self) -> None:
        """Test CurveType enum values."""
        assert CurveType.LOSSY.value == "lossy"
        assert CurveType.COLORS.value == "colors"

    def test_dataset_split_values(self) -> None:
        """Test DatasetSplit enum values."""
        assert DatasetSplit.TRAIN.value == "train"
        assert DatasetSplit.VAL.value == "val"
        assert DatasetSplit.TEST.value == "test"
