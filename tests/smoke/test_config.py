"""Tests for giflab.config module."""

import os
from pathlib import Path
from unittest.mock import patch

from giflab.config import (
    DEFAULT_COMPRESSION_CONFIG,
    DEFAULT_ENGINE_CONFIG,
    DEFAULT_PATH_CONFIG,
    CompressionConfig,
    EngineConfig,
    PathConfig,
)


class TestCompressionConfig:
    """Tests for CompressionConfig class."""

    def test_default_initialization(self):
        """Test that default values are set correctly."""
        config = CompressionConfig()

        # Prediction-focused defaults: frame reduction handled separately
        assert config.FRAME_KEEP_RATIOS == [1.0]
        # 5 color counts for prediction granularity
        assert len(config.COLOR_KEEP_COUNTS) >= 5
        assert 256 in config.COLOR_KEEP_COUNTS
        # Lossy levels include 0 (no compression)
        assert 0 in config.LOSSY_LEVELS
        # At least one engine available
        assert len(config.ENGINES) >= 1

    def test_custom_initialization(self):
        """Test initialization with custom values."""
        config = CompressionConfig(
            FRAME_KEEP_RATIOS=[1.0, 0.5],
            COLOR_KEEP_COUNTS=[128, 64],
            LOSSY_LEVELS=[0, 80],
            ENGINES=["gifsicle"],
        )

        assert config.FRAME_KEEP_RATIOS == [1.0, 0.5]
        assert config.COLOR_KEEP_COUNTS == [128, 64]
        assert config.LOSSY_LEVELS == [0, 80]
        assert config.ENGINES == ["gifsicle"]


class TestPathConfig:
    """Tests for PathConfig class."""

    def test_default_paths(self):
        """Test default path configuration."""
        config = PathConfig()

        assert config.RAW_DIR == Path("data/raw")
        assert config.RENDERS_DIR == Path("data/renders")
        assert config.CSV_DIR == Path("data/csv")
        assert config.BAD_GIFS_DIR == Path("data/bad_gifs")
        assert config.TMP_DIR == Path("data/tmp")
        assert config.SEED_DIR == Path("seed")
        assert config.LOGS_DIR == Path("logs")

    def test_custom_paths(self):
        """Test custom path configuration."""
        config = PathConfig(
            RAW_DIR=Path("/custom/raw"), RENDERS_DIR=Path("/custom/renders")
        )

        assert config.RAW_DIR == Path("/custom/raw")
        assert config.RENDERS_DIR == Path("/custom/renders")
        # Other paths should remain default
        assert config.CSV_DIR == Path("data/csv")


class TestEngineConfig:
    """Tests for EngineConfig class."""

    def test_default_initialization(self):
        """Test that default engine paths are set correctly."""
        config = EngineConfig()

        assert config.GIFSICLE_PATH == "gifsicle"
        # Animately path is set to local development path by default
        assert "launcher" in config.ANIMATELY_PATH or config.ANIMATELY_PATH == "animately"
        assert config.IMAGEMAGICK_PATH == "magick"
        assert config.FFMPEG_PATH == "ffmpeg"
        assert config.FFPROBE_PATH == "ffprobe"
        assert config.GIFSKI_PATH == "gifski"

    def test_custom_initialization(self):
        """Test initialization with custom engine paths."""
        config = EngineConfig(
            GIFSICLE_PATH="/custom/gifsicle",
            ANIMATELY_PATH="/custom/animately",
            IMAGEMAGICK_PATH="/custom/magick",
            FFMPEG_PATH="/custom/ffmpeg",
            FFPROBE_PATH="/custom/ffprobe",
            GIFSKI_PATH="/custom/gifski",
        )

        assert config.GIFSICLE_PATH == "/custom/gifsicle"
        assert config.ANIMATELY_PATH == "/custom/animately"
        assert config.IMAGEMAGICK_PATH == "/custom/magick"
        assert config.FFMPEG_PATH == "/custom/ffmpeg"
        assert config.FFPROBE_PATH == "/custom/ffprobe"
        assert config.GIFSKI_PATH == "/custom/gifski"

    def test_environment_variable_overrides(self):
        """Test that environment variables override default paths."""
        env_vars = {
            "GIFLAB_GIFSICLE_PATH": "/env/gifsicle",
            "GIFLAB_ANIMATELY_PATH": "/env/animately",
            "GIFLAB_IMAGEMAGICK_PATH": "/env/magick",
            "GIFLAB_FFMPEG_PATH": "/env/ffmpeg",
            "GIFLAB_FFPROBE_PATH": "/env/ffprobe",
            "GIFLAB_GIFSKI_PATH": "/env/gifski",
        }

        with patch.dict(os.environ, env_vars):
            config = EngineConfig()

            assert config.GIFSICLE_PATH == "/env/gifsicle"
            assert config.ANIMATELY_PATH == "/env/animately"
            assert config.IMAGEMAGICK_PATH == "/env/magick"
            assert config.FFMPEG_PATH == "/env/ffmpeg"
            assert config.FFPROBE_PATH == "/env/ffprobe"
            assert config.GIFSKI_PATH == "/env/gifski"

    def test_partial_environment_overrides(self):
        """Test that only set environment variables are overridden."""
        env_vars = {
            "GIFLAB_GIFSICLE_PATH": "/env/gifsicle",
            "GIFLAB_FFMPEG_PATH": "/env/ffmpeg",
        }

        with patch.dict(os.environ, env_vars):
            config = EngineConfig()

            # Overridden paths
            assert config.GIFSICLE_PATH == "/env/gifsicle"
            assert config.FFMPEG_PATH == "/env/ffmpeg"

            # Default paths (not overridden) - Animately uses dev path
            assert "launcher" in config.ANIMATELY_PATH or config.ANIMATELY_PATH == "animately"
            assert config.IMAGEMAGICK_PATH == "magick"
            assert config.FFPROBE_PATH == "ffprobe"
            assert config.GIFSKI_PATH == "gifski"

    def test_environment_overrides_with_custom_initialization(self):
        """Test that environment variables override even custom initialization values."""
        env_vars = {
            "GIFLAB_GIFSICLE_PATH": "/env/gifsicle",
        }

        with patch.dict(os.environ, env_vars):
            config = EngineConfig(GIFSICLE_PATH="/custom/gifsicle")

            # Environment variable should win over custom initialization
            assert config.GIFSICLE_PATH == "/env/gifsicle"


def test_default_configurations():
    """Test that default configuration instances work."""
    assert isinstance(DEFAULT_COMPRESSION_CONFIG, CompressionConfig)
    assert isinstance(DEFAULT_PATH_CONFIG, PathConfig)
    assert isinstance(DEFAULT_ENGINE_CONFIG, EngineConfig)

    # Verify defaults are properly set
    assert DEFAULT_COMPRESSION_CONFIG.FRAME_KEEP_RATIOS is not None
    assert DEFAULT_COMPRESSION_CONFIG.COLOR_KEEP_COUNTS is not None
    assert DEFAULT_COMPRESSION_CONFIG.LOSSY_LEVELS is not None
    assert DEFAULT_COMPRESSION_CONFIG.ENGINES is not None

    # Prediction-focused defaults
    assert len(DEFAULT_COMPRESSION_CONFIG.FRAME_KEEP_RATIOS) >= 1
    assert len(DEFAULT_COMPRESSION_CONFIG.COLOR_KEEP_COUNTS) >= 5
    assert len(DEFAULT_COMPRESSION_CONFIG.LOSSY_LEVELS) >= 1
    assert len(DEFAULT_COMPRESSION_CONFIG.ENGINES) >= 1

    # Verify engine config defaults
    assert DEFAULT_ENGINE_CONFIG.GIFSICLE_PATH == "gifsicle"
    # Animately uses local dev path or default
    assert "launcher" in DEFAULT_ENGINE_CONFIG.ANIMATELY_PATH or \
           DEFAULT_ENGINE_CONFIG.ANIMATELY_PATH == "animately"
    assert DEFAULT_ENGINE_CONFIG.IMAGEMAGICK_PATH == "magick"
    assert DEFAULT_ENGINE_CONFIG.FFMPEG_PATH == "ffmpeg"
    assert DEFAULT_ENGINE_CONFIG.FFPROBE_PATH == "ffprobe"
    assert DEFAULT_ENGINE_CONFIG.GIFSKI_PATH == "gifski"
