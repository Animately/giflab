"""Integration test configuration.

Integration tests use real engines, real GIFs, and real metrics.
Process-isolated via pytest-xdist --dist loadfile.
Target: <5min with xdist.
"""

import shutil

import pytest


def _engine_available(name: str) -> bool:
    """Check if an external engine binary is available."""
    return shutil.which(name) is not None


@pytest.fixture
def requires_gifsicle():
    """Skip test if gifsicle is not installed."""
    if not _engine_available("gifsicle"):
        pytest.skip("gifsicle not installed")


@pytest.fixture
def requires_animately():
    """Skip test if animately is not installed."""
    if not _engine_available("animately"):
        pytest.skip("animately not installed")


@pytest.fixture
def requires_ffmpeg():
    """Skip test if ffmpeg is not installed."""
    if not _engine_available("ffmpeg"):
        pytest.skip("ffmpeg not installed")


@pytest.fixture
def requires_gifski():
    """Skip test if gifski is not installed."""
    if not _engine_available("gifski"):
        pytest.skip("gifski not installed")


@pytest.fixture
def requires_imagemagick():
    """Skip test if convert (ImageMagick) is not installed."""
    if not _engine_available("convert"):
        pytest.skip("ImageMagick (convert) not installed")
