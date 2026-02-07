"""Functional test configuration.

Functional tests use mocked engines and small synthetic GIFs.
No real compression binaries, no ML model loading.
Target: <30s total, <1s each.
"""

import pytest
from PIL import Image


@pytest.fixture
def tiny_gif(tmp_path):
    """Create a minimal 10x10 single-frame GIF for testing."""
    path = tmp_path / "tiny.gif"
    img = Image.new("RGB", (10, 10), (128, 64, 200))
    img.save(path, format="GIF")
    return path


@pytest.fixture
def tiny_gif_multiframe(tmp_path):
    """Create a minimal 10x10 4-frame GIF for testing."""
    path = tmp_path / "tiny_multi.gif"
    frames = []
    for i in range(4):
        val = int(i * 255 / 3)
        frames.append(Image.new("RGB", (10, 10), (val, 0, 255 - val)))
    frames[0].save(
        path, save_all=True, append_images=frames[1:], duration=100, loop=0
    )
    return path
