"""Root test configuration.

Provides shared fixtures used across multiple test layers.
Layer-specific fixtures live in each layer's own conftest.py.
"""

import shutil
from pathlib import Path

import pytest
from PIL import Image


def pytest_collection_modifyitems(items):
    """Enforce serial execution for tests marked with @pytest.mark.serial.

    Maps the custom ``serial`` marker to ``xdist_group("serial")`` so that
    pytest-xdist schedules all serial-marked tests on the same worker,
    preventing CPU-contention issues for tight timing assertions.
    """
    for item in items:
        if item.get_closest_marker("serial"):
            item.add_marker(pytest.mark.xdist_group("serial"))


def _create_dummy_gif(path: Path, frames: int = 1, colors: int = 2) -> None:
    """Create a small dummy GIF at *path* with *frames* and *colors*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    size = (10, 10)
    imgs = []
    for i in range(frames):
        val = int((i % colors) * 255 / max(colors - 1, 1))
        imgs.append(Image.new("RGB", size, (val, 0, 255 - val)))
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=100, loop=0)


def _ensure_required_gif_fixtures() -> None:
    """Regenerate core GIF fixtures if they were cleaned from git."""
    fixtures = {
        "simple_4frame.gif": (4, 4),
        "single_frame.gif": (1, 2),
        "many_colors.gif": (2, 16),
    }
    base = Path(__file__).parent / "fixtures"
    for name, (frames, colors) in fixtures.items():
        gif_path = base / name
        if not gif_path.exists():
            _create_dummy_gif(gif_path, frames=frames, colors=colors)


# Run at import time so all downstream fixtures see the files
_ensure_required_gif_fixtures()


@pytest.fixture
def fast_compress(monkeypatch):
    """Stub out gifsicle / Animately invocations for fast tests.

    Replaces compression functions with no-op copies that return
    minimal metadata dictionaries mimicking the real wrappers.
    """

    def _noop_copy(input_path, output_path, *args, **kwargs):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(input_path, output_path)

        try:
            size_kb = output_path.stat().st_size / 1024.0
        except FileNotFoundError:
            size_kb = 0.0

        return {
            "render_ms": 1,
            "engine": "noop",
            "command": "noop-copy",
            "kilobytes": size_kb,
            "ssim": 1.0,
            "lossy_level": kwargs.get("lossy_level", 0),
            "frame_keep_ratio": kwargs.get("frame_keep_ratio", 1.0),
            "color_keep_count": kwargs.get("color_keep_count", None),
        }

    monkeypatch.setattr("giflab.lossy.compress_with_gifsicle", _noop_copy, raising=True)
    monkeypatch.setattr(
        "giflab.lossy.compress_with_animately", _noop_copy, raising=True
    )
    monkeypatch.setattr(
        "giflab.tool_wrappers.compress_with_gifsicle", _noop_copy, raising=False
    )
    monkeypatch.setattr(
        "giflab.tool_wrappers.compress_with_animately", _noop_copy, raising=False
    )

    yield
