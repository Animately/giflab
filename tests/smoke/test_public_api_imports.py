"""Smoke tests for the public API imports and lazy-import contract.

The public API must be importable without triggering torch / lpips /
heavyweight model loads. Consumers who only want compress() should not
pay LPIPS cost just by importing.

US2 (measure) expands this file with measure-related names.
"""

from __future__ import annotations

import subprocess
import sys


def test_us1_public_names_importable_from_top_level() -> None:
    """US1 public names import from the top-level giflab package."""
    from giflab import (  # noqa: F401
        SUPPORTED_ENGINES,
        CompressResult,
        EngineIdentifier,
        EngineUnavailableError,
        UnknownEngineError,
        compress,
    )


def test_supported_engines_tuple_shape() -> None:
    from giflab import SUPPORTED_ENGINES

    assert SUPPORTED_ENGINES == (
        "animately",
        "gifsicle",
        "gifski",
        "imagemagick",
        "ffmpeg",
    )


def test_lazy_import_does_not_pull_in_heavy_deps() -> None:
    """Importing the public API must not pull torch/lpips into sys.modules.

    Runs in a subprocess so the test gets a clean import space.
    """
    code = (
        "import sys; "
        "from giflab import compress, CompressResult; "
        "forbidden = [m for m in ('torch', 'lpips', 'open_clip') if m in sys.modules]; "
        "print('FORBIDDEN_MODULES=' + ','.join(forbidden))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    forbidden = completed.stdout.strip().split("=", 1)[1]
    assert (
        forbidden == ""
    ), f"Heavy modules leaked into sys.modules on import: {forbidden!r}"
