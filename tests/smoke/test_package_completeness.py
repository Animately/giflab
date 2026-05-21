"""Guard against source files being silently dropped from the package.

The v0.3.0 release shipped without `src/giflab/deep_perceptual_metrics.py`
because an over-broad `.gitignore` rule (`deep_*`, unanchored) swallowed it.
This test imports every .py file present under `src/giflab/` as a
`giflab.*` module so a future regression — file on disk but not picked up
by the package, or file referenced but missing — fails CI immediately.
"""
from __future__ import annotations

import importlib
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src" / "giflab"


def _module_names() -> list[str]:
    names: list[str] = []
    for p in sorted(SRC.rglob("*.py")):
        if "__pycache__" in p.parts or p.name == "__init__.py":
            continue
        rel = p.relative_to(SRC).with_suffix("")
        names.append("giflab." + ".".join(rel.parts))
    return names


@pytest.mark.parametrize("modname", _module_names())
def test_module_importable(modname: str) -> None:
    importlib.import_module(modname)
