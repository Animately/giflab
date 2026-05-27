"""Deterministic GIF fixture generators for the test suite.

All functions produce the same bytes on every run (fixed seed / fixed content).
Importable by other generator scripts (e.g. scripts/fixtures/generate.py) so
that the sibling fixture-generation infrastructure can call these directly.

Usage
-----
From tests (auto-called via conftest.py)::

    from tests.fixtures._generators import ensure_test_4_frames_gif
    path = ensure_test_4_frames_gif(fixtures_dir)

From a script::

    python -m tests.fixtures._generators          # regenerates all managed fixtures
    python scripts/fixtures/generate.py           # equivalent entry-point
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

# ---------------------------------------------------------------------------
# test_4_frames.gif
# ---------------------------------------------------------------------------

#: Expected properties of the generated fixture (used in smoke tests).
TEST_4_FRAMES_SPEC = {
    "n_frames": 4,
    "size": (80, 80),
    "duration_ms": 150,
    "loop": 0,
}


def generate_test_4_frames_gif(output_path: Path) -> Path:
    """Create *tests/fixtures/test_4_frames.gif* deterministically.

    Produces a 4-frame, 80×80 px GIF with one solid-colour frame per
    colour (red → green → blue → yellow), 150 ms per frame, infinite loop.
    Content is purely arithmetic — no random seed required; the output is
    byte-stable across platforms and Pillow versions (palette quantisation
    is deterministic for flat solid-colour images).

    Parameters
    ----------
    output_path:
        Where to write the GIF.  Parent directory is created if needed.

    Returns
    -------
    Path
        The *output_path* argument, for convenience.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    colours = [
        (255, 0, 0),    # frame 1 — red
        (0, 255, 0),    # frame 2 — green
        (0, 0, 255),    # frame 3 — blue
        (255, 255, 0),  # frame 4 — yellow
    ]

    frames: list[Image.Image] = []
    for bg in colours:
        img = Image.new("RGB", TEST_4_FRAMES_SPEC["size"], bg)

        # Inner 40×40 white square — gives the compressors real edges to work
        # with while keeping colour count minimal.
        inner_x0, inner_y0 = 20, 20
        inner_x1, inner_y1 = 60, 60
        for y in range(inner_y0, inner_y1):
            for x in range(inner_x0, inner_x1):
                img.putpixel((x, y), (255, 255, 255))

        frames.append(img)

    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=TEST_4_FRAMES_SPEC["duration_ms"],
        loop=TEST_4_FRAMES_SPEC["loop"],
        optimize=False,  # keep palette predictable
    )
    return output_path


def ensure_test_4_frames_gif(fixtures_dir: Path) -> Path:
    """Return path to *test_4_frames.gif*, generating it when absent.

    Parameters
    ----------
    fixtures_dir:
        The ``tests/fixtures/`` directory.

    Returns
    -------
    Path
        Absolute path to the GIF file.
    """
    path = fixtures_dir / "test_4_frames.gif"
    if not path.exists():
        generate_test_4_frames_gif(path)
    return path


# ---------------------------------------------------------------------------
# Registry — add new managed fixtures here so scripts/fixtures/generate.py
# can iterate over them.
# ---------------------------------------------------------------------------

#: Mapping of fixture filename → (generator_fn, kwargs).
#: ``generator_fn(fixtures_dir / filename, **kwargs)``
MANAGED_FIXTURES: dict[str, tuple] = {
    "test_4_frames.gif": (generate_test_4_frames_gif, {}),
}


def regenerate_all(fixtures_dir: Path | None = None) -> None:
    """(Re)generate every fixture in :data:`MANAGED_FIXTURES`.

    Parameters
    ----------
    fixtures_dir:
        Defaults to the ``tests/fixtures/`` directory relative to this file.
    """
    if fixtures_dir is None:
        fixtures_dir = Path(__file__).parent

    for name, (fn, kwargs) in MANAGED_FIXTURES.items():
        out = fixtures_dir / name
        fn(out, **kwargs)
        print(f"  generated {out}")


if __name__ == "__main__":
    regenerate_all()
    print("Done.")
