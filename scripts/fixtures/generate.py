#!/usr/bin/env python3
"""Deterministic test-fixture generator for GifLab.

Produces every gitignored GIF in tests/fixtures/ from fixed seeds so that
a fresh git worktree (or a new contributor machine) can regenerate all
fixtures with one command:

    make fixtures
    # or directly:
    poetry run python scripts/fixtures/generate.py

Each sub-generator writes to the *output_dir* passed to ``generate_all()``.
The default output directory is ``tests/fixtures/`` relative to the project
root (the directory two levels above this file).

All PIL Image.save() calls use fixed colour palettes and fixed random seeds
so output is byte-identical across runs on the same Python / Pillow version.

Design notes
------------
- No external tools required — only Pillow + NumPy (both already in the
  dev dependencies).
- ``generate_all(output_dir)`` is the single public entry point; tests import
  and call it directly to redirect output to a tmp dir.
- Each individual generator function is public (``create_*``) so the sibling
  task's fixture (test_4_frames.gif, etc.) can be imported individually if
  needed.
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


# ---------------------------------------------------------------------------
# Project root + default output dir
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
DEFAULT_OUTPUT_DIR = _PROJECT_ROOT / "tests" / "fixtures"


# ---------------------------------------------------------------------------
# Engine-testing fixtures  (originally: tests/fixtures/generate_fixtures.py)
# ---------------------------------------------------------------------------


def create_simple_4frame_gif(output_dir: Path) -> Path:
    """4-frame, 16-colour, 64×64px GIF for basic engine functionality tests."""
    frames = []
    colors = [
        (255, 0, 0),    # red
        (0, 255, 0),    # green
        (0, 0, 255),    # blue
        (255, 255, 0),  # yellow
    ]

    for i, color in enumerate(colors):
        img = Image.new("RGB", (64, 64), color)
        square_x = 10 + i * 10
        square_y = 10 + i * 5
        for x in range(square_x, min(square_x + 16, 64)):
            for y in range(square_y, min(square_y + 16, 64)):
                img.putpixel((x, y), (255, 255, 255))
        frames.append(img)

    output_path = output_dir / "simple_4frame.gif"
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=250,
        loop=0,
        optimize=False,
    )
    return output_path


def create_single_frame_gif(output_dir: Path) -> Path:
    """1-frame, 8-colour, 32×32px GIF for edge-case testing."""
    img = Image.new("RGB", (32, 32))
    colors = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
        (128, 128, 128),
        (255, 255, 255),
    ]
    for x in range(32):
        for y in range(32):
            color_idx = ((x // 4) + (y // 4)) % len(colors)
            img.putpixel((x, y), colors[color_idx])

    output_path = output_dir / "single_frame.gif"
    img.save(output_path, optimize=False)
    return output_path


def create_many_colors_gif(output_dir: Path) -> Path:
    """4-frame, 256-colour, 64×64px GIF for palette stress testing."""
    frames = []
    for frame_idx in range(4):
        img = Image.new("RGB", (64, 64))
        for x in range(64):
            for y in range(64):
                r = (x * 4 + frame_idx * 16) % 256
                g = (y * 4 + frame_idx * 32) % 256
                b = ((x + y) * 2 + frame_idx * 8) % 256
                img.putpixel((x, y), (r, g, b))
        frames.append(img)

    output_path = output_dir / "many_colors.gif"
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=200,
        loop=0,
        optimize=False,
    )
    return output_path


# ---------------------------------------------------------------------------
# Validation / wrapper-testing fixtures  (originally: create_test_fixtures.py)
# ---------------------------------------------------------------------------


def create_test_10_frames_gif(output_dir: Path) -> Path:
    """10-frame, 100×100px GIF at 10 FPS for frame-reduction tests."""
    frames = []
    for i in range(10):
        img = Image.new("RGB", (100, 100), "white")
        draw = ImageDraw.Draw(img)
        x = i * 10
        color = (255, 0, 0) if i < 5 else (0, 255, 0)
        draw.rectangle([x, 40, x + 20, 60], fill=color)
        draw.text((10, 10), f"Frame {i + 1}", fill="black")
        frames.append(img)

    output_path = output_dir / "test_10_frames.gif"
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=100,
        loop=0,
    )
    return output_path


def create_test_4_frames_gif(output_dir: Path) -> Path:
    """4-frame, 80×80px GIF for quick wrapper tests.

    NOTE: The sibling task (giflab-missing-test-fixture-test-4-frames-gif)
    also creates this file.  This generator keeps it in sync so `make
    fixtures` covers the full set from a single command.
    """
    frames = []
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    for i in range(4):
        img = Image.new("RGB", (80, 80), "white")
        draw = ImageDraw.Draw(img)
        draw.ellipse([20, 20, 60, 60], fill=colors[i])
        draw.text((5, 5), f"F{i + 1}", fill="black")
        frames.append(img)

    output_path = output_dir / "test_4_frames.gif"
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=150,
        loop=0,
    )
    return output_path


def create_test_256_colors_gif(output_dir: Path) -> Path:
    """5-frame, 128×128px, many-colour GIF for colour-reduction tests."""
    frames = []
    for frame_idx in range(5):
        img = Image.new("RGB", (128, 128))
        pixels = []
        for y in range(128):
            for x in range(128):
                r = int((x + frame_idx * 25) % 256)
                g = int((y + frame_idx * 25) % 256)
                b = int((x + y + frame_idx * 25) % 256)
                pixels.append((r, g, b))
        img.putdata(pixels)
        draw = ImageDraw.Draw(img)
        draw.text((5, 5), f"Colors F{frame_idx + 1}", fill="white")
        frames.append(img)

    output_path = output_dir / "test_256_colors.gif"
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=200,
        loop=0,
    )
    return output_path


def create_test_2_colors_gif(output_dir: Path) -> Path:
    """6-frame, 60×60px, 2-colour GIF for palette-minimum tests."""
    frames = []
    for i in range(6):
        img = Image.new("P", (60, 60))
        img.putpalette(
            [0, 0, 0, 255, 255, 255] + [0] * (256 - 2) * 3
        )
        draw = ImageDraw.Draw(img)
        if i % 2 == 0:
            draw.rectangle([10, 10, 50, 50], fill=1)
        else:
            draw.ellipse([15, 15, 45, 45], fill=1)
        frames.append(img.convert("RGB"))

    output_path = output_dir / "test_2_colors.gif"
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=250,
        loop=0,
    )
    return output_path


def create_test_30_frames_gif(output_dir: Path) -> Path:
    """30-frame, 64×64px GIF with progress-bar animation for frame-reduction tests."""
    frames = []
    for i in range(30):
        img = Image.new("RGB", (64, 64), "lightgray")
        draw = ImageDraw.Draw(img)
        progress = i / 29.0
        bar_width = int(50 * progress)
        draw.rectangle([7, 25, 57, 35], outline="black", fill="white")
        if bar_width > 0:
            draw.rectangle([8, 26, 8 + bar_width - 1, 34], fill="green")
        draw.text((2, 2), f"{i + 1}/30", fill="black")
        frames.append(img)

    output_path = output_dir / "test_30_frames.gif"
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=80,
        loop=0,
    )
    return output_path


# ---------------------------------------------------------------------------
# Temporal-artifact fixtures  (originally: generate_temporal_artifact_fixtures.py)
# ---------------------------------------------------------------------------


def create_flicker_high_gif(output_dir: Path) -> Path:
    """10-frame GIF with high-severity flicker for temporal-artifact detection tests."""
    frames = []
    base_color = (100, 100, 100)
    for i in range(10):
        img = Image.new("RGB", (64, 64), base_color)
        if i % 2 == 0:
            rng = random.Random(42 + i)
            for _ in range(5):
                x, y = rng.randint(0, 48), rng.randint(0, 48)
                for dx in range(16):
                    for dy in range(16):
                        if x + dx < 64 and y + dy < 64:
                            img.putpixel((x + dx, y + dy), (255, 255, 255))
        else:
            rng = random.Random(42 + i)
            for _ in range(5):
                x, y = rng.randint(0, 48), rng.randint(0, 48)
                for dx in range(16):
                    for dy in range(16):
                        if x + dx < 64 and y + dy < 64:
                            img.putpixel((x + dx, y + dy), (0, 0, 0))
        frames.append(img)

    output_path = output_dir / "flicker_high.gif"
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=100,
        loop=0,
        disposal=2,
    )
    return output_path


def create_flicker_low_gif(output_dir: Path) -> Path:
    """10-frame GIF with low-severity (smooth) flicker."""
    frames = []
    base_color = (100, 100, 100)
    for i in range(10):
        img = Image.new("RGB", (64, 64), base_color)
        offset = i * 5
        for x in range(64):
            for y in range(64):
                val = min(255, max(0, base_color[0] + offset % 20))
                img.putpixel((x, y), (val, val, val))
        frames.append(img)

    output_path = output_dir / "flicker_low.gif"
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=100,
        loop=0,
        disposal=2,
    )
    return output_path


def create_background_stable_gif(output_dir: Path) -> Path:
    """10-frame GIF with stable background edges (no flicker)."""
    frames = []
    for i in range(10):
        img = Image.new("RGB", (64, 64))
        edge_color = (50, 50, 50)
        for x in range(64):
            for y in range(10):
                img.putpixel((x, y), edge_color)
            for y in range(54, 64):
                img.putpixel((x, y), edge_color)
        for y in range(10, 54):
            for x in range(10):
                img.putpixel((x, y), edge_color)
            for x in range(54, 64):
                img.putpixel((x, y), edge_color)
        center_color = (200, 100 + i * 20 % 100, 100)
        for x in range(20, 44):
            for y in range(20, 44):
                img.putpixel((x, y), center_color)
        frames.append(img)

    output_path = output_dir / "background_stable.gif"
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=100,
        loop=0,
    )
    return output_path


def create_background_flickering_gif(output_dir: Path) -> Path:
    """10-frame GIF with flickering background edges."""
    frames = []
    for i in range(10):
        img = Image.new("RGB", (64, 64))
        edge_color = (50 + (i * 20) % 100, 50, 50)
        for x in range(64):
            for y in range(10):
                img.putpixel((x, y), edge_color)
            for y in range(54, 64):
                img.putpixel((x, y), edge_color)
        for y in range(10, 54):
            for x in range(10):
                img.putpixel((x, y), edge_color)
            for x in range(54, 64):
                img.putpixel((x, y), edge_color)
        center_color = (200, 100 + i * 20 % 100, 100)
        for x in range(20, 44):
            for y in range(20, 44):
                img.putpixel((x, y), center_color)
        frames.append(img)

    output_path = output_dir / "background_flickering.gif"
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=100,
        loop=0,
    )
    return output_path


def create_pumping_yes_gif(output_dir: Path) -> Path:
    """12-frame GIF with temporal quality oscillation (pumping effect)."""
    frames = []
    for i in range(12):
        img = Image.new("RGB", (64, 64))
        quality_factor = 1.0 if i % 3 == 0 else 0.3
        for x in range(64):
            for y in range(64):
                base = int(((x + y) / 128) * 255)
                if quality_factor < 1.0:
                    base = (base // 32) * 32
                img.putpixel((x, y), (base, base, base))
        if quality_factor == 1.0:
            for j in range(0, 64, 4):
                if j < 64:
                    img.putpixel((j, j), (255, 0, 0))
                    if j + 1 < 64:
                        img.putpixel((j + 1, j), (0, 255, 0))
        frames.append(img)

    output_path = output_dir / "pumping_yes.gif"
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=100,
        loop=0,
    )
    return output_path


def create_pumping_no_gif(output_dir: Path) -> Path:
    """12-frame GIF with consistent quality (no pumping)."""
    frames = []
    for i in range(12):
        img = Image.new("RGB", (64, 64))
        for x in range(64):
            for y in range(64):
                base = int(((x + y) / 128) * 255)
                img.putpixel((x, y), (base, base, base))
        for j in range(0, 64, 8):
            if j < 64:
                img.putpixel((j, j), (255, 0, 0))
        frames.append(img)

    output_path = output_dir / "pumping_no.gif"
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=100,
        loop=0,
    )
    return output_path


def create_disposal_corrupted_gif(output_dir: Path) -> Path:
    """8-frame GIF with disposal artifacts (previous frames bleeding through)."""
    frames = []
    accumulated_artifacts = None

    for i in range(8):
        img = Image.new("RGB", (64, 64), (100, 100, 100))
        obj_x = i * 8
        obj_y = i * 4
        for x in range(obj_x, min(obj_x + 10, 64)):
            for y in range(obj_y, min(obj_y + 10, 64)):
                img.putpixel((x, y), (255, 0, 0))

        if i > 0:
            if accumulated_artifacts is None:
                accumulated_artifacts = np.array(img)
            else:
                current = np.array(img)
                accumulated_artifacts = (
                    0.2 * accumulated_artifacts + 0.8 * current
                ).astype(np.uint8)
                img = Image.fromarray(accumulated_artifacts)
                if i % 2 == 0:
                    rng = random.Random(99 + i)
                    for _ in range(3):
                        x, y = rng.randint(0, 54), rng.randint(0, 54)
                        for dx in range(10):
                            for dy in range(10):
                                if x + dx < 64 and y + dy < 64:
                                    img.putpixel((x + dx, y + dy), (128, 64, 64))

        frames.append(img)

    output_path = output_dir / "disposal_corrupted.gif"
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=100,
        loop=0,
        disposal=1,
    )
    return output_path


def create_disposal_clean_gif(output_dir: Path) -> Path:
    """8-frame GIF with clean disposal (no bleeding)."""
    frames = []
    for i in range(8):
        img = Image.new("RGB", (64, 64), (100, 100, 100))
        obj_x = i * 8
        obj_y = i * 4
        for x in range(obj_x, min(obj_x + 10, 64)):
            for y in range(obj_y, min(obj_y + 10, 64)):
                img.putpixel((x, y), (255, 0, 0))
        frames.append(img)

    output_path = output_dir / "disposal_clean.gif"
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=100,
        loop=0,
        disposal=2,
    )
    return output_path


def create_smooth_animation_gif(output_dir: Path) -> Path:
    """16-frame GIF with smooth circular motion for baseline comparison."""
    frames = []
    for i in range(16):
        img = Image.new("RGB", (64, 64), (50, 50, 50))
        angle = (i / 16.0) * 2 * math.pi
        center_x, center_y = 32, 32
        radius = 15
        obj_x = int(center_x + radius * math.cos(angle))
        obj_y = int(center_y + radius * math.sin(angle))
        for dx in range(-5, 6):
            for dy in range(-5, 6):
                x, y = obj_x + dx, obj_y + dy
                if 0 <= x < 64 and 0 <= y < 64:
                    distance = math.sqrt(dx * dx + dy * dy)
                    if distance <= 5:
                        intensity = max(0, 1.0 - distance / 5.0)
                        color_val = int(200 * intensity)
                        img.putpixel((x, y), (color_val, color_val, 255))
        frames.append(img)

    output_path = output_dir / "smooth_animation.gif"
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=80,
        loop=0,
        disposal=2,
    )
    return output_path


def create_static_with_noise_gif(output_dir: Path) -> Path:
    """8-frame GIF with static base + small noise patch (stable background test)."""
    base_pattern = np.random.RandomState(42).randint(
        0, 255, (64, 64, 3), dtype=np.uint8
    )
    frames = []
    for i in range(8):
        frame_array = base_pattern.copy()
        noise_area = np.random.RandomState(42 + i).randint(
            -10, 11, (20, 20, 3), dtype=np.int16
        )
        frame_array[22:42, 22:42] = np.clip(
            frame_array[22:42, 22:42].astype(np.int16) + noise_area, 0, 255
        ).astype(np.uint8)
        frames.append(Image.fromarray(frame_array))

    output_path = output_dir / "static_with_noise.gif"
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=150,
        loop=0,
    )
    return output_path


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def generate_all(output_dir: Path | None = None) -> list[Path]:
    """Generate every gitignored GIF fixture into *output_dir*.

    Parameters
    ----------
    output_dir:
        Directory to write fixtures into.  Defaults to
        ``<project_root>/tests/fixtures/``.  The directory is created if it
        does not exist.

    Returns
    -------
    list[Path]
        Paths of all files written.
    """
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generators = [
        # Engine-testing set
        create_simple_4frame_gif,
        create_single_frame_gif,
        create_many_colors_gif,
        # Validation / wrapper set
        create_test_10_frames_gif,
        create_test_4_frames_gif,
        create_test_256_colors_gif,
        create_test_2_colors_gif,
        create_test_30_frames_gif,
        # Temporal-artifact set
        create_flicker_high_gif,
        create_flicker_low_gif,
        create_background_stable_gif,
        create_background_flickering_gif,
        create_pumping_yes_gif,
        create_pumping_no_gif,
        create_disposal_corrupted_gif,
        create_disposal_clean_gif,
        create_smooth_animation_gif,
        create_static_with_noise_gif,
    ]

    created: list[Path] = []
    for gen in generators:
        path = gen(output_dir)
        created.append(path)
        print(f"  created {path.name}")

    return created


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate all gitignored GIF test fixtures for GifLab."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write fixtures into (default: tests/fixtures/)",
    )
    args = parser.parse_args()

    print(f"Generating fixtures into {args.output_dir} ...")
    files = generate_all(args.output_dir)
    print(f"Done — {len(files)} fixtures written.")
