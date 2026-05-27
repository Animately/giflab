#!/usr/bin/env python3
"""Reproduce the corpus-wide frame-drop sweep used in the 2026-05-27 audit.

Run with Poetry:
    poetry run python docs/metrics-audit/animately-frame-drops-2026-05-27/sweep_frame_drops.py

Outputs `sweep.csv` alongside this script. The CSV is what the README's
percentages and histogram numbers were computed from.

Methodology:
- For every GIF under ~/Documents/GIFs/, read frame count and total duration
  via PIL (img.info["duration"] — same units as the source file, no fallback).
- Skip single-frame GIFs (nothing to deduplicate) and files larger than the
  configurable --max-kb threshold (default 5000 KB = decimal MB).
- Run animately at the configured --lossy level with a per-file --timeout.
- Record original frames, compressed frames, percent dropped, and whether
  PIL-measured total duration is preserved.

The PIL duration here is the source-file duration (literal info["duration"]
sum). It can be 0 if the GIF was authored with no inter-frame delays — that
is treated as 0=0 (preserved). This script does NOT apply the 100ms-fallback
used by `src/giflab/wrapper_validation/timing_validation.py` because that
fallback produces synthetic duration_diff values for 0-delay GIFs (see the
README for the worked example).
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

from PIL import Image  # type: ignore[import-untyped]

DEFAULT_ANIMATELY = (
    "/Users/lachlants/repos/animately/animately-engine-releases/archive/"
    "animately_1.1.20.0"
)
DEFAULT_CORPUS = Path.home() / "Documents" / "GIFs"


def count_frames_and_duration(path: Path) -> tuple[int, int]:
    """Return (frame_count, total_duration_ms) using literal info["duration"].

    Zero-delay GIFs (commonly produced by some authoring tools) return 0;
    we do NOT substitute a 100ms fallback. The point of this script is to
    measure whether PIL-observable duration is preserved by animately.
    """
    img = Image.open(path)
    n = 0
    total_dur = 0
    try:
        while True:
            total_dur += img.info.get("duration", 0)
            n += 1
            img.seek(img.tell() + 1)
    except EOFError:
        pass
    return n, total_dur


def get_gif_info(path: Path) -> dict[str, object] | None:
    try:
        img = Image.open(path)
        n_frames, total_dur = count_frames_and_duration(path)
        img.seek(0)
        return {
            "n_frames": n_frames,
            "width": img.size[0],
            "height": img.size[1],
            "has_transparency": img.info.get("transparency", None) is not None,
            "disposal": img.info.get("disposal", 0),
            "total_duration_ms": total_dur,
            "file_size_kb": os.path.getsize(path) // 1024,
        }
    except Exception:
        return None


def run_sweep(
    corpus: Path,
    animately: Path,
    lossy: int,
    max_kb: int,
    timeout: int,
    output_csv: Path,
) -> None:
    all_gifs = sorted(corpus.rglob("*.gif"))
    print(f"Found {len(all_gifs)} GIFs under {corpus}", file=sys.stderr)

    out_dir = output_csv.parent / f"animately_sweep_tmp_lossy{lossy}"
    out_dir.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "gif_path",
                "orig_frames",
                "comp_frames",
                "dropped",
                "drop_pct",
                "width",
                "height",
                "has_transparency",
                "disposal",
                "orig_duration_ms",
                "comp_duration_ms",
                "duration_preserved",
                "file_size_kb",
                "status",
            ]
        )

        for i, gif in enumerate(all_gifs):
            info = get_gif_info(gif)
            if info is None:
                continue

            # Skip single-frame GIFs (nothing to dedup)
            if info["n_frames"] <= 1:
                continue

            if info["file_size_kb"] > max_kb:
                writer.writerow(
                    [
                        str(gif),
                        info["n_frames"],
                        -1, -1, -1,
                        info["width"], info["height"],
                        info["has_transparency"], info["disposal"],
                        info["total_duration_ms"], -1, False,
                        info["file_size_kb"], "skipped_too_large",
                    ]
                )
                continue

            out_path = out_dir / f"sweep_{i}.gif"
            try:
                subprocess.run(
                    [
                        str(animately),
                        "--input", str(gif),
                        "--output", str(out_path),
                        "--lossy", str(lossy),
                    ],
                    capture_output=True,
                    timeout=timeout,
                )

                if out_path.exists():
                    comp_frames, comp_dur = count_frames_and_duration(out_path)
                    out_path.unlink()
                    status = "ok"
                else:
                    comp_frames, comp_dur = -1, -1
                    status = "engine_silent_failure"
            except subprocess.TimeoutExpired:
                comp_frames, comp_dur = -1, -1
                status = "timeout"
                if out_path.exists():
                    out_path.unlink()
            except Exception:
                comp_frames, comp_dur = -1, -1
                status = "error"

            dropped = info["n_frames"] - comp_frames if comp_frames >= 0 else -1
            drop_pct = (
                round(dropped / info["n_frames"] * 100, 1)
                if dropped >= 0 else -1.0
            )
            duration_preserved = (
                bool(info["total_duration_ms"] == comp_dur)
                if comp_dur >= 0 else False
            )

            writer.writerow(
                [
                    str(gif),
                    info["n_frames"], comp_frames, dropped, drop_pct,
                    info["width"], info["height"],
                    info["has_transparency"], info["disposal"],
                    info["total_duration_ms"], comp_dur, duration_preserved,
                    info["file_size_kb"], status,
                ]
            )
            fh.flush()

    out_dir.rmdir()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus", type=Path, default=DEFAULT_CORPUS,
        help="GIF corpus root (default: ~/Documents/GIFs)",
    )
    parser.add_argument(
        "--animately", type=Path, default=Path(DEFAULT_ANIMATELY),
        help="Path to animately binary",
    )
    parser.add_argument(
        "--lossy", type=int, default=60,
        help="Animately --lossy level (default: 60)",
    )
    parser.add_argument(
        "--max-kb", type=int, default=5000,
        help="Skip files larger than this many KB (default: 5000 = 5 MB decimal)",
    )
    parser.add_argument(
        "--timeout", type=int, default=30,
        help="Per-file animately timeout in seconds (default: 30)",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).parent / "sweep.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    run_sweep(
        corpus=args.corpus,
        animately=args.animately,
        lossy=args.lossy,
        max_kb=args.max_kb,
        timeout=args.timeout,
        output_csv=args.output,
    )
    print(f"Wrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
