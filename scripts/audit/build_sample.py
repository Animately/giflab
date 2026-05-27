"""Stratified sample of a GIF corpus for the metrics audit.

Walks --source recursively, measures each GIF's (file size, frame count,
dimensions) cheaply via PIL (no full frame decode), then samples N items
distributed across buckets so the sample spans the corpus variety.

Usage:
    poetry run python scripts/audit/build_sample.py \\
        --source ~/Documents/GIFs --n 100 --seed 42 \\
        --output sample_manifest.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

# Bucket edges. Anything above the last edge falls in the top bucket.
# Picked to span the typical Animately corpus (few KB → tens of MB,
# 1–500 frames, small icons → 1080p+).
SIZE_KB_EDGES = [50, 250, 1000, 5000]          # 5 buckets
FRAME_EDGES = [1, 5, 20, 80]                    # 5 buckets
DIM_EDGES = [100, 320, 720]                     # 4 buckets (by max(w,h))


def _bucket(value: float, edges: list[float]) -> int:
    """Return bucket index 0..len(edges)."""
    for i, edge in enumerate(edges):
        if value <= edge:
            return i
    return len(edges)


def probe_gif(path: Path) -> dict[str, Any] | None:
    """Return cheap metadata for a GIF, or None if it can't be parsed."""
    try:
        size_bytes = path.stat().st_size
        with Image.open(path) as im:
            if im.format != "GIF":
                return None
            width, height = im.size
            frame_count = getattr(im, "n_frames", 1)
        return {
            "path": str(path),
            "kb": round(size_bytes / 1024.0, 2),
            "frames": int(frame_count),
            "width": int(width),
            "height": int(height),
        }
    except (OSError, UnidentifiedImageError, Image.DecompressionBombError):
        return None


def walk_corpus(source: Path) -> list[dict[str, Any]]:
    print(f"[sample] Walking {source}...", flush=True)
    entries: list[dict[str, Any]] = []
    skipped = 0
    for path in source.rglob("*.gif"):
        if not path.is_file():
            continue
        meta = probe_gif(path)
        if meta is None:
            skipped += 1
            continue
        entries.append(meta)
    print(f"[sample] Found {len(entries)} GIFs ({skipped} skipped/unreadable)", flush=True)
    return entries


def stratify(entries: list[dict[str, Any]]) -> dict[tuple[int, int, int], list[dict[str, Any]]]:
    buckets: dict[tuple[int, int, int], list[dict[str, Any]]] = defaultdict(list)
    for e in entries:
        key = (
            _bucket(e["kb"], SIZE_KB_EDGES),
            _bucket(e["frames"], FRAME_EDGES),
            _bucket(max(e["width"], e["height"]), DIM_EDGES),
        )
        buckets[key].append(e)
    return buckets


def sample_balanced(
    buckets: dict[tuple[int, int, int], list[dict[str, Any]]],
    n: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Round-robin across non-empty buckets, sampling without replacement."""
    bucket_keys = list(buckets.keys())
    rng.shuffle(bucket_keys)
    pools = {k: list(buckets[k]) for k in bucket_keys}
    for k in pools:
        rng.shuffle(pools[k])

    selected: list[dict[str, Any]] = []
    while len(selected) < n:
        progress = False
        for k in bucket_keys:
            if not pools[k]:
                continue
            selected.append(pools[k].pop())
            progress = True
            if len(selected) >= n:
                break
        if not progress:
            break
    return selected


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True, type=Path)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()

    source = args.source.expanduser().resolve()
    if not source.exists():
        print(f"source dir does not exist: {source}", file=sys.stderr)
        return 2

    entries = walk_corpus(source)
    if not entries:
        print(f"No readable GIFs under {source}", file=sys.stderr)
        return 2

    buckets = stratify(entries)
    print(f"[sample] {len(buckets)} non-empty buckets (size × frames × dims)", flush=True)

    rng = random.Random(args.seed)
    sample = sample_balanced(buckets, args.n, rng)
    print(f"[sample] Selected {len(sample)} / {args.n} requested", flush=True)

    # Tag each entry with source and content_type (real GIFs are uncategorised).
    # `kb_orig` matches the downstream sweep CSV schema.
    manifest = [
        {
            "path": e["path"],
            "source": "real",
            "content_type": None,
            "kb_orig": e["kb"],
            "frames": e["frames"],
            "width": e["width"],
            "height": e["height"],
        }
        for e in sample
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"entries": manifest}, indent=2))
    print(f"[sample] Wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
