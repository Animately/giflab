"""Phase 2 of the metrics audit — calibrate the lossy level for the main sweep.

Compresses 10 representative GIFs (5 real stratified + 5 synthetic) across a
wide lossy grid, measures all metrics, and analyses where the metrics actually
move and where they most disagree. Picks 1-3 lossy levels for Phase 3.

Usage:
    poetry run python scripts/audit/pilot.py \\
        --source ~/Documents/GIFs \\
        --output-csv pilot.csv \\
        --output-decisions pilot_decisions.json

The CSV schema matches sweep.py so report.py can ingest both.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

# Local helpers
sys.path.insert(0, str(Path(__file__).parent))
from _common import (  # noqa: E402
    classify_frame_reduction,
    compress_animately,
    import_giflab,
    load_existing_keys,
    measure_pair,
    open_csv_for_append,
    read_gif_timing,
)
from build_sample import sample_balanced, stratify, walk_corpus  # noqa: E402

# Synthetic specs to include in the pilot (must exist in
# SyntheticGifGenerator.synthetic_specs).
SYNTHETIC_PILOT_NAMES = [
    "smooth_gradient",
    "photographic_noise",
    "animation_heavy",
    "few_colors",
    "mixed_content",
]

DEFAULT_LOSSY_GRID = [0, 20, 40, 60, 80, 100, 140, 200]


# ---------------------------------------------------------------------------
# CSV row schema
# ---------------------------------------------------------------------------


BASE_FIELDS = [
    "path",
    "source",
    "content_type",
    "frames",
    "width",
    "height",
    "lossy",
    "kb_orig",
    "kb_compressed",
    "compression_ratio",
    "success",
    "error",
    "runtime_s",
    # Frame-reduction classification (see _common.classify_frame_reduction):
    # distinguishes benign temporal dedup (duration preserved) from possible
    # frame-loss bugs (duration not preserved) so operators can filter sweep
    # rows instead of investigating every frame-count mismatch.
    "frame_reduction",
    "frame_reduction_class",
    "frames_deduplicated",
    "frame_dedup",
    "frame_loss",
]


def build_pilot_sample(
    source: Path,
    synth_dir: Path,
    n_real: int,
    rng: random.Random,
    gl: dict[str, Any],
    max_orig_kb: float | None = None,
) -> list[dict[str, Any]]:
    """Pick the pilot's 10 GIFs. Returns a manifest list.

    `max_orig_kb` filters out real GIFs above the given size before
    stratification — useful because animately has a 10s wall-clock timeout
    that very large real GIFs blow through, and a stalled real GIF can
    consume most of the pilot's runtime budget.
    """
    manifest: list[dict[str, Any]] = []

    # Real: stratified from --source
    entries = walk_corpus(source)
    if max_orig_kb is not None:
        n_before = len(entries)
        entries = [e for e in entries if e["kb"] <= max_orig_kb]
        print(f"[pilot] Filtered real corpus to kb<={max_orig_kb}: {len(entries)}/{n_before}", flush=True)
    buckets = stratify(entries)
    real_sample = sample_balanced(buckets, n_real, rng)
    for e in real_sample:
        manifest.append(
            {
                "path": e["path"],
                "source": "real",
                "content_type": None,
                "kb_orig": e["kb"],
                "frames": e["frames"],
                "width": e["width"],
                "height": e["height"],
            }
        )

    # Synthetic — written to a persistent dir so report.py can render thumbs
    synth_dir.mkdir(parents=True, exist_ok=True)
    gen = gl["SyntheticGifGenerator"](synth_dir)
    paths = gen.generate_gifs(use_targeted_set=False)
    name_to_spec = {s.name: s for s in gen.synthetic_specs}
    path_by_name = {p.stem: p for p in paths}
    for name in SYNTHETIC_PILOT_NAMES:
        if name not in path_by_name:
            print(f"[pilot] WARN: synthetic spec '{name}' not generated", flush=True)
            continue
        spec = name_to_spec[name]
        p = path_by_name[name]
        manifest.append(
            {
                "path": str(p),
                "source": "synthetic",
                "content_type": spec.content_type,
                "kb_orig": round(p.stat().st_size / 1024.0, 2),
                "frames": spec.frames,
                "width": spec.size[0],
                "height": spec.size[1],
            }
        )

    return manifest


def sweep_pilot(
    manifest: list[dict[str, Any]],
    lossy_grid: list[int],
    csv_path: Path,
    workdir: Path,
    gl: dict[str, Any],
) -> None:
    # Build header once we know all metric keys. We don't a priori — discover
    # via a single measure() and lock the order.
    first_orig = Path(manifest[0]["path"])
    probe_compressed = workdir / "_probe.gif"
    ok, err = compress_animately(first_orig, probe_compressed, lossy_grid[0], gl)
    if not ok:
        raise RuntimeError(
            f"Pilot probe failed on {first_orig} at lossy={lossy_grid[0]}: {err}"
        )
    probe_metrics, _ = measure_pair(first_orig, probe_compressed, gl)
    # Drop any metric key that collides with BASE_FIELDS (e.g.
    # 'compression_ratio' which we compute ourselves above).
    base_set = set(BASE_FIELDS)
    metric_fields = sorted(k for k in probe_metrics if k not in base_set)
    header = BASE_FIELDS + metric_fields

    existing_keys = load_existing_keys(csv_path, key_fields=("path", "lossy"))
    fh, writer, _ = open_csv_for_append(csv_path, header)

    total_pairs = len(manifest) * len(lossy_grid)
    done = 0
    try:
        for entry in manifest:
            orig = Path(entry["path"])
            for lossy in lossy_grid:
                done += 1
                key = (str(orig), str(lossy))
                if key in existing_keys:
                    continue
                compressed = (
                    workdir / f"{orig.stem}_lossy{lossy}{orig.suffix}"
                )
                ok, err = compress_animately(orig, compressed, lossy, gl)
                if not ok:
                    row: dict[str, Any] = {
                        **entry,
                        "path": str(orig),
                        "lossy": lossy,
                        "kb_compressed": None,
                        "compression_ratio": None,
                        "success": False,
                        "error": err,
                        "runtime_s": None,
                    }
                    writer.writerow(row)
                    fh.flush()
                    print(f"[pilot] [{done}/{total_pairs}] FAIL {orig.name} lossy={lossy}: {err}", flush=True)
                    continue
                metrics, runtime_s = measure_pair(orig, compressed, gl)
                kb_compressed = round(compressed.stat().st_size / 1024.0, 2)
                # Classify any frame-count reduction (dedup vs frame_loss)
                # while both GIFs still exist on disk — the compressed
                # artifact is unlinked a few lines below.
                orig_frames, orig_dur = read_gif_timing(orig)
                comp_frames, comp_dur = read_gif_timing(compressed)
                frame_class = classify_frame_reduction(
                    orig_frames=orig_frames,
                    orig_duration_ms=orig_dur,
                    comp_frames=comp_frames,
                    comp_duration_ms=comp_dur,
                )
                row = {
                    **entry,
                    "path": str(orig),
                    "lossy": lossy,
                    "kb_compressed": kb_compressed,
                    "compression_ratio": (
                        round(entry["kb_orig"] / kb_compressed, 3)
                        if kb_compressed > 0
                        else None
                    ),
                    "success": True,
                    "error": "",
                    "runtime_s": round(runtime_s, 2),
                    **frame_class,
                    **metrics,
                }
                writer.writerow(row)
                fh.flush()
                if done % 5 == 0 or done == total_pairs:
                    print(
                        f"[pilot] [{done}/{total_pairs}] {orig.name} lossy={lossy} ok ({runtime_s:.1f}s)",
                        flush=True,
                    )
                # Clean up the compressed artifact — we don't need to keep it
                try:
                    compressed.unlink()
                except OSError:
                    pass
    finally:
        fh.close()


# ---------------------------------------------------------------------------
# Pilot analysis: pick lossy levels for the main sweep
# ---------------------------------------------------------------------------


def _load_pilot_csv(csv_path: Path) -> tuple[list[str], list[dict[str, Any]]]:
    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        rows = [r for r in reader if r.get("success") == "True"]
    if not rows:
        raise RuntimeError("Pilot CSV has no successful rows")
    metric_fields = [
        f
        for f in reader.fieldnames or []
        if f not in BASE_FIELDS
    ]
    return metric_fields, rows


def _to_float(s: str) -> float | None:
    if s is None or s == "":
        return None
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def analyse_pilot(
    csv_path: Path, lossy_grid: list[int]
) -> dict[str, Any]:
    metric_fields, rows = _load_pilot_csv(csv_path)

    # values_by_metric[metric][lossy] -> list of values (one per GIF)
    values_by_metric: dict[str, dict[int, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for r in rows:
        lossy = int(r["lossy"])
        for m in metric_fields:
            v = _to_float(r.get(m, ""))
            if v is None or not np.isfinite(v):
                continue
            values_by_metric[m][lossy].append(v)

    per_metric_summary: dict[str, dict[str, Any]] = {}
    for m, by_lossy in values_by_metric.items():
        per_lossy_mean: dict[int, float] = {
            l: float(np.mean(vs)) for l, vs in by_lossy.items() if vs
        }
        if len(per_lossy_mean) < 2:
            continue
        sorted_levels = sorted(per_lossy_mean.keys())
        all_vals = [per_lossy_mean[l] for l in sorted_levels]
        full_range = max(all_vals) - min(all_vals)
        per_metric_summary[m] = {
            "per_lossy_mean": per_lossy_mean,
            "full_range": full_range,
            "min": min(all_vals),
            "max": max(all_vals),
        }

    # Cross-metric disagreement at each lossy level: rank-normalize each
    # metric's values across the 10 pilot GIFs (1 = best for that metric),
    # then compute mean(max_rank - min_rank) across metrics. Higher = more
    # disagreement at that lossy level.
    disagreement_by_lossy: dict[int, float] = {}
    for lossy in lossy_grid:
        # gif_path -> per-metric normalised rank in [0,1]
        gif_metric_rank: dict[str, dict[str, float]] = defaultdict(dict)
        for m, by_lossy in values_by_metric.items():
            vals = by_lossy.get(lossy, [])
            if len(vals) < 3:
                continue
            # Map by gif path: find rows at this lossy
            paths_at_lossy = [
                r["path"] for r in rows if int(r["lossy"]) == lossy
            ]
            vals_at_lossy = [
                _to_float(r.get(m, ""))
                for r in rows
                if int(r["lossy"]) == lossy
            ]
            # Skip metrics that are constant at this lossy (e.g. all 1.0)
            usable = [
                (p, v)
                for p, v in zip(paths_at_lossy, vals_at_lossy)
                if v is not None and np.isfinite(v)
            ]
            if len(usable) < 3:
                continue
            v_arr = np.array([v for _, v in usable])
            if v_arr.max() - v_arr.min() < 1e-6:
                continue
            # Rank-normalize: 1 = best (highest value here; direction
            # doesn't matter since we take spread)
            order = np.argsort(v_arr)
            ranks = np.empty_like(order, dtype=float)
            ranks[order] = np.linspace(0.0, 1.0, len(order))
            for (p, _), r_val in zip(usable, ranks):
                gif_metric_rank[p][m] = float(r_val)

        if not gif_metric_rank:
            continue
        spreads: list[float] = []
        for _gif, per_metric in gif_metric_rank.items():
            if len(per_metric) < 2:
                continue
            ranks = list(per_metric.values())
            spreads.append(max(ranks) - min(ranks))
        if spreads:
            disagreement_by_lossy[lossy] = float(np.mean(spreads))

    # Pick lossy levels for the main sweep.
    # Rule: take the top disagreement level. If 2nd-best is within 0.75x of
    # top, include it too (up to 3 total).
    sorted_disagreement = sorted(
        disagreement_by_lossy.items(), key=lambda kv: kv[1], reverse=True
    )
    chosen: list[int] = []
    rationale: list[str] = []
    if not sorted_disagreement:
        # Fallback: middle of grid
        midpoint = lossy_grid[len(lossy_grid) // 2]
        chosen = [midpoint]
        rationale.append(
            f"No usable disagreement signal; defaulting to grid midpoint lossy={midpoint}"
        )
    else:
        top_lossy, top_score = sorted_disagreement[0]
        chosen.append(top_lossy)
        rationale.append(
            f"lossy={top_lossy} maximises cross-metric disagreement "
            f"(score {top_score:.3f})"
        )
        for cand_lossy, cand_score in sorted_disagreement[1:]:
            if len(chosen) >= 3:
                break
            if cand_score < 0.75 * top_score:
                continue
            chosen.append(cand_lossy)
            rationale.append(
                f"lossy={cand_lossy} also high disagreement "
                f"(score {cand_score:.3f}, {cand_score/top_score:.2f}x of peak)"
            )
        chosen.sort()

    return {
        "chosen_lossy_levels": chosen,
        "rationale": rationale,
        "disagreement_by_lossy": disagreement_by_lossy,
        "per_metric_full_range": {
            m: s["full_range"] for m, s in per_metric_summary.items()
        },
        "lossy_grid_used": lossy_grid,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--source",
        required=True,
        type=Path,
        help="Source dir for real GIFs (e.g. ~/Documents/GIFs)",
    )
    ap.add_argument("--output-csv", required=True, type=Path)
    ap.add_argument("--output-decisions", required=True, type=Path)
    ap.add_argument(
        "--lossy-grid",
        type=str,
        default=",".join(str(x) for x in DEFAULT_LOSSY_GRID),
        help="Comma-separated lossy levels to sweep",
    )
    ap.add_argument("--n-real", type=int, default=5)
    ap.add_argument(
        "--max-orig-kb",
        type=float,
        default=2000.0,
        help="Skip real GIFs larger than this (avoids animately timeouts)",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--workdir",
        type=Path,
        default=None,
        help="Working dir for compressed artifacts (defaults to temp dir)",
    )
    ap.add_argument(
        "--synthetic-dir",
        type=Path,
        default=None,
        help="Where to write synthetic GIFs (defaults to <output-csv>.parent/synthetic)",
    )
    args = ap.parse_args()

    lossy_grid = [int(x) for x in args.lossy_grid.split(",") if x.strip()]
    source = args.source.expanduser().resolve()

    if args.workdir is None:
        wd_ctx = tempfile.TemporaryDirectory(prefix="giflab_pilot_")
        workdir = Path(wd_ctx.name)
    else:
        wd_ctx = None
        workdir = args.workdir
        workdir.mkdir(parents=True, exist_ok=True)

    try:
        gl = import_giflab()
        rng = random.Random(args.seed)
        synth_dir = args.synthetic_dir or args.output_csv.parent / "synthetic"
        manifest = build_pilot_sample(
            source, synth_dir, args.n_real, rng, gl, max_orig_kb=args.max_orig_kb
        )
        print(f"[pilot] Manifest: {len(manifest)} GIFs ({sum(1 for m in manifest if m['source']=='real')} real + {sum(1 for m in manifest if m['source']=='synthetic')} synthetic)", flush=True)

        sweep_pilot(manifest, lossy_grid, args.output_csv, workdir, gl)
        print(f"[pilot] Wrote {args.output_csv}", flush=True)

        decisions = analyse_pilot(args.output_csv, lossy_grid)
        args.output_decisions.parent.mkdir(parents=True, exist_ok=True)
        args.output_decisions.write_text(json.dumps(decisions, indent=2))
        print(f"[pilot] Wrote {args.output_decisions}")
        print(f"[pilot] Chosen lossy levels: {decisions['chosen_lossy_levels']}")
        for r in decisions["rationale"]:
            print(f"[pilot]   - {r}")
    finally:
        if wd_ctx is not None:
            wd_ctx.cleanup()

    return 0


if __name__ == "__main__":
    sys.exit(main())
