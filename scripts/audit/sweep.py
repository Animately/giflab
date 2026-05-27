"""Phase 3 of the metrics audit — main corpus sweep at pilot-chosen lossy levels.

For each (manifest entry × lossy level): compress via animately → measure all
metrics → append row to CSV. Resume-safe: skips (path, lossy) pairs already in
the CSV.

Usage:
    poetry run python scripts/audit/sweep.py \\
        --manifest sample_manifest.json \\
        --decisions pilot_decisions.json \\
        --output sweep_results.csv
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

# Local helpers
sys.path.insert(0, str(Path(__file__).parent))
from _common import (  # noqa: E402
    compress_animately,
    import_giflab,
    load_existing_keys,
    measure_pair,
    open_csv_for_append,
)
from pilot import BASE_FIELDS  # noqa: E402


def _augment_manifest_with_synthetic(
    manifest: list[dict[str, Any]],
    synth_dir: Path,
    gl: dict[str, Any],
) -> list[dict[str, Any]]:
    """Append the full synthetic set (25 specs) to the manifest. Synthetic
    GIFs come from SyntheticGifGenerator and add labelled content_type."""
    synth_dir.mkdir(parents=True, exist_ok=True)
    gen = gl["SyntheticGifGenerator"](synth_dir)
    paths = gen.generate_gifs(use_targeted_set=False)
    name_to_spec = {s.name: s for s in gen.synthetic_specs}
    augmented = list(manifest)
    for p in paths:
        spec = name_to_spec[p.stem]
        augmented.append(
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
    return augmented


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument(
        "--decisions",
        type=Path,
        default=None,
        help="pilot_decisions.json (provides lossy levels); ignored if --lossy is given",
    )
    ap.add_argument(
        "--lossy",
        type=str,
        default=None,
        help="Comma-separated lossy levels override (e.g. '40,100')",
    )
    ap.add_argument("--output", required=True, type=Path, help="sweep_results.csv")
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
        help="Where to write synthetic GIFs (defaults to <output>.parent/synthetic)",
    )
    ap.add_argument(
        "--skip-synthetic",
        action="store_true",
        help="Do not augment manifest with full synthetic set",
    )
    args = ap.parse_args()

    # Resolve lossy levels.
    if args.lossy:
        lossy_levels = [int(x) for x in args.lossy.split(",") if x.strip()]
    elif args.decisions:
        decisions = json.loads(args.decisions.read_text())
        lossy_levels = list(decisions["chosen_lossy_levels"])
    else:
        print("Either --lossy or --decisions must be provided", file=sys.stderr)
        return 2
    if not lossy_levels:
        print("No lossy levels resolved", file=sys.stderr)
        return 2
    print(f"[sweep] Lossy levels: {lossy_levels}", flush=True)

    manifest_data = json.loads(args.manifest.read_text())
    manifest = manifest_data["entries"]
    print(f"[sweep] Manifest: {len(manifest)} real entries", flush=True)

    if args.workdir is None:
        wd_ctx = tempfile.TemporaryDirectory(prefix="giflab_sweep_")
        workdir = Path(wd_ctx.name)
    else:
        wd_ctx = None
        workdir = args.workdir
        workdir.mkdir(parents=True, exist_ok=True)

    try:
        gl = import_giflab()

        if not args.skip_synthetic:
            synth_dir = args.synthetic_dir or args.output.parent / "synthetic"
            manifest = _augment_manifest_with_synthetic(manifest, synth_dir, gl)
            n_real = sum(1 for m in manifest if m["source"] == "real")
            n_synth = sum(1 for m in manifest if m["source"] == "synthetic")
            print(f"[sweep] Manifest after synthetic augment: {n_real} real + {n_synth} synthetic", flush=True)

        # Probe to discover metric keys
        first_orig = Path(manifest[0]["path"])
        probe = workdir / "_probe.gif"
        ok, err = compress_animately(first_orig, probe, lossy_levels[0], gl)
        if not ok:
            raise RuntimeError(
                f"Sweep probe failed on {first_orig} at lossy={lossy_levels[0]}: {err}"
            )
        probe_metrics, _ = measure_pair(first_orig, probe, gl)
        # Drop any metric key that collides with BASE_FIELDS (e.g.
        # 'compression_ratio' which we compute ourselves above).
        base_set = set(BASE_FIELDS)
        metric_fields = sorted(k for k in probe_metrics if k not in base_set)
        header = BASE_FIELDS + metric_fields

        existing_keys = load_existing_keys(args.output, key_fields=("path", "lossy"))
        if existing_keys:
            print(f"[sweep] Resume: skipping {len(existing_keys)} already-done (path,lossy) combos", flush=True)
        fh, writer, _ = open_csv_for_append(args.output, header)

        total = len(manifest) * len(lossy_levels)
        done = 0
        try:
            for entry in manifest:
                orig = Path(entry["path"])
                if not orig.exists():
                    print(f"[sweep] WARN: missing {orig}", flush=True)
                    done += len(lossy_levels)
                    continue
                for lossy in lossy_levels:
                    done += 1
                    if (str(orig), str(lossy)) in existing_keys:
                        continue
                    compressed = workdir / f"{orig.stem}_lossy{lossy}{orig.suffix}"
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
                        print(f"[sweep] [{done}/{total}] FAIL {orig.name} lossy={lossy}: {err}", flush=True)
                        continue
                    metrics, runtime_s = measure_pair(orig, compressed, gl)
                    kb_compressed = round(compressed.stat().st_size / 1024.0, 2)
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
                        **metrics,
                    }
                    writer.writerow(row)
                    fh.flush()
                    if done % 10 == 0 or done == total:
                        print(f"[sweep] [{done}/{total}] {orig.name} lossy={lossy} ok ({runtime_s:.1f}s)", flush=True)
                    try:
                        compressed.unlink()
                    except OSError:
                        pass
        finally:
            fh.close()

        print(f"[sweep] Wrote {args.output}")
    finally:
        if wd_ctx is not None:
            wd_ctx.cleanup()

    return 0


if __name__ == "__main__":
    sys.exit(main())
