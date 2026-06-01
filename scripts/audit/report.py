"""Render the metrics-audit markdown report from sanity + pilot + sweep outputs.

Produces a self-contained directory:
    <output>/
        report.md
        figures/*.png       (histograms, response curves, correlation matrix)
        thumbnails/*.png    (first frame thumbnails for outlier callouts)

Usage:
    poetry run python scripts/audit/report.py \\
        --sanity sanity_results.json \\
        --pilot pilot.csv \\
        --decisions pilot_decisions.json \\
        --sweep sweep_results.csv \\
        --output docs/metrics-audit/2026-05-22/
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

# Curated set of metrics to chart in detail. Histograms, correlations, and
# outlier callouts focus on these — everything else still appears in the
# sanity verdict table.
KEY_METRICS = [
    "ssim",
    "ms_ssim",
    "psnr",
    "lpips_quality_mean",
    "fsim",
    "gmsd",
    "chist",
    "edge_similarity",
    "texture_similarity",
    "composite_quality",
    "temporal_consistency",
    "ssimulacra2_mean",
    "deltae_mean",
    "banding_score_mean",
]

# Direction of "better" — used for ranking outliers and disagreement.
# Higher-is-better metrics get rank 1 = highest value; lower-is-better get
# rank 1 = lowest value.
HIGHER_IS_BETTER = {
    "ssim",
    "ms_ssim",
    "psnr",  # internal psnr is normalised [0,1], higher = better
    "fsim",
    "chist",
    "edge_similarity",
    "texture_similarity",
    "composite_quality",
    "temporal_consistency",
    "ssimulacra2_mean",
    "sharpness_similarity",
}
LOWER_IS_BETTER = {
    "lpips_quality_mean",
    "gmsd",
    "deltae_mean",
    "banding_score_mean",
    "mse",
    "rmse",
    "flicker_excess",
    "temporal_pumping_score",
}


def _metric_direction(name: str) -> str:
    if name in HIGHER_IS_BETTER:
        return "higher_better"
    if name in LOWER_IS_BETTER:
        return "lower_better"
    return "unknown"


def frame_reduction_summary(sweep_df: pd.DataFrame) -> list[str]:
    """Markdown lines summarising frame-reduction classification in the sweep.

    Operators triaging sweep results need to separate benign temporal dedup
    (fewer frames, total duration preserved — no quality concern) from
    possible frame-loss bugs (fewer frames, duration NOT preserved).  The
    sweep CSV carries a ``frame_reduction_class`` column for exactly this; this
    helper rolls it up into counts and lists the frame-loss / unknown rows that
    actually warrant investigation.

    Returns an empty list when the column is absent (old CSVs) so the report
    degrades gracefully.
    """
    if "frame_reduction_class" not in sweep_df.columns:
        return []

    md: list[str] = ["### Frame-reduction classification", ""]
    cls = sweep_df["frame_reduction_class"].astype(str)
    counts = cls.value_counts()
    n_none = int(counts.get("none", 0))
    n_dedup = int(counts.get("dedup", 0))
    n_loss = int(counts.get("frame_loss", 0))
    n_unknown = int(counts.get("unknown", 0))

    md.append(
        "Distinguishes benign temporal **dedup** (fewer frames, total "
        "duration preserved) from possible **frame_loss** (duration not "
        "preserved — needs investigation)."
    )
    md.append("")
    md.append(f"- No reduction: {n_none}")
    md.append(f"- Dedup (benign, duration preserved): {n_dedup}")
    md.append(f"- Frame loss (possible bug): {n_loss}")
    md.append(f"- Unknown (durations unreadable): {n_unknown}")
    md.append("")

    # List the rows that actually warrant attention.
    flagged = sweep_df[cls.isin(["frame_loss", "unknown"])]
    if not flagged.empty:
        md.append("#### Rows needing investigation (frame_loss / unknown)")
        md.append("")
        rows = []
        for _, r in flagged.iterrows():
            rows.append(
                [
                    Path(str(r["path"])).name[:48],
                    str(r["lossy"]) if "lossy" in flagged.columns else "",
                    str(r["frame_reduction_class"]),
                    str(r["frame_reduction"]) if "frame_reduction" in flagged.columns else "",
                ]
            )
        md.append(md_table(rows, ["gif", "lossy", "class", "frames_removed"]))
        md.append("")

    return md


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_sanity(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def load_pilot(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "success" in df.columns:
        df = df[df["success"].astype(str).isin(["True", "true", "1"])]
    return df


def load_sweep(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "success" in df.columns:
        df = df[df["success"].astype(str).isin(["True", "true", "1"])]
    return df


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def render_response_curves(pilot_df: pd.DataFrame, out_dir: Path) -> list[Path]:
    """One figure per key metric: lossy on x, metric value on y, one line per GIF."""
    out_dir.mkdir(parents=True, exist_ok=True)
    figs: list[Path] = []
    for metric in KEY_METRICS:
        if metric not in pilot_df.columns:
            continue
        fig, ax = plt.subplots(figsize=(7, 4))
        for gif_path, sub in pilot_df.groupby("path"):
            label = Path(str(gif_path)).stem[:24]
            sub = sub.sort_values("lossy")
            ax.plot(sub["lossy"], sub[metric], marker="o", linewidth=1, label=label)
        ax.set_xlabel("animately --lossy")
        ax.set_ylabel(metric)
        ax.set_title(f"Pilot response curve: {metric}")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=6, loc="best", ncol=2)
        fig.tight_layout()
        p = out_dir / f"pilot_curve_{metric}.png"
        fig.savefig(p, dpi=110)
        plt.close(fig)
        figs.append(p)
    return figs


def render_histograms(sweep_df: pd.DataFrame, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    figs: list[Path] = []
    for metric in KEY_METRICS:
        if metric not in sweep_df.columns:
            continue
        vals = sweep_df[metric].dropna().to_numpy()
        if len(vals) == 0:
            continue
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.hist(vals, bins=30, color="#4a7bd6", edgecolor="white")
        ax.axvline(np.mean(vals), color="red", linestyle="--", label=f"mean={np.mean(vals):.3f}")
        ax.axvline(np.median(vals), color="green", linestyle="--", label=f"median={np.median(vals):.3f}")
        ax.set_xlabel(metric)
        ax.set_ylabel("count")
        ax.set_title(f"Sweep distribution: {metric}  (n={len(vals)})")
        ax.legend(fontsize=8)
        fig.tight_layout()
        p = out_dir / f"hist_{metric}.png"
        fig.savefig(p, dpi=110)
        plt.close(fig)
        figs.append(p)
    return figs


def render_correlation_matrix(sweep_df: pd.DataFrame, out_dir: Path) -> Path | None:
    cols = [m for m in KEY_METRICS if m in sweep_df.columns]
    if len(cols) < 2:
        return None
    sub = sweep_df[cols].dropna()
    if sub.empty:
        return None
    corr = sub.corr(method="pearson")
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(cols, fontsize=8)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=7, color="black")
    fig.colorbar(im, ax=ax, fraction=0.04)
    ax.set_title("Cross-metric Pearson correlation (sweep)")
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "correlation_matrix.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    return p


# ---------------------------------------------------------------------------
# Thumbnails
# ---------------------------------------------------------------------------


def make_thumbnail(gif_path: Path, out_path: Path, max_dim: int = 200) -> bool:
    try:
        with Image.open(gif_path) as im:
            frame = im.convert("RGB").copy()
        frame.thumbnail((max_dim, max_dim))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        frame.save(out_path)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Outlier / disagreement analysis
# ---------------------------------------------------------------------------


def top_outliers_per_metric(
    sweep_df: pd.DataFrame, top_n: int = 5
) -> dict[str, pd.DataFrame]:
    """For each key metric, return top-N rows by absolute z-score."""
    out: dict[str, pd.DataFrame] = {}
    for metric in KEY_METRICS:
        if metric not in sweep_df.columns:
            continue
        sub = sweep_df[["path", "lossy", "source", "content_type", metric]].dropna(subset=[metric])
        if len(sub) < 3:
            continue
        mu = sub[metric].mean()
        sigma = sub[metric].std()
        if sigma <= 0:
            continue
        sub = sub.copy()
        sub["z"] = (sub[metric] - mu) / sigma
        sub["abs_z"] = sub["z"].abs()
        out[metric] = sub.sort_values("abs_z", ascending=False).head(top_n)
    return out


def rank_normalise(vals: np.ndarray, higher_better: bool) -> np.ndarray:
    """Map values to [0,1] where 1 = best for the metric."""
    order = np.argsort(vals)
    ranks = np.empty_like(order, dtype=float)
    if len(order) > 1:
        ranks[order] = np.linspace(0.0, 1.0, len(order))
    else:
        ranks[order] = np.array([0.5])
    if not higher_better:
        ranks = 1.0 - ranks
    return ranks


def disagreement_table(
    sweep_df: pd.DataFrame, top_n: int = 10, min_metrics: int = 5
) -> pd.DataFrame:
    """For each row, compute per-metric rank in [0,1] (1=best) and return
    top-N rows by (max_rank - min_rank). Only considers metrics with a
    known direction in HIGHER_IS_BETTER / LOWER_IS_BETTER."""
    metrics_to_use = [
        m
        for m in KEY_METRICS
        if m in sweep_df.columns and _metric_direction(m) != "unknown"
    ]
    if len(metrics_to_use) < min_metrics:
        return pd.DataFrame()

    df = sweep_df.copy()
    rank_cols: list[str] = []
    for m in metrics_to_use:
        vals = df[m].to_numpy(dtype=float)
        mask = np.isfinite(vals)
        ranks = np.full(len(vals), np.nan)
        if mask.sum() >= 3:
            ranks[mask] = rank_normalise(vals[mask], _metric_direction(m) == "higher_better")
        rc = f"_rank_{m}"
        df[rc] = ranks
        rank_cols.append(rc)

    rank_arr = df[rank_cols].to_numpy()
    finite_mask = np.isfinite(rank_arr)
    finite_count = finite_mask.sum(axis=1)
    # Replace NaN with sentinel values for arg max/min so all-NaN rows don't
    # raise. Their spreads stay NaN below and we mask them out at the end.
    rank_for_max = np.where(finite_mask, rank_arr, -np.inf)
    rank_for_min = np.where(finite_mask, rank_arr, np.inf)
    with np.errstate(invalid="ignore"):
        max_vals = np.where(finite_count > 0, np.nanmax(rank_arr, axis=1), np.nan)
        min_vals = np.where(finite_count > 0, np.nanmin(rank_arr, axis=1), np.nan)
    spreads = np.where(finite_count >= min_metrics, max_vals - min_vals, np.nan)
    df["rank_spread"] = spreads
    arg_high = rank_for_max.argmax(axis=1)
    arg_low = rank_for_min.argmin(axis=1)
    df["best_metric"] = [
        metrics_to_use[i] if not np.isnan(spreads[k]) else ""
        for k, i in enumerate(arg_high)
    ]
    df["worst_metric"] = [
        metrics_to_use[i] if not np.isnan(spreads[k]) else ""
        for k, i in enumerate(arg_low)
    ]

    return (
        df.dropna(subset=["rank_spread"])
        .sort_values("rank_spread", ascending=False)
        .head(top_n)
        [["path", "lossy", "source", "content_type", "rank_spread", "best_metric", "worst_metric"]]
    )


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def md_table(rows: list[list[str]], header: list[str]) -> str:
    out = ["| " + " | ".join(header) + " |"]
    out.append("|" + "|".join(["---"] * len(header)) + "|")
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def render_report(
    sanity: dict[str, Any],
    pilot_df: pd.DataFrame,
    decisions: dict[str, Any],
    sweep_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = out_dir / "figures"
    thumbs_dir = out_dir / "thumbnails"

    # --- Figures ---
    print("[report] Rendering response curves...", flush=True)
    response_curves = render_response_curves(pilot_df, figures_dir)
    print("[report] Rendering histograms...", flush=True)
    histograms = render_histograms(sweep_df, figures_dir)
    print("[report] Rendering correlation matrix...", flush=True)
    corr_fig = render_correlation_matrix(sweep_df, figures_dir)

    # --- Outliers / disagreement ---
    print("[report] Computing outliers...", flush=True)
    outliers = top_outliers_per_metric(sweep_df, top_n=5)
    print("[report] Computing disagreement table...", flush=True)
    disagree = disagreement_table(sweep_df, top_n=10)

    # --- Thumbnails for outlier callouts ---
    unique_paths: set[str] = set()
    for df in outliers.values():
        unique_paths.update(df["path"].astype(str).tolist())
    if not disagree.empty:
        unique_paths.update(disagree["path"].astype(str).tolist())
    print(f"[report] Generating {len(unique_paths)} thumbnails...", flush=True)
    thumb_for: dict[str, str] = {}
    for p_str in unique_paths:
        p = Path(p_str)
        rel = p.stem.replace("/", "_") + ".png"
        out_path = thumbs_dir / rel
        if make_thumbnail(p, out_path):
            thumb_for[p_str] = f"thumbnails/{rel}"

    # ---------- Build report.md ----------
    md: list[str] = []
    md.append("# GifLab metrics audit")
    md.append("")
    md.append(
        "Three-phase audit: (1) sanity tests on synthetic content, "
        "(2) lossy calibration pilot, (3) corpus sweep at the chosen "
        "lossy levels. Findings are advisory — the goal is to surface "
        "anything weird in metric behaviour, not to ship fixes."
    )
    md.append("")

    # Pilot calibration
    md.append("## Phase 2 — pilot calibration")
    md.append("")
    md.append(f"**Chosen lossy levels for the main sweep:** `{decisions['chosen_lossy_levels']}`")
    md.append("")
    md.append("Rationale:")
    for r in decisions.get("rationale", []):
        md.append(f"- {r}")
    md.append("")
    md.append("**Cross-metric disagreement by lossy level:**")
    md.append("")
    dis_by = decisions.get("disagreement_by_lossy", {})
    if dis_by:
        # JSON keys are strings, but pilot.py may also use int keys.
        norm = {int(k): float(v) for k, v in dis_by.items()}
        rows = [[str(l), f"{norm[l]:.3f}"] for l in sorted(norm.keys())]
        md.append(md_table(rows, ["lossy", "disagreement"]))
        md.append("")

    md.append("**Response curves (one line per pilot GIF):**")
    md.append("")
    for fig in response_curves:
        md.append(f"![{fig.stem}](figures/{fig.name})")
        md.append("")

    # Sanity verdicts
    md.append("## Phase 1 — sanity verdicts")
    md.append("")
    md.append(
        "Identity = metric on (gif, gif). Pathological = metric on (white, "
        "black). Direction inferred from those two reference points. "
        "Verdict PASS if monotonicity holds across all degradation kinds "
        "(noise / blur / quantize / animately lossy). SUSPICIOUS if any "
        "kind shows an inversion. INCONCLUSIVE if identity == pathological "
        "(usually a single-stream metric where this pair can't discriminate)."
    )
    md.append("")
    verdicts = sanity.get("verdicts", [])
    rows = []
    for v in verdicts:
        rows.append(
            [
                v["metric"],
                f"{v['identity_mean']:.4f}",
                f"{v['pathological']:.4f}",
                v["direction"],
                v["verdict"],
                v.get("note", ""),
            ]
        )
    md.append(md_table(rows, ["metric", "identity_mean", "pathological", "direction", "verdict", "note"]))
    md.append("")

    susp = [v for v in verdicts if v["verdict"] == "SUSPICIOUS"]
    if susp:
        md.append("### Monotonicity violations (SUSPICIOUS detail)")
        md.append("")
        for v in susp:
            md.append(f"**{v['metric']}** — {len(v['monotonicity_failures'])} failures")
            for f in v["monotonicity_failures"][:5]:
                vals_str = ", ".join(f"{x:.4f}" for x in f["values"])
                md.append(f"- `{f['kind']}` on `{f['base']}`: [{vals_str}]")
            md.append("")

    # Sweep summary
    md.append("## Phase 3 — corpus sweep")
    md.append("")
    md.append(f"Rows: **{len(sweep_df)}** successful (path × lossy) pairs.")
    if "source" in sweep_df.columns:
        md.append("")
        n_real = (sweep_df["source"] == "real").sum()
        n_synth = (sweep_df["source"] == "synthetic").sum()
        md.append(f"- Real GIFs: {n_real}")
        md.append(f"- Synthetic GIFs: {n_synth}")
    md.append("")

    md.extend(frame_reduction_summary(sweep_df))

    if corr_fig is not None:
        md.append("### Cross-metric correlation")
        md.append("")
        md.append(f"![correlation matrix](figures/{corr_fig.name})")
        md.append("")
        md.append(
            "Metric pairs with |r| < 0.2 within the same family (e.g. SSIM "
            "vs MS-SSIM) suggest the metrics are measuring different things "
            "than expected."
        )
        md.append("")

    md.append("### Distributions (key metrics)")
    md.append("")
    for fig in histograms:
        md.append(f"![{fig.stem}](figures/{fig.name})")
        md.append("")

    # Outliers
    md.append("### Top outliers per metric (|z| highest)")
    md.append("")
    for metric, df in outliers.items():
        md.append(f"#### {metric}")
        md.append("")
        rows = []
        for _, r in df.iterrows():
            thumb = thumb_for.get(str(r["path"]), "")
            thumb_md = f"![]({thumb})" if thumb else ""
            rows.append(
                [
                    thumb_md,
                    Path(str(r["path"])).name[:48],
                    str(r["lossy"]),
                    str(r["source"]) if r["source"] else "",
                    str(r["content_type"]) if pd.notna(r["content_type"]) else "",
                    f"{r[metric]:.4f}",
                    f"{r['z']:+.2f}",
                ]
            )
        md.append(md_table(rows, ["thumb", "gif", "lossy", "source", "content_type", "value", "z"]))
        md.append("")

    # Disagreement clusters
    if not disagree.empty:
        md.append("### Cross-metric disagreement (top-spread GIFs)")
        md.append("")
        md.append(
            "GIFs where the best-ranked metric and the worst-ranked metric "
            "are far apart. Inspect these to see whether one of the metrics "
            "is mis-firing on this kind of content."
        )
        md.append("")
        rows = []
        for _, r in disagree.iterrows():
            thumb = thumb_for.get(str(r["path"]), "")
            thumb_md = f"![]({thumb})" if thumb else ""
            rows.append(
                [
                    thumb_md,
                    Path(str(r["path"])).name[:48],
                    str(r["lossy"]),
                    str(r["source"]) if pd.notna(r["source"]) else "",
                    str(r["content_type"]) if pd.notna(r["content_type"]) else "",
                    f"{r['rank_spread']:.2f}",
                    str(r["best_metric"]),
                    str(r["worst_metric"]),
                ]
            )
        md.append(md_table(rows, ["thumb", "gif", "lossy", "source", "content_type", "spread", "best→", "worst→"]))
        md.append("")

    # Per-content-type breakdown (synthetic only)
    if "source" in sweep_df.columns:
        synth = sweep_df[sweep_df["source"] == "synthetic"]
        if not synth.empty and "content_type" in synth.columns:
            md.append("### Synthetic per-content-type means (key metrics)")
            md.append("")
            available = [m for m in KEY_METRICS if m in synth.columns]
            grouped = synth.groupby("content_type")[available].mean().round(4)
            header = ["content_type"] + available
            rows = []
            for ct, row in grouped.iterrows():
                rows.append([str(ct)] + [f"{row[m]:.3f}" if pd.notna(row[m]) else "" for m in available])
            md.append(md_table(rows, header))
            md.append("")

    # Footer
    md.append("---")
    md.append("")
    md.append(
        "Generated by `scripts/audit/report.py`. Source CSVs: "
        f"`{Path(sys.argv[0]).name}` arguments. Re-run any of the audit "
        "scripts to refresh."
    )

    (out_dir / "report.md").write_text("\n".join(md))
    print(f"[report] Wrote {out_dir / 'report.md'}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sanity", required=True, type=Path)
    ap.add_argument("--pilot", required=True, type=Path)
    ap.add_argument("--decisions", required=True, type=Path)
    ap.add_argument("--sweep", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()

    sanity = load_sanity(args.sanity)
    pilot_df = load_pilot(args.pilot)
    decisions = json.loads(args.decisions.read_text())
    sweep_df = load_sweep(args.sweep)

    render_report(sanity, pilot_df, decisions, sweep_df, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
