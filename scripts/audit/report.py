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

# Local helpers (scripts/audit/ is not a package)
sys.path.insert(0, str(Path(__file__).parent))
from _common import tie_average_unit_ranks  # noqa: E402
from sanity import _classify_metric  # noqa: E402

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
    "temporal_consistency_compressed",
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
    "temporal_consistency_compressed",
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
    "flicker_excess_compressed",
    "temporal_pumping_score_compressed",
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
                    Path(str(r["path"])).name[:48] if "path" in flagged.columns else "",
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
    """Map values to [0,1] where 1 = best for the metric.

    Tie-aware: tied values share their average rank (see
    ``_common.tie_average_unit_ranks``). The sweep corpus is heavily
    tie-saturated (232/371 rows tied at ``banding_score_mean == 0.0``), and
    the previous argsort+linspace scheme assigned arbitrary distinct ranks
    across tie blocks — pinning the disagreement-table spread near 1.0 with
    tie noise instead of genuine metric-family disagreement.
    """
    ranks = tie_average_unit_ranks(vals)
    if not higher_better:
        ranks = 1.0 - ranks
    return ranks


def disagreement_table(
    sweep_df: pd.DataFrame, top_n: int = 10, min_metrics: int = 5
) -> pd.DataFrame:
    """For each row, compute per-metric rank in [0,1] (1=best) and return
    top-N rows by (max_rank - min_rank). Only considers metrics with a
    known direction in HIGHER_IS_BETTER / LOWER_IS_BETTER.

    Only pairwise-quality metrics vote: single-stream ``_compressed`` keys
    (e.g. ``temporal_consistency_compressed``) don't measure fidelity to the
    original, so their "disagreement" with pair metrics is structural, not a
    signal that a metric is mis-firing. They keep their outlier tables and
    response curves — this exclusion is disagreement-only (mirrors the PR #54
    sanity DIAGNOSTIC triage; see ``sanity._classify_metric``)."""
    metrics_to_use = [
        m
        for m in KEY_METRICS
        if m in sweep_df.columns
        and _metric_direction(m) != "unknown"
        and _classify_metric(m) == "pairwise_quality"
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
    # Absolute values of the best/worst metric per row, so range-compression
    # cases stay visible (e.g. chist's whole corpus range is [0.9866, 1.0] —
    # its rank-0.0 "worst" is a near-perfect absolute value).
    metric_arr = df[metrics_to_use].to_numpy(dtype=float)
    row_idx = np.arange(len(df))
    df["best_value"] = metric_arr[row_idx, arg_high]
    df["worst_value"] = metric_arr[row_idx, arg_low]

    return (
        df.dropna(subset=["rank_spread"])
        .sort_values("rank_spread", ascending=False)
        .head(top_n)
        [[
            "path", "lossy", "source", "content_type", "rank_spread",
            "best_metric", "best_value", "worst_metric", "worst_value",
        ]]
    )


# ---------------------------------------------------------------------------
# SSIM vs composite_quality divergence (control-relative paired deltas)
# ---------------------------------------------------------------------------

# EPS mirrors gifprep's ``EPSILON_QUALITY`` (gifprep/src/gifprep/bench/
# pareto.py) and the EPS in gifprep's corpus-local divergence pass
# (analyses/2026-06-12-corpus-local-composite-rerun/divergence.py), so these
# findings transfer 1:1 to the downstream consumer
# [[gifprep-corpus-local-composite-rerun]]. GUARD = 4x EPS bounds the verdict
# (held-vs-degraded) criterion; the zone between EPS and GUARD is a
# deliberate *indeterminate* band that is never flagged — the cliff-edge
# lesson applied to the audit criterion itself (no boundary jitter reported
# as disagreement).
DIVERGENCE_EPS = 0.005
DIVERGENCE_GUARD = 0.02

_DIVERGENCE_CELL_COLUMNS = [
    "path",
    "lossy",
    "source",
    "content_type",
    "frame_reduction_class",
    "ssim_control",
    "ssim_cell",
    "comp_control",
    "comp_cell",
    "d_ssim",
    "d_comp",
    "divergence_score",
    "flag",
]

# The 15 named composite contributors, mirroring the exact order and
# normalisation of ``giflab.enhanced_metrics.calculate_composite_quality``.
# ``kind`` selects the normalisation: "standard" routes through
# ``normalize_metric`` (which carries the psnr/mse/gmsd/banding/deltae
# special cases), the other kinds replicate the three inline normalisations
# that live in ``calculate_composite_quality`` itself. This bounded
# duplication is kept honest by ``composite_attribution``'s mandatory
# round-trip check against the real function — a decomposition that drifts
# from production is flagged non-attributable, never silently trusted.
_COMPOSITE_CONTRIBUTORS: list[tuple[str, str, str]] = [
    ("ssim_mean", "ENHANCED_SSIM_WEIGHT", "standard"),
    ("ms_ssim_mean", "ENHANCED_MS_SSIM_WEIGHT", "standard"),
    ("psnr_mean", "ENHANCED_PSNR_WEIGHT", "standard"),
    ("mse_mean", "ENHANCED_MSE_WEIGHT", "standard"),
    ("fsim_mean", "ENHANCED_FSIM_WEIGHT", "standard"),
    ("edge_similarity_mean", "ENHANCED_EDGE_WEIGHT", "standard"),
    ("gmsd_mean", "ENHANCED_GMSD_WEIGHT", "standard"),
    ("chist_mean", "ENHANCED_CHIST_WEIGHT", "standard"),
    ("sharpness_similarity_mean", "ENHANCED_SHARPNESS_WEIGHT", "standard"),
    ("texture_similarity_mean", "ENHANCED_TEXTURE_WEIGHT", "standard"),
    ("temporal_consistency_delta", "ENHANCED_TEMPORAL_WEIGHT", "one_minus_clamped"),
    ("lpips_quality_mean", "ENHANCED_LPIPS_WEIGHT", "one_minus"),
    ("ssimulacra2_mean", "ENHANCED_SSIMULACRA2_WEIGHT", "clamp"),
    ("banding_score_mean", "ENHANCED_BANDING_WEIGHT", "standard"),
    ("deltae_mean", "ENHANCED_DELTAE_WEIGHT", "standard"),
]


def _opt_str(row: pd.Series, col: str) -> str:
    """Optional string column with graceful absence."""
    if col in row.index and pd.notna(row[col]):
        return str(row[col])
    return ""


def ssim_composite_divergence(
    sweep_df: pd.DataFrame,
    eps: float = DIVERGENCE_EPS,
    guard: float = DIVERGENCE_GUARD,
    control_lossy: int = 0,
    ssim_col: str = "ssim",
    comp_col: str = "composite_quality",
) -> dict[str, Any]:
    """Control-relative SSIM-vs-composite divergence over a sweep DataFrame.

    Each GIF's ``lossy == control_lossy`` row is its own control. Per cell
    (gif, L != control):

        d_ssim = ssim(L) - ssim(control)
        d_comp = composite_quality(L) - composite_quality(control)

    Flags (mutually exclusive, ``direction`` takes precedence):

    - ``direction`` — the deltas point in opposite directions, both beyond
      ``eps`` (identical rule to gifprep's divergence pass).
    - ``verdict`` — one metric held (``|d| <= eps``) while the other moved
      beyond ``guard``. Movement into the (eps, guard] zone is deliberately
      indeterminate and never flagged.

    ``divergence_score = |d_comp - d_ssim|`` is computed for EVERY analysable
    cell (continuous severity; cannot tie-saturate like rank spreads).

    NaN honesty: cells where either metric is non-finite in either row are
    excluded and counted — never imputed, never flagged. Duplicate
    (path, lossy) rows (sweep resume artifacts) are deduped keep-first.

    Returns a dict::

        {
          "cells":    DataFrame[_DIVERGENCE_CELL_COLUMNS], sorted by
                      divergence_score descending,
          "controls": DataFrame[path, source, content_type, ssim_control,
                      comp_control, identity_gap]  (identity_gap =
                      comp_control - ssim_control; both metrics have identity
                      value exactly 1.0, so this exposes palette-axis
                      divergence at the control point itself),
          "excluded": {"no_control": int, "nan_control": int, "nan_cell": int},
          "eps": eps, "guard": guard, "control_lossy": control_lossy,
        }
    """
    df = sweep_df.drop_duplicates(subset=["path", "lossy"], keep="first").copy()
    df["lossy"] = df["lossy"].astype(int)

    control_df = df[df["lossy"] == int(control_lossy)]
    cell_df = df[df["lossy"] != int(control_lossy)]

    control_paths: set[str] = set(control_df["path"].astype(str))
    good_controls: dict[str, tuple[float, float]] = {}
    control_records: list[dict[str, Any]] = []
    for _, crow in control_df.iterrows():
        s0 = float(crow[ssim_col])
        c0 = float(crow[comp_col])
        if np.isfinite(s0) and np.isfinite(c0):
            good_controls[str(crow["path"])] = (s0, c0)
            control_records.append(
                {
                    "path": str(crow["path"]),
                    "source": _opt_str(crow, "source"),
                    "content_type": _opt_str(crow, "content_type"),
                    "ssim_control": s0,
                    "comp_control": c0,
                    "identity_gap": c0 - s0,
                }
            )

    excluded = {"no_control": 0, "nan_control": 0, "nan_cell": 0}
    records: list[dict[str, Any]] = []
    for _, row in cell_df.iterrows():
        path = str(row["path"])
        if path not in control_paths:
            excluded["no_control"] += 1
            continue
        if path not in good_controls:
            excluded["nan_control"] += 1
            continue
        s_l = float(row[ssim_col])
        c_l = float(row[comp_col])
        if not (np.isfinite(s_l) and np.isfinite(c_l)):
            excluded["nan_cell"] += 1
            continue
        s0, c0 = good_controls[path]
        d_ssim = s_l - s0
        d_comp = c_l - c0
        direction = (d_comp > eps and d_ssim < -eps) or (
            d_comp < -eps and d_ssim > eps
        )
        verdict = (abs(d_ssim) <= eps and abs(d_comp) > guard) or (
            abs(d_comp) <= eps and abs(d_ssim) > guard
        )
        flag = "direction" if direction else ("verdict" if verdict else "")
        records.append(
            {
                "path": path,
                "lossy": int(row["lossy"]),
                "source": _opt_str(row, "source"),
                "content_type": _opt_str(row, "content_type"),
                "frame_reduction_class": _opt_str(row, "frame_reduction_class"),
                "ssim_control": s0,
                "ssim_cell": s_l,
                "comp_control": c0,
                "comp_cell": c_l,
                "d_ssim": d_ssim,
                "d_comp": d_comp,
                "divergence_score": abs(d_comp - d_ssim),
                "flag": flag,
            }
        )

    cells = pd.DataFrame(records, columns=_DIVERGENCE_CELL_COLUMNS)
    if not cells.empty:
        cells = cells.sort_values(
            "divergence_score", ascending=False
        ).reset_index(drop=True)
    controls = pd.DataFrame(
        control_records,
        columns=[
            "path",
            "source",
            "content_type",
            "ssim_control",
            "comp_control",
            "identity_gap",
        ],
    )
    return {
        "cells": cells,
        "controls": controls,
        "excluded": excluded,
        "eps": eps,
        "guard": guard,
        "control_lossy": control_lossy,
    }


def adjacent_level_divergence(
    sweep_df: pd.DataFrame,
    eps: float = DIVERGENCE_EPS,
    ssim_col: str = "ssim",
    comp_col: str = "composite_quality",
) -> pd.DataFrame:
    """Adjacent-lossy-level monotonicity disagreement per GIF.

    For each GIF, walk its available lossy levels in ascending order and
    compare the step deltas sign(Δssim) vs sign(Δcomposite) between
    consecutive analysable levels. Catches "composite recovers while SSIM
    keeps falling" shapes (the historical smooth_gradient dip-then-recover)
    that the control-relative framing can smooth over.

    Rows with a non-finite value in either metric are skipped (the step is
    taken between the nearest analysable levels, labelled honestly via
    ``lossy_from``/``lossy_to``). Returns one row per step with
    ``sign_disagree`` True when the deltas point in opposite directions,
    both beyond ``eps``.
    """
    df = sweep_df.drop_duplicates(subset=["path", "lossy"], keep="first").copy()
    df["lossy"] = df["lossy"].astype(int)

    records: list[dict[str, Any]] = []
    for path, sub in df.groupby("path"):
        sub = sub.sort_values("lossy")
        prev: tuple[int, float, float] | None = None
        for _, row in sub.iterrows():
            s = float(row[ssim_col])
            c = float(row[comp_col])
            if not (np.isfinite(s) and np.isfinite(c)):
                continue
            lossy = int(row["lossy"])
            if prev is not None:
                p_lossy, p_s, p_c = prev
                d_ssim = s - p_s
                d_comp = c - p_c
                records.append(
                    {
                        "path": str(path),
                        "source": _opt_str(row, "source"),
                        "content_type": _opt_str(row, "content_type"),
                        "lossy_from": p_lossy,
                        "lossy_to": lossy,
                        "d_ssim": d_ssim,
                        "d_comp": d_comp,
                        "divergence_score": abs(d_comp - d_ssim),
                        "sign_disagree": (d_comp > eps and d_ssim < -eps)
                        or (d_comp < -eps and d_ssim > eps),
                    }
                )
            prev = (lossy, s, c)

    return pd.DataFrame(
        records,
        columns=[
            "path",
            "source",
            "content_type",
            "lossy_from",
            "lossy_to",
            "d_ssim",
            "d_comp",
            "divergence_score",
            "sign_disagree",
        ],
    )


def _normalise_contribution(kind: str, name: str, value: float, em: Any) -> float:
    """Normalise one contributor value exactly as the production composite."""
    if kind == "standard":
        return float(em.normalize_metric(name, float(value)))
    if kind == "one_minus_clamped":
        # temporal_consistency_delta: clamp delta to [0, 1] then invert.
        return max(0.0, min(1.0, 1.0 - max(0.0, min(1.0, float(value)))))
    if kind == "one_minus":
        # lpips: lower = better.
        return max(0.0, min(1.0, 1.0 - float(value)))
    if kind == "clamp":
        # ssimulacra2: already 0-1, higher = better.
        return max(0.0, min(1.0, float(value)))
    raise ValueError(f"unknown contribution kind: {kind}")


def _nan_equal(a: float, b: float, tol: float) -> bool:
    """Equality within tol, treating NaN == NaN as equal (honest round-trip)."""
    a_nan = isinstance(a, float) and np.isnan(a)
    b_nan = isinstance(b, float) and np.isnan(b)
    if a_nan or b_nan:
        return a_nan and b_nan
    return abs(a - b) <= tol


def composite_attribution(
    control_row: Any,
    cell_row: Any,
    tol: float = 1e-6,
) -> dict[str, Any]:
    """Decompose a cell's composite movement into named contributor terms.

    For each of the 15 composite contributors, reports
    ``weighted_delta = w_i * (norm_i(cell) - norm_i(control))`` plus the
    surviving (measurable) weight of each row, so NaN-driven weight
    *redistribution* surfaces as its own named driver instead of hiding as
    phantom quality movement.

    Verification is mandatory and two-layered; failure of either check marks
    the result non-attributable rather than silently trusting it:

    1. **Internal consistency** — the local decomposition, resolved through
       the real ``_resolve_composite_from_contributions``, must match the
       real ``calculate_composite_quality`` on the same row.
    2. **CSV round-trip** — the real function's value must match the row's
       stored ``composite_quality`` within ``tol``.

    Rows may be dicts or pandas Series (any Mapping of metric key -> value).
    """
    from giflab import enhanced_metrics as em  # lazy: heavy import chain

    cfg = em.DEFAULT_METRICS_CONFIG
    control = dict(control_row)
    cell = dict(cell_row)

    def decompose(row: dict[str, Any]) -> tuple[dict[str, tuple[float, float, bool]], float]:
        """Per-row contributor map: name -> (norm_or_nan, weight, missing)."""
        per_metric: dict[str, tuple[float, float, bool]] = {}
        contributions: list[tuple[float, float]] = []
        use_delta = getattr(cfg, "USE_TEMPORAL_DELTA_FOR_COMPOSITE", True)
        for name, weight_attr, kind in _COMPOSITE_CONTRIBUTORS:
            weight = float(getattr(cfg, weight_attr))
            key, use_kind = name, kind
            if name == "temporal_consistency_delta":
                # Mirror the production delta-vs-compressed branch.
                if not (use_delta and name in row):
                    if "temporal_consistency_compressed" in row:
                        key = "temporal_consistency_compressed"
                        use_kind = "standard"
                    else:
                        # Key absent entirely: no contribution at all.
                        per_metric[name] = (float("nan"), weight, True)
                        continue
            elif key not in row:
                per_metric[name] = (float("nan"), weight, True)
                continue
            raw = row[key]
            if em._is_missing(raw if raw is None else float(raw)):
                per_metric[name] = (float("nan"), weight, True)
                contributions.append((float("nan"), weight))
                continue
            norm = _normalise_contribution(use_kind, key, float(raw), em)
            per_metric[name] = (norm, weight, False)
            contributions.append((norm, weight))
        recomputed = em._resolve_composite_from_contributions(contributions)
        return per_metric, recomputed

    result: dict[str, Any] = {
        "attributable": True,
        "reason": "",
        "drivers": [],
    }

    for label, row in (("control", control), ("cell", cell)):
        per_metric, local = decompose(row)
        real = em.calculate_composite_quality(row)
        stored = row.get("composite_quality", float("nan"))
        stored = float(stored) if stored is not None else float("nan")
        result[f"per_metric_{label}"] = per_metric
        result[f"recomputed_{label}"] = real
        result[f"stored_{label}"] = stored
        result[f"surviving_weight_{label}"] = sum(
            w for (n, w, missing) in per_metric.values() if not missing
        )
        if not _nan_equal(local, real, 1e-9):
            result["attributable"] = False
            result["reason"] = (
                f"decomposition drift ({label}): local={local!r} vs "
                f"calculate_composite_quality={real!r}"
            )
            return result
        if not _nan_equal(real, stored, tol):
            result["attributable"] = False
            result["reason"] = (
                f"round-trip mismatch ({label}): stored={stored!r} vs "
                f"recomputed={real!r} (tol={tol})"
            )
            return result

    drivers: list[dict[str, Any]] = []
    for name, _weight_attr, _kind in _COMPOSITE_CONTRIBUTORS:
        n0, w, missing_0 = result["per_metric_control"][name]
        n_l, _, missing_l = result["per_metric_cell"][name]
        weighted_delta = (
            w * (n_l - n0) if not (missing_0 or missing_l) else float("nan")
        )
        drivers.append(
            {
                "metric": name,
                "weight": w,
                "norm_control": n0,
                "norm_cell": n_l,
                "weighted_delta": weighted_delta,
                "missing_control": missing_0,
                "missing_cell": missing_l,
            }
        )
    # Finite drivers ranked by |weighted_delta| descending; missing-driven
    # (redistribution) entries follow.
    drivers.sort(
        key=lambda d: (
            0 if np.isfinite(d["weighted_delta"]) else 1,
            -abs(d["weighted_delta"]) if np.isfinite(d["weighted_delta"]) else 0.0,
        )
    )
    result["drivers"] = drivers

    # Residual between the true composite delta and the first-order
    # per-contributor sum (scaled by the control's surviving weight):
    # captures weight redistribution plus any [0, 1] clamping.
    d_comp = result["recomputed_cell"] - result["recomputed_control"]
    finite_sum = sum(
        d["weighted_delta"] for d in drivers if np.isfinite(d["weighted_delta"])
    )
    w0 = result["surviving_weight_control"]
    result["redistribution_residual"] = (
        d_comp - finite_sum / w0 if w0 > 0 else float("nan")
    )
    return result


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def md_table(rows: list[list[str]], header: list[str]) -> str:
    out = ["| " + " | ".join(header) + " |"]
    out.append("|" + "|".join(["---"] * len(header)) + "|")
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def _render_divergence_scatter(
    cells: pd.DataFrame, eps: float, guard: float, out_dir: Path
) -> Path | None:
    if cells.empty:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6.5))
    palette = {"": ("#9aa5b1", "agree / indeterminate"),
               "verdict": ("#e8930c", "verdict divergence"),
               "direction": ("#d12d2d", "direction divergence")}
    for flag, (color, label) in palette.items():
        sub = cells[cells["flag"] == flag]
        if sub.empty:
            continue
        ax.scatter(
            sub["d_ssim"], sub["d_comp"], s=18, alpha=0.65, color=color,
            label=f"{label} (n={len(sub)})", zorder=3 if flag else 2,
        )
    for v, style in ((eps, {"linestyle": ":", "alpha": 0.7}),
                     (guard, {"linestyle": "--", "alpha": 0.5})):
        for sign in (1, -1):
            ax.axvline(sign * v, color="black", linewidth=0.8, **style)
            ax.axhline(sign * v, color="black", linewidth=0.8, **style)
    lim = max(
        0.05,
        float(np.nanmax(np.abs(cells[["d_ssim", "d_comp"]].to_numpy()))) * 1.05,
    )
    ax.plot([-lim, lim], [-lim, lim], color="#4a7bd6", linewidth=0.8, alpha=0.5)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("d_ssim = ssim(L) − ssim(0)")
    ax.set_ylabel("d_comp = composite(L) − composite(0)")
    ax.set_title(
        f"SSIM vs composite paired deltas (dotted = ±EPS {eps}, "
        f"dashed = ±GUARD {guard})"
    )
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    p = out_dir / "divergence_scatter.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    return p


def _render_divergence_score_hist(
    cells: pd.DataFrame, eps: float, out_dir: Path
) -> Path | None:
    if cells.empty:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    vals = cells["divergence_score"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    ax.hist(vals, bins=40, color="#4a7bd6", edgecolor="white")
    ax.axvline(eps, color="red", linestyle=":", label=f"EPS={eps}")
    ax.axvline(
        np.median(vals), color="green", linestyle="--",
        label=f"median={np.median(vals):.4f}",
    )
    ax.set_yscale("log")
    ax.set_xlabel("divergence_score = |d_comp − d_ssim|")
    ax.set_ylabel("count (log)")
    ax.set_title(f"Continuous divergence severity, all cells (n={len(vals)})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = out_dir / "divergence_score_hist.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    return p


def _attribution_summary(attr: dict[str, Any], top_n: int = 3) -> str:
    """One-line driver summary for a flagged cell's attribution."""
    if not attr["attributable"]:
        return f"NON-ATTRIBUTABLE: {attr['reason']}"
    parts: list[str] = []
    finite = [d for d in attr["drivers"] if np.isfinite(d["weighted_delta"])]
    for d in finite[:top_n]:
        parts.append(f"{d['metric']} {d['weighted_delta']:+.4f}")
    # Redistribution drivers: contributors missing in exactly one row.
    for d in attr["drivers"]:
        if d["missing_control"] != d["missing_cell"]:
            side = "cell" if d["missing_cell"] else "control"
            parts.append(f"{d['metric']} (missing@{side}, w={d['weight']:.2f})")
    resid = attr.get("redistribution_residual", float("nan"))
    if np.isfinite(resid) and abs(resid) > 1e-4:
        parts.append(f"redistribution/clamp residual {resid:+.4f}")
    return "; ".join(parts) if parts else "no measurable drivers"


def render_divergence_report(
    sweep_df: pd.DataFrame,
    out_dir: Path,
    eps: float = DIVERGENCE_EPS,
    guard: float = DIVERGENCE_GUARD,
    control_lossy: int = 0,
    max_table_rows: int = 50,
) -> list[str]:
    """Render the standalone SSIM-vs-composite divergence report.

    Writes ``<out_dir>/ssim-composite-divergence.md`` plus its figures,
    thumbnails and ``data/divergence_cells.csv``, and returns a short
    markdown summary block for embedding in the main ``report.md``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = out_dir / "figures"
    thumbs_dir = out_dir / "thumbnails"
    data_dir = out_dir / "data"

    div = ssim_composite_divergence(
        sweep_df, eps=eps, guard=guard, control_lossy=control_lossy
    )
    cells = div["cells"]
    controls = div["controls"]
    excluded = div["excluded"]
    steps = adjacent_level_divergence(sweep_df, eps=eps)

    n_cells = len(cells)
    n_direction = int((cells["flag"] == "direction").sum()) if n_cells else 0
    n_verdict = int((cells["flag"] == "verdict").sum()) if n_cells else 0
    n_flagged = n_direction + n_verdict
    if n_cells:
        both_held = (
            (cells["d_ssim"].abs() <= eps) & (cells["d_comp"].abs() <= eps)
        ).sum()
        agree_direction = (
            (cells["d_ssim"].abs() > eps)
            & (cells["d_comp"].abs() > eps)
            & (np.sign(cells["d_ssim"]) == np.sign(cells["d_comp"]))
        ).sum()
    else:
        both_held = agree_direction = 0
    n_indeterminate = n_cells - n_flagged - int(both_held) - int(agree_direction)
    n_excluded = sum(excluded.values())

    # --- Figures ---
    scatter_fig = _render_divergence_scatter(cells, eps, guard, figures_dir)
    hist_fig = _render_divergence_score_hist(cells, eps, figures_dir)

    # --- Persist per-cell deltas for downstream consumers (gifprep rerun) ---
    data_dir.mkdir(parents=True, exist_ok=True)
    cells.to_csv(data_dir / "divergence_cells.csv", index=False)

    # --- Attribution for flagged cells ---
    flagged = cells[cells["flag"] != ""].head(max_table_rows)
    full = sweep_df.drop_duplicates(subset=["path", "lossy"], keep="first").copy()
    full["lossy"] = full["lossy"].astype(int)
    full = full.set_index(["path", "lossy"])
    attributions: dict[tuple[str, int], dict[str, Any]] = {}
    for _, cell in flagged.iterrows():
        key = (str(cell["path"]), int(cell["lossy"]))
        try:
            attributions[key] = composite_attribution(
                full.loc[(key[0], int(control_lossy))], full.loc[key]
            )
        except Exception as e:  # missing contributor columns, etc.
            attributions[key] = {
                "attributable": False,
                "reason": f"{type(e).__name__}: {e}",
                "drivers": [],
            }

    # --- Thumbnails for flagged cells ---
    thumb_for: dict[str, str] = {}
    for p_str in sorted(set(flagged["path"].astype(str))):
        rel = Path(p_str).stem.replace("/", "_") + ".png"
        if make_thumbnail(Path(p_str), thumbs_dir / rel):
            thumb_for[p_str] = f"thumbnails/{rel}"

    # --- Sensitivity at EPS x {0.5, 1, 2} (GUARD scales as 4x EPS) ---
    sensitivity_rows: list[list[str]] = []
    for factor in (0.5, 1.0, 2.0):
        e = eps * factor
        res = ssim_composite_divergence(
            sweep_df, eps=e, guard=4 * e, control_lossy=control_lossy
        )
        rc = res["cells"]
        nd = int((rc["flag"] == "direction").sum()) if len(rc) else 0
        nv = int((rc["flag"] == "verdict").sum()) if len(rc) else 0
        agree_pct = 100.0 * (1 - (nd + nv) / len(rc)) if len(rc) else float("nan")
        sensitivity_rows.append(
            [
                f"{e:.4f}",
                f"{4 * e:.4f}",
                str(nd),
                str(nv),
                f"{agree_pct:.1f}%" if np.isfinite(agree_pct) else "n/a",
            ]
        )

    # ---------- Standalone markdown ----------
    md: list[str] = []
    md.append("# SSIM vs composite_quality divergence")
    md.append("")
    md.append(
        "Control-relative paired-delta audit: each GIF's "
        f"`lossy={control_lossy}` row (animately recompress, no lossy) is its "
        "own control; per cell `d_ssim = ssim(L) − ssim(0)` and "
        "`d_comp = composite_quality(L) − composite_quality(0)`. "
        f"**EPS = {eps}** (identical to gifprep's `EPSILON_QUALITY` verdict "
        f"gate, so findings transfer 1:1 downstream), **GUARD = {guard}** "
        "(4×EPS). *Direction divergence*: deltas point in opposite "
        "directions, both beyond EPS. *Verdict divergence*: one metric held "
        "(|d| ≤ EPS) while the other moved beyond GUARD. The zone between "
        "EPS and GUARD is a deliberate indeterminate band — never flagged. "
        "`divergence_score = |d_comp − d_ssim|` is computed for every "
        "analysable cell (continuous severity; cannot tie-saturate). Cells "
        "with NaN in either metric in either row are excluded and counted, "
        "never imputed. Run provenance: `data/sweep_run.log`."
    )
    md.append("")
    md.append("## Summary")
    md.append("")
    md.append(f"- Analysable cells: **{n_cells}** "
              f"(GIFs with usable control: {len(controls)})")
    md.append(f"- Direction-divergent: **{n_direction}**")
    md.append(f"- Verdict-divergent (held-vs-degraded): **{n_verdict}**")
    md.append(f"- Same direction beyond EPS: {int(agree_direction)}")
    md.append(f"- Both held (|d| ≤ EPS): {int(both_held)}")
    md.append(f"- Indeterminate (sub-GUARD movement): {n_indeterminate}")
    md.append(
        f"- Unanalysable cells (excluded, never imputed): {n_excluded} — "
        f"no control: {excluded['no_control']}, NaN control: "
        f"{excluded['nan_control']}, NaN cell: {excluded['nan_cell']}"
    )
    if n_cells and not n_flagged:
        md.append(
            "- **Zero flagged cells at this EPS — composite tracks SSIM "
            "direction on 100% of analysable cells.** (Itself a reportable "
            "finding; see sensitivity below.)"
        )
    md.append("")
    md.append("### Sensitivity (EPS × {0.5, 1, 2}, GUARD = 4×EPS)")
    md.append("")
    md.append(
        md_table(
            sensitivity_rows,
            ["eps", "guard", "direction", "verdict", "agreement rate"],
        )
    )
    md.append("")

    if scatter_fig is not None:
        md.append(f"![divergence scatter](figures/{scatter_fig.name})")
        md.append("")
    if hist_fig is not None:
        md.append(f"![divergence score histogram](figures/{hist_fig.name})")
        md.append("")

    md.append("## Flagged cells (ranked by divergence_score)")
    md.append("")
    if flagged.empty:
        md.append("None at this EPS/GUARD.")
        md.append("")
    else:
        md.append(
            "Per-cell attribution decomposes the composite movement into its "
            "15 named contributor terms `w_i·(norm_i(L) − norm_i(0))`; "
            "contributors missing in exactly one row are listed as "
            "redistribution drivers. Every attribution is round-trip "
            "verified against the production `calculate_composite_quality` — "
            "mismatches are marked NON-ATTRIBUTABLE, never silently trusted. "
            "`frame_class` annotates frame-reduction: `frame_loss` cells "
            "carry an alignment caveat (frame misalignment can move SSIM and "
            "temporal-delta differently — treat the deltas with suspicion)."
        )
        md.append("")
        rows = []
        for _, r in flagged.iterrows():
            attr = attributions[(str(r["path"]), int(r["lossy"]))]
            thumb = thumb_for.get(str(r["path"]), "")
            rows.append(
                [
                    f"![]({thumb})" if thumb else "",
                    Path(str(r["path"])).name[:48],
                    str(r["lossy"]),
                    str(r["source"]),
                    str(r["content_type"]),
                    str(r["frame_reduction_class"]),
                    f"{r['d_ssim']:+.4f}",
                    f"{r['d_comp']:+.4f}",
                    f"{r['divergence_score']:.4f}",
                    str(r["flag"]),
                    _attribution_summary(attr),
                ]
            )
        md.append(
            md_table(
                rows,
                [
                    "thumb", "gif", "lossy", "source", "content_type",
                    "frame_class", "d_ssim", "d_comp", "score", "flag",
                    "top composite drivers",
                ],
            )
        )
        md.append("")
        n_total_flagged = int((cells["flag"] != "").sum())
        if n_total_flagged > max_table_rows:
            md.append(
                f"(Showing top {max_table_rows} of {n_total_flagged} flagged "
                "cells; the full set is in `data/divergence_cells.csv`.)"
            )
            md.append("")

    md.append("## Adjacent-level monotonicity disagreement")
    md.append("")
    md.append(
        "sign(Δssim) vs sign(Δcomposite) over consecutive lossy levels — "
        "catches \"composite recovers while SSIM keeps falling\" shapes that "
        "the control-relative deltas can smooth over."
    )
    md.append("")
    flagged_steps = steps[steps["sign_disagree"]].sort_values(
        "divergence_score", ascending=False
    )
    if flagged_steps.empty:
        md.append(
            f"No adjacent-level sign disagreement beyond EPS={eps} across "
            f"{len(steps)} steps."
        )
        md.append("")
    else:
        rows = [
            [
                Path(str(r["path"])).name[:48],
                f"{r['lossy_from']}→{r['lossy_to']}",
                str(r["source"]),
                str(r["content_type"]),
                f"{r['d_ssim']:+.4f}",
                f"{r['d_comp']:+.4f}",
            ]
            for _, r in flagged_steps.head(20).iterrows()
        ]
        md.append(
            md_table(
                rows,
                ["gif", "step", "source", "content_type", "Δssim", "Δcomposite"],
            )
        )
        md.append("")
        if len(flagged_steps) > 20:
            md.append(f"(Top 20 of {len(flagged_steps)} disagreeing steps.)")
            md.append("")

    md.append("## Identity-relative divergence at the control point")
    md.append("")
    md.append(
        "Both metrics have identity value exactly 1.0 (sanity.json), so "
        f"comparing `1 − ssim({control_lossy})` vs "
        f"`1 − composite({control_lossy})` exposes palette-axis divergence "
        "(lossy=0 still palette-quantises) that the lossy-delta framing "
        "hides. `identity_gap = composite(0) − ssim(0)`; large |gap| means "
        "the two metrics already disagree about the palette/recompress step."
    )
    md.append("")
    if controls.empty:
        md.append("No usable control rows.")
        md.append("")
    else:
        top_controls = controls.reindex(
            controls["identity_gap"].abs().sort_values(ascending=False).index
        ).head(10)
        rows = [
            [
                Path(str(r["path"])).name[:48],
                str(r["source"]),
                str(r["content_type"]),
                f"{r['ssim_control']:.4f}",
                f"{r['comp_control']:.4f}",
                f"{r['identity_gap']:+.4f}",
            ]
            for _, r in top_controls.iterrows()
        ]
        md.append(
            md_table(
                rows,
                ["gif", "source", "content_type", "ssim(0)", "composite(0)",
                 "identity_gap"],
            )
        )
        md.append("")

    md.append("---")
    md.append("")
    md.append(
        "Generated by `scripts/audit/report.py` (divergence pass). "
        "Per-cell deltas: `data/divergence_cells.csv`."
    )
    (out_dir / "ssim-composite-divergence.md").write_text("\n".join(md))
    print(f"[report] Wrote {out_dir / 'ssim-composite-divergence.md'}")

    # ---------- Summary block for the main report ----------
    summary: list[str] = []
    summary.append("### SSIM vs composite_quality divergence")
    summary.append("")
    summary.append(
        f"Control-relative paired deltas vs each GIF's lossy={control_lossy} "
        f"row (EPS={eps}, GUARD={guard}): **{n_direction} direction-divergent"
        f"** and **{n_verdict} verdict-divergent** of {n_cells} analysable "
        f"cells ({n_excluded} unanalysable, excluded and counted). Full pass: "
        "[ssim-composite-divergence.md](ssim-composite-divergence.md)."
    )
    summary.append("")
    return summary


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
        rows = [[str(lv), f"{norm[lv]:.3f}"] for lv in sorted(norm.keys())]
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
        "(noise / blur / quantize / animately lossy). SUSPICIOUS if a "
        "**pairwise-quality** metric shows an inversion. DIAGNOSTIC if a "
        "structurally-non-monotonic-by-design metric (dispersion `_std`/`_min`/"
        "`_max` siblings, single-stream `_compressed` keys, or diagnostic/"
        "system metrics like render_ms / kilobytes / efficiency) shows an "
        "inversion — real but expected, so segregated from SUSPICIOUS. "
        "INCONCLUSIVE if identity == pathological (the pair can't discriminate)."
    )
    md.append("")
    verdicts = sanity.get("verdicts", [])
    _counts: dict[str, int] = {}
    for _v in verdicts:
        _counts[_v["verdict"]] = _counts.get(_v["verdict"], 0) + 1
    md.append(
        "**Verdict summary:** "
        + ", ".join(
            f"{_counts.get(k, 0)} {k}"
            for k in ("PASS", "SUSPICIOUS", "DIAGNOSTIC", "INCONCLUSIVE")
        )
        + ". Only SUSPICIOUS (pairwise-quality inversions) warrants "
        "investigation; DIAGNOSTIC metrics are non-monotonic by design."
    )
    md.append("")
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

    # SSIM-vs-composite divergence pass — only when the sweep carries
    # control rows at lossy=0 (graceful absence otherwise, same pattern as
    # frame_reduction_summary; older sweeps without a lossy-0 grid simply
    # skip the section).
    if "lossy" in sweep_df.columns and (sweep_df["lossy"].astype(int) == 0).any():
        print("[report] Rendering SSIM-vs-composite divergence pass...", flush=True)
        md.extend(render_divergence_report(sweep_df, out_dir))

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
            "is mis-firing on this kind of content. Ranks are tie-aware "
            "(tied values share their average rank) and only pairwise-quality "
            "metrics vote — single-stream `_compressed` keys are excluded. "
            "The absolute best/worst values are shown so range-compressed "
            "metrics (rank extreme, near-perfect absolute value) are visible."
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
                    f"{r['best_metric']} ({r['best_value']:.4f})",
                    f"{r['worst_metric']} ({r['worst_value']:.4f})",
                ]
            )
        md.append(md_table(rows, ["thumb", "gif", "lossy", "source", "content_type", "spread", "best→ (value)", "worst→ (value)"]))
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
    # --sanity/--pilot/--decisions are required for the full report but
    # optional under --divergence-only, which runs the SSIM-vs-composite
    # divergence pass on the sweep CSV alone.
    ap.add_argument("--sanity", type=Path)
    ap.add_argument("--pilot", type=Path)
    ap.add_argument("--decisions", type=Path)
    ap.add_argument("--sweep", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument(
        "--divergence-only",
        action="store_true",
        help="Render only the SSIM-vs-composite divergence pass "
        "(ssim-composite-divergence.md + figures) from --sweep; "
        "--sanity/--pilot/--decisions are not needed.",
    )
    ap.add_argument(
        "--eps",
        type=float,
        default=DIVERGENCE_EPS,
        help="Divergence noise band (default mirrors gifprep's "
        "EPSILON_QUALITY = 0.005)",
    )
    ap.add_argument(
        "--guard",
        type=float,
        default=DIVERGENCE_GUARD,
        help="Held-vs-degraded guard band (default 0.02 = 4x EPS)",
    )
    ap.add_argument(
        "--control-lossy",
        type=int,
        default=0,
        help="Lossy level used as each GIF's control row (default 0)",
    )
    args = ap.parse_args()

    sweep_df = load_sweep(args.sweep)

    if args.divergence_only:
        render_divergence_report(
            sweep_df,
            args.output,
            eps=args.eps,
            guard=args.guard,
            control_lossy=args.control_lossy,
        )
        return 0

    missing = [
        f"--{name}"
        for name in ("sanity", "pilot", "decisions")
        if getattr(args, name) is None
    ]
    if missing:
        ap.error(
            f"{', '.join(missing)} required unless --divergence-only is given"
        )

    sanity = load_sanity(args.sanity)
    pilot_df = load_pilot(args.pilot)
    decisions = json.loads(args.decisions.read_text())

    render_report(sanity, pilot_df, decisions, sweep_df, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
