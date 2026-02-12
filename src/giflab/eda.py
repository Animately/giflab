"""EDA utilities to generate exploratory artefacts from metrics CSV.

This module produces basic visual diagnostics:
1. Histograms for each numeric metric (up to *max_columns*)
2. Correlation heat-map (Pearson)
3. PCA scree plot (explained variance)

The artefacts are saved as PNG files inside *output_dir* and can be
embedded in notebooks or HTML reports. The function avoids heavy
interactive dependencies (sets matplotlib Agg backend).
"""
from __future__ import annotations

import warnings
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Headless backend for CI / servers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Lazy import sklearn to reduce startup time
# PCA will be imported only when generate_eda is called with PCA analysis needed

__all__ = ["generate_eda"]


def _numeric_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include=[np.number]).columns.tolist()


def _sanitize_filename(name: str) -> str:
    """Sanitize a column name to be safe for use as a filename."""
    import re

    # Replace problematic characters with underscores
    safe_name = re.sub(r"[^\w\-_.]", "_", name)
    # Remove consecutive underscores
    safe_name = re.sub(r"_+", "_", safe_name)
    # Remove leading/trailing underscores
    safe_name = safe_name.strip("_")
    # Ensure it's not empty
    if not safe_name:
        safe_name = "metric"
    return safe_name


def generate_eda(
    csv_path: Path | str,
    output_dir: Path | str,
    max_columns: int = 20,
) -> dict[str, Path]:
    """Generate EDA artefacts for the metrics CSV.

    Returns a mapping of artefact names → file paths.
    """
    csv_path = Path(csv_path)
    output_dir = Path(output_dir)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    num_cols = _numeric_columns(df)
    if not num_cols:
        raise ValueError("CSV contains no numeric columns to analyse")

    artefacts: dict[str, Path] = {}

    # Limit number of columns for histogram generation
    sel_cols: Sequence[str] = num_cols[:max_columns]

    # Histograms
    for col in sel_cols:
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.hist(df[col].dropna(), bins=30, color="steelblue", alpha=0.8)
        ax.set_title(f"Histogram: {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")
        safe_col_name = _sanitize_filename(col)
        file_path = output_dir / f"hist_{safe_col_name}.png"
        fig.tight_layout()
        fig.savefig(file_path, dpi=150)
        plt.close(fig)
        artefacts[f"hist_{safe_col_name}"] = file_path

    # Correlation heatmap
    if len(num_cols) >= 2:
        corr = df[num_cols].corr(method="pearson")
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(num_cols)))
        ax.set_xticklabels(num_cols, rotation=90, fontsize=6)
        ax.set_yticks(range(len(num_cols)))
        ax.set_yticklabels(num_cols, fontsize=6)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title("Correlation Heat-map (Pearson)")
        fig.tight_layout()
        heat_path = output_dir / "correlation_heatmap.png"
        fig.savefig(heat_path, dpi=150)
        plt.close(fig)
        artefacts["correlation_heatmap"] = heat_path

    # PCA scree plot
    try:
        # Import PCA only when needed (lazy loading for performance)
        from sklearn.decomposition import PCA

        pca = PCA()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            pca.fit(df[num_cols].fillna(0))
        explained = pca.explained_variance_ratio_
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.plot(range(1, len(explained) + 1), explained, marker="o", lw=1.5)
        ax.set_xlabel("Principal Component")
        ax.set_ylabel("Explained Variance Ratio")
        ax.set_title("PCA Scree Plot")
        ax.set_ylim(0, 1)
        fig.tight_layout()
        scree_path = output_dir / "pca_scree_plot.png"
        fig.savefig(scree_path, dpi=150)
        plt.close(fig)
        artefacts["pca_scree_plot"] = scree_path
    except Exception as exc:
        # Do not fail the whole pipeline if PCA fails (e.g., singular matrix)
        warnings.warn(f"PCA scree plot generation failed: {exc}", stacklevel=2)

    return artefacts
