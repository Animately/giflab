"""Phase 1 of the metrics audit — sanity-check each metric on controlled inputs.

Three tests per metric:
- Identity: compare a GIF against itself. Records the metric's ceiling/floor value.
- Pathological: compare a white GIF against a black GIF of matching shape.
  Records the metric's worst-case value.
- Monotonicity: take a base GIF, apply 4 ordered degradation levels of each
  kind (noise, blur, quantize, animately lossy), check that the metric ordering
  matches the (identity -> pathological) direction with no inversions.

Output:
- JSON file with raw values + per-metric verdict.
- Terminal table summarising verdicts.

Usage:
    poetry run python scripts/audit/sanity.py --output sanity_results.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter

# Project root: scripts/audit/sanity.py → scripts/audit/ → scripts/ → project root
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_FIXTURES_DIR = _PROJECT_ROOT / "tests" / "fixtures"

# Lazy giflab imports (heavy)
def _import_giflab():
    from giflab.config import MetricsConfig
    from giflab.metrics import calculate_comprehensive_metrics
    from giflab.public_api import compress
    from giflab.synthetic_gifs import SyntheticGifGenerator

    return {
        "MetricsConfig": MetricsConfig,
        "calculate_comprehensive_metrics": calculate_comprehensive_metrics,
        "compress": compress,
        "SyntheticGifGenerator": SyntheticGifGenerator,
    }


# Base GIFs used for monotonicity (spec.name values from synthetic_gifs.py).
# Picked to span different content responses: smooth gradient (color-sensitive),
# noise/texture (high-frequency), high_contrast (edge-sensitive), geometric
# (structure-sensitive), motion (temporal). Falls back gracefully if a spec
# isn't present.
MONOTONICITY_BASES = [
    "smooth_gradient",
    "photographic_noise",
    "high_contrast",
    "geometric_patterns",
    "animation_heavy",
]

# Additional fixture-based monotonicity bases.  These are deterministically
# generated GIFs (via ``make fixtures`` / ``scripts/fixtures/generate.py``)
# that guard specific regression classes discovered in the audit.  They are
# loaded by name from ``tests/fixtures/`` and injected into the monotonicity
# loop alongside the synthetic specs so they exercise the same degradation
# sequences (noise, blur, quantize, lossy).
#
# - transparency_bearing: guards PR #8's alpha-compositing bug.
#   A palette-P GIF with a ``transparency`` index in its header.  With the
#   buggy ``Image.convert('RGB')`` path, transparent regions resolve to a
#   palette-dependent colour that changes across re-encodings, corrupting
#   every pair-comparison metric.  Running degradation sequences through this
#   fixture means a regression would produce palette-noise-driven inversions
#   that the monotonicity check flags as SUSPICIOUS.
FIXTURE_MONOTONICITY_BASES: dict[str, str] = {
    # logical name → filename in tests/fixtures/
    "transparency_bearing": "transparency_bearing_monotonicity.gif",
}

# Identity sample: 1 GIF per major content category, picked to keep runtime
# reasonable. The sanity verdict only needs a handful to detect "metric
# returns 0 on identity" style bugs.
IDENTITY_SAMPLE = [
    "smooth_gradient",
    "high_contrast",
    "photographic_noise",
    "geometric_patterns",
    "solid_blocks",
]

# Degradation grids. Wider ranges than typical so we exercise the metric's
# response across its full dynamic range.
NOISE_SIGMAS = [5, 15, 30, 60]
BLUR_SIGMAS = [0.5, 1.5, 3.0, 6.0]
QUANTIZE_COLORS = [256, 64, 16, 4]
# animately's --lossy parameter saturates earlier than the documented [0, 200]
# range on low-complexity content. Empirically, on the smooth_gradient synthetic
# the byte output stabilises around lossy ~125 and stays bit-identical at
# higher levels; on high_contrast / geometric_patterns / animation_heavy it
# already saturates by lossy ~100. The grid still spans the engine's effective
# range for noisy content (photographic_noise keeps shrinking past lossy=200),
# so the saturation only affects a subset of bases.
#
# Audit-pipeline gotcha (2026-05-22 audit, smooth_gradient): when two
# consecutive lossy grid points both land past saturation but produce slightly
# different byte outputs (e.g. animately 100 vs 130), their local-window
# similarity scores (ssim, ms_ssim, mse_max, psnr_min, etc.) can differ by
# fractions of a percent — sometimes in the "improving" direction. The
# monotonicity check then flags a SUSPICIOUS verdict that is actually benign:
# the metric is faithfully reporting a real difference between two compressed
# outputs neither of which corresponds to additional information loss.
# This is documented in src/giflab/metrics.py near the SSIM clamp site.
LOSSY_LEVELS = [20, 60, 100, 160]


# ---------------------------------------------------------------------------
# GIF I/O helpers
# ---------------------------------------------------------------------------


def read_gif_frames(path: Path) -> tuple[list[Image.Image], int]:
    """Return (frames as RGB PIL Images, frame duration in ms)."""
    frames: list[Image.Image] = []
    duration = 100
    with Image.open(path) as im:
        duration = int(im.info.get("duration", 100))
        try:
            while True:
                frames.append(im.convert("RGB").copy())
                im.seek(im.tell() + 1)
        except EOFError:
            pass
    return frames, duration


def save_gif(frames: list[Image.Image], path: Path, duration_ms: int = 100) -> None:
    if not frames:
        raise ValueError(f"No frames to save for {path}")
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )


# ---------------------------------------------------------------------------
# Degradation helpers
# ---------------------------------------------------------------------------


def add_gaussian_noise(frames: list[Image.Image], sigma: float, rng: np.random.Generator) -> list[Image.Image]:
    out = []
    for f in frames:
        arr = np.asarray(f, dtype=np.float32)
        noise = rng.normal(0.0, sigma, arr.shape).astype(np.float32)
        noised = np.clip(arr + noise, 0, 255).astype(np.uint8)
        out.append(Image.fromarray(noised, mode="RGB"))
    return out


def gaussian_blur(frames: list[Image.Image], sigma: float) -> list[Image.Image]:
    return [f.filter(ImageFilter.GaussianBlur(radius=sigma)) for f in frames]


def quantize_palette(frames: list[Image.Image], n_colors: int) -> list[Image.Image]:
    # Quantize each frame independently to a `n_colors`-entry palette, then
    # convert back to RGB for consistent downstream handling.
    out = []
    for f in frames:
        q = f.quantize(colors=n_colors, method=Image.Quantize.MEDIANCUT)
        out.append(q.convert("RGB"))
    return out


def make_solid_gif(color: tuple[int, int, int], size: tuple[int, int], frames: int, path: Path) -> None:
    imgs = [Image.new("RGB", size, color=color) for _ in range(frames)]
    save_gif(imgs, path)


# ---------------------------------------------------------------------------
# Metric runner
# ---------------------------------------------------------------------------


def run_metrics(
    original: Path, compressed: Path, *, gl: dict[str, Any]
) -> dict[str, float]:
    """Run the comprehensive metrics suite and return a flat float dict."""
    config = gl["MetricsConfig"]()
    config.ENABLE_DEEP_PERCEPTUAL = True
    config.ENABLE_TEMPORAL_ARTIFACTS = True
    raw = gl["calculate_comprehensive_metrics"](
        original, compressed, config=config, force_all_metrics=True
    )
    return {k: float(v) for k, v in raw.items() if isinstance(v, (int, float))}


# ---------------------------------------------------------------------------
# Test orchestration
# ---------------------------------------------------------------------------


@dataclass
class PerMetricVerdict:
    metric: str
    identity_mean: float
    identity_std: float
    pathological: float
    direction: str  # "higher_better", "lower_better", "flat"
    monotonicity_failures: list[dict[str, Any]]
    verdict: str  # "PASS", "SUSPICIOUS", "INCONCLUSIVE"
    note: str = ""


def _gif_path_by_name(gifs: list[Path], name: str) -> Path | None:
    for p in gifs:
        if p.stem == name:
            return p
    return None


def _direction(identity_v: float, pathological_v: float, tol: float = 1e-3) -> str:
    """Infer the direction of 'better' for a metric.

    Pathological is, by construction, the worse comparison. If its value is
    lower than identity, lower-numbers-equal-worse, i.e. higher-is-better.
    If higher, the opposite.
    """
    delta = pathological_v - identity_v
    if abs(delta) < tol:
        return "flat"
    return "higher_better" if delta < 0 else "lower_better"


def _range_derived_tol(values: list[float], tol_floor: float = 1e-4, relative_eps: float = 0.05) -> float:
    """Compute a content-aware monotonicity tolerance from the observed range.

    A fixed tolerance is too tight for metrics that vary over a wide range
    (e.g. MSE on noisy content) and too loose for metrics that barely move
    at all (e.g. chist on non-gradient content).  Scaling by the observed
    range lets the threshold track the metric's effective resolution.

    Formula: ``max(tol_floor, observed_range * relative_eps)``

    - ``tol_floor`` (default 1e-4): minimum epsilon regardless of range, so
      metrics that return identical values don't produce a zero tolerance.
    - ``relative_eps`` (default 0.05): 5% of the observed range is allowed
      as noise before an inversion is flagged.  Inversions larger than this
      are treated as real non-monotonicity rather than noise-floor artefacts.

    Args:
        values: The ordered metric values for a single degradation sweep.
            NaN values are excluded from the range calculation.
        tol_floor: Minimum tolerance (absolute).  Defaults to 1e-4.
        relative_eps: Fraction of the observed range to use as tolerance.
            Defaults to 0.05 (5%).

    Returns:
        Content-aware tolerance >= tol_floor.
    """
    finite = [v for v in values if not math.isnan(v)]
    if len(finite) < 2:
        return tol_floor
    observed_range = max(finite) - min(finite)
    return max(tol_floor, observed_range * relative_eps)


def _monotonicity_check(
    values: list[float], direction: str, tol: float | None = None
) -> list[tuple[int, float, float]]:
    """Return inversions: (level_idx, value_at_idx, value_at_idx+1) where the
    pair is worse-ordered than expected given the direction. Empty list = OK.

    direction "higher_better" expects values to be non-increasing as severity
    increases. "lower_better" expects non-decreasing. "flat" means we can't
    really test monotonicity — return empty so it doesn't false-flag.

    Tolerance is content-aware by default: ``tol`` is derived from the
    observed value range via :func:`_range_derived_tol` (``max(1e-4,
    range * 0.05)``).  This prevents metrics that are saturated near their
    ceiling from accumulating false SUSPICIOUS verdicts due to noise-floor
    flips that are well within the metric's effective resolution.

    Pass an explicit ``tol`` to override the range-derived value (e.g. in
    tests that verify the raw check logic against a known epsilon).
    """
    if direction == "flat":
        return []
    effective_tol = tol if tol is not None else _range_derived_tol(values)
    inversions: list[tuple[int, float, float]] = []
    for i in range(len(values) - 1):
        a, b = values[i], values[i + 1]
        if any(math.isnan(x) for x in (a, b)):
            continue
        if direction == "higher_better":
            # Expect b <= a + effective_tol
            if b > a + effective_tol:
                inversions.append((i, a, b))
        else:  # lower_better
            if b < a - effective_tol:
                inversions.append((i, a, b))
    return inversions


def _coalesce_byte_identical_levels(
    paths: list[Path],
    per_metric: dict[str, list[float]],
) -> tuple[list[Path], dict[str, list[float]]]:
    """Remove consecutive byte-identical entries from a lossy compression sweep.

    Animately's --lossy parameter saturates around level ~125 on low-complexity
    content, producing literally identical compressed bytes past the saturation
    knee. Monotonicity testing on such a sequence is spurious — identical bytes
    produce identical (or floating-point-noise-bumped) metric values, which can
    trigger false inversions or false positives.

    Only *consecutive* duplicates are coalesced: if bytes A appear at levels
    [0, 2, 3] with a different content B at level 1, levels 0 and 2 are kept
    (they are not adjacent), while level 3 is dropped (it repeats level 2).

    Args:
        paths: Ordered list of compressed-output paths for each lossy level.
        per_metric: Dict mapping metric name to a list of float values aligned
            with ``paths`` (same length and same order).

    Returns:
        (kept_paths, kept_per_metric) with consecutive byte-identical entries
        removed. The first occurrence of each run is always kept.
    """
    if not paths:
        return [], {}

    kept_indices: list[int] = [0]
    prev_bytes = paths[0].read_bytes()

    for i in range(1, len(paths)):
        current_bytes = paths[i].read_bytes()
        if current_bytes != prev_bytes:
            kept_indices.append(i)
            prev_bytes = current_bytes
        # else: consecutive byte-identical — skip

    kept_paths = [paths[i] for i in kept_indices]
    kept_per_metric = {
        metric: [vals[i] for i in kept_indices]
        for metric, vals in per_metric.items()
    }
    return kept_paths, kept_per_metric


def run_sanity(workdir: Path, *, skip_lossy: bool = False) -> dict[str, Any]:
    gl = _import_giflab()
    gen_dir = workdir / "synthetic"
    gen_dir.mkdir(parents=True, exist_ok=True)

    print(f"[sanity] Generating targeted synthetic GIFs into {gen_dir}", flush=True)
    gen = gl["SyntheticGifGenerator"](gen_dir)
    gifs = gen.generate_gifs(use_targeted_set=True)
    name_to_spec = {s.name: s for s in gen.synthetic_specs}
    print(f"[sanity] Generated {len(gifs)} GIFs", flush=True)

    # ---- Identity ----
    print("[sanity] Running identity tests...", flush=True)
    identity_results: dict[str, list[float]] = defaultdict(list)
    identity_per_gif: dict[str, dict[str, float]] = {}
    for name in IDENTITY_SAMPLE:
        path = _gif_path_by_name(gifs, name)
        if path is None:
            print(f"[sanity]   skip identity {name}: not in targeted set", flush=True)
            continue
        m = run_metrics(path, path, gl=gl)
        identity_per_gif[name] = m
        for k, v in m.items():
            identity_results[k].append(v)

    # All metric keys observed
    metric_keys = sorted(identity_results.keys())
    print(f"[sanity] Observed {len(metric_keys)} metric keys", flush=True)

    # ---- Pathological (two pairs to exercise both color and structure metrics) ----
    print("[sanity] Running pathological (solid: white vs black)...", flush=True)
    patho_dir = workdir / "patho"
    patho_dir.mkdir(exist_ok=True)
    white = patho_dir / "white.gif"
    black = patho_dir / "black.gif"
    size = (96, 96)
    frames = 4
    make_solid_gif((255, 255, 255), size, frames, white)
    make_solid_gif((0, 0, 0), size, frames, black)
    patho_solid = run_metrics(white, black, gl=gl)

    # Structure-rich pathological pair: a synthetic gradient GIF and its
    # pixel-wise inverse (255 - pixel). Every pixel differs maximally but
    # local structure is preserved — should drive structural metrics
    # (fsim, gmsd, edge_similarity, texture_similarity) away from identity.
    print("[sanity] Running pathological (structural: gradient vs inverse)...", flush=True)
    grad_path = _gif_path_by_name(gifs, "smooth_gradient")
    patho_structural: dict[str, float] = {}
    if grad_path is not None:
        grad_frames, dur = read_gif_frames(grad_path)
        inverted_frames = [
            Image.fromarray(255 - np.asarray(f, dtype=np.uint8), mode="RGB")
            for f in grad_frames
        ]
        inv_path = patho_dir / "smooth_gradient_inverse.gif"
        save_gif(inverted_frames, inv_path, duration_ms=dur)
        patho_structural = run_metrics(grad_path, inv_path, gl=gl)
    else:
        print("[sanity]   skipping structural pathological: smooth_gradient missing", flush=True)

    # Pick the more useful pathological value per metric: the one with
    # the bigger delta from identity. That's the one that actually
    # discriminates this metric.
    patho_metrics: dict[str, float] = {}
    patho_source: dict[str, str] = {}
    for k in set(patho_solid) | set(patho_structural):
        ident = float(np.mean(identity_results.get(k, [float("nan")]))) if identity_results.get(k) else float("nan")
        solid_v = patho_solid.get(k, float("nan"))
        struct_v = patho_structural.get(k, float("nan"))
        solid_delta = abs(solid_v - ident) if not (math.isnan(solid_v) or math.isnan(ident)) else 0.0
        struct_delta = abs(struct_v - ident) if not (math.isnan(struct_v) or math.isnan(ident)) else 0.0
        if struct_delta > solid_delta:
            patho_metrics[k] = struct_v
            patho_source[k] = "structural"
        else:
            patho_metrics[k] = solid_v
            patho_source[k] = "solid"

    # ---- Monotonicity ----
    print("[sanity] Running monotonicity tests...", flush=True)
    rng = np.random.default_rng(seed=42)

    # results[(degradation_kind, base_name)] -> {metric: [v_level0, v_level1, v_level2, v_level3]}
    monotonicity: dict[tuple[str, str], dict[str, list[float]]] = {}

    for base_name in MONOTONICITY_BASES:
        base_path = _gif_path_by_name(gifs, base_name)
        if base_path is None:
            print(f"[sanity]   skip monotonicity {base_name}: not in targeted set", flush=True)
            continue
        print(f"[sanity]   base: {base_name}", flush=True)
        base_frames, dur = read_gif_frames(base_path)

        # ---- Noise ----
        kind = "noise"
        per_metric_levels: dict[str, list[float]] = defaultdict(list)
        for sigma in NOISE_SIGMAS:
            degraded_path = workdir / f"{base_name}_noise_{sigma}.gif"
            degraded = add_gaussian_noise(base_frames, sigma, rng)
            save_gif(degraded, degraded_path, duration_ms=dur)
            m = run_metrics(base_path, degraded_path, gl=gl)
            for k, v in m.items():
                per_metric_levels[k].append(v)
        monotonicity[(kind, base_name)] = dict(per_metric_levels)

        # ---- Blur ----
        kind = "blur"
        per_metric_levels = defaultdict(list)
        for sigma in BLUR_SIGMAS:
            degraded_path = workdir / f"{base_name}_blur_{sigma}.gif"
            degraded = gaussian_blur(base_frames, sigma)
            save_gif(degraded, degraded_path, duration_ms=dur)
            m = run_metrics(base_path, degraded_path, gl=gl)
            for k, v in m.items():
                per_metric_levels[k].append(v)
        monotonicity[(kind, base_name)] = dict(per_metric_levels)

        # ---- Quantize ----
        kind = "quantize"
        per_metric_levels = defaultdict(list)
        for ncolors in QUANTIZE_COLORS:
            degraded_path = workdir / f"{base_name}_q_{ncolors}.gif"
            degraded = quantize_palette(base_frames, ncolors)
            save_gif(degraded, degraded_path, duration_ms=dur)
            m = run_metrics(base_path, degraded_path, gl=gl)
            for k, v in m.items():
                per_metric_levels[k].append(v)
        monotonicity[(kind, base_name)] = dict(per_metric_levels)

        # ---- Real lossy ----
        if not skip_lossy:
            kind = "lossy"
            per_metric_levels = defaultdict(list)
            # Track paths in the same order as per_metric_levels so we can
            # coalesce consecutive byte-identical outputs (animately's --lossy
            # saturates around level ~125 on low-complexity content, producing
            # literally identical bytes which would otherwise generate spurious
            # monotonicity inversions from floating-point noise).
            degraded_paths: list[Path] = []
            for level in LOSSY_LEVELS:
                degraded_path = workdir / f"{base_name}_lossy_{level}.gif"
                try:
                    # apply_content_ceiling=False: this IS the monotonicity
                    # lossy sweep that calibrates the ceilings. Clamping
                    # photographic bases (smooth_gradient, photographic_noise)
                    # down to 20/30 would collapse levels 60/100/160 into
                    # byte-identical outputs and degenerate the curve to a
                    # single point. See scripts/audit/_common.py docstring.
                    gl["compress"](
                        base_path,
                        degraded_path,
                        "animately",
                        {"lossy_level": level},
                        apply_content_ceiling=False,
                    )
                except Exception as e:
                    print(f"[sanity]     animately --lossy {level} failed on {base_name}: {e}", flush=True)
                    continue
                m = run_metrics(base_path, degraded_path, gl=gl)
                for k, v in m.items():
                    per_metric_levels[k].append(v)
                degraded_paths.append(degraded_path)
            if per_metric_levels:
                # Drop consecutive byte-identical outputs before storing for
                # monotonicity check — see _coalesce_byte_identical_levels.
                _, coalesced_metrics = _coalesce_byte_identical_levels(
                    degraded_paths, dict(per_metric_levels)
                )
                dropped = len(degraded_paths) - len(
                    next(iter(coalesced_metrics.values()), [])
                )
                if dropped > 0:
                    print(
                        f"[sanity]     coalesced {dropped} byte-identical lossy level(s) on {base_name}",
                        flush=True,
                    )
                monotonicity[(kind, base_name)] = coalesced_metrics

    # ---- Fixture-based monotonicity bases (FIXTURE_MONOTONICITY_BASES) ----
    # These are pre-generated regression-guard fixtures from tests/fixtures/.
    # They exercise specific bug classes (e.g. alpha-compositing) that synthetic
    # specs don't cover.  Run the same noise/blur/quantize/lossy sequences.
    #
    # The canonical copies of these fixtures are COMMITTED to tests/fixtures/
    # (with a .gitignore exception) so the sanity check has a real regression
    # guard even on a fresh clone where `make fixtures` has not been run.
    #
    # ``skipped_fixture_checks`` surfaces any fixture that could not be loaded
    # (file or directory absent) so downstream CI / reviewers can gate on it
    # rather than silently treating "no monotonicity checks ran" as "all PASS".
    skipped_fixture_checks: list[dict[str, str]] = []
    if FIXTURE_MONOTONICITY_BASES and _FIXTURES_DIR.exists():
        for logical_name, filename in FIXTURE_MONOTONICITY_BASES.items():
            base_path = _FIXTURES_DIR / filename
            if not base_path.exists():
                msg = (
                    f"{base_path} not found — fixture should be committed in-tree; "
                    "run `make fixtures` if it was deleted"
                )
                print(
                    f"[sanity]   skip fixture monotonicity {logical_name}: {msg}",
                    flush=True,
                )
                skipped_fixture_checks.append(
                    {"logical_name": logical_name, "reason": msg}
                )
                continue
            print(f"[sanity]   fixture base: {logical_name} ({filename})", flush=True)
            base_frames, dur = read_gif_frames(base_path)

            # Noise
            kind = "noise"
            per_metric_levels = defaultdict(list)
            for sigma in NOISE_SIGMAS:
                degraded_path = workdir / f"{logical_name}_noise_{sigma}.gif"
                degraded = add_gaussian_noise(base_frames, sigma, rng)
                save_gif(degraded, degraded_path, duration_ms=dur)
                m = run_metrics(base_path, degraded_path, gl=gl)
                for k, v in m.items():
                    per_metric_levels[k].append(v)
            monotonicity[(kind, logical_name)] = dict(per_metric_levels)

            # Blur
            kind = "blur"
            per_metric_levels = defaultdict(list)
            for sigma in BLUR_SIGMAS:
                degraded_path = workdir / f"{logical_name}_blur_{sigma}.gif"
                degraded = gaussian_blur(base_frames, sigma)
                save_gif(degraded, degraded_path, duration_ms=dur)
                m = run_metrics(base_path, degraded_path, gl=gl)
                for k, v in m.items():
                    per_metric_levels[k].append(v)
            monotonicity[(kind, logical_name)] = dict(per_metric_levels)

            # Quantize
            kind = "quantize"
            per_metric_levels = defaultdict(list)
            for ncolors in QUANTIZE_COLORS:
                degraded_path = workdir / f"{logical_name}_q_{ncolors}.gif"
                degraded = quantize_palette(base_frames, ncolors)
                save_gif(degraded, degraded_path, duration_ms=dur)
                m = run_metrics(base_path, degraded_path, gl=gl)
                for k, v in m.items():
                    per_metric_levels[k].append(v)
            monotonicity[(kind, logical_name)] = dict(per_metric_levels)

            # Lossy
            if not skip_lossy:
                kind = "lossy"
                per_metric_levels = defaultdict(list)
                degraded_paths: list[Path] = []
                for level in LOSSY_LEVELS:
                    degraded_path = workdir / f"{logical_name}_lossy_{level}.gif"
                    try:
                        # apply_content_ceiling=False: same audit-bypass
                        # contract as the synthetic lossy sweep above — the
                        # fixture monotonicity sweep must run the requested
                        # lossy grid verbatim (see _common.py docstring).
                        gl["compress"](
                            base_path,
                            degraded_path,
                            "animately",
                            {"lossy_level": level},
                            apply_content_ceiling=False,
                        )
                    except Exception as e:
                        print(
                            f"[sanity]     animately --lossy {level} failed on {logical_name}: {e}",
                            flush=True,
                        )
                        continue
                    m = run_metrics(base_path, degraded_path, gl=gl)
                    for k, v in m.items():
                        per_metric_levels[k].append(v)
                    degraded_paths.append(degraded_path)
                if per_metric_levels:
                    _, coalesced_metrics = _coalesce_byte_identical_levels(
                        degraded_paths, dict(per_metric_levels)
                    )
                    dropped = len(degraded_paths) - len(
                        next(iter(coalesced_metrics.values()), [])
                    )
                    if dropped > 0:
                        print(
                            f"[sanity]     coalesced {dropped} byte-identical lossy level(s)"
                            f" on {logical_name}",
                            flush=True,
                        )
                    monotonicity[(kind, logical_name)] = coalesced_metrics
    elif FIXTURE_MONOTONICITY_BASES:
        msg = (
            f"{_FIXTURES_DIR} not found — fixtures directory absent; "
            "run `make fixtures` or check tests/fixtures/ exists in the repo"
        )
        print(f"[sanity]   skip fixture monotonicity bases: {msg}", flush=True)
        for logical_name in FIXTURE_MONOTONICITY_BASES:
            skipped_fixture_checks.append(
                {"logical_name": logical_name, "reason": msg}
            )

    # ---- Verdict per metric ----
    print("[sanity] Computing verdicts...", flush=True)
    verdicts: list[PerMetricVerdict] = []
    for metric in metric_keys:
        ident_vals = identity_results.get(metric, [])
        ident_mean = float(np.mean(ident_vals)) if ident_vals else float("nan")
        ident_std = float(np.std(ident_vals)) if ident_vals else float("nan")
        patho_val = float(patho_metrics.get(metric, float("nan")))
        direction = _direction(ident_mean, patho_val)

        # Collect monotonicity failures across (kind, base) pairs
        failures: list[dict[str, Any]] = []
        for (kind, base_name), per_metric in monotonicity.items():
            vals = per_metric.get(metric)
            if not vals or len(vals) < 2:
                continue
            invs = _monotonicity_check(vals, direction)
            if invs:
                failures.append(
                    {
                        "kind": kind,
                        "base": base_name,
                        "values": vals,
                        "inversions": [
                            {"from_level": i, "v_from": a, "v_to": b}
                            for i, a, b in invs
                        ],
                    }
                )

        # Decide verdict
        note = ""
        src = patho_source.get(metric, "solid")
        if direction == "flat" and abs(patho_val - ident_mean) < 1e-3:
            verdict = "INCONCLUSIVE"
            note = "identity and both pathological pairs (solid + structural) gave identical values — metric doesn't discriminate even on (gradient vs inverse)"
        elif failures:
            verdict = "SUSPICIOUS"
            note = f"{len(failures)} monotonicity violation(s); patho={src}"
        else:
            verdict = "PASS"
            note = f"patho={src}"

        verdicts.append(
            PerMetricVerdict(
                metric=metric,
                identity_mean=ident_mean,
                identity_std=ident_std,
                pathological=patho_val,
                direction=direction,
                monotonicity_failures=failures,
                verdict=verdict,
                note=note,
            )
        )

    return {
        "identity_per_gif": identity_per_gif,
        "pathological": patho_metrics,
        "pathological_source": patho_source,
        "pathological_solid": patho_solid,
        "pathological_structural": patho_structural,
        "monotonicity": {
            f"{kind}::{base}": per_metric for (kind, base), per_metric in monotonicity.items()
        },
        "skipped_fixture_checks": skipped_fixture_checks,
        "verdicts": [asdict(v) for v in verdicts],
        "config": {
            "identity_sample": IDENTITY_SAMPLE,
            "monotonicity_bases": MONOTONICITY_BASES,
            "fixture_monotonicity_bases": list(FIXTURE_MONOTONICITY_BASES.keys()),
            "noise_sigmas": NOISE_SIGMAS,
            "blur_sigmas": BLUR_SIGMAS,
            "quantize_colors": QUANTIZE_COLORS,
            "lossy_levels": LOSSY_LEVELS if not skip_lossy else [],
        },
    }


def print_verdict_table(results: dict[str, Any]) -> None:
    rows = results["verdicts"]
    print()
    print(f"{'Metric':<40} {'Identity':>12} {'Patho':>12} {'Dir':<14} {'Verdict':<12} Note")
    print("-" * 120)
    for r in rows:
        print(
            f"{r['metric']:<40} "
            f"{r['identity_mean']:>12.4f} "
            f"{r['pathological']:>12.4f} "
            f"{r['direction']:<14} "
            f"{r['verdict']:<12} "
            f"{r['note']}"
        )
    print()

    # Loudly surface any fixture-based checks that were skipped — these are
    # regression guards that didn't run, so a clean PASS table above does not
    # mean "everything was checked".  Downstream CI gates should treat a
    # non-empty skipped list as a warning or failure.
    skipped = results.get("skipped_fixture_checks", [])
    if skipped:
        print("⚠️  SKIPPED FIXTURE CHECKS (regression guards that did not run):")
        for entry in skipped:
            print(f"  - {entry['logical_name']}: {entry['reason']}")
        print()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", required=True, type=Path, help="Path to sanity_results.json")
    ap.add_argument(
        "--workdir",
        type=Path,
        default=None,
        help="Working directory for generated GIFs (defaults to a temp dir)",
    )
    ap.add_argument(
        "--skip-lossy",
        action="store_true",
        help="Skip the animately --lossy monotonicity arm (e.g. if binary unavailable)",
    )
    args = ap.parse_args()

    if args.workdir is None:
        wd_ctx = tempfile.TemporaryDirectory(prefix="giflab_sanity_")
        workdir = Path(wd_ctx.name)
    else:
        wd_ctx = None
        workdir = args.workdir
        workdir.mkdir(parents=True, exist_ok=True)

    try:
        results = run_sanity(workdir, skip_lossy=args.skip_lossy)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2, default=str))
        print(f"[sanity] Wrote {args.output}")
        print_verdict_table(results)
    finally:
        if wd_ctx is not None:
            wd_ctx.cleanup()

    return 0


if __name__ == "__main__":
    sys.exit(main())
