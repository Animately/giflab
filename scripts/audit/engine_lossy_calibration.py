"""Calibrate whether a lossy engine needs a content-aware lossy ceiling.

The content-aware ceiling in ``giflab.public_api.compress`` clamps the requested
lossy level DOWN for photographic / gradient / data-viz content, because
**animately's re-quantising lossy posterises that content into a quality
cliff**. This script measures, per engine, how ``composite_quality`` /
``deltae`` / ``banding_score`` degrade across the public lossy range on the
content archetypes the ceiling targets — so the decision "does engine X need a
ceiling?" is grounded in data, not a guess (see CLAUDE.md "metric accuracy is
load-bearing").

It is deterministic and self-contained — it generates its own synthetic content
(no external corpus), so it reproduces anywhere and can calibrate a new engine.

Usage:
    poetry run python scripts/audit/engine_lossy_calibration.py
    poetry run python scripts/audit/engine_lossy_calibration.py --engines animately gifsicle gifski

2026-06-05 finding (animately vs gifsicle):

    Rich RGB gradient — composite_quality by public lossy level
       engine   L=0    L=40   L=80   L=120
    animately  1.000  0.730  0.700  0.698   <- CLIFF (posterisation): needs ceiling
     gifsicle  1.000  0.969  0.941  0.924   <- GRADUAL, no cliff: needs NO ceiling

    Across smooth-gradient / photographic-noise / data-viz-flat / rich-gradient,
    gifsicle's error-bounded lossy produced banding_score == 0 at every level and
    a smooth composite curve (>=0.90 at lossy 100), while animately cliffs early.
    Conclusion: gifsicle does not exhibit the posterisation failure mode the
    ceiling guards against, so it gets NO content ceiling.

2026-06-09 finding (gifski / ffmpeg / imagemagick) — all THREE get NO ceiling,
but for two structurally different reasons. Be honest about which:

    Rich RGB gradient — composite_quality by public lossy level (banding == 0.00
    at EVERY measurement for all three, across all four content archetypes):
       engine   L=0    L=40   L=80   L=100
       gifski  0.920  0.734  0.518  0.398   <- real axis, GRADUAL, banding-free
       ffmpeg  0.492  0.492  0.492  0.492   <- FLAT: lossy_level is INERT
  imagemagick  1.000  1.000  1.000  1.000   <- FLAT: lossy_level is INERT

    - gifski has a REAL lossy axis (4 distinct output md5s; bytes 15.0->2.3 kB
      over L=0->100). Its composite declines smoothly with banding_score == 0 at
      every level — gifsicle's profile, no posterisation cliff. NO ceiling.
    - ffmpeg / imagemagick ``lossy_level`` is INERT for GIF output: the wrappers
      map it to ``-q:v`` (a video-DCT knob) / ``-quality`` (a PNG/JPEG zlib knob)
      respectively, NEITHER of which affects GIF pixels. Verified by md5 probe:
      output is BYTE-IDENTICAL at every level (same hash, same size). So their
      flat curves are NOT "graceful degradation" — no lossy axis is exercised at
      all, hence they cannot cliff. imagemagick's flat composite == 1.000 means
      "nothing changed" (re-saved as-is), NOT "perfect lossy". Do not misread a
      flat/declining curve as evidence of graceful degradation unless the output
      bytes actually vary across levels.

    Caveat (evidence quality, not a calibration error): on tiny 4-frame
    synthetic GIFs the metrics pipeline logs frame-count / FPS-drift warnings for
    gifski; these slightly perturb gifski's ABSOLUTE composite but not the
    verdict (banding == 0, monotone, real axis). Fixing the two inert wrappers to
    actually drive GIF lossiness is a separate engine-fidelity task, NOT a
    ceiling-calibration change — out of scope here.

    Conclusion: all four non-animately engines are now calibrated and none needs
    a content ceiling. The ``engine == "animately"`` gate in public_api.compress
    is correct and fully data-backed.

2026-06-12 finding (ffmpeg / imagemagick, POST-FIX — lossy axis made real):

    The two inert wrappers were fixed: ``lossy_level`` now maps geometrically
    (256→16 colours, halving every 25 levels) onto palette-size reduction +
    dithering — ffmpeg via two-pass ``palettegen=max_colors`` /
    ``paletteuse=dither=sierra2_4a``, imagemagick via ``-dither Riemersma
    -colors N`` (plain re-save at L=0). Re-running this harness now exercises
    a REAL axis: distinct md5s, bytes decline monotonically, deltae rises
    monotonically, and banding_score == 0.00 at EVERY measurement for both
    engines across all four content archetypes.

    Rich RGB gradient — composite_quality by public lossy level:
       engine   L=0    L=10   L=40   L=80   L=100
       ffmpeg  0.686  0.664  0.602  0.505  0.458   <- GRADUAL (max adjacent
                                  grid step 0.052): no cliff, needs NO ceiling
  imagemagick  1.000  0.758  0.663  0.545  0.508   <- ENTRY STEP -0.242 at
                                  L=0→10, then gradual — see below

    - ffmpeg's L=0 base is 0.686, not 1.000: it always re-encodes through a
      single palettegen-optimised global palette, slightly lossy for
      >256-colour content (still an improvement over its old generic-palette
      GIF encode, which measured 0.492). Its curve never drops >=0.2 between
      adjacent grid levels — gifsicle-profile gradual degradation, NO ceiling.
    - imagemagick's L=0 is an honest plain re-save (1.000). Its FIRST
      quantisation step costs ~0.24 composite on gradient archetypes
      (1.000→0.758 rich, 1.000→0.756 smooth, at L=10) with banding == 0 —
      an entry-step discontinuity from ImageMagick's quantiser remapping
      colours even when the target palette exceeds the unique-colour count,
      NOT animately-style progressive posterisation (after the step the curve
      is gradual: 0.758→0.508 over L=10→100). By the pre-registered decision
      rule (any >=0.2 adjacent-grid-level drop by L=40 raises the ceiling
      question), this is deferred to the follow-up calibration task
      ``giflab-imagemagick-lossy-entry-step-ceiling-calibration`` rather than
      granting a ceiling here (ClassifierConfig mandates a data-backed
      per-engine dimension before any non-animately ceiling).
    - data-viz-flat content is byte-stable across all levels for both engines
      (palette reduction is honestly a no-op when the 16-colour floor still
      covers every unique colour) — a flat 1.000 there now means "nothing to
      remove", which IS graceful, unlike the pre-fix flat curves that meant
      "nothing was attempted".

    Conclusion: the content ceiling remains animately-only. ffmpeg is
    confirmed no-ceiling on a real axis; imagemagick's entry step is tracked
    in the follow-up task above.

Note: the default ``LEVELS`` grid stops at 100 (the public lossy range).
Levels outside 0-100 now raise a clear range error from BOTH the ffmpeg and
imagemagick wrappers (``validate_lossy_level_for_engine``: "lossy_level must
be between 0 and 100 for <engine>") — replacing ffmpeg's old silent clamp and
imagemagick's old confusing ``quality must be in 0-100 range`` crash at level
120. Pass ``--levels`` explicitly to probe other grids within 0-100.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image


def _save_gif(frames: list[np.ndarray], path: Path) -> None:
    imgs = [
        Image.fromarray(f, "RGB").convert("P", palette=Image.ADAPTIVE, colors=256)
        for f in frames
    ]
    imgs[0].save(
        path,
        save_all=True,
        append_images=imgs[1:],
        duration=100,
        loop=0,
        optimize=False,
    )


def _smooth_gradient(n: int = 4) -> list[np.ndarray]:
    out = []
    for k in range(n):
        row = np.linspace(0, 255, 128, dtype=np.float32)
        f = np.clip(np.stack([row] * 96, axis=0) + k * 2, 0, 255)
        out.append(np.stack([f, f, f], -1).astype(np.uint8))
    return out


def _photographic_noise(n: int = 4) -> list[np.ndarray]:
    rng = np.random.RandomState(0)
    base = np.stack([np.linspace(0, 255, 128, dtype=np.float32)] * 96, axis=0)
    out = []
    for k in range(n):
        f = np.clip(base + rng.normal(0, 18, base.shape) + k * 2, 0, 255)
        out.append(np.stack([f, f, f], -1).astype(np.uint8))
    return out


def _data_viz_flat(n: int = 4) -> list[np.ndarray]:
    cols = [
        (31, 119, 180),
        (255, 127, 14),
        (44, 160, 44),
        (214, 39, 40),
        (148, 103, 189),
    ]
    out = []
    for k in range(n):
        f = np.zeros((96, 128, 3), np.uint8)
        for i, c in enumerate(cols):
            h = 20 + i * 12 + k
            f[96 - h : 96, i * 24 : (i + 1) * 24] = c
        out.append(f)
    return out


def _rich_gradient(n: int = 4) -> list[np.ndarray]:
    """2D RGB gradient — many colours; the kind that posterises under coarse
    re-quantisation (the animately failure mode the ceiling targets)."""
    yy, xx = np.mgrid[0:96, 0:160]
    out = []
    for k in range(n):
        r = xx / 160 * 255
        g = yy / 96 * 255
        b = ((xx + yy) / 256 * 255 + k * 3) % 256
        out.append(np.clip(np.stack([r, g, b], -1), 0, 255).astype(np.uint8))
    return out


CONTENTS = {
    "smooth_gradient": _smooth_gradient,
    "photographic_noise": _photographic_noise,
    "data_viz_flat": _data_viz_flat,
    "rich_gradient": _rich_gradient,
}
# Default grid stops at 100 (the public lossy range). Levels outside 0-100
# raise a clear range error from every 0-100 engine wrapper
# (validate_lossy_level_for_engine), so probing above 100 is gifsicle-only.
LEVELS = [0, 10, 20, 30, 40, 60, 80, 100]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--engines", nargs="+", default=["animately", "gifsicle"])
    ap.add_argument(
        "--levels", nargs="+", type=int, default=LEVELS, help="public lossy levels"
    )
    args = ap.parse_args()

    # Heavy imports deferred so --help is instant.
    from giflab.config import MetricsConfig
    from giflab.metrics import calculate_comprehensive_metrics
    from giflab.public_api import compress

    cfg = MetricsConfig()
    cfg.ENABLE_DEEP_PERCEPTUAL = False  # banding/deltae/composite don't need LPIPS
    cfg.ENABLE_TEMPORAL_ARTIFACTS = False

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for name, factory in CONTENTS.items():
            orig = td / f"{name}.gif"
            _save_gif(factory(), orig)
            print(f"\n=== {name} ===")
            print(
                f"{'engine':>10} {'pub':>4} {'composite':>9} {'deltae':>7} {'banding':>8} {'kB':>7}"
            )
            for engine in args.engines:
                for level in args.levels:
                    outp = td / f"{name}_{engine}_{level}.gif"
                    try:
                        compress(
                            orig,
                            outp,
                            engine=engine,
                            params={"lossy_level": level},
                            apply_content_ceiling=False,
                        )
                        m = calculate_comprehensive_metrics(
                            orig, outp, config=cfg, force_all_metrics=True
                        )
                        print(
                            f"{engine:>10} {level:>4} "
                            f"{m.get('composite_quality', float('nan')):>9.4f} "
                            f"{m.get('deltae_mean', float('nan')):>7.3f} "
                            f"{m.get('banding_score_mean', float('nan')):>8.2f} "
                            f"{outp.stat().st_size / 1024:>7.1f}"
                        )
                    except Exception as exc:  # noqa: BLE001 — calibration probe
                        print(
                            f"{engine:>10} {level:>4}  ERROR {type(exc).__name__}: {exc}"
                        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
