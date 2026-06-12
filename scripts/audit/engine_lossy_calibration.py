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
      question), this raised the ceiling question; the follow-up calibration
      ran the same day and CLOSED it with NO ceiling — see the entry-step
      finding below (ClassifierConfig mandates a data-backed per-engine
      dimension before any non-animately ceiling, and the data says no).
    - data-viz-flat content is byte-stable across all levels for both engines
      (palette reduction is honestly a no-op when the 16-colour floor still
      covers every unique colour) — a flat 1.000 there now means "nothing to
      remove", which IS graceful, unlike the pre-fix flat curves that meant
      "nothing was attempted".

    Conclusion: the content ceiling remains animately-only. ffmpeg is
    confirmed no-ceiling on a real axis; imagemagick's entry step was
    calibrated and closed the same day — next finding.

2026-06-12 entry-step finding (imagemagick follow-up — CLOSED: NO ceiling,
no mapping change; ImageMagick 7.1.2-0 Q16-HDRI):

    The ~0.24 entry step flagged above was isolated with a fine grid plus
    direct ``magick`` probes. Verdict: it is a one-off, content-dependent
    cost of ENTERING the quantise+dither axis at all — paid in full at the
    FIRST positive level — not a level-dependent cliff a ceiling could avoid.

    Fine grid (this harness: ``--engines imagemagick --levels 0 1 2 3 5 7
    10``; palette targets 249→194 over L=1→10) — composite_quality:

       content              L=0    L=1    L=2    L=3    L=5    L=7   L=10
       rich_gradient       1.000  0.812  0.806  0.795  0.784  0.772  0.758
       smooth_gradient     1.000  0.756  0.756  0.756  0.756  0.756  0.756
       photographic_noise  1.000  0.908  0.908  0.908  0.908  0.908  0.909
       data_viz_flat       1.000  1.000  1.000  1.000  1.000  1.000  1.000

    - rich_gradient (per-frame uniques 256, union across frames 1024): the
      whole step lands at L=1 (colors=249); after it the curve is GRADUAL
      (max adjacent fine-grid step 0.014; 0.812→0.508 over L=1→100).
    - smooth_gradient (union 129 unique colours): output is BYTE-IDENTICAL
      at every level L=1–10 (one md5 across all targets 249→194 — each
      exceeds the 129 uniques; deltae flat at 0.798). The flat L1–10
      segment is NOT an inert axis (bytes DO change vs L=0): the step is
      invocation cost, level-independent below the unique-count threshold.
    - Direct probes (exact commands, mirroring lossy_compress's invocation):
        magick rich.gif -dither Riemersma -colors 255 out.gif  -> 0.8169
          (the gentlest possible target still pays the step; colour union
          1024→839 — the quantiser remaps colours even when the target
          exceeds the per-frame unique count of 256)
        magick rich.gif -dither None -colors 255 out.gif       -> 0.8928
          (quantiser-only cost ~0.107; Riemersma dither adds ~0.076)
        magick rich.gif -dither None -colors 194 out.gif       -> 0.8100
          (vs Riemersma at 194: 0.7579)
        45-unique-colour spread-palette control: ``-colors 255`` AND
          ``-colors 194`` -> composite 1.0000, byte-identical outputs,
          palette untouched. Small well-separated palettes pay ZERO.

    Why NO ceiling (and no mapping change):
    - A ceiling clamps lossy_level DOWN to a POSITIVE value (existing
      animately ceilings: 20–40); the step is paid in full at L=1, so every
      positive ceiling value still pays it. Only L=0 avoids it, and "clamp
      to 0" is disabling lossy, not a ceiling (the PR #40 over-trigger
      lesson).
    - A smoother entry mapping cannot help: the cost is internal to
      ImageMagick's quantise+dither invocation, paid even at colors=255/249
      when the target exceeds the content's unique-colour count (smooth:
      129 uniques, target 249, still −0.244). There is no gentler entry
      than colors=255; the mapping is already geometric and smooth.
    - It is not the posterisation failure mode the ceiling exists for:
      after entry the curve is gradual with banding_score == 0 at every
      measurement. Structurally this is ffmpeg's verdict — ffmpeg's L=0
      base is 0.686 on identical content because it always pays its
      global-palette cost — and imagemagick at L=1 (0.812) is strictly
      better than ffmpeg at ANY level; a ceiling for imagemagick but not
      ffmpeg would be incoherent.
    - A "skip -colors when target >= unique count" guard is also ruled out:
      the spread-palette control shows IM already no-ops byte-identically
      there; on dense palettes the perturbation is the real cost of
      entering the axis, not waste.

    Recorded, not acted on:
    - banding_score == 0 at every measurement: the entry step is invisible
      to the banding metric; the SSIM-family composite terms carry the
      entire signal (composite −0.244 for a deltae of 0.8 on smooth
      gradients — the ssim-composite-divergence audit's territory, not
      chased here).
    - ``-dither None -colors 194`` scores BOTH higher composite (0.810 vs
      0.758) AND smaller bytes (19.9 vs 21.4 kB) than Riemersma on
      rich_gradient. Riemersma stays: it is the banding guard chosen by the
      dithering research, and a level-conditional dither switch would be a
      discrete cliff (continuous-over-discrete). Candidate future
      dither-calibration task.

    Conclusion: all four non-animately engines are now calibrated CLOSED
    with no ceiling. The ``engine == "animately"`` gate in
    public_api.compress is correct and fully data-backed.

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
