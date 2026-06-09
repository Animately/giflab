# 2026-06-03 metrics audit — post-fix verdict & follow-up actions

**Date of analysis:** 2026-06-05
**Audit run:** `report.md` + `data/` (sanity + pilot + 371-pair corpus sweep, animately 1.1.30.0)

This is the human verdict accompanying the generated `report.md`. The 2026-06-03
audit ran *after* audit-fixes #38–#52 landed, so it is the **definitive
post-fix baseline** (no corpus re-run planned — #52 only touches the Phase-6
optimised path on a zero-aligned-pairs edge case, off by default). A deep
cross-check of the audit against every landed fix found the **metric layer
clean**; the actionable items were audit-tooling and one load-bearing cliff.

## Verdict: the metric layer is clean

- **Encoder healthy.** Corpus sweep: 371 pairs, `frame_loss = 0`, `unknown = 0`.
  Frame reductions are benign, duration-preserving dedup (22 pairs) — not
  corruption. No Linear ticket against the animately encoder.
- **Every Phase-3 outlier is *correct* metric behaviour** on genuinely hard
  content, not a bug:
  - `texture_similarity` z=−11.45 on the e926f7a2 vintage portrait (film grain
    smoothed by quantisation → LBP collapses);
  - `edge_similarity → 0.0` on gradients (Canny map destroyed / banding
    introduced);
  - `lpips_quality_mean` spike on NoN25 header (real perceptual error);
  - `ssim`/`ms_ssim` dips on photographic content at lossy=60.
  These are the metrics correctly identifying genuine compression failure modes.
- **Two suspicious-looking values spot-checked and confirmed HONEST (not
  sentinels):**
  - `banding_score = 100.0` (f19df0a2, VnK4p8) is a legitimate clamp of a
    *continuous* severity score —
    `gradient_color_artifacts.py` → `min(100.0, max(0.0, hist_diff*30 +
    contour_excess*10))`. A maximally-banded frame genuinely saturates at 100.
  - `temporal_consistency_compressed = 0.0000` (f19df0a2, VnK4p8) is honest
    `exp(-relative_variance * 2) ≈ 0` rounding for high-variance content
    (`metrics.py` `calculate_temporal_consistency`), not a floor or sentinel.
- **No remaining sentinel / cliff-edge / mislabel violations** of the
  "metric accuracy is load-bearing" rules across `metrics.py`,
  `enhanced_metrics.py`, `temporal_artifacts.py`, `gradient_color_artifacts.py`,
  and `optimized_metrics.py`. The #38–#52 wave (NaN-over-sentinel, dual
  white+black compositing, single-stream `_compressed` honesty, optimized
  temporal NaN) holds up.

## Actions taken (this follow-up)

### A. Sanity-harness verdict triage — `74 → 18 SUSPICIOUS`
The sanity harness flagged **74** SUSPICIOUS metrics, but ~70 were
structurally non-monotonic *by design* and buried the ~1 signal that mattered
(`composite_quality`). Added `_classify_metric` / `_decide_verdict` to
`scripts/audit/sanity.py`: dispersion siblings (`_std`/`_min`/`_max`/`_first`/
`_last`/`_middle`/`_positional_variance`), single-stream `_compressed` temporal
keys, and diagnostic/system metrics (`render_ms`/`kilobytes`/`efficiency`/
counts) with monotonicity failures now get a **DIAGNOSTIC** verdict, not
SUSPICIOUS. `_delta` keys (the genuine original-vs-compressed change signal)
stay pairwise. Re-derived over the committed `data/sanity.json`: **SUSPICIOUS
74 → 18** (56 → DIAGNOSTIC: 37 dispersion, 15 single-stream, 4 diagnostic), and
`composite_quality` + `edge_similarity` now surface instead of being buried.

### B. `edge_similarity` `union==0` cliff → NaN (root-cause fix)
`composite_quality` was non-monotonic on smooth_gradient lossy
`[0.97, 0.77, 0.73, 0.76]` — it dips then **recovers**. Driver:
`edge_similarity` `[1.00, 0.97, 0.48, 0.99]`. Root cause (confirmed by probe):
the `union == 0` guard (no Canny edges in *either* frame) returned a
fabricated-perfect **1.0**. On edgeless gradients at high lossy, most frames go
edgeless → 1.0, and those outliers rebound the `nanmedian` aggregate upward.

Fix (`metrics.py` `edge_similarity`): return `float("nan")` on `union == 0` —
edge similarity is *undefined* when there are no edges to compare, not perfect.
NaN is dropped by the `nanmedian` aggregation; if every frame is edgeless the
metric aggregates to NaN and `composite_quality` redistributes its 6% weight
(`_resolve_composite_from_contributions`). Compression that *invents* banding
edges absent from an edgeless original still gives `union > 0, intersection ==
0` → ~0.0 (artifact, correctly penalised) — the genuine signal is unaffected.

This is the **root-cause fix the median aggregation (#30/#32) was working
around**: the 1.0 outliers it tolerated are now removed at the source. Validated
by unit tests (`TestEdgeSimilaritySparseEdgeAggregation`, rewritten to the
corrected contract). Corpus-wide re-validation is deferred with the declined
corpus re-run; the change only affects edgeless content and is strictly more
honest.

### C. This document — close-out verdict.

### D. Per-engine (gifsicle) lossy-ceiling calibration — finding: no ceiling needed
The open TODO ("calibrate per-engine ceilings") was resolved with data, not by
building speculative machinery. The content-aware ceiling exists to prevent the
posterisation **quality cliff** that animately's re-quantising lossy produces on
photographic / gradient / data-viz content. A 2026-06-05 calibration
(`scripts/audit/engine_lossy_calibration.py`, deterministic on synthetic
content) measured `composite_quality` per engine across the public lossy range:

| content (rich RGB gradient) | L=0 | L=40 | L=80 | L=120 |
|---|---|---|---|---|
| **animately** composite | 1.000 | 0.730 | 0.700 | 0.698 |
| **gifsicle** composite | 1.000 | 0.969 | 0.941 | 0.924 |

animately **cliffs** (composite → 0.73 by lossy 40 — exactly why its photographic
ceiling is 20); gifsicle's error-bounded lossy degrades **gradually** with no
cliff (≥0.92 to lossy 120) and `banding_score = 0` across all four content
archetypes tested (smooth gradient, photographic noise, data-viz flat, rich
gradient). **Conclusion: gifsicle does not exhibit the failure mode the ceiling
guards against, so it gets NO content ceiling** — clamping it would discard good
compression for no benefit. The `engine == "animately"` gate is therefore
correct and now data-backed (config.py / public_api.py comments updated; the
`test_ceiling_skipped_for_non_animately_engine` regression test locks it).

**Evidence scope (honest):** the conclusion rests on four synthetic content
archetypes spanning the ceiling's target failure modes plus a direct same-content
animately-vs-gifsicle comparison — it is a mechanism-level finding about
gifsicle's error-bounded lossy, not content-specific. A broad real-corpus
gifsicle sweep would further harden it. **Still deferred:** gifski / imagemagick
/ ffmpeg remain uncalibrated and skip the ceiling; re-run the calibration script
with `--engines` to assess them before granting any of them a ceiling.

## What was deliberately NOT changed

- The remaining **18 SUSPICIOUS** pairwise-quality metrics (`gmsd`/`chist`/
  `fsim` on blur, `texture_similarity` on photographic noise, `mse` on
  transparency, `palette_distance`, `disposal_artifacts_delta`, etc.) are
  **content-sensitivity, not bugs** — the metrics correctly track genuine
  compression difficulty on hard synthetic content. No change.

## Methodology caveat for the next pilot

The pilot's cross-metric disagreement score **saturates at 1.000 from lossy=20
onward** (0.952 at lossy=0). It does not discriminate between lossy levels, so
the chosen `[20, 40, 60]` grid is only weakly grounded ("maximises
disagreement" is true of almost every level ≥20). A future pilot should use a
disagreement statistic that doesn't saturate — e.g. a rank-spread normalised so
intermediate levels remain distinguishable — so it can actually select the most
*informative* lossy levels rather than tying at the ceiling.
