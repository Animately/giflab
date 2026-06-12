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
gifsicle sweep would further harden it.

### 2026-06-09 follow-up: gifski / ffmpeg / imagemagick (deferral closed)

The three remaining non-animately engines were calibrated with the same harness
(`scripts/audit/engine_lossy_calibration.py`). **All three get NO content
ceiling**, but for two structurally different reasons — and the honest
distinction matters (a flat curve is only evidence of graceful degradation if the
output bytes actually vary across levels):

| content (rich RGB gradient) | L=0 | L=40 | L=80 | L=100 | reading |
|---|---|---|---|---|---|
| **gifski** composite | 0.920 | 0.734 | 0.518 | 0.398 | real axis, GRADUAL, banding-free |
| **ffmpeg** composite | 0.492 | 0.492 | 0.492 | 0.492 | FLAT — `lossy_level` is INERT |
| **imagemagick** composite | 1.000 | 1.000 | 1.000 | 1.000 | FLAT — `lossy_level` is INERT |

`banding_score = 0.00` at **every** measurement for all three, across all four
content archetypes.

- **gifski** has a REAL lossy axis (4 distinct output md5s; bytes 15.0→2.3 kB over
  L=0→100). Its composite declines smoothly with `banding_score = 0` at every
  level — gifsicle's profile, no posterisation cliff. **NO ceiling.**
- **ffmpeg / imagemagick** `lossy_level` is **INERT** for GIF output: the
  wrappers map it to `-q:v` (a video-DCT knob) / `-quality` (a PNG/JPEG zlib
  knob) respectively, neither of which affects GIF pixels. Verified by md5 probe:
  output is BYTE-IDENTICAL at every level (same hash, same size) on both content
  types tested. So their flat curves are **not** "graceful degradation" — no
  lossy axis is exercised at all, hence they cannot cliff. imagemagick's flat
  `composite == 1.000` means "nothing changed" (re-saved as-is), **not** "perfect
  lossy". **NO ceiling.** Fixing the two inert wrappers to actually drive GIF
  lossiness is a separate engine-fidelity task, out of scope for ceiling
  calibration.

**Conclusion: all four non-animately engines are now calibrated and none needs a
content ceiling.** The `engine == "animately"` gate in `public_api.compress` is
correct and fully data-backed; the parametrized
`test_ceiling_skipped_for_non_animately_engine` regression test locks the
invariant for all four. The default `LEVELS` grid was capped at 100 (the public
lossy range) — it previously included 120, which crashed the imagemagick column
(`quality = 100 - 120 = -20 → ValueError`).

### E. 2026-06-12 sanity re-run — fix B corpus-validated, SUSPICIOUS 18 → 15, all adjudicated

A sanity-only re-run on merged `main` (PR #54 in; animately 1.1.30.0, seed 42 —
same binary version and config as this audit) validated fix B at the harness
level. Across all 143 metrics exactly **four verdicts changed**, every one
attributable to the `union==0 → NaN` change:

| metric | pre-fix | post-fix |
|---|---|---|
| `composite_quality` | SUSPICIOUS | **PASS** |
| `edge_similarity` | SUSPICIOUS | **PASS** |
| `edge_similarity_mean` | SUSPICIOUS | **PASS** |
| `edge_similarity_std` | DIAGNOSTIC | INCONCLUSIVE |

`composite_quality` on `lossy::smooth_gradient` went `[0.970, 0.769, 0.730,
0.760]` (the dip-then-recover that motivated the fix) → `[0.970, 0.711, 0.701,
0.700]` — monotone non-increasing on all four degradation arms. Identity is
unharmed. **No new SUSPICIOUS appeared**: 18 → **15** (fresh totals: PASS 46,
SUSPICIOUS 15, DIAGNOSTIC 55, INCONCLUSIVE 27). The remaining 15 (9 distinct
metrics; bare + `_mean` are the same signal) were adjudicated per-metric and
are **all content-sensitivity at degradation extremes or metric-saturation
float noise — zero metric bugs**: `chist` (histogram-correlation properties at
out-of-domain extremes), `deltae_pct_gt2` (step function of content geometry;
the continuous `deltae_mean` sibling PASSes), `disposal_artifacts_delta`
(detector out of domain at extremes), `fsim` (saturation noise + extreme-blur
flattening), `gmsd` (deviation-pooling property: std of the GMS map falls as
degradation becomes spatially *uniform* — don't use it to order severe uniform
degradations), `mse` (honest arithmetic on two different distortions),
`palette_distance` (noise floor, <0.1% of range), `sharpness_similarity`
(false sharpness from σ60 noise), `texture_similarity` (known LBP collapse,
PR #44/#48). Reproduction is bit-for-bit deterministic against the 2026-06-03
baseline. Evidence: `audit-rerun-2026-06-12/` (repo root, gitignored); full
per-metric adjudication table in the giflab-validate-edge-fix task record.

## Staleness note: this report's disagreement table (added 2026-06-12)

The 2026-06-03 `report.md` cross-metric disagreement table is **stale at the
top** — keep it as the historical baseline but do not re-litigate its rows:

- Its #1 row (gradient_small, spread 1.00, best=`edge_similarity`) rests on
  the pre-#54 `union==0 → 1.0` sentinel that fix B removed. Current code
  returns NaN on those frames, so the row cannot reproduce on any future sweep.
- 9/10 of its best-metric attributions are **tie-block noise**: the old
  `rank_normalise` (argsort + linspace) assigned arbitrary distinct ranks
  across tie blocks, and the corpus is heavily tie-saturated (232/371 rows
  tied at `banding_score_mean == 0.0`, 65 at `edge_similarity == 1.0`).
  Tie-aware average ranks + single-stream `_compressed` exclusion landed in
  `scripts/audit/report.py` / `pilot.py` (2026-06-12); a re-render on the same
  CSV drops the max spread to ~0.92, and the residual top entries are genuine
  metric-family disagreement (binary `edge_similarity` vs continuous
  `fsim`/`gmsd` on gradients) — acceptable for a diagnostic aid.

## What was deliberately NOT changed

- The remaining **15 SUSPICIOUS** pairwise-quality metrics (`gmsd`/`chist`/
  `fsim` on blur, `texture_similarity` on photographic noise, `mse` on
  transparency, `palette_distance`, `disposal_artifacts_delta`, etc.) are
  **content-sensitivity, not bugs** — verified per-metric on the 2026-06-12
  sanity re-run (section E above). No change.

## Methodology caveat for the next pilot

The pilot's cross-metric disagreement score **saturates at 1.000 from lossy=20
onward** (0.952 at lossy=0). It does not discriminate between lossy levels, so
the chosen `[20, 40, 60]` grid is only weakly grounded ("maximises
disagreement" is true of almost every level ≥20). A future pilot should use a
disagreement statistic that doesn't saturate — e.g. a rank-spread normalised so
intermediate levels remain distinguishable — so it can actually select the most
*informative* lossy levels rather than tying at the ceiling.

**Resolved 2026-06-12:** the saturation was root-caused to the same tie-handling
artifact as the report table (partial tie blocks at low lossy got arbitrary
distinct linspace ranks; the constant-metric skip only caught *fully* constant
metrics). `scripts/audit/pilot.py` now uses tie-aware average ranks and only
lets pairwise-quality metrics vote (dispersion siblings, single-stream
`_compressed` keys and diagnostic/system metrics are excluded, mirroring the
sanity triage). The next pilot's disagreement-by-lossy curve should
discriminate between levels.
