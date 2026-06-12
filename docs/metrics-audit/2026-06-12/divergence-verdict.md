# 2026-06-12 SSIM-vs-composite_quality divergence audit — verdict

**Task:** giflab-ssim-composite-divergence-audit (rollout giflab-rollout-2026-06-12-2, Wave 2).
**Question:** gifprep adopted `composite_quality` as its benchmark's single Pareto verdict gate
(gifprep PR #8) — *where do SSIM and composite disagree on real content, and is each
disagreement defensible?*
**Data:** fresh 500-cell sweep (this directory, `data/sweep.csv`): giflab's 100-GIF audit corpus
+ 25 synthetics × lossy {0, 20, 40, 60}, animately 1.1.30.0, post-PR #54 metrics code.
Generated analysis: [ssim-composite-divergence.md](ssim-composite-divergence.md); per-cell
deltas: `data/divergence_cells.csv`; provenance: `data/sweep_run.log`.

## Verdict: composite_quality never contradicts SSIM — it is strictly more sensitive

Across **371 analysable cells** (496 successful rows, 125/125 GIFs with a usable lossy=0
control, **zero** NaN-excluded cells):

1. **Zero direction divergences.** No cell where composite and SSIM moved in opposite
   directions beyond EPS = 0.005 (gifprep's `EPSILON_QUALITY`). The failure mode the audit was
   designed to catch — composite *improving* while SSIM degrades, or vice versa — does not
   occur on this corpus at any swept lossy level.
2. **Zero adjacent-level sign disagreements** across 371 level-steps. The historical
   smooth_gradient "dip-then-recover" shape (composite recovering at higher lossy while SSIM
   keeps falling) is gone — corpus-level confirmation that PR #54's `edge_similarity
   union==0 → NaN` fix removed it, consistent with the 2026-06-03 sanity re-run
   (`2026-06-03/post-fix-verdict.md` §E).
3. **Zero identity-relative divergence at the control point.** All 125 controls came back
   `ssim == composite_quality == 1.0` exactly: animately's lossy=0 recompress is a
   pixel-identical round-trip on this entire corpus, so the palette-axis caveat ("lossy 0 is
   not identity") turned out moot here — every delta in this audit is measured against the
   original pixels.
4. **The two metrics differ in *rate*, not direction.** Composite falls further than SSIM in
   324/371 cells (87%); median `d_comp` = −0.147 vs median `d_ssim` = −0.033. The continuous
   `divergence_score = |d_comp − d_ssim|` has median 0.083, p95 0.152, max 0.209.
5. **29 verdict divergences (7.8% of cells)** — and every single one is the same shape:
   **SSIM held (|d| ≤ 0.005) while composite degraded beyond GUARD = 0.02**. The mirror shape
   (composite held, SSIM moved) never occurs. All 29 passed the attribution round-trip check
   (0 non-attributable), so each one decomposes exactly into named contributor terms.

Practical reading for the downstream verdict gate: on the animately lossy axis,
`composite_quality` is a **conservative superset** of SSIM — everything SSIM flags, composite
flags in the same direction; composite additionally flags pixel-level degradation that SSIM's
local luminance/contrast/structure normalisation discounts. Of the 97 cells where SSIM said
"held", composite registered real movement beyond EPS in 45 (29 beyond GUARD) — an SSIM-only
gate would have called ~12% of these cells clean that the composite (correctly, see below)
does not.

## Adjudication of the 29 flagged cells

Clustered by top attribution driver (`w_i·(norm_i(L) − norm_i(0))`, full table in the
generated report). Triage classes per the task note: (1) defensible-by-design,
(2) content-sensitivity, (3) suspected bug.

| cluster (top driver) | n | cells | triage | adjudication |
|---|---|---|---|---|
| `mse_mean` (+`psnr_mean`) | 15 | real email GIFs at L20–60, `noise_small/large`, `unnamed copy 2` | **(1) defensible-by-design** | The signal-quality block (MSE 5% + PSNR 8% weight) registers genuine pixel drift — dither speckle and palette shifts from animately's lossy path — that SSIM's per-window normalisation discounts by construction. `2026-06-03/post-fix-verdict.md` §E adjudicated `mse` as "honest arithmetic on two different distortions". Largest cell: −0.025 composite on flat SSIM; visually plausible mild degradation. |
| `gmsd_mean` | 9 | real email GIFs, `gradient_medium`, `rcs-evolution` | **(1)/(2) defensible, content-sensitive on gradients** | GMSD reacts to gradient-magnitude changes (dither speckle on flat/gradient regions) at mild lossy — in-domain for the metric (§E's deviation-pooling caveat applies to *severe uniform* degradation, not these). On gradient synthetics this is exactly the shape the Wave-1 calibration close-out predicted (`scripts/audit/engine_lossy_calibration.py:176`): the SSIM-family/gradient composite terms carry the whole gradient-damage signal while `banding_score_mean` stays silent — its weighted delta is ≈ 0 in **all 29** flagged cells, again confirming the banding metric's blindness to this axis (recorded, not chased, same as Wave 1). |
| `edge_similarity_mean` | 5 | `single_pixel_anim` ×3 (top-3 by score, composite −0.12..−0.13), `gradient_xlarge` L20 (−0.099), one real GIF | **(2) content-sensitivity — composite's direction is right; magnitude rests on small-support edge maps** | `single_pixel_anim` is the audit's sharpest example of the *composite being more correct than SSIM*: a one-pixel animation whose moving pixel gets mangled is destroyed *as an animation* (edge −0.046, temporal_consistency_delta −0.034..−0.040), yet SSIM ≈ 1.0 because 99.99% of pixels match. On near-edgeless content (`gradient_xlarge`), the post-#54 code returns NaN only at `union == 0`; tiny-but-nonzero edge unions still produce low-support, noisy similarity values — direction trustworthy, magnitude less so. Watch-item, not a bug (continuous small-support damping would be the eventual fix shape if it ever matters). |

**No suspected metric bug was found; no follow-up bug task is filed.** Every flagged cell is
the composite blend behaving as designed (catching something SSIM misses) or a documented
content sensitivity already adjudicated in the 2026-06-03 audit. The 4 unswept cells
(animately `--lossy` timeout at 10s on 3 oversized GIFs, see `data/sweep_run.log`) are absent
rather than imputed; their GIFs keep their other levels.

## Criterion (for reproducibility)

Control-relative paired deltas, mirroring the downstream consumer exactly:
`d_ssim = ssim(L) − ssim(0)`, `d_comp = composite_quality(L) − composite_quality(0)`.
**EPS = 0.005** — byte-identical to gifprep's `EPSILON_QUALITY`
(`gifprep/src/gifprep/bench/pareto.py`) and to gifprep's own
`analyses/2026-06-12-corpus-local-composite-rerun/divergence.py`, so there is exactly one
divergence semantics across both repos. **GUARD = 0.02** (4×EPS) bounds the held-vs-degraded
criterion; the (EPS, GUARD] zone is a deliberate indeterminate band that is never flagged
(16 cells landed there). Sensitivity at EPS×{0.5, 1, 2}: 23 / 29 / 41 verdict divergences,
0 direction divergences at every setting — the zero-direction-divergence verdict is not an
artifact of the constant. NaN cells are excluded-and-counted (0 this run), duplicates deduped
keep-first. Implementation: `scripts/audit/report.py` (`ssim_composite_divergence`,
`adjacent_level_divergence`, `composite_attribution`), smoke-tested in
`tests/smoke/test_audit_divergence.py`.

## What was deliberately NOT changed

- **No production metric edits** — `src/giflab/` is untouched by this task (scope guard).
- The banding metric's silence on gradient lossy damage is recorded (again), not patched.
- `edge_similarity`'s small-support noise on near-edgeless content is a watch-item; the
  union==0 case is already NaN (PR #54) and nothing here shows the residual matters at
  verdict level.
- gifprep's 455-GIF `corpus-local` was deliberately NOT swept here — the downstream rerun
  already produced per-cell metrics over it and applies its own divergence.py to them; this
  doc is the interpretive input it was waiting on.

## Execution notes (deviation from the approved plan, documented)

The plan estimated "~15–20s metrics time per cell ≈ 2.5–3.5h" and prescribed a serial
`~/bin/gentle` run. Both halves of that estimate were wrong in practice: the 2026-06-03
sweep's own `runtime_s` column shows this corpus costs **mean 137s/cell** (median 27s — the
heavy tail dominates; 14.2h summed for 371 cells), and `taskpolicy -b` pins work to the 4
E-cores (~5× slower again, measured on the same GIF). Serial+gentle extrapolated to multiple
days. The sweep was therefore executed as **6 cost-balanced shards** (greedy LPT on the
2026-06-03 per-path runtimes), each under plain `nice -n 15` (no E-core pinning; still yields
to foreground use) with `GIFLAB_MAX_PARALLEL_WORKERS=2` — same corpus, same grid, same
`sweep.py` code path, only the job scheduling changed. Matched-cell check: median **0.92×**
the 2026-06-03 per-cell runtimes (no slowdown from sharding). Wall clock ≈ 4.7h overnight.
Details in `data/sweep_run.log`. `sweep.py` itself needed **zero changes**, as the plan
predicted.

## Hand-off to gifprep-corpus-local-composite-rerun

When interpreting corpus-local cells where the composite-gated verdict diverges from SSIM:

1. **Expect the asymmetry.** On a lossy-like axis, composite degrading while SSIM holds is
   the *normal* shape (29/29 here) and is usually mse/psnr- or gmsd-driven (pixel drift /
   dither speckle that SSIM discounts). It is not evidence the gate is broken — in the
   sharpest case (`single_pixel_anim`) the composite is the metric getting it right.
2. **Opposite-direction divergence would be new information.** We observed zero across 371
   cells; if corpus-local strategy cells show composite and SSIM moving in opposite
   directions beyond 0.005, that is a shape this audit did not see and worth a per-cell
   attribution (the `composite_attribution` round-trip tooling in
   `scripts/audit/report.py` works on any row carrying the 15 contributor columns).
3. **Check the edge/gradient caveat first** on flat or near-edgeless content: small-support
   `edge_similarity` and gmsd-on-gradients move composite hardest when SSIM is flat.
4. Cross-reference cells against `data/divergence_cells.csv` (all 371 deltas + flags) —
   the same EPS semantics apply 1:1.

Escalation options if corpus-local needs more coverage than this audit provides (deliberately
not executed here to avoid scope creep): add lossy=100 cells to this corpus (+~45 min
sharded), or sample specific gifprep categories into a follow-up sweep.
