# Phase 0 Research: Metric Validation Gate

All decisions reuse existing giflab internals and keep the gate a thin, single-run study. No production metric code is modified.

## R1 — Rank-agreement statistic

**Decision**: **Spearman's ρ per GIF**, averaged across the study set; report Kendall's τ alongside as a secondary column.

**Rationale**: The unit of judgement is "did the verdict order *this GIF's* 4–6 compressions like the human did" — a small per-GIF ranking. Spearman's ρ is the standard, interpretable rank-correlation for that, and the SC-002 threshold (≥ 0.85) is expressed on its [-1, 1] scale. Per-GIF then averaged (not one global correlation across all outputs) is correct because absolute `composite_quality` values are not comparable across different GIFs — only the *within-GIF ordering* is meaningful. `scipy.stats.spearmanr`/`kendalltau` are available via the existing metrics stack.

**Alternatives considered**: One pooled correlation across all outputs (rejected — conflates within-GIF ordering with cross-GIF scale, exactly the thing that is not comparable); pairwise concordance only (Kendall) as primary (kept as secondary — ρ is more familiar and matches the threshold already written).

**Tie handling** (spec edge case): Spearman with average ranks for ties; the audit harness already learned the tie-rank lesson (`_common.py::tie_average_unit_ranks`) — reuse that helper so ties don't inflate/deflate agreement.

## R2 — Per-bucket quality floor derivation

**Decision**: For each GIF, the floor candidate = the `composite_quality` of the **lowest-quality output the rater still marked acceptable** (the single boundary, FR-002). Per bucket, aggregate the candidates as **median + IQR** (the median is the reported floor; the IQR is the reported spread, SC-003).

**Rationale**: The boundary is exactly "the composite_quality at the acceptable/not-acceptable cutover," which is the leaderboard's iso-quality operating point. Median + IQR is robust to the small per-bucket N and to a single odd GIF; it gives a usable floor with an honest spread. Buckets with too few boundary-crossings are flagged low-confidence rather than reported precisely (spec edge case + SC-003).

**Alternatives considered**: Mean + std (rejected — not robust at N≈3–5 per bucket); a logistic fit of acceptable-vs-not against composite_quality (rejected for v1 — over-engineered for a single-boundary-per-GIF design and too few points per bucket); a single global floor (rejected — contradicts the per-content-type requirement and the photo-vs-flat-graphic motivation).

## R3 — Offline ranking-sheet format

**Decision**: A self-contained **HTML contact sheet** rendered per study GIF — the original's representative frame plus each compressed output shown inline, in a randomised order — with a tiny embedded form (drag-to-order or numbered inputs + one "acceptable down to here" marker) that the rater fills in; on save it writes/produces a **CSV** of `(gif, output_id, human_rank, accepted)`. No web server — a local file opened in a browser, or a CSV the rater edits directly as the fallback.

**Rationale**: Honours "offline, no web service" (spec/plan constraint). Reuses `report.py`'s existing thumbnail/figure rendering for the inline images. Randomised output order prevents the rater anchoring on the `composite_quality` order (which would bias agreement upward — the exact thing being measured). CSV output plugs straight into `ingest_rankings.py`.

**Alternatives considered**: A live web app (rejected — violates the offline/thin constraint, scope creep); ranking from filenames in a folder (rejected — no inline visual, error-prone); reusing an external labelling tool (rejected — new dependency for a one-off study).

## R4 — Quality-spread generation

**Decision**: Produce 4–6 outputs per GIF along a **fixed compression ladder** that reliably spans best→worst, generated through the shipped `compress()` / existing wrappers. The ladder deliberately includes outputs that vary **one operation axis at a time** (a lossy sweep, a colour-reduction sweep, a frame-reduction sweep) so divergences can be attributed to an operation type (FR-007).

**Rationale**: A fixed, recorded ladder makes the study reproducible (FR-010) and guarantees a visible quality range so the rater's ranking is meaningful and a boundary is likely to be crossed. Varying one axis at a time is what makes the operation-type disagreement grouping (clarified answer) computable without cross-engine chaining (out of scope).

**Alternatives considered**: Random params (rejected — not reproducible, may not span quality); full grid (rejected — too many outputs for one rating session, exceeds SC-005); leaderboard-style cross-engine chains (rejected — explicitly out of scope for the gate).

## R5 — `composite_quality` configuration ("near the operating point")

**Decision**: Compute via `giflab.metrics.calculate_comprehensive_metrics(orig, comp, force_all_metrics=True)` with `config.ENABLE_DEEP_PERCEPTUAL = True` and `ENABLE_TEMPORAL_ARTIFACTS = True` — the **same full-stack configuration the audit harness uses and the leaderboard will use to rank**. Record the config + giflab commit in the study manifest.

**Rationale**: The gate must validate the verdict the leaderboard will actually trust, not a cheaper variant (spec Assumption). This matches the documented audit-harness invocation (thread "Known quirks": the audit uses `force_all_metrics=True` to get the full key set, not the narrow public `measure()`).

**Alternatives considered**: The public `measure(["composite_quality"])` surface (rejected — it forces LPIPS but the gate also wants the sub-metric breakdown for R6, which the comprehensive call exposes directly); a cheap SSIM-only proxy (rejected — would validate the wrong number).

## R6 — Attributing a disagreement to a "contributing component of the verdict"

**Decision**: When the verdict's rank of an output diverges from the human's by more than a set margin, attribute it to the **sub-metric whose own ranking deviates most from the human's at that output**, weighted by that sub-metric's `composite_quality` weight. `calculate_comprehensive_metrics` already returns all 11 weighted sub-metrics; the composite weights are a documented public contract (`docs/public-api.md`). Report the dominant contributor per divergence, grouped by content bucket × operation type.

**Rationale**: Turns "the metric is wrong here" into "edge_similarity over-rewarded this heavy colour-reduction output," which is directly actionable for a metric fix (US3). Uses only existing outputs (the sub-metric values + their published weights); computes nothing new in the metric layer.

**Alternatives considered**: Leave-one-out recomputation (drop each sub-metric, see which flips the rank) — more rigorous but heavier; kept as a possible refinement, not needed for v1's "name the likely culprit" goal. Reporting raw sub-metric tables without attribution (rejected — pushes the diagnosis onto the reader).

## R7 — Scope guard (what this gate must NOT touch)

**Decision**: The gate is **read-only with respect to `src/giflab/`**. It imports `compress` and `calculate_comprehensive_metrics` and changes no metric, config-default, or wrapper code. A NO-GO result produces a disagreement report that *feeds* a separate metric-fix effort; it does not itself fix the metric.

**Rationale**: Constitution Principle VII + the repo's load-bearing metric-accuracy discipline. Keeping the gate read-only means it can never be the thing that silently changes a metric while "validating" it.
