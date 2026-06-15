# Compression-Pipeline Leaderboard — direction & design

**Status**: Direction doc. The leaderboard instrument is being **rebuilt**; this records the conceptual design agreed 2026-06-13. Sections mark **shipped** vs **planned** explicitly.
**Owner**: Lachy
**Related**: [public-api.md](../public-api.md) · [gifprep-integration.md](../gifprep-integration.md) · [metrics-system.md](./metrics-system.md)

---

## Why this exists (the reframe)

The goal: *feed in GIFs, test every compression combination, score on quality + size, and find which pipeline wins for which kind of GIF.*

This was **built and run in mid-2025** (`src/giflab/core/runner.py`, `core/pareto.py`, `core/sampling.py`, the preset system — results still under `data/experimental/results/`), then **deleted in commit `648db9a`** ("Remove legacy subsystems") in favour of a prediction-dataset → ML pipeline that was never finished (the `train` CLI is a stub; no models/DB exist).

Since then, a year of `audit-fix` work (see the metrics-audit thread + `docs/metrics-audit/`) has hardened the one thing that experiment always lacked: a **trustworthy quality metric** (`composite_quality`, an 11-metric weighted verdict). So the next step is to **rebuild the leaderboard lean, now that the metric is good enough to rank on.**

## Where this lives, and the gifprep boundary

The leaderboard lives **in giflab** (giflab owns "parameter grids, matrix benchmarks, engines, metrics" per [gifprep-integration.md](../gifprep-integration.md)). giflab carries two roles:

1. **Measurement library** — `compress()` / `measure()`, consumed by gifprep.
2. **Compression-pipeline leaderboard instrument** — this document.

gifprep is **orthogonal**: it owns the *preprocessing* axis (denoise / cleanup / AI transforms applied *before* compression) and its own Pareto harness, and it consumes giflab as a library. One-way dependency (gifprep → giflab). The leaderboard ranks *compression* pipelines; it does not do preprocessing.

```
        ┌─────────── giflab ───────────┐
raw GIF │ engines + metrics + leaderboard│ → best compression pipeline per content-type
        └───────────────▲───────────────┘
                        │ compress() / measure()  (library contract)
                 ┌──────┴──────┐
   raw GIF  →    │   gifprep    │ → "does preprocessing X improve the q/size tradeoff?"
                 └──────────────┘
```

## Design decisions (locked 2026-06-13)

| Dimension | Decision |
|---|---|
| **Deliverable** | A *trustworthy leaderboard* of the best compression **pipelines** per GIF content-type. (ML predictor deferred.) |
| **Pivot** | **Iso-quality** — smallest file at a calibrated quality floor — as the headline ranking; **Pareto-frequency** (how often a pipeline is on the frontier across the corpus) as an exchange-rate-free cross-check. |
| **Unit** | Multi-step **chained pipelines** (frame / colour / lossy), cross-engine allowed, with **PNG-sequence intermediates** between hops to avoid the GIF-recompression tax. Step *order* is a variable. |
| **Build** | **Rebuild fresh** against today's APIs; deleted `core/runner.py` + `core/pareto.py` are *reference only*. |
| **Search** | **Two-stage screening**: run ~1,200 pipeline structures on a small stratified GIF sample → eliminate the dominated → full param-sweep only the survivors. |
| **Step order** | Canonical (frame→colour→lossy) in phase-1; survivors get 2–3 *theory-justified* orderings (the colour↔lossy swap is the interesting one). |
| **Trust** | **Human validation gate first** — prove `composite_quality` at the floor agrees with human ranking *before* any irreversible elimination. |
| **Labels** | Auto-label (CLIP + the 25 hand-crafted features, surfacing disagreement) → human reviews confidence-sorted contact sheets → start with 5 compression-behaviour buckets → let the data merge/split to the final taxonomy. |
| **End-state** | Research instrument now (lean, re-runnable, clean pipeline-plugin seam); platform / predictor later. |

## The pivot (how (quality, size) becomes a ranking)

The product question is iso-quality: *"the smallest file that still looks fine."* So fix a **quality floor** (a `composite_quality` value, likely per content-type — a photo at SSIM 0.95 looks fine, flat vector art at 0.95 can band) and rank pipelines by the **file size** they achieve at that floor. Alongside it, a **Pareto-frequency** view (exchange-rate-free) cross-checks the headline; when the two disagree, that disagreement is itself a signal about metric fragility.

This choice also relaxes the demand on the metric: it only needs to be **rank-correct near the floor**, not absolutely calibrated.

## The unit (chained pipelines)

A pipeline is an ordered chain of operations, each fillable by any capable tool:

- Within one engine → single pass, no intermediate file.
- Across engines → materialise an intermediate; a **PNG sequence** beats an intermediate GIF (no recompression tax per hop). The deleted `core/runner.py::_execute_pipeline_with_metrics` already solved this — it tracked **connection methods between steps (G=GIF, P=PNG)** and used PNG when both tools supported it. The rebuild must reintroduce this abstraction, because the current `tool_interfaces.py::apply()` is GIF-path → GIF-path only.

`src/giflab/dynamic_pipeline.py::generate_all_pipelines()` (shipped) enumerates **1,200 structures** (5 frame × 20 colour × 12 lossy slots, including `NoOp` slots and 13 ffmpeg dither variants) — before parameter sweeps. Step order is currently hardcoded.

## The search (two-stage screening)

Exhaustive is infeasible: 1,200 structures × param grid (lossy ∈ 7, colour ∈ 5, frame ∈ ~3) × corpus × the full metric stack (LPIPS/SSIMULACRA2 are seconds each) is millions of runs. So:

1. **Phase 1 — screen.** Run all structures on a small stratified GIF sample with a coarse grid (optionally a cheap metric like SSIM-only). Eliminate dominated pipelines, margin-conservatively (elimination is **irreversible** — a metric wrong about a pipeline kills it forever).
2. **Phase 2 — sweep survivors.** Full param grid + the 2–3 step-order variants on the surviving handful, across the full corpus, with the full metric stack.

## Trust: the validation gate (do this first)

Because phase-1 elimination is irreversible and rides entirely on `composite_quality`, validate the metric **before** building the runner: generate ~50–100 outputs spanning quality levels and content types, human-rank them, measure rank-correlation against `composite_quality` near the floor, and calibrate the per-content-type floor. If it diverges systematically → fix the metric first. Tracked as the first task (`giflab-metric-validation-gate`); reuses `scripts/audit/`.

## Content-type taxonomy

Cut by **compression behaviour**, not semantics (a 3D render and a photo are semantically different but compress identically). Start with 5 buckets — photographic/continuous-tone, flat-graphic/vector, animation/cartoon, text/screen-capture, pixel-art — auto-labelled by an ensemble (CLIP content scores + the 25 features, with disagreement flagged for human review), then **let the data decide the final count** (merge buckets with statistically indistinguishable winners; split buckets with high internal disagreement). Labels become ground truth + a CLIP validation set for scaling later.

## Current state vs planned

| Component | State |
|---|---|
| 7 engines + 35 tool wrappers (incl. dither variants, NoOp slots) | **shipped** (`tool_wrappers.py`, `external_engines/`, `capability_registry.py`) |
| `generate_all_pipelines()` structure enumeration | **shipped** but only used to register DB rows (`dynamic_pipeline.py`, `storage.py`) |
| Quality metric stack + `composite_quality` verdict | **shipped** (`metrics.py`, `enhanced_metrics.py`); de-bugged via the audit |
| `compress()` / `measure()` public API | **shipped** (`public_api.py`, v0.4.0) |
| Audit sweep/report machinery | **shipped** (`scripts/audit/`) |
| Metric validation gate (human calibration of the floor) | **planned — first task** |
| G/P connection abstraction (PNG-seq hops) | **planned** (reference: deleted `core/runner.py`) |
| Two-stage screening runner + Pareto elimination | **planned** (reference: deleted `core/runner.py`, `core/pareto.py`) |
| Content-type auto-labeller + contact-sheet review | **planned** |
| Leaderboard report (per-bucket ranking) | **planned** |
| ML curve-predictor (`giflab train`) | **deferred** (currently a stub) |

## Reference (recovering the deleted harness)

The 2025 implementation is *reference only* but solved real problems. Recover via git:

```bash
git show 648db9a~1:src/giflab/core/runner.py    # G/P hop mechanism, _execute_pipeline_with_metrics
git show 648db9a~1:src/giflab/core/pareto.py    # Pareto frontier elimination
git show 648db9a~1:src/giflab/core/sampling.py  # representative/quick sampling
```
