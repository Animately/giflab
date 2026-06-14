# Quickstart: Running the Metric Validation Gate

A single-run study. Five thin scripts under `scripts/audit/validation_gate/`, one human step in the middle. All commands via Poetry; run the heavy generation under `gentle`.

Output folder for a run: `docs/metrics-audit/<date>/validation-gate/` (set `OUT` below).

```bash
OUT=docs/metrics-audit/2026-06-14/validation-gate
```

## 1. Select + hand-label the study GIFs

Selects ~15–20 GIFs stratified across content (reusing `build_sample.py`) and writes a manifest with a **blank `content_bucket` column** to fill in by hand (FR-008):

```bash
poetry run python -m scripts.audit.validation_gate.build_study select --out "$OUT" --n 18
# → $OUT/study_manifest.csv  (edit it: assign each GIF a bucket from the 5)
```

Open `study_manifest.csv`, set each row's `content_bucket` (`photographic` / `flat_graphic` / `animation` / `text_screen` / `pixel_art`). Re-running any later step fails loudly if a row is still unlabelled.

## 2. Generate the quality-spread outputs + compute the verdict

For each labelled GIF, produces 4–6 compressions along one-operation-at-a-time ladders (R4) via the shipped `compress()`, then computes `composite_quality` + the 11 sub-metrics with the full-stack config (R5):

```bash
gentle poetry run python -m scripts.audit.validation_gate.build_study generate --out "$OUT"
# → $OUT/study_outputs/*.gif , appends composite_quality + submetrics to study_manifest.csv
```

## 3. Render the offline ranking sheet

```bash
poetry run python -m scripts.audit.validation_gate.ranking_sheet --out "$OUT"
# → $OUT/ranking_sheet.html  (outputs shown per-GIF in RANDOMISED order — no verdict leak)
```

## 4. Rate (the human step — single rater)

Open `$OUT/ranking_sheet.html` in a browser. For each GIF: order its outputs best→worst, and mark the **lowest one still acceptable** (the single boundary). Save → `$OUT/human_judgements.csv` (or edit the CSV directly as a fallback).

## 5. Ingest + analyse

```bash
poetry run python -m scripts.audit.validation_gate.ingest_rankings --out "$OUT"
poetry run python -m scripts.audit.validation_gate.analyse --out "$OUT"
# → $OUT/agreement.csv  (per-GIF + per-bucket Spearman/Kendall)
# → $OUT/floors.csv     (per-bucket floor median + IQR)
# → $OUT/disagreements.csv (per divergence: bucket × operation_axis × dominant submetric)
```

## 6. Render the verdict report

```bash
poetry run python -m scripts.audit.validation_gate.render_report --out "$OUT"
# → $OUT/report.md  (GO / NO_GO / LOW_CONFIDENCE + agreement tables + floors + disagreement report + figures)
```

## 7. Read the verdict, commit the report

- **GO** (mean Spearman ≥ 0.85, no bucket ≤ 0.5) → the metric is rankable; the per-bucket floors in `floors.csv` are the leaderboard's operating points. Proceed to M2.
- **NO_GO** → the disagreement report names where/why; fixing the metric becomes the next priority (separate effort — the gate does not touch metric code).
- **LOW_CONFIDENCE** (near-threshold, single rater) → bring in a second rater for the marginal buckets before committing.

Commit `report.md` (and `floors.csv` / `agreement.csv`); raw `study_outputs/*.gif` stay gitignored.

## Acceptance check (maps to spec)

- A single go/no-go verdict + per-bucket agreement numbers exist in `report.md` (SC-001, US1).
- Every represented bucket has a floor + spread in `floors.csv` (SC-003, US2).
- Every beyond-margin divergence appears in `disagreements.csv` with bucket + operation axis + contributing sub-metric (SC-004, US3).
- The whole rating step fits one session (≤ ~20 GIFs / ~100 outputs) (SC-005).
- Re-running from the committed `study_manifest.csv` reproduces the verdict (FR-010).
