# Contracts: Validation-Gate Script Interfaces

This feature exposes **no network/REST API**. Its "contracts" are the command-line interfaces of the five thin study scripts under `scripts/audit/validation_gate/` and the file artifacts they exchange. Each script is pure-ish: deterministic given its inputs, writes named outputs under the run folder, and re-runs idempotently (resume-safe, reusing `_common.py` helpers).

Common option: `--out <dir>` — the run folder, `docs/metrics-audit/<date>/validation-gate/`.

## `build_study select`
- **In**: `--out`, `--n <int>` (study size), optional `--source <dir>` (default `~/Documents/GIFs`).
- **Out**: `study_manifest.csv` with columns `gif_id, path, content_bucket(blank), orig_frames, orig_kb`.
- **Contract**: selects a stratified sample (reuses `build_sample.py`); does NOT assign buckets (FR-008 — human fills them).

## `build_study generate`
- **In**: `--out` (reads `study_manifest.csv`, requires every `content_bucket` filled).
- **Out**: `study_outputs/*.gif`; appends `output_id, engine, params, operation_axis, output_kb, composite_quality, submetrics, metric_config_id, giflab_commit` rows.
- **Contract**: 4–6 outputs/GIF along one-axis-at-a-time ladders (R4); `composite_quality` via `calculate_comprehensive_metrics(force_all_metrics=True)` full-stack config (R5); NaN verdicts recorded as NaN, never a sentinel. Fails loudly on any unlabelled GIF. Does not modify `src/giflab/`.

## `ranking_sheet`
- **In**: `--out` (reads manifest + outputs).
- **Out**: `ranking_sheet.html` — per-GIF, outputs inline in **randomised** order, with order + boundary inputs.
- **Contract**: offline, self-contained (no server); randomisation seed recorded for reproducibility; never displays the `composite_quality` value or order.

## `ingest_rankings`
- **In**: `--out` (reads the filled `ranking_sheet.html` export or hand-edited `human_judgements.csv`).
- **Out**: validated `human_judgements.csv` (`gif_id, ranking, acceptability_boundary_output_id, rater_id, notes`).
- **Contract**: validates each ranking covers exactly its GIF's outputs and the boundary id is one of them; ties allowed.

## `analyse`
- **In**: `--out` (manifest + judgements).
- **Out**: `agreement.csv`, `floors.csv`, `disagreements.csv`.
- **Contract**: per-GIF Spearman ρ (+ Kendall τ) with tie-average ranks (R1); per-bucket floor = median + IQR of boundary `composite_quality` (R2); disagreements beyond margin attributed to dominant weighted sub-metric (R6), grouped by bucket × operation_axis. NaN verdicts excluded and counted.

## `render_report`
- **In**: `--out` (all of the above).
- **Out**: `report.md` (+ figures) containing the `GateVerdict` and tables.
- **Contract**: applies the decision rule (`GO` iff mean ρ ≥ 0.85 and worst bucket > 0.5; else `NO_GO`; `LOW_CONFIDENCE` near-threshold under single rater); reuses `report.py` rendering helpers.
