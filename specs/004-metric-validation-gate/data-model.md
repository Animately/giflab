# Phase 1 Data Model: Metric Validation Gate

All records are small, file-backed (CSV/JSON under `docs/metrics-audit/<date>/validation-gate/`), and validated with `pydantic` (consistent with `prediction/schemas.py`). These are study artifacts, not training-data pipeline rows (see plan Constitution Check / Principle IV deviation).

## ContentBucket (enum)

The 5 compression-behaviour buckets from `docs/technical/compression-pipeline-leaderboard.md`:
`photographic` · `flat_graphic` · `animation` · `text_screen` · `pixel_art`.

## OperationAxis (enum)

Which single operation a study output varies (for FR-007 attribution): `lossy` · `colour` · `frame`.

## StudyGif

One original GIF selected for the study, hand-assigned to a bucket.

| Field | Type | Notes |
|---|---|---|
| `gif_id` | str | stable id (content hash or slug) |
| `path` | Path | source under `~/Documents/GIFs` |
| `content_bucket` | ContentBucket | **hand-assigned** (FR-008); blank in the generated stub until the rater fills it |
| `orig_frames` | int | from metadata |
| `orig_kb` | float | original size |

Validation: `content_bucket` must be set before output generation proceeds (build fails loudly on unlabelled rows).

## StudyOutput

One compressed result of a StudyGif, with its verdict and sub-metrics.

| Field | Type | Notes |
|---|---|---|
| `output_id` | str | `{gif_id}__{engine}__{axis}__{step}` |
| `gif_id` | str | FK → StudyGif |
| `engine` | str | wrapper/engine used |
| `params` | dict | e.g. `{"lossy": 60}` / `{"colors": 32}` / `{"frame_ratio": 0.5}` |
| `operation_axis` | OperationAxis | the single axis this output varies |
| `output_kb` | float | result size |
| `composite_quality` | float | may be **NaN** (excluded from agreement/floor maths, recorded — never treated as a score) |
| `submetrics` | dict[str, float] | the 11 weighted contributors (for R6 attribution) |
| `metric_config_id` | str | records the full-stack config used (R5) |
| `giflab_commit` | str | reproducibility (FR-010) |

Relationship: each StudyGif has 4–6 StudyOutputs spanning quality (R4).

## HumanJudgement

One rater's judgement of a single GIF's outputs.

| Field | Type | Notes |
|---|---|---|
| `gif_id` | str | FK → StudyGif |
| `ranking` | list[str] | output_ids ordered best→worst (ties allowed → equal rank) |
| `acceptability_boundary_output_id` | str \| null | the **lowest-quality output still acceptable** (FR-002); null = "no boundary crossed" (all or none acceptable) |
| `rater_id` | str | single rater by design (clarification) |
| `notes` | str | optional rater uncertainty note |

Validation: `ranking` must contain exactly the GIF's outputs; boundary id (if set) must be one of them.

## AgreementResult

Per-GIF and per-bucket rank agreement (R1).

| Field | Type | Notes |
|---|---|---|
| `scope` | str | a `gif_id` or a bucket name |
| `scope_kind` | str | `gif` \| `bucket` |
| `spearman_rho` | float | primary measure |
| `kendall_tau` | float | secondary |
| `n_outputs` | int | outputs compared (NaN verdicts excluded) |
| `confidence` | str | `ok` \| `low` (low if too few outputs/GIFs) |

## QualityFloor

Per-bucket calibrated acceptability threshold (R2, SC-003).

| Field | Type | Notes |
|---|---|---|
| `content_bucket` | ContentBucket | |
| `floor_composite_quality` | float | median of per-GIF boundary composite_quality values |
| `spread_iqr` | float | reported spread |
| `n_boundaries` | int | GIFs that crossed a boundary in this bucket |
| `confidence` | str | `ok` \| `low` (low if `n_boundaries` below a small minimum) |

## DisagreementRecord

One systematic human↔verdict divergence (FR-007, R6).

| Field | Type | Notes |
|---|---|---|
| `gif_id` | str | |
| `output_id` | str | |
| `content_bucket` | ContentBucket | grouping key |
| `operation_axis` | OperationAxis | grouping key (clarification) |
| `human_rank` | int | |
| `verdict_rank` | int | |
| `rank_delta` | int | beyond margin → recorded |
| `dominant_submetric` | str | the contributor that most drove the divergence (R6) |
| `submetric_weight` | float | its composite_quality weight |

## GateVerdict

The single go/no-go outcome (US1, SC-001/002).

| Field | Type | Notes |
|---|---|---|
| `decision` | str | `GO` \| `NO_GO` \| `LOW_CONFIDENCE` |
| `mean_spearman` | float | across study-set GIFs |
| `worst_bucket_rho` | float | the minimum per-bucket agreement |
| `threshold_mean` | float | 0.85 (clarified) |
| `threshold_inversion` | float | 0.5 (clarified) |
| `rater_count` | int | 1 (clarified) |
| `date` | str | run date (study folder) |
| `giflab_commit` | str | reproducibility |
| `rationale` | str | which criterion drove the decision |

Decision rule: `GO` iff `mean_spearman ≥ 0.85` AND `worst_bucket_rho > 0.5`; `NO_GO` if any bucket `≤ 0.5` or `mean_spearman` clearly below; `LOW_CONFIDENCE` if near-threshold under a single rater (clarified edge case).
