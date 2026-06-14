# Feature Specification: Metric Validation Gate

**Feature Branch**: `004-metric-validation-gate`
**Created**: 2026-06-14
**Status**: Draft
**Input**: User description: "A metric validation gate that establishes whether giflab's composite_quality verdict is trustworthy enough to rank compression outputs, and calibrates the per-content-type quality floor the compression-pipeline leaderboard ranks at. Generate a study set of compressed GIF outputs spanning quality levels across content-type buckets; collect human quality rankings and acceptability cutoffs; measure agreement between human judgement and composite_quality near the operating point; emit a go/no-go decision on the metric plus a calibrated per-bucket quality floor. First gated milestone of the compression-pipeline leaderboard rebuild (see docs/technical/compression-pipeline-leaderboard.md) — no leaderboard runner, cross-engine chaining, or auto-labeller is built here."

## Overview

The compression-pipeline leaderboard ranks compression outputs by quality using the `composite_quality` verdict, and it eliminates losing pipelines based on that ranking. Elimination is **irreversible** — once a pipeline is judged worse and dropped, it never reappears. Therefore the leaderboard is only as trustworthy as `composite_quality` is *rank-correct*: it must order a GIF's compressions the way a human would, especially near the "still looks acceptable" boundary where elimination decisions are tightest.

This feature is a **gate**, not a pipeline. It answers one question before any leaderboard machinery is built — *can we trust `composite_quality` to rank?* — and produces the one number the leaderboard needs to operate: the **quality floor** (the `composite_quality` value at the boundary of acceptable quality), calibrated per content type. It is the first gated milestone of the leaderboard rebuild; if the metric fails the gate, fixing the metric becomes the next priority instead of building the leaderboard.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Decide whether the quality verdict can be trusted to rank (Priority: P1)

A researcher preparing to build the compression-pipeline leaderboard needs to know, before investing in any elimination machinery, whether the `composite_quality` verdict orders a GIF's compressions the way a person does. They assemble a study set of compressed outputs spanning a range of visible quality across several content types, record how a human ranks those outputs by quality, and compare that ranking to the verdict's ranking. The outcome is a clear go/no-go decision backed by per-content-type agreement numbers.

**Why this priority**: This is the foundational, gating capability. Every downstream elimination decision in the leaderboard rests on the verdict being rank-correct, and elimination is irreversible. Without a trustworthy verdict there is no defensible basis for building the leaderboard. It is the cheapest experiment that can invalidate the entire plan, so it must come first and stand alone.

**Independent Test**: Assemble the study set, capture a human ranking for each study GIF's compressions, and run the comparison. The story is satisfied if it yields a single go/no-go verdict plus a per-content-type table of how closely the human and the verdict agree — with no leaderboard, runner, or model required.

**Acceptance Scenarios**:

1. **Given** a study set of compressed outputs across multiple content types and a human quality ranking for each study GIF's compressions, **When** the gate compares the human ranking to the `composite_quality` ranking, **Then** it reports a per-content-type agreement measure and a single overall go/no-go verdict against a stated agreement threshold.
2. **Given** the human and the verdict agree closely in every content bucket, **When** the gate evaluates the threshold, **Then** the verdict is "GO" and the result records the agreement evidence that justified it.
3. **Given** at least one content bucket where the verdict's ranking is systematically inverted relative to the human, **When** the gate evaluates the threshold, **Then** the verdict is "NO-GO" and the offending bucket(s) are named.

---

### User Story 2 - Calibrate the per-content-type quality floor (Priority: P2)

The leaderboard ranks at an iso-quality operating point — "the smallest file that still looks acceptable." To do that it needs a concrete acceptability threshold expressed as a `composite_quality` value, and that threshold differs by content type (a photo and a flat graphic tolerate different amounts of degradation before they look bad). The researcher captures, for each study GIF, the point at which the human judges quality to drop from acceptable to unacceptable, and from those judgements derives a quality floor for each content bucket.

**Why this priority**: Necessary to operate the leaderboard, but only meaningful once the verdict has passed the trust gate (Story 1). A floor calibrated on an untrustworthy verdict would be meaningless, so this is strictly downstream of Story 1.

**Independent Test**: From the human acceptability cutoffs collected for the study set, derive one quality floor per content bucket and report each with its spread/uncertainty. Satisfied if every represented bucket has a floor value usable as an operating point.

**Acceptance Scenarios**:

1. **Given** human "lowest still-acceptable" markers for the study GIFs in a content bucket, **When** the gate calibrates that bucket, **Then** it produces a single quality-floor value for the bucket together with a measure of its spread across the GIFs.
2. **Given** content buckets that demonstrably tolerate different degradation, **When** floors are calibrated, **Then** the per-bucket floors differ accordingly rather than collapsing to one global value.
3. **Given** a bucket with too few study GIFs to calibrate confidently, **When** the gate calibrates, **Then** it flags that bucket's floor as low-confidence rather than reporting a falsely precise value.

---

### User Story 3 - Surface systematic verdict biases for follow-up (Priority: P3)

When the human and the verdict disagree, the researcher needs to understand *why* and *where*, so that any metric fix can be targeted. The gate characterises systematic disagreements — grouped by content type and by the kind of compression that produced them — and attributes each to the contributing part of the verdict, producing a short report that becomes the input to metric-fix work.

**Why this priority**: Diagnostic value that turns a "NO-GO" into actionable next steps, and adds confidence to a "GO". Useful but not required for the gate's primary decision, so lowest priority.

**Independent Test**: On a study set with known disagreements, confirm the gate emits a report listing each systematic divergence with its content bucket, the compression family involved, and the contributing component of the verdict.

**Acceptance Scenarios**:

1. **Given** outputs where the human and verdict rankings diverge beyond a set margin, **When** the gate analyses them, **Then** each systematic divergence is listed with its content bucket and the compression family it involves.
2. **Given** a listed divergence, **When** the researcher reads the report, **Then** it names the contributing component of the verdict so a fix can be scoped.

---

### Edge Cases

- A human ranking contains ties (two compressions judged equal quality) → the agreement measure handles ties without inflating or deflating agreement.
- A study GIF where every compression is judged acceptable (the floor is never crossed) or none is acceptable → the gate records "no floor crossing observed" for that GIF rather than inventing a floor.
- The verdict cannot be computed for an output (returns an undefined / `NaN` value) → that output is excluded from agreement and floor calculations transparently, and the exclusion is recorded, never silently treated as a score.
- The human rater is uncertain about an ordering → the uncertainty is captured rather than forced into a false-precise rank.
- A content bucket has too few study GIFs to support a verdict or a floor → results for that bucket are flagged low-confidence rather than reported as firm.
- Only a single rater is available → the gate still produces a verdict, and the single-rater limitation is recorded as a stated confidence caveat.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The gate MUST assemble a study set of compressed outputs that spans a range of visible quality for each study GIF, drawn from multiple content-type buckets.
- **FR-002**: The gate MUST capture, for each study GIF, a human **quality ranking** of that GIF's compressions and a human **"lowest still-acceptable"** marker.
- **FR-003**: The gate MUST obtain a `composite_quality` verdict for every study output, computed under the same configuration the leaderboard will use near the operating point.
- **FR-004**: The gate MUST quantify the agreement between the human ranking and the `composite_quality` ranking, reported per content-type bucket.
- **FR-005**: The gate MUST emit a single go/no-go verdict on whether `composite_quality` is trustworthy enough to rank, evaluated against a stated, recorded agreement threshold.
- **FR-006**: The gate MUST derive a per-content-type quality **floor** — the `composite_quality` value at the human acceptability boundary — and report each floor with its spread / uncertainty.
- **FR-007**: The gate MUST characterise systematic disagreements between human and verdict (grouped by content bucket and compression family) and attribute each to a contributing component of the verdict.
- **FR-008**: Study GIFs MUST be assigned to content-type buckets by hand for this gate; the gate MUST NOT depend on an automated content classifier.
- **FR-009**: The gate MUST persist its outputs — the verdict, the per-bucket floors, the agreement tables, and the disagreement report — to a dated, versioned location alongside prior metric-audit records.
- **FR-010**: The study MUST be reproducible: the GIF set, the compression settings used to produce each output, and the verdict configuration and software version MUST be recorded so the result can be regenerated.

### Key Entities

- **Study GIF**: an original GIF selected for the study, with a hand-assigned content-type bucket.
- **Study Output**: one compressed result of a study GIF, with its compression settings, its file size, and its `composite_quality` verdict.
- **Human Judgement**: for one study GIF, the human's ranking of its outputs by quality plus the "lowest still-acceptable" marker.
- **Agreement Result**: per content bucket, how closely the human ranking and the verdict ranking match.
- **Quality Floor**: per content bucket, the calibrated acceptability threshold expressed as a `composite_quality` value, with its spread.
- **Gate Verdict**: the overall go/no-go decision with its supporting evidence and the threshold it was judged against.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: The gate produces an unambiguous go/no-go decision accompanied by a per-content-type agreement number for every represented bucket (no decision rests on unquantified judgement).
- **SC-002**: A "GO" decision requires the human and verdict rankings to agree strongly across the study set — a mean per-GIF rank agreement of at least 0.85 — AND no content bucket exhibiting systematic inversion (rank agreement at or below 0.5). *(Thresholds are the recorded starting values and may be tuned in clarification.)*
- **SC-003**: Every content bucket represented in the study set receives a quality-floor value with a stated spread, usable directly as the leaderboard's iso-quality operating point.
- **SC-004**: Every human-versus-verdict disagreement beyond the set margin appears in the disagreement report, attributed to its content bucket and a contributing component of the verdict.
- **SC-005**: The human ranking effort required to run the gate fits within a single focused session (on the order of 20 GIFs and 100 compressed outputs).

## Assumptions

- A single experienced rater is sufficient to produce the go/no-go decision; a second rater is optional and would only raise confidence. The single-rater limitation is recorded as a caveat.
- The study set is on the order of 15–20 GIFs with roughly 4–6 compressions each (≈ 60–120 outputs), enough to span quality and cover buckets within one session.
- The content-type buckets are the five compression-behaviour buckets defined in `docs/technical/compression-pipeline-leaderboard.md` (photographic/continuous-tone, flat-graphic/vector, animation/cartoon, text/screen-capture, pixel-art), hand-assigned for this gate.
- The agreement thresholds in SC-002 are reasonable starting values, not settled policy; they may be adjusted during clarification before the gate is run.
- Study GIFs are drawn from the existing local GIF collection used for prior metric-audit work.
- "Near the operating point" means the verdict configuration matches what the leaderboard will use to rank at the acceptability boundary, rather than a different or cheaper configuration.

## Dependencies

- The `composite_quality` verdict and the underlying quality metrics must be available and computable for arbitrary reference / compressed GIF pairs.
- The ability to produce compressed outputs spanning a quality range from study GIFs.
- The content-type taxonomy defined in `docs/technical/compression-pipeline-leaderboard.md`.
- A dated metric-audit output location consistent with prior audit records, for persisting results (FR-009).

## Out of Scope

- The compression-pipeline leaderboard runner, Pareto elimination, and cross-engine pipeline chaining (later milestones M2 / M3).
- Any automated content-type classifier or contact-sheet review tooling (later milestone).
- The machine-learning curve predictor.
- Building production tooling or a reusable harness beyond what the gate itself needs to run once and produce its verdict and floors.
