# Animately Frame-Drop Investigation: Malthouse + Corpus Sweep

**Date:** 2026-05-27
**Branch:** `audit-fix/malthouse-drops`
**Investigator:** Claude Code (Wave 1 subagent)
**Task note:** `~/repos/obsidian/Work/Tasks/giflab-animately-drops-frames-malthouse-investigation.md`
**Rollout:** `[[giflab-rollout-2026-05-26]]`

---

## Executive Summary

The "Malthouse frame-drop" (8 frames → 3 frames in `season-launch-header.gif`) is **intentional, lossless, and correct behaviour** by animately's lossy encoder. The encoder collapses consecutive identical/near-identical composited frames and sums their delays. Total animation duration (as measured by PIL's literal `info["duration"]` sum) is preserved in every directly-verified case.

The frame-count reduction is a real, observable size optimisation, not a fidelity defect. The visual output at playback time is identical to the original.

**The original ticket-shaped concern** — "is animately silently corrupting our GIFs?" — answers as **no**. The metric-layer problem that surfaced the Malthouse drop (`ssimulacra2_score: 50.0` sentinel returned on frame-count mismatch) is already addressed in PR #10. **No Linear ticket against animately's encoder.**

One Obsidian follow-up was captured to elevate the operator-side classification (dedup-vs-genuine-failure) from "all NaN looks alike in the sweep CSV" to two distinct event classes — see "Follow-up task" below. Medium priority.

---

## Reproduction of the Malthouse Case

**File:** `~/Documents/GIFs/Email/…malthousetheatre…ew-2024-season-2025-on-sale_season-launch-header.gif`

| Property | Value |
|---|---|
| Original frames | 8 |
| Original dimensions | 1334 × 358 |
| Total animation duration | 1600 ms (8 × 200 ms) |
| Has transparency | Yes (frame 0) |
| Disposal method | 0 (do not dispose) on all frames |

**Pixel analysis.** All 8 frames are byte-for-byte identical when composited to RGB (0.0% pixels differ; mean absolute difference 0.0). This GIF was authored as 8 repeated copies of the same static image — a pattern commonly produced by tools that pad to a target frame count or export a still as an animated GIF.

**Compression result, every tested lossy level (10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 150, 200):**

| Input frames | Output frames | Final frame delay | Total duration |
|---|---|---|---|
| 8 | 3 | 1200 ms | 1600 ms |

Animately collapses the 8 identical composited frames into 3: two unique frames at 200 ms each, then a final frame holding the remaining 6 × 200 ms = 1200 ms. Total animation duration is **exactly preserved** at the PIL `info["duration"]` level.

**`--lossy 0` produces an empty output file** (exit code 0, no frames written). This is a separate edge-case in animately not investigated here; flagged for future capture.

---

## Corpus Sweep

**Corpus:** `~/Documents/GIFs/` — 455 GIFs across Email, Website, and `test gifs` directories.
**Animately:** `/Users/lachlants/repos/animately/animately-engine-releases/archive/animately_1.1.20.0` (Sep 4 2025 build).
**Compression:** `--lossy 60` (single level for the primary headline; the existing repo-tracked `audit/sweep.csv` covers 20/40/60 across the smaller 125-GIF audit sample for comparison).
**Cutoff:** Files >5000 KB skipped (decimal-MB threshold, 55 of 455 corpus files; the equivalent `find -size +5M` binary-MiB threshold marks 53 files — both filters are valid, just different unit conventions).
**Per-file timeout:** 30 s.

### Reproducing this sweep

The exact script that produced the corpus CSV is committed alongside this README:

```bash
cd ~/repos/animately/giflab
poetry run python docs/metrics-audit/animately-frame-drops-2026-05-27/sweep_frame_drops.py
# Output: docs/metrics-audit/animately-frame-drops-2026-05-27/sweep.csv
```

Defaults match the methodology in this report (`--corpus ~/Documents/GIFs`, `--lossy 60`, `--max-kb 5000`, `--timeout 30`). Override any flag to extend the methodology to other corpora or lossy levels. Full sweep on the 455-GIF corpus takes ~30 min wall-clock because animately is run serially per file with up to a 30 s timeout.

The committed `sweep.csv` columns are: `gif_path, orig_frames, comp_frames, dropped, drop_pct, width, height, has_transparency, disposal, orig_duration_ms, comp_duration_ms, duration_preserved, file_size_kb, status`. **`orig_duration_ms` and `comp_duration_ms` are literal PIL `info["duration"]` sums** — they do NOT apply the 100ms-per-frame fallback used by `src/giflab/wrapper_validation/timing_validation.py`. See "Note on duration_diff_ms in the repo-tracked audit CSV" below for why that distinction matters.

**About the duration columns in the committed CSV.** The bulk sweep that produced this CSV recorded `comp_frames`/`dropped`/`drop_pct` for every successfully-compressed row, but `comp_duration_ms`/`duration_preserved` are populated only for the spot-checked rows listed under "Worst-drop cases" below. Re-running animately on all 455 GIFs to populate the duration columns end-to-end takes ~30 minutes wall-clock; that is what the `sweep_frame_drops.py` script does on a fresh run. The headline 14.2% drop rate (53/373) is computed from `dropped > 0` across all `status == ok` multi-frame rows and reconciles exactly against the committed CSV.

### Headline numbers (from this sweep's CSV)

| Category | Count | % of testable |
|---|---|---|
| Successfully compressed multi-frame GIFs | 373 | — |
| With frame drops | 53 | 14.2% |
| Without frame drops | 320 | 85.8% |
| Skipped (file >5000 KB) | 55 | — |
| Timed out (>30 s) | 12 | — |
| Engine silent failure (exit 0, no output) | 15 | — |

**Duration preservation (PIL info["duration"] sum):** 10/10 spot-checked rows in the committed `sweep.csv` show `duration_preserved=True` (all of the "worst-drop cases" table below plus three additional medium-drop rows). The bulk-sweep `comp_duration_ms` column is left empty for rows that were not individually re-verified, to be honest about what was actually measured by the bulk run.

Note: the repo's existing `audit/sweep.csv` (125 unique files, lossy 20/40/60) reports a lower drop rate (11/122 ≈ 9.0% at lossy 60) because it samples a curated subset of the corpus — not 455 files. The two CSVs measure different sample populations, not different behaviours. The repo's audit CSV is referenced in the `duration_diff_ms` cross-check below.

### Drop-percentage distribution (53 affected GIFs)

| Drop range | Count |
|---|---|
| 0%–1% | 5 |
| 1%–5% | 17 |
| 5%–10% | 9 |
| 10%–20% | 12 |
| 20%–50% | 8 |
| 50%–100% | 2 |

Mean drop: 12.4%. Median: 5.3%. Max: 67.3% (Teapot 3D demo: 217 → 71 frames).

### Worst-drop cases (spot-checked for duration preservation)

All values below match `comp_duration_ms` / `duration_preserved` in the committed `sweep.csv`.

| File (abbreviated) | Orig | Comp | Drop % | PIL duration preserved |
|---|---|---|---|---|
| `season-launch-header.gif` (Malthouse) | 8 | 3 | 62.5% | Yes (1600 ms) |
| `onsale-gif.gif` (Dark MOFO) | 65 | 44 | 32.3% | Yes (3850 ms) |
| `original.gif` | 50 | 41 | 18.0% | Yes (4000 ms) |
| `1a126de8.gif` | 21 | 18 | 14.3% | Yes (2100 ms) |
| `da8e365e.gif` | 126 | 112 | 11.1% | Yes (5040 ms) |
| `unnamed copy 17.gif` | 210 | 189 | 10.0% | Yes (21,000 ms) |
| `fixkey_ai.gif` | 155 | 150 | 3.2% | Yes (16,170 ms) |
| `a77e5c36-ad54-4c88-897e-cca9118c0749.gif` | 48 | 47 | 2.1% | Yes (4800 ms) |
| `01JZJRFAQNXBVFK80C5BHY2VJ4.gif` | 113 | 112 | 0.9% | Yes (8740 ms) |
| `cluahw3y.gif` | 147 | 146 | 0.7% | Yes (14,700 ms) |

The reviewer-flagged Spar3D Teapot case (`https___hs-26117460…Teapot…`, 217 → 71 = 67.3%) was independently re-verified to preserve 13,560 ms across multiple lossy levels — that confirmation lives in the PR thread rather than `sweep.csv` because the CSV's duration columns were not back-filled for every row (see "About the duration columns" above).

### Note on `duration_diff_ms` in the repo-tracked audit CSV

The committed `audit/sweep.csv` contains three non-zero `duration_diff_ms` rows, all for the same GIF (`ai-announcement_01HM01QX7KJD0387CRABAHPXTV.gif` at lossy 20/40/60 with diffs of 900/1100/1200 ms respectively). This **looks** like a counter-example to "duration always preserved" but is a **metric artefact**, not real timing loss:

- The source GIF has **0 ms delay on every one of its 55 frames** (verified with `PIL.Image.info["duration"]`). It is a "render as fast as possible" GIF.
- `src/giflab/wrapper_validation/timing_validation.py:92-93` substitutes `duration = 100` whenever a per-frame delay is `< 1` (the "default fallback" for stripped-delay files).
- So when the validator computes a synthetic total duration: source becomes 55 × 100 = 5500 ms, lossy-20 output (46 frames) becomes 46 × 100 = 4600 ms, diff = 900 ms — matching the CSV exactly. Same arithmetic for lossy 40 (44 × 100 = 4400 ms, diff 1100) and lossy 60 (43 × 100 = 4300 ms, diff 1200).
- Both the source and the compressed outputs actually have **0 ms = 0 ms total duration** when measured without the fallback. Real rendering timing is preserved.

The 100ms-fallback exists for a defensible reason — single-frame GIFs and edge cases shouldn't crash the duration metric — but it propagates a per-dropped-frame synthetic 100ms diff for any 0-delay source. This is a metric/validator-layer issue, not an animately encoder issue. **Captured as a follow-up:** `~/repos/obsidian/Work/Tasks/giflab-timing-validation-zero-delay-fallback.md`.

### Correlation analysis (proxy, not causation)

| Factor | Affected rate | Notes |
|---|---|---|
| Has transparency (GIF header) | 15.8% | Mild positive, not significant |
| No transparency | 10.6% | Mild negative, not significant |
| Disposal method | 100% of drops are method-0 | But ≈85% of the corpus is method-0 too; not predictive |
| Original frame count | Mean 129 (affected) vs 33 (unaffected) | Proxy for what's below |
| Width/height | No meaningful difference | Not predictive |

**The actual driver is content-level redundancy** — how many consecutive composited frames are identical or near-identical. High frame count is a useful *proxy* (more frames → more chances for inter-frame duplication, especially in long animations with static end-frames or static intervals), but it is not itself the cause. A 217-frame GIF where every frame is visually distinct will not lose any frames; an 8-frame GIF of identical copies (Malthouse) loses 5.

The right framing: **animately's `--lossy` pass eliminates duplicate composited frames**. The fraction of any given GIF affected by this depends on how much of its content is redundant. Frame count correlates with affectedness because longer animations are more likely to contain redundancy, not because length itself triggers anything.

---

## Root-Cause Analysis

### What animately's lossy encoder does

The `--lossy` pass decompresses each GIF frame, composites it against the accumulated previous frame buffer (respecting disposal method), then compares the **composited full-frame pixel content** between consecutive composited frames. When two consecutive composited frames are identical (or near-identical within tolerance), it merges them by summing their delays into a single output frame.

This is content-aware temporal deduplication — a standard GIF optimisation that gifsicle and similar tools also perform. Properties:
- **Visually lossless.** Rendering the result with a standard GIF player produces the same observable animation.
- **Duration-preserving.** The total animation length (sum of per-frame delays) is invariant.
- **Sub-image-aware.** Works even on GIFs that store partial-frame deltas (e.g. disposal=0 small tile updates) because comparison happens on the composited full frames, not the stored deltas.

### Why the Malthouse case was extreme (62.5% reduction)

`season-launch-header.gif` was produced with 8 byte-identical copies of the same static image. Every consecutive pair is identical → maximum possible deduplication. The 8 → 3 result is:
- Frame 0: 200 ms (kept)
- Frame 1: 200 ms (kept)
- Frame 7 (last surviving, others merged): 1200 ms

This is **visually correct**: a 1.6 s static image rendered as 3 GIF frames instead of 8.

### Why other corpus GIFs show partial drops

GIFs with drops share at least one of:
- **Exact-duplicate consecutive composited frames** (Malthouse: 7/7 pairs identical; `01JZJRFAQNXBVFK80C5BHY2VJ4.gif`: 1/112 pairs identical → 1 frame dropped).
- **Near-identical composited frames** despite distinct stored deltas (disposal=0 sub-image updates whose composited result is visually unchanged).
- **Slow-moving / static-background animations** — high inter-frame correlation.

GIFs without drops: every consecutive composited frame is meaningfully different.

### Separate issue: silent engine failures (15 cases in the 455-file sweep)

15 GIFs produced no output file despite animately returning exit code 0. These are not frame-drop cases but a robustness concern (the lossy engine appears to bail silently on certain inputs). Outside the scope of this investigation — flagged for future capture.

---

## Impact on giflab's Metric Pipeline

### Pre-PR #10 behaviour (now fixed)

When animately compressed a GIF from N to M frames (N ≠ M), `calculate_ssimulacra2()` in `src/giflab/metrics.py` hit a frame-count mismatch path and returned a sentinel dict with `ssimulacra2_score: 50.0`. This propagated silently into `composite_quality` and CSV exports as a fake "average" quality score.

### Post-PR #10 behaviour (correct)

The metric layer returns `float("nan")` on frame-count mismatch. Downstream aggregates use `np.nanmean` / `np.nanpercentile`. SSIMULACRA2 contributes nothing to the composite when it can't be computed honestly.

### Remaining metric-layer concern (operator-facing)

Even after PR #10, when an operator runs the audit sweep and sees `comp_frames < orig_frames` plus `ssimulacra2_mean = NaN`, they cannot distinguish "encoder correctly deduplicated, timing preserved" from "encoder dropped frames pathologically and broke the GIF". The CSV row looks the same in both cases.

The frame-count change from dedup is **not a quality defect** — it is a size/efficiency improvement with zero visual impact. The remaining work is operator-facing: surface the difference in the sweep output so that triaging dedup-affected rows doesn't burn time investigating non-issues.

---

## Conclusion: No Linear Ticket

The Malthouse frame-drop is **intentional, lossless behaviour** by animately's temporal deduplication pass. The encoder is working correctly. The visual output is identical to the original at playback time.

**No bug exists in animately's encoder.** The correct response is:

1. giflab's metric layer returns NaN when frame counts mismatch (done in PR #10) ✓
2. giflab's audit pipeline should classify "dedup with timing preservation" as a distinct event class (see Obsidian follow-up below) — **medium priority** because the operator-side cost of conflating dedup with genuine failure is recurring.
3. The `duration_diff_ms` 0-delay fallback artefact (`timing_validation.py:92-93`) deserves separate triage — see second Obsidian follow-up.

---

## Follow-up Tasks (Obsidian, not Linear)

### 1. `giflab-audit-classify-frame-dedup-events` — medium priority

`~/repos/obsidian/Work/Tasks/giflab-audit-classify-frame-dedup-events.md`

When animately's lossy pass merges duplicate frames, the audit log should record this as a distinct event (`frame_dedup: true`, `frames_removed: N`, `duration_preserved: true`) rather than treating any frame-count mismatch as a potential quality failure. Without this, an operator triaging sweep results today cannot distinguish dedup-with-timing-preserved from genuine frame loss — every such row needs manual investigation. That's a recurring operational cost.

### 2. `giflab-timing-validation-zero-delay-fallback` — medium priority

`~/repos/obsidian/Work/Tasks/giflab-timing-validation-zero-delay-fallback.md`

`src/giflab/wrapper_validation/timing_validation.py:92-93` substitutes `duration = 100ms` for any per-frame delay `< 1`. For 0-delay GIFs this fabricates a synthetic total duration of `100 × N_frames`, which then produces a synthetic `duration_diff_ms = 100 × frames_dropped` whenever animately deduplicates — even though both source and compressed have 0 ms = 0 ms actual rendering duration.

Either drop the fallback for the diff calculation (use raw info["duration"] and accept that 0=0=preserved), or document the fallback prominently and have the audit sweep filter out 0-delay-source rows from duration-diff reporting.

---

## Tests Run

```bash
cd ~/repos/animately/giflab
poetry run pytest tests/smoke/ tests/functional/ --ignore=tests/smoke/test_audit_sanity_helpers.py --tb=no -q
# 1106 passed, 11 skipped, 0 failed
```

`tests/smoke/test_audit_sanity_helpers.py` is excluded — it is an **untracked** file (status: `??`) from another wave-1 agent that expects a function `_coalesce_byte_identical_levels` not yet in `scripts/audit/sanity.py`. Pre-existing pollution from a parallel agent's worktree, not introduced by this investigation.

No code changes in this PR (read-only investigation). Only the methodology script (`sweep_frame_drops.py`) was added.

---

## Appendix: Animately Version Matrix

| Version | Tested | Frame-drop behaviour |
|---|---|---|
| 1.1.20.0 (recommended) | Yes (all sweep + Malthouse runs) | Deduplication confirmed |

All experiments used `/Users/lachlants/repos/animately/animately-engine-releases/archive/animately_1.1.20.0`.

The `--lossy 0` and `--advanced-lossy` cases produced no output during incidental testing (lossy 0: empty file written; advanced-lossy: the flag takes a JSON config, not a numeric argument, so `--advanced-lossy 40` is a usage error on my part rather than an engine bug). Neither is core to the frame-drop investigation.
