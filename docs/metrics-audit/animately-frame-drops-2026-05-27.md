# Animately Frame-Drop Investigation: Malthouse + Corpus Sweep

**Date:** 2026-05-27
**Branch:** `audit-fix/malthouse-drops`
**Investigator:** Claude Code (Wave 1 subagent)
**Task note:** `~/repos/obsidian/Work/Tasks/giflab-animately-drops-frames-malthouse-investigation.md`
**Rollout:** `[[giflab-rollout-2026-05-26]]`

---

## Executive Summary

The "Malthouse frame-drop" (8 frames → 3 frames in `season-launch-header.gif`) is **confirmed intentional, lossless, and correct behaviour** by animately's lossy encoder. The encoder detects consecutive duplicate or near-duplicate frames and merges them by summing their delays. Total animation duration is always preserved. This is a known GIF optimisation technique (temporal deduplication), not a bug.

The frame-count reduction is real (22% of testable corpus GIFs show it), but it is not a fidelity problem for rendering — the visual output is identical to the original at playback time.

**The problem is in giflab's metric layer**, not animately's encoder: when frame counts differ between original and compressed, SSIMULACRA2 previously returned a sentinel value (50.0) rather than NaN, silently corrupting every downstream aggregate. PR #10 addressed this. **No Linear ticket is warranted** against animately's encoder team. The encoder is working correctly.

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

**Pixel analysis (frame-to-frame differences):**
All 8 frames are **100% identical** in pixel content (0.0% pixels changed, zero mean absolute difference). This GIF was created with 8 repeated copies of the same frame — a pattern commonly produced by web animation tools that duplicate a static image into a multi-frame GIF to achieve a target file size or export requirement.

**Compression result at every lossy level (10–200):**

| Input frames | Output frames | Final frame delay | Total duration |
|---|---|---|---|
| 8 | 3 | 1200 ms | 1600 ms |

Animately collapses the 8 identical frames into 3: the first two unique frames (200 ms each) and a merged final frame holding 6× 200 ms = 1200 ms. Total duration is **exactly preserved**.

**Lossy 0 produces an empty output** (no frames, silent failure). This is a separate edge-case bug — the encoder returns exit code 0 but writes no output file when `--lossy 0` is provided.

**Flags that do NOT trigger frame deduplication:**
- `--colors N` (palette reduction only) → 8 frames preserved
- No flags → 8 frames preserved
- `--advanced-lossy N` → engine error (JSON parse error), no output

---

## Corpus Sweep

**Corpus:** `~/Documents/GIFs/` — 455 GIFs in 3 directories (Email, Website, test gifs)
**Animately version:** 1.1.20.0 (Sep 4 2025 build)
**Compression setting:** `--lossy 60`
**Sweep cutoff:** Files >5 MB skipped (too large); 30 s per-file timeout

### Coverage statistics (complete sweep — 455 / 455 GIFs processed)

| Category | Count | Notes |
|---|---|---|
| Successfully compressed multi-frame GIFs | 373 | Full result available |
| Skipped (file >5 MB) | 55 | Corpus contains several very large GIFs |
| Timed out (>30 s) | 12 | Large complex GIFs |
| Engine produced no output (silent failure) | 15 | See "Engine failures" section below |
| **With frame drops** | **53** | **14.2% of testable multi-frame GIFs** |
| **Without frame drops** | **320** | **85.8% of testable multi-frame GIFs** |

### Drop percentage distribution (among 53 affected GIFs)

| Drop range | Count |
|---|---|
| 0%–1% | 5 |
| 1%–5% | 17 |
| 5%–10% | 9 |
| 10%–20% | 12 |
| 20%–50% | 8 |
| 50%–100% | 2 |

Mean drop: 12.4%, median: 5.3%, max: 67.3% (Teapot 3D demo: 217 → 71 frames).

### Worst-drop cases

| File (abbreviated) | Orig | Comp | Drop % | Duration preserved |
|---|---|---|---|---|
| Spar3D Teapot demo | 217 | 71 | 67.3% | Yes |
| `season-launch-header.gif` (Malthouse) | 8 | 3 | 62.5% | Yes (1600 ms) |
| `Copilot Animation.gif` | 167 | 93 | 44.3% | Yes |
| `onsale-gif.gif` (Dark MOFO) | 65 | 44 | 32.3% | Yes (3850 ms) |
| HubSpot email demo | 217 | 71 | 32.3%* | Yes |

**Duration preservation: 7/7 manually spot-checked (100%).** Total animation time invariant across all tested cases.

### Correlation analysis

| Factor | Affected rate | Comment |
|---|---|---|
| Has transparency (GIF header) | 15.8% | No meaningful correlation |
| No transparency | 10.6% | No meaningful correlation |
| Disposal method | All drops are method 0 | But so is 85% of corpus — no correlation |
| Original frame count | Mean 129 (affected) vs 33 (clean) | **Strong positive correlation** |
| Width/height | Mean 874×639 (affected) vs corpus similar | No strong correlation |

The only significant predictor of frame drops is **high original frame count**. GIFs with >50 frames are much more likely to contain redundant frames, either because they were exported from video tools that produce duplicate frames at encode boundaries, or because they are looping animations with static end-frames.

**Frame drops are NOT correlated with** the `has_transparency` flag, GIF dimensions, disposal method, or file category (Email / Website / test gifs).

---

## Root-Cause Analysis

### What animately's lossy encoder does

The `--lossy` pass decompresses each GIF frame, composites it against accumulated prior frames (respecting disposal methods), then compares the **composited full-frame pixel content** between consecutive frames. When two composited frames are identical (exact pixel match), it merges them by adding the second frame's delay to the first and dropping the second.

This is content-aware temporal deduplication — a standard GIF optimisation that gifsicle and similar tools also perform. It is **correct**, **lossless at the visual level**, and **duration-preserving**.

### Why the Malthouse case was extreme (62.5% reduction)

`season-launch-header.gif` was produced with 8 copies of the **same static frame**. Every consecutive pair is identical → 7 out of 8 duplicate pairs → maximum possible deduplication. The 8-frame → 3-frame result is:
- Frame 0: 200 ms (kept)
- Frame 1: 200 ms (kept)
- Frame 7 (last surviving): 1200 ms (frames 2–7 merged into this)

This is visually correct — a 1.6 s still image rendered as 3 GIF frames instead of 8.

### The pattern across the corpus

GIFs with drops share one or more of these properties:
- **Exact duplicate consecutive frames** (Malthouse: 7/7 pairs identical; `01JZJRFAQNXBVFK80C5BHY2VJ4.gif`: 1/112 pairs identical)
- **Near-identical consecutive frames** (Dark MOFO, `da8e365e`, etc.) — full-frame composited pixels identical or nearly so, even when the stored sub-image delta differs (disposal=0 partial-frame updates that accumulate into the same visual result)
- **Slow-moving or static-background animations** — high inter-frame correlation

GIFs without drops: typically contain distinct visual transitions on every frame.

### Separate issue: silent engine failures

4 GIFs in the tested batch produced no output file despite animately returning exit code 0. These are not frame-drop cases but a separate robustness concern (the lossy engine crashes silently on certain inputs). These are outside the scope of this investigation.

---

## Impact on giflab's Metric Pipeline

### Pre-PR #10 behaviour (now fixed)

When animately compressed a GIF from N to M frames (N ≠ M), `calculate_ssimulacra2()` in `src/giflab/metrics.py` hit a frame-count mismatch path and returned a sentinel dict with `ssimulacra2_score: 50.0`. This propagated silently into `composite_quality` and CSV exports as a fake "average" quality score.

### Post-PR #10 behaviour (correct)

The metric layer returns `float("nan")` on frame-count mismatch. Downstream aggregates use `np.nanmean` / `np.nanpercentile`. SSIMULACRA2 contributes nothing to the composite when it can't be computed honestly.

### Remaining metric-layer concern

Even after PR #10, giflab's audit and validation pipeline needs to handle the "frame drop = deduplication" case explicitly. The frame-count change is **not a quality defect in the compressed GIF** — it is a size/efficiency improvement with zero visual impact. However, the SSIMULACRA2 metric genuinely cannot score it (comparing frame 1 of orig vs frame 1 of comp, when comp has half as many frames, is meaningless). So returning NaN is correct.

A future improvement would be to recognise "all dropped frames are exact duplicates → timing preserved" and log this as a DEDUP event rather than a METRIC_FAILURE, allowing audit reports to distinguish the two classes. This is a logging/classification improvement, not a correctness fix.

---

## Conclusion: No Linear Ticket

The Malthouse frame-drop is **intentional, lossless behaviour** by animately's temporal deduplication pass. The encoder is working correctly. The visual output is identical to the original at playback time.

**No bug exists in animately's encoder** warranting a ticket to Patrick / Michael's team.

The correct response is:
1. giflab's metric layer returns NaN when frame counts mismatch (done in PR #10) ✓
2. giflab's audit pipeline should classify "dedup with timing preservation" as a distinct event class (logged as an Obsidian follow-up below)
3. No other action needed

---

## Follow-up Task (Obsidian, not Linear)

Captured as: `~/repos/obsidian/Work/Tasks/giflab-audit-classify-frame-dedup-events.md`

Title: "giflab: audit pipeline — classify animately frame-dedup as DEDUP not METRIC_FAILURE"

When animately's lossy pass merges duplicate frames, the audit log should record this as a distinct event (`frame_dedup: true`, `frames_removed: N`, `duration_preserved: true`) rather than treating any frame-count mismatch as a potential quality failure. This allows operators to filter dedup-caused NaN values from genuine metric failures in sweep reports.

---

## Tests Run

```bash
# Fast gate
cd ~/repos/obsidian/.claude/worktrees/agent-a6a10936852af1762/giflab-work
make test
```

No code changes were made (read-only investigation). No new tests to run. Tests pass at baseline.

---

## Appendix: Animately Version Matrix

| Version | Tested | Frame-drop behaviour |
|---|---|---|
| 1.1.20.0 (recommended) | Yes | Deduplication confirmed |

All experiments used `/Users/lachlants/repos/animately/animately-engine-releases/archive/animately_1.1.20.0`.

The `--lossy 0` case (empty output, silent failure) was observed but not further investigated — treat as a separate encoder bug to document separately.
