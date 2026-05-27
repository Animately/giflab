# GifLab Metrics Accuracy-Lens Review — 2026-05-26

**Auditor**: Claude Code (automated static review)
**Branch**: `audit-fix/accuracy-lens-audit`
**Scope**: `src/giflab/metrics.py`, `enhanced_metrics.py`, `temporal_artifacts.py`,
`gradient_color_artifacts.py`, `ssimulacra2_metrics.py`, `optimized_metrics.py`,
`optimization_validation/validation_checker.py`, `wrapper_validation/quality_validation.py`
**Not in scope**: tests, CLI, pipeline orchestration, prediction features

This review applies the five accuracy-lens rules from `CLAUDE.md`:
1. **Continuous over discrete** — no cliff-edge thresholds
2. **NaN over fabricated values** — no sentinel substitutes for can't-compute
3. **Pair-wise over single-stream, labelled honestly**
4. **Honest error paths end-to-end** — NaN-aware comparisons throughout
5. **Same key shape across paths** — all code paths emit the same keys with the same semantics

---

## File: `src/giflab/metrics.py`

### `calculate_safe_psnr` — HIGH
**Rule violated**: Rule 2 — NaN over fabricated values
**Where**: `metrics.py:482`
```python
except Exception as e:
    logger.warning(f"PSNR calculation failed: {e}")
    return 0.0
```
**What**: When PSNR calculation fails (corrupt frame, mismatched dtypes, etc.) the function returns `0.0` instead of `float("nan")`. PSNR of 0.0 dB is a legitimate measurement on a nearly-destroyed image. A downstream consumer reading `psnr_mean = 0.0` cannot distinguish "calculation exploded" from "image is maximally corrupted". The same 0.0 value is also appended to the raw list at `metrics.py:2424` on exception, so even the unnormalized PSNR average is contaminated.

Also, `metrics.py:475`: returning `100.0` for MSE=0 (perfect match) is an explicit sentinel. Perfect-match PSNR is genuinely infinite (∞); the commonly cited figure of 100 dB is an arbitrary cap. The cap is self-documented, but downstream code that scales by `PSNR_MAX_DB` (e.g., `min(frame_psnr / float(config.PSNR_MAX_DB), 1.0)`) maps 100 dB back to 1.0 correctly, so this specific sentinel is functionally harmless — but it is a sentinel nonetheless.
**Downstream impact**: `composite_quality` via `enhanced_metrics.normalize_metric("psnr_mean", ...)` treats 0.0 as a very-poor-quality signal. `wrapper_validation/quality_validation.py:277` fails the gate `psnr_value >= 10.0` — potentially rejecting a valid output for a calculation bug. CSV exports record a misleading 0.0 PSNR.
**Suggested direction**: Return `float("nan")` on exception. Aggregate with `np.nanmean`. Let `composite_quality` redistribute weight when PSNR is NaN (already done via the total_weight pattern in `enhanced_metrics.py`). Fix the `validation_checker` and `quality_validation` guards to use NaN-aware comparisons (covered by [[giflab-validation-checker-nan-aware-refactor]]).

---

### `_aggregate_metric` — HIGH
**Rule violated**: Rule 2 — NaN over fabricated values
**Where**: `metrics.py:1536–1541`
```python
if not values:
    return {
        metric_name: 0.0,
        f"{metric_name}_std": 0.0,
        f"{metric_name}_min": 0.0,
        f"{metric_name}_max": 0.0,
    }
```
**What**: When a metric's value list is empty (because every per-frame calculation failed and appended 0.0, or because `aligned_pairs` was empty for this metric), the aggregator returns four 0.0 values rather than NaN. The empty-list case is genuinely "could not measure" — not "measured as zero". Every metric in the loop at `metrics.py:2388–2493` shares the same append-0.0-on-exception pattern, so all NaN-warranted failures reach `_aggregate_metric` as a list of 0.0s, which then become valid-looking non-NaN aggregates.
**Downstream impact**: `composite_quality` weighs 0.0 metrics as if they were real measurements. `validation_checker._validate_quality_thresholds` triggers false-fail on a technically-valid result. The CSV row records 0.0 for every key from a failed metric, silently masking calculation errors.
**Suggested direction**: Audit all `except` blocks that `append(0.0)` in the per-frame loops and change them to `append(float("nan"))`. Change `_aggregate_metric` to use `np.nanmean`/`np.nanstd`/`np.nanmin`/`np.nanmax`, and return NaN for the empty-list case. The `total_weight` accounting in `calculate_composite_quality` already handles missing keys gracefully — it just needs to see a missing key or NaN rather than a fabricated 0.0.

---

### `calculate_temporal_consistency` — MEDIUM (single-stream mislabelling)
**Rule violated**: Rule 3 — Pair-wise over single-stream, labelled honestly
**Where**: `metrics.py:655–712`; result key at `metrics.py:2889`:
```python
result["temporal_consistency"] = float(temporal_post)
```
**What**: The function takes a single list of frames and measures their internal frame-to-frame consistency — it is a single-stream metric. The key `temporal_consistency` implies an original-vs-compressed comparison, as do the aggregate keys `temporal_consistency_std`, `temporal_consistency_min`, `temporal_consistency_max` (which are set to `temporal_post` three times at lines 2890-2892, not to actual statistics). The `_pre` and `_post` variants (lines 2895-2896) are correctly named and expose separate streams. However, `temporal_consistency` (bare key) remains an alias for `temporal_post` while being passed into `composite_quality` via `enhanced_metrics.py:173` as a quality signal — a post-only measurement is weighted as if it reflects original-vs-compressed quality.
**Downstream impact**: A static-black compressed output scores `temporal_consistency = 1.0` (perfectly consistent single stream), which registers as "good temporal quality" in `composite_quality` even when the original had rich motion. The pre/post/delta pattern addresses this for the named keys but the bare key is still ambiguous. `_validate_temporal_consistency` in `validation_checker.py` reads `temporal_consistency_post` directly — that consumer is correctly named. The composite quality formula is the victim.
**Suggested direction**: Either (a) remove `temporal_consistency` from the `composite_quality` formula and use `temporal_consistency_delta` instead (pair comparison), or (b) define `temporal_consistency` as the min(pre, post) so it reflects "how consistent is the worst of the two streams". Document the choice in the key's inline comment. May be partially addressed by [[giflab-single-stream-metrics-misuse-audit]] (PR #14). **Note**: confirmed this is in `audit-fix/single-stream-metrics-misuse-audit` worktree — check whether PR #14 removed `temporal_consistency` from the composite formula.

---

### `detect_disposal_artifacts` / `detect_structural_artifacts` — MEDIUM
**Rule violated**: Rule 2 — NaN over fabricated values; Rule 1 — cliff edge
**Where**: `metrics.py:1137–1139`
```python
if edge_ratio > 1.2:
    edge_increase_score = max(0.0, 1.0 - ((edge_ratio - 1.2) * 1.5))
else:
    edge_increase_score = 1.0
```
**What**: Hard threshold at `edge_ratio = 1.2` produces a cliff: `edge_ratio = 1.199` → score 1.0; `edge_ratio = 1.201` → score ≈ 0.9997. The penalty is continuous above 1.2 but the 0→-penalty discontinuity at the threshold is still there. Additionally, the fallback at line 1168 returns `1.0` ("no artifacts") on any exception, which is a sentinel: an exception during detection is not the same as clean disposal.
**Downstream impact**: `disposal_artifacts_post` feeds `validation_checker._validate_disposal_artifacts` (line 488–504) at threshold `artifact_threshold`. An image landing just below 1.2 passes; just above fails. This is a classic "just over threshold" inversion problem — a 1% edge density increase can change the validation outcome.
**Suggested direction**: Replace the binary `else: edge_increase_score = 1.0` with a smooth function that starts penalizing before 1.2 (e.g., sigmoid centred at 1.2). The exception path should return `float("nan")`, and `detect_disposal_artifacts` should propagate NaN through its aggregate rather than silently replacing it with 1.0.

---

### `detect_background_color_stability` — LOW
**Rule violated**: Rule 1 — hardcoded empirical threshold
**Where**: `metrics.py:1001`
```python
stability_score = max(0.0, 1.0 - (avg_shift / 50.0))
```
**What**: The `50.0` divisor is an empirical constant that determines when the metric reaches 0.0 (avg_shift ≥ 50). This is not a cliff-edge in the strict sense (the function is continuous), but it is a hardcoded scale with no clear perceptual grounding. A content type with naturally large edge shifts (dark-on-dark) will always score poorly regardless of whether the shift is an artifact. The comment "100 is empirical threshold" at line 1072 (in `detect_color_fidelity_corruption`) is the same pattern.
**Downstream impact**: These two functions contribute to the weighted average in `_detect_partial_animation_artifacts_enhanced`. Systematic bias against dark-content GIFs. Related to [[giflab-alpha-background-configurability-dark-content]].
**Suggested direction**: Calibrate against a dataset rather than hard-coding. At minimum, document that these are empirical constants needing recalibration and tag them with `# TODO: calibrate`.

---

### Per-frame exception `append(0.0)` pattern — HIGH (systemic)
**Rule violated**: Rule 2 — NaN over fabricated values
**Where**: `metrics.py:2402, 2410, 2424, 2435, 2436, 2444, 2452, 2460, 2473, 2483, 2493` (sequential path) and `metrics.py:1670, 1682, 1698, 1709, 1724, 1734` (selected-metrics path)
**What**: Every `except` block in the per-frame metric loops appends `0.0` to the values list. For SSIM, this means a corrupt frame reads as "SSIM=0.0 (identical to completely different image)", not as "failed to measure". The 0.0 then enters `_aggregate_metric` and is averaged as if real. For PSNR the 0.0 is also appended to `raw_metric_values["psnr"]` so the unnormalized value exported via `psnr_raw` is also contaminated.
**Downstream impact**: Same as `calculate_safe_psnr` — all metrics are affected when frames are corrupt. The effect is worst for SSIM (0.0 is a real score implying total dissimilarity) and PSNR (0.0 dB is below any threshold check in `quality_validation.py`). Silent corruption of `composite_quality` and all downstream consumers.
**Suggested direction**: Change all `append(0.0)` on-exception paths to `append(float("nan"))`. Change `_aggregate_metric` to use NaN-aware aggregation. This is the highest-impact single change in the codebase — it fixes SSIM, PSNR, FSIM, GMSD, chist, edge_similarity, texture_similarity, sharpness_similarity, ms_ssim, mse, and rmse simultaneously. Proposed new task: `giflab-per-frame-exception-nan-sentinel`.

---

### `calculate_ms_ssim` — MEDIUM
**Rule violated**: Rule 2 — NaN over fabricated values
**Where**: `metrics.py:607-608`
```python
except (ValueError, RuntimeError) as e:
    ssim_values.append(0.0)
```
Also `metrics.py:652`:
```python
else:
    return 0.0  # No ssim_values collected
```
**What**: Same sentinel-on-failure pattern as the per-frame loops. The MS-SSIM function is called per-frame so the 0.0 propagates into `metric_values["ms_ssim"]`. The return 0.0 at line 652 is also reachable if the frame is too small for a single SSIM calculation.
**Downstream impact**: Contributes to the aggregate `ms_ssim_mean` with weight `ENHANCED_MS_SSIM_WEIGHT` in composite quality. A MS-SSIM calculation failure on a tiny GIF masquerades as a very-low-quality result.
**Suggested direction**: Return `float("nan")` on failure. Covered by the same `giflab-per-frame-exception-nan-sentinel` fix.

---

### Optimized conditional path — SSIMULACRA2 sentinels — HIGH
**Rule violated**: Rule 2 — NaN over fabricated values
**Where**: `metrics.py:2159–2165`
```python
default_ssimulacra2_metrics: dict[str, float | str] = {
    "ssimulacra2_mean": 50.0,
    "ssimulacra2_p95": 50.0,
    "ssimulacra2_min": 50.0,
    "ssimulacra2_frame_count": 0.0,
    "ssimulacra2_triggered": 0.0,
}
```
This same literal dict appears at lines 2216-2225, 2231-2233, and in the main path at 2704-2758 (seven more sites total).
**What**: SSIMULACRA2 scores in its native range span -inf to 100, where 50 means "medium quality". The sentinel `50.0` is indistinguishable from "SSIMULACRA2 ran and reported medium quality". A consumer cannot tell whether the value reflects a real measurement or a fallback. This is the exact bug that PR #10 ([[giflab-ssimulacra2-scale-inconsistency]]) was written to fix in the scale dimension, but the sentinel value problem remains. **May be addressed by PR #10 if it replaced 50.0 with NaN** — need to verify against the PR #10 branch.
**Downstream impact**: `_validate_ssimulacra2_metrics` in `validation_checker.py:848` applies `ssimulacra2_mean < low_threshold (0.3)` — at 50.0 sentinel this evaluates False, silently passing quality validation for a metric that wasn't computed. The `ssimulacra2_triggered == 0.0` guard at line 824 is meant to catch this but relies on the caller correctly setting `triggered=0.0` in the fallback (which it does), so the guard works today — but it's fragile and creates redundant signaling. Covered by [[giflab-dry-ssimulacra2-fallback-dict]] and [[giflab-validation-checker-nan-aware-refactor]].
**Suggested direction**: Replace all SSIMULACRA2 fallback dicts with NaN-valued scores. Consolidate into a single `_DEFAULT_SSIMULACRA2_FALLBACK` constant. This is already proposed in [[giflab-dry-ssimulacra2-fallback-dict]].

---

### LPIPS `lpips_quality_mean = 0.5` sentinel — HIGH
**Rule violated**: Rule 2 — NaN over fabricated values
**Where**: `metrics.py:1758` (selected-metrics path) and `metrics.py:2596–2598` (default_deep_perceptual_metrics)
```python
results["lpips_quality_mean"] = 0.5
# ...
"lpips_quality_mean": 0.5,
"lpips_quality_p95": 0.5,
"lpips_quality_max": 0.5,
```
**What**: LPIPS scores of 0.5 represent "moderate perceptual difference" on a real 0-1 scale. The sentinel is indistinguishable from a real measurement. A failed LPIPS load silently reports medium-quality rather than "unknown". Note that in `enhanced_metrics.py:185`, LPIPS is inverted: `normalized_lpips = 1.0 - lpips_score`. So sentinel 0.5 contributes `0.5 × ENHANCED_LPIPS_WEIGHT` to composite quality — neither a help nor a hurt, which conceals the failure.
**Downstream impact**: `_validate_deep_perceptual_metrics` in `validation_checker.py:729` guards with `not any([lpips_quality_mean, lpips_quality_p95, lpips_quality_max])`. In Python, `any([0.5, 0.5, 0.5])` is `True`, so sentinel LPIPS values pass the guard and trigger the full validation path — comparing `0.5 > lpips_threshold (0.3)` and potentially appending a spurious `perceptual_quality_degradation` issue for a result that actually had LPIPS disabled. The flag `deep_perceptual_device == "fallback"` in the guard on line 731 mitigates this for the main path, but only if the sentinel dict correctly sets `"deep_perceptual_device": "fallback"` (it does in the main path; unclear in the selected-metrics path at line 1758 which only sets `lpips_quality_mean = 0.5` without setting `deep_perceptual_device`).
**Suggested direction**: Return NaN for all LPIPS keys when LPIPS is unavailable. Consolidate fallback dicts (parallel to the SSIMULACRA2 fix). Proposed new task: `giflab-lpips-fallback-nan-sentinel`.

---

### `calculate_selected_metrics` — incomplete SSIMULACRA2 keys — HIGH
**Rule violated**: Rule 5 — Same key shape across paths
**Where**: `metrics.py:1769–1772`
```python
except Exception as e:
    logger.warning(f"SSIMULACRA2 calculation failed: {e}")
    results["ssimulacra2_mean"] = 50.0
```
**What**: When SSIMULACRA2 fails in `calculate_selected_metrics`, only `ssimulacra2_mean` is set. The other four keys (`ssimulacra2_p95`, `ssimulacra2_min`, `ssimulacra2_frame_count`, `ssimulacra2_triggered`) are absent. The main path at lines 2704-2758 always writes all five keys. This is a key-shape mismatch between the conditional-metrics-enabled path and the standard path.
**Downstream impact**: `validation_checker._validate_ssimulacra2_metrics` at line 820 retrieves `ssimulacra2_triggered` with a default of `0.0` — so it falls through to the "SSIMULACRA2 unavailable" warning branch, which is the wrong behaviour (SSIMULACRA2 ran but failed, vs SSIMULACRA2 was never attempted). `ssimulacra2_p95` and `ssimulacra2_min` are `None` which confusingly differs from the main path's 50.0. CSV rows from this path have empty cells where other paths have 50.0/NaN. Covered by [[giflab-dry-ssimulacra2-fallback-dict]].
**Suggested direction**: Apply the same fix as `giflab-dry-ssimulacra2-fallback-dict` — use a shared constant so every exception path emits the same five keys.

---

### SSIMULACRA2 `normalize_score` — MEDIUM
**Rule violated**: Rule 2 — NaN over fabricated values; Rule 1 — cliff edge
**Where**: `ssimulacra2_metrics.py:87–95`
```python
if raw_score >= SSIMULACRA2_EXCELLENT_SCORE:
    return 1.0
elif raw_score <= SSIMULACRA2_POOR_SCORE:
    return 0.0
else:
    return (raw_score - SSIMULACRA2_POOR_SCORE) / (...)
```
**What**: Scores below `SSIMULACRA2_POOR_SCORE (10.0)` are clamped to `0.0`. SSIMULACRA2 can return values down to -∞; a score of `-30` (severely corrupted) and a score of `10` (just-bad) both map to `0.0`, collapsing the severity signal. The function at line 85 correctly returns `0.0` for non-finite (inf/NaN) inputs, which is a sentinel but acceptable since SSIMULACRA2 returning infinity implies a binary failure mode. The cliff at `POOR_SCORE` is the main issue.
**Downstream impact**: `ssimulacra2_min` will be 0.0 for any frame with score ≤ 10, regardless of whether it's -10 or -100. Validation at `validation_checker.py:872` only checks `< low_threshold (0.3)` — since 0.0 < 0.3 it does trigger correctly, so the validation outcome is right, but the reported value (0.0) misleads about severity.
**Suggested direction**: Extend the linear interpolation below `POOR_SCORE` using a smooth asymptotic approach toward 0 (e.g., `max(0, score / POOR_SCORE)` for scores below POOR_SCORE). Covered by [[giflab-ssimulacra2-scale-inconsistency]].

---

### `ssimulacra2_metrics.py:215–216` — frame fallback sentinel — HIGH
**Rule violated**: Rule 2 — NaN over fabricated values
**Where**: `ssimulacra2_metrics.py:215–216`
```python
except Exception as e:
    logger.error(f"SSIMULACRA2 failed for frame {frame_idx}: {e}")
    scores.append(0.5)
```
Also `ssimulacra2_metrics.py:220–221`:
```python
if not scores:
    logger.error("No SSIMULACRA2 scores calculated")
    scores = [0.5]  # Fallback
```
**What**: Per-frame failures and the all-frames-failed case both substitute `0.5` (normalized score ≈ 50 SSIMULACRA2 units, "medium quality"). The resulting `ssimulacra2_mean = 0.5` is indistinguishable from a real measurement. The `ssimulacra2_triggered = 1.0` flag at line 227 will be set even for all-failure cases (since the function always returns this value), which defeats the "was SSIMULACRA2 actually used?" signalling in `validation_checker.py:824`. **This is the specific bug described in [[giflab-validation-checker-nan-aware-refactor]]**.
**Downstream impact**: `any([0.5, 0.5, 0.5])` is `True`, so the early-return guard in `validation_checker._validate_ssimulacra2_metrics` does not fire. Comparisons `0.5 < 0.3` → False → no issue filed → **silent PASS for failed SSIMULACRA2 computation**. This is the exact class of silent-corruption the CLAUDE.md principle warns about. Covered by [[giflab-ssimulacra2-scale-inconsistency]] (PR #10) and [[giflab-validation-checker-nan-aware-refactor]].
**Suggested direction**: Replace per-frame sentinel with `float("nan")`. Replace all-failure fallback with NaN. Use `np.nanmean`/`np.nanpercentile` for aggregation. Set `ssimulacra2_triggered = 0.0` when all frames failed. Validation guard becomes `all(_is_missing(v) for v in ...) or triggered == 0.0`.

---

## File: `src/giflab/enhanced_metrics.py`

### `calculate_composite_quality` — LPIPS/SSIMULACRA2 NaN-unawareness — HIGH
**Rule violated**: Rule 4 — Honest error paths end-to-end
**Where**: `enhanced_metrics.py:181–196`
```python
if "lpips_quality_mean" in metrics:
    lpips_score = metrics["lpips_quality_mean"]
    normalized_lpips = max(0.0, min(1.0, 1.0 - lpips_score))
    composite_quality += config.ENHANCED_LPIPS_WEIGHT * normalized_lpips
    total_weight += config.ENHANCED_LPIPS_WEIGHT

if "ssimulacra2_mean" in metrics:
    ssimulacra2_score = metrics["ssimulacra2_mean"]
    normalized_ssimulacra2 = max(0.0, min(1.0, ssimulacra2_score))
    composite_quality += config.ENHANCED_SSIMULACRA2_WEIGHT * normalized_ssimulacra2
    total_weight += config.ENHANCED_SSIMULACRA2_WEIGHT
```
**What**: If `lpips_quality_mean` or `ssimulacra2_mean` is `float("nan")`, then:
- `max(0.0, min(1.0, 1.0 - nan))` → `nan` (NaN propagates through min/max in Python only with `np.maximum`; with `max()` on scalars, `nan < 0.0` returns False, so `max(0.0, nan)` returns `0.0` — **this swallows the NaN and injects a fake 0.0 into composite quality**.
- `total_weight += weight` adds the metric's weight even though the measurement was NaN — the weight denominator is wrong.

If sentinel 0.5/50.0 values are replaced with NaN (the recommended fix), this NaN-unawareness in `calculate_composite_quality` will silently corrupt the composite score. The fix must be coordinated — NaN-propagation in the inputs requires NaN-awareness in the composite formula.
**Downstream impact**: After the sentinel→NaN fix, composite quality will be systematically biased when LPIPS or SSIMULACRA2 fail. All downstream consumers inherit the wrong composite. This is the "honest error paths end-to-end" failure mode.
**Suggested direction**: Add a NaN guard before each metric block:
```python
if "lpips_quality_mean" in metrics and not math.isnan(metrics["lpips_quality_mean"]):
    ...
```
This is a coordinated fix with the sentinel→NaN change in `metrics.py`. Proposed task: `giflab-composite-quality-nan-guard`.

---

### `calculate_composite_quality` — NaN key check vs NaN value — MEDIUM
**Rule violated**: Rule 4 — Honest error paths end-to-end
**Where**: `enhanced_metrics.py:103–178` (all metric blocks)
```python
if "ssim_mean" in metrics:
    raw_value = metrics["ssim_mean"]
    normalized = normalize_metric("ssim_mean", raw_value)
    ...
```
**What**: The guard is `key in metrics`, which is True even when `metrics["ssim_mean"] = float("nan")`. After the recommended sentinel→NaN fix, all exception paths will emit NaN rather than 0.0, and these guards will pass while `normalize_metric` receives NaN. For most metrics, `normalize_metric` simply does arithmetic on the NaN which propagates — ultimately `composite_quality` becomes NaN and the composite formula's `max(0.0, min(1.0, composite_quality))` returns `0.0` (since NaN comparisons return False). **Silent composite quality of 0.0** — worse than the sentinel problem it was meant to fix.
**Downstream impact**: As above — all downstream consumers. Must be fixed in concert with the sentinel→NaN change.
**Suggested direction**: Same NaN guard for all metric blocks. Or refactor `normalize_metric` to return NaN for NaN inputs and handle NaN before the weight accumulation.

---

### `calculate_legacy_composite_quality` — sentinel-propagation — MEDIUM
**Rule violated**: Rule 2 — fabricated values
**Where**: `enhanced_metrics.py:254–258`
```python
composite_quality = (
    config.SSIM_WEIGHT * metrics.get("ssim_mean", 0.0)
    + config.MS_SSIM_WEIGHT * metrics.get("ms_ssim_mean", 0.0)
    + config.PSNR_WEIGHT * metrics.get("psnr_mean", 0.0)
    + config.TEMPORAL_WEIGHT * metrics.get("temporal_consistency", 0.0)
)
```
**What**: The `.get(key, 0.0)` pattern treats missing keys as 0.0 (which is a sentinel) rather than redistributing weight. If `ssim_mean` is absent, this formula silently gives the SSIM dimension 0% of quality contribution (which looks like very poor SSIM) rather than proportionally redistributing its weight.
**Downstream impact**: Legacy composite path is used when `USE_ENHANCED_COMPOSITE_QUALITY = False`. If an individual metric fails and the key is absent, the legacy composite is biased downward.
**Suggested direction**: Use the same `total_weight` redistribution approach as the enhanced path, or use `metrics.get(key, None)` and skip the term when None.

---

## File: `src/giflab/temporal_artifacts.py`

### `calculate_enhanced_temporal_metrics` — single-stream measurements — MEDIUM
**Rule violated**: Rule 3 — Pair-wise over single-stream, labelled honestly
**Where**: `temporal_artifacts.py:949–954`
```python
flicker_metrics = detector.detect_flicker_excess(compressed_frames, ...)
flat_flicker_metrics = detector.detect_flat_region_flicker(compressed_frames)
pumping_metrics = detector.detect_temporal_pumping(compressed_frames)
```
**What**: All three temporal metrics — `flicker_excess`, `flat_flicker_ratio`, `temporal_pumping_score` — are computed on `compressed_frames` only. The function signature accepts `original_frames` and `compressed_frames`, but `original_frames` is used only to determine `min_frame_count` for truncation. The results are stored as bare keys (`flicker_excess`, not `flicker_excess_compressed`), which implies original-vs-compressed comparison.
**Downstream impact**: A highly flickery original that is well-compressed will report low `flicker_excess` (the compressed output is stable) as a quality success, when what the metric should measure is "did compression introduce flicker beyond the original?" An original that itself has high flicker will make compressed look good unconditionally. `validation_checker._validate_temporal_artifacts` reads these bare keys and makes pass/fail decisions based on compressed-only signals.
**Suggested direction**: Compute flicker metrics on both streams: `flicker_excess_original = detector.detect_flicker_excess(original_frames)`, `flicker_excess_compressed = detector.detect_flicker_excess(compressed_frames)`, then report `flicker_excess_delta = max(0, compressed - original)` as the quality-relevant signal. This is the correct pair-wise approach. Covered by [[giflab-single-stream-metrics-misuse-audit]] (PR #14) — confirm whether PR #14 addresses temporal_artifacts or only disposal/temporal_consistency.

---

### `detect_temporal_pumping` — undefined variable risk — LOW
**Rule violated**: Rule 4 — honest error paths
**Where**: `temporal_artifacts.py:870–874`
```python
return {
    "temporal_pumping_score": oscillation_score,
    "quality_oscillation_frequency": oscillation_frequency
    if "oscillation_frequency" in locals()
    else 0.0,
    ...
}
```
**What**: `oscillation_frequency` is set inside the `if len(quality_diffs) >= 2:` block but referenced outside it via a `"variable_in_locals()"` check. The `oscillation_frequency` variable is always set when `len(quality_diffs) >= 2`, but the `locals()` check is the wrong way to guard this — it's fragile and unusual Python. If `quality_diffs` is empty (0 or 1 frame), `oscillation_frequency` is never set and the guard triggers the `else: 0.0` path, which is correct but the pattern is non-obvious.
**Downstream impact**: Minor. The 0.0 default is reasonable for <2 frames. But the pattern suggests the author was uncertain about the flow — this warrants clarity.
**Suggested direction**: Initialize `oscillation_frequency = 0.0` before the conditional block and remove the `locals()` check.

---

## File: `src/giflab/gradient_color_artifacts.py`

### `calculate_dither_quality_metrics` exception path — MEDIUM
**Rule violated**: Rule 2 — NaN over fabricated values
**Where**: `gradient_color_artifacts.py:779–785`
```python
except Exception as e:
    logger.error(f"Failed to analyze dither quality: {e}")
    return {
        "dither_ratio_mean": 0.0,
        "dither_ratio_p95": 0.0,
        "dither_quality_score": 0.0,
        "flat_region_count": 0,
    }
```
**What**: On exception, all dither quality metrics return 0.0. `dither_quality_score = 0.0` is the worst possible score (0 on a 0-100 scale), implying severe over-dithering or under-dithering. A calculation exception is not the same as detected dithering failure.
**Downstream impact**: The dither metrics are not currently weighted into `composite_quality` — they are diagnostic-only outputs in `calculate_gradient_color_metrics`. Impact is therefore limited to CSV exports and any future consumer that reads these keys. The 0.0 will look like "dither analysis failed or found no dithering" which is ambiguous.
**Suggested direction**: Return NaN for `dither_ratio_mean`, `dither_ratio_p95`, `dither_quality_score` and 0 for `flat_region_count` on exception. Lower priority since these aren't currently composite-weighted.

---

### `calculate_transparency_artifact_score` and `calculate_posterization_score` — LOW
**Rule violated**: Rule 2 — NaN over fabricated values
**Where**: `gradient_color_artifacts.py:986–987`, `gradient_color_artifacts.py:1064–1066`
```python
return {"posterization_score": 0.0}  # on exception
return {"transparency_artifact_score": 0.0}  # on exception
```
**What**: Exception paths return 0.0 (no artifacts), identical to the "ran successfully and found no artifacts" result. A calculation failure is indistinguishable from a clean result. Same class of issue as the dither metrics.
**Downstream impact**: Same as dither — diagnostic-only, not composite-weighted today. Lower priority.
**Suggested direction**: Return NaN on exception.

---

### `calculate_deltae2000` approximation quality — LOW
**Rule violated**: Rule 1 (precision concern, not cliff-edge)
**Where**: `gradient_color_artifacts.py:365–418`
**What**: The function is self-described as "simplified CIEDE2000 approximation" that "omits the complex weighting functions for simplicity". The full CIEDE2000 formula includes chroma corrections, RT rotation term, and a6-parameter weighting structure. The simplified version here uses only lightness/chroma/hue terms with simple weighting (SL, SC, SH) but omits the `a*` adjustment (C' calculation), the hue angle (H'), the RT cross-term, and the `f(C')` function. The result is closer to CIE94 than CIEDE2000.
**Downstream impact**: `deltae_mean` values will be systematically different from true CIEDE2000 values, particularly for saturated colours and hue changes near primary axes. `enhanced_metrics.normalize_metric("deltae_mean", ...)` applies `max(0, 1 - value/10.0)` — since the approximate deltae may be proportionally different, the normalization constant 10.0 may not be correctly calibrated for this approximation.
**Suggested direction**: Either use `colormath` library for full CIEDE2000 (already in the code comments), or clearly document that this is CIE94-approximation and calibrate the normalization constant accordingly. This is a measurement-accuracy issue rather than an error-path issue — LOW priority unless deltae is being relied upon for production decisions.

---

## File: `src/giflab/optimized_metrics.py`

### `calculate_optimized_comprehensive_metrics` — key-shape mismatch — HIGH
**Rule violated**: Rule 5 — Same key shape across paths
**Where**: `optimized_metrics.py:534–557`
```python
default_metrics = {
    "composite_quality": ...,
    "efficiency": 1.0,  # Simplified
    "compression_ratio": 1.0,
    "kilobytes": 0.0,
    "banding_score_mean": 0.0,
    "deltae_mean": 0.0,
    "color_patch_count": 0,
    "has_text_ui_content": False,
    "text_ui_edge_density": 0.0,
    "text_ui_component_count": 0,
    "ssimulacra2_mean": 50.0,
    "ssimulacra2_triggered": 0.0,
}
```
**What**: The Phase 6 optimized path emits a substantially different key schema than the main path. Missing entirely: `ssimulacra2_p95`, `ssimulacra2_min`, `ssimulacra2_frame_count`, all LPIPS keys (`lpips_quality_mean`, `lpips_quality_p95`, `lpips_quality_max`), all gradient metrics beyond `banding_score_mean` and `deltae_mean`, text/UI metrics beyond the three listed, all temporal artifact keys (`flicker_excess`, `flat_flicker_ratio`, `temporal_pumping_score`, etc.), `temporal_consistency_pre`, `temporal_consistency_post`, `temporal_consistency_delta`. The `composite_quality` is computed from a 2-metric average (`(ssim_mean + 1 - mse_mean/10000) / 2`) rather than the 11-metric weighted formula in `enhanced_metrics.py`.

Also: `efficiency = 1.0` is a sentinel — the function doesn't have access to file size, so it substitutes 1.0 (perfect efficiency) unconditionally. A consumer reading `efficiency` cannot distinguish "great efficiency" from "Phase 6 was used". Covered by [[giflab-phase6-optimized-path-metric-alias-parity]].
**Downstream impact**: When `GIFLAB_ENABLE_PHASE6_OPTIMIZATION=true`, `validation_checker` reads `ssimulacra2_p95 = None` (key absent), `lpips_quality_mean = None`, etc. The `any([None, None, None])` guard fires False for all-None, so the "metric unavailable" warning is correctly triggered. But `ssimulacra2_mean = 50.0` with absent `ssimulacra2_triggered` causes the same triggered-default-0.0 issue as above. The composite quality formula is completely different, so scores are not comparable between Phase 6 and standard paths even on the same input.
**Suggested direction**: Phase 6 should either (a) produce the same key schema with NaN for keys it can't compute, or (b) be removed/deprecated in favour of the conditional-metrics optimization path (`GIFLAB_ENABLE_CONDITIONAL_METRICS`) which preserves schema fidelity. The current Phase 6 is a research prototype that violates Rule 5 systematically. Covered by [[giflab-phase6-optimized-path-metric-alias-parity]].

---

### `FastTemporalConsistency.calculate_optimized` — algorithm divergence — MEDIUM
**Rule violated**: Rule 5 — Same key shape (algorithmic divergence)
**Where**: `optimized_metrics.py:258–307`
**What**: The standard `calculate_temporal_consistency` in `metrics.py` uses an exponential-decay formula (`consistency = np.exp(-relative_variance * 2.0)`). The `FastTemporalConsistency.calculate_optimized` uses `1 / (1 + CoV)`. These produce different values on the same input:
- Static content: both return 1.0 (same)
- Uniform animation: standard returns 1.0 (uniform variance = 0), optimized returns `1 / (1 + 0)` = 1.0 (same)
- Highly variable animation: standard returns `exp(-large)` ≈ 0, optimized returns `1 / (1 + large)` > 0 — different floor behaviour

The functions share the same output key `temporal_consistency`. When Phase 6 is enabled, `temporal_consistency` means something different than when it's disabled.
**Downstream impact**: `validation_checker._validate_temporal_consistency` threshold checks against the same threshold regardless of which formula produced the value. Comparisons between Phase 6 results and standard results are apples-to-oranges. Covered by [[giflab-phase6-optimized-path-metric-alias-parity]].
**Suggested direction**: Either call the standard `calculate_temporal_consistency` function directly from the Phase 6 path (consistency), or document the divergence and flag Phase 6 results with a `_temporal_algorithm: "optimized"` key.

---

## File: `src/giflab/optimization_validation/validation_checker.py`

### `_validate_temporal_artifacts` — `any()` NaN-brittleness — CRITICAL
**Rule violated**: Rule 4 — Honest error paths end-to-end
**Where**: `validation_checker.py:641–645`
```python
if not any(
    [flicker_excess, flat_flicker_ratio, temporal_pumping, lpips_t_mean]
):
    result.warnings.append(...)
    return
```
**What**: In Python, `any([nan, nan, nan, nan])` returns `True` because NaN is truthy. After the recommended sentinel→NaN fix in `temporal_artifacts.py`, this guard will PASS (any() returns True for NaN inputs) and fall through to the validation logic — where `nan > threshold` evaluates to `False` (NaN comparisons always return False). **No issue is filed for a metric that failed entirely.** This is a silent PASS for a measurement failure.
**Downstream impact**: Systematic silent PASS for all temporal artifact validations when the LPIPS model fails to load or temporal computation crashes. This is the highest-severity finding because it inverts the error behaviour: failures become passes.
**Suggested direction**: Replace `any(...)` with `any(v is not None and not (isinstance(v, float) and math.isnan(v)) for v in ...)` — or use a `_is_present(v)` helper. Then also fix the per-metric comparisons to use NaN-aware guards. This is the core fix in [[giflab-validation-checker-nan-aware-refactor]].

---

### `_validate_deep_perceptual_metrics` — `any()` NaN-brittleness — CRITICAL
**Rule violated**: Rule 4 — Honest error paths end-to-end
**Where**: `validation_checker.py:729–731`
```python
if (
    not any([lpips_quality_mean, lpips_quality_p95, lpips_quality_max])
    or not deep_perceptual_used
):
```
**What**: Same `any()` NaN-brittleness as above. Additionally: `deep_perceptual_used` is determined by `compression_metrics.get("deep_perceptual_device", "fallback") != "fallback"`. The sentinel dict at `metrics.py:2601` sets `"deep_perceptual_device": "fallback"`, so this guard correctly catches the main fallback path. But after the sentinel→NaN fix, the selected-metrics path (`calculate_selected_metrics`) at `metrics.py:1758` only sets `results["lpips_quality_mean"] = 0.5` without setting `"deep_perceptual_device"`. This key will be absent; `compression_metrics.get("deep_perceptual_device", "fallback")` returns `"fallback"`; `deep_perceptual_used` = False; the guard fires correctly. So this specific code path is protected by the `deep_perceptual_device` flag — but the `any()` NaN issue still applies for the case where the full path ran but produced NaN values.
**Downstream impact**: After sentinel→NaN fix: LPIPS failures that produce NaN values will pass the `any()` guard and trigger false validation issues. Already partially covered by [[giflab-validation-checker-nan-aware-refactor]].
**Suggested direction**: Same as `_validate_temporal_artifacts` — replace `any(...)` with NaN-aware check.

---

### `_validate_ssimulacra2_metrics` — `any()` NaN-brittleness + triggered=0.0 gap — CRITICAL
**Rule violated**: Rule 4 — Honest error paths end-to-end
**Where**: `validation_checker.py:823–825`
```python
if (
    not any([ssimulacra2_mean, ssimulacra2_p95, ssimulacra2_min])
    or ssimulacra2_triggered == 0.0
):
```
**What**: Same `any([nan, nan, nan])` = True issue. After the sentinel→NaN fix in `ssimulacra2_metrics.py`, a fully-failed SSIMULACRA2 run that returns all-NaN metrics with `triggered=1.0` (incorrect — the all-failure case should set `triggered=0.0`) will pass the guard. But even if `triggered=0.0` is correctly set: `not any([nan, nan, nan]) or triggered == 0.0` = `not True or True` = `True` — the guard fires correctly for that case. The deeper problem is when triggered=1.0 with NaN scores (partial failure). This is the specific scenario described in [[giflab-validation-checker-nan-aware-refactor]].
**Downstream impact**: As documented in [[giflab-validation-checker-nan-aware-refactor]]: silent PASS for all-NaN SSIMULACRA2 with triggered=1.0. Highest-priority finding in the validation layer.
**Suggested direction**: Per [[giflab-validation-checker-nan-aware-refactor]] — implement `_is_missing(v)` and check `all(_is_missing(v) for v in (...))`. Also ensure `ssimulacra2_triggered = 0.0` is set when all frames fail in `ssimulacra2_metrics.py`.

---

### `_validate_disposal_artifacts` — falsy-value guard issue — MEDIUM
**Rule violated**: Rule 4 — Honest error paths end-to-end
**Where**: `validation_checker.py:479`
```python
if not disposal_pre or not disposal_post:
```
**What**: `disposal_artifacts_pre` and `disposal_artifacts_post` are scores between 0.0 and 1.0 where 0.0 means "severe artifacts". The guard `if not disposal_pre` treats `disposal_pre = 0.0` (severe artifacts) as "data unavailable" and skips validation entirely, filing a warning instead of an issue. A perfectly-corrupt disposal result would be silently ignored.
**Downstream impact**: Any GIF with disposal_artifacts_pre = 0.0 (perfectly catastrophic disposal) or disposal_artifacts_post = 0.0 skips the check. The probability of exactly 0.0 is low but non-zero for extreme cases. More practically, this is a conceptual error — a missing value should use `is None`, not falsy check.
**Suggested direction**: Replace `not disposal_pre` with `disposal_pre is None`. Same for `disposal_post`.

---

### `_validate_multi_metric_combinations` — falsy-value guards throughout — MEDIUM
**Rule violated**: Rule 4 — Honest error paths end-to-end
**Where**: `validation_checker.py:552, 573–580, 599–600, 611–613`
```python
if composite_quality and compression_ratio:
if original_fps and compressed_fps and original_frames and compressed_frames ...:
if disposal_pre and temporal_consistency:
if original_frames and compressed_frames and composite_quality:
```
**What**: All these guards use truthiness rather than `is not None` checks. For metrics that can legitimately be 0 (e.g., `compression_ratio = 0.0` is unusual but possible for a degenerate case, `temporal_consistency = 0.0` is worst-case but valid), the guard incorrectly treats "measured as zero" as "data unavailable". The `disposal_pre and temporal_consistency` check at line 600 creates the paradox: `disposal_pre = 0.0` (severe artifacts) skips the `animation_corruption` check — the exact case where it should fire.
**Downstream impact**: Combination check 3 (animation corruption detection) silently skips when `disposal_pre = 0.0`, precisely when the corruption is worst. Other checks similarly skip on zero values.
**Suggested direction**: Replace all truthiness checks with explicit `is not None` (and NaN-awareness after the sentinel→NaN fix).

---

## File: `src/giflab/wrapper_validation/quality_validation.py`

### `_check_metric_outliers` — NaN comparisons — MEDIUM
**Rule violated**: Rule 4 — Honest error paths end-to-end
**Where**: `quality_validation.py:244–248`
```python
is_acceptable = (
    isinstance(ssim_value, int | float)
    and ssim_value >= self.catastrophic_thresholds["min_ssim_mean"]
)
```
**What**: `isinstance(nan, float)` is `True`. `nan >= 0.2` is `False`. So a NaN SSIM value produces `is_acceptable = False` — triggering a "catastrophic SSIM failure" for a measurement that didn't happen. This is better than a silent PASS but it's the wrong signal: the failure message says "SSIM below threshold" when actually "SSIM could not be computed".

Same pattern for MSE at line 260 (`nan <= 10000` → False → "catastrophic MSE" even though MSE was never measured) and PSNR at line 277.
**Downstream impact**: After sentinel→NaN fix, all three outlier checks may spuriously fire for normal failed frames, causing `quality_acceptable = False` and rejecting valid outputs on computation-failure grounds rather than quality grounds. This is a false negative (rejecting good outputs).
**Suggested direction**: Add explicit NaN check before each threshold comparison:
```python
is_acceptable = (
    isinstance(ssim_value, int | float)
    and not math.isnan(ssim_value)
    and ssim_value >= threshold
) or (ssim_value is None or (isinstance(ssim_value, float) and math.isnan(ssim_value)))
```
Or: treat NaN as "unable to assess" and return a neutral check result rather than fail/pass.

---

## Uncertain Findings

### `texture_similarity` single-stream concern (uncertain)
**Where**: `metrics.py:1471–1497`
**Note**: `texture_similarity` computes LBP histograms for both `frame1` and `frame2` and correlates them — this IS pair-wise (correct). The key name is consistent. The sentinel at line 1493–1494 (`return 1.0` for both-uniform textures) is a reasonable edge case (two uniform textures are identically similar), not a measurement fabrication. No issue to flag.

### `sharpness_similarity` edge case (uncertain)
**Where**: `metrics.py:1514–1519`
**Note**: `max(var1, var2) == 0` returning `0.0` at line 1517 could be NaN-preferable, but the case is "both frames are perfectly flat except one has zero Laplacian variance and the other doesn't" — `var1 = 0, var2 = 0` is handled by line 1514 (both flat = identical = 1.0). `max(var1, var2) == 0` is only reached if one is nonzero... wait, `max(0, nonzero) = nonzero ≠ 0`. So line 1516 is logically unreachable after line 1514. This is dead code, not a bug.

### `detect_global_animation_pattern` threshold (uncertain)
**Where**: `metrics.py:778`
```python
avg_change_ratio > 0.7  # >70% pixels changed = global animation
```
**Note**: Hard threshold, but this is a content-classification branch selector — it determines which disposal-artifact detection algorithm to run, not a metric score itself. The downstream impact of getting this wrong is using the slightly-wrong detector, not corrupting a value. Medium concern but not in scope of the five rules as a metric output.

---

## Cross-Cutting Patterns

### Rules violated most often

| Rule | Violations | Files |
|------|-----------|-------|
| Rule 2 — NaN over sentinel | 15 | metrics.py (11), ssimulacra2_metrics.py (2), gradient_color_artifacts.py (3) |
| Rule 4 — Honest error paths | 8 | validation_checker.py (6), quality_validation.py (2) |
| Rule 5 — Same key shape | 3 | optimized_metrics.py (2), metrics.py:calculate_selected_metrics (1) |
| Rule 3 — Pair-wise labelled | 2 | temporal_artifacts.py (1), metrics.py:temporal_consistency (1) |
| Rule 1 — Continuous over discrete | 2 | metrics.py:detect_structural_artifacts (1), ssimulacra2_metrics.py:normalize_score (1) |

The dominant failure mode is sentinel values hiding calculation errors. Rule 4 violations in the validator are a secondary amplifier — they prevent sentinels from being detected even when callers emit correctly.

---

### Sentinels in use

| Sentinel value | Sites emitting it | Consumer behaviour | Consumer distinguishes from real data? |
|---|---|---|---|
| `0.0` (SSIM/FSIM/etc. on frame exception) | metrics.py:2402, 2444, 2473, 2483, 2493 (×5) | Averaged into metric mean; fed to composite_quality | NO — looks like measured 0.0 |
| `0.0` (PSNR on exception) | metrics.py:2423–2424 | Triggers quality_validation catastrophic PSNR check | NO |
| `0.0` (MS-SSIM on exception) | metrics.py:2410 | Averaged into ms_ssim_mean | NO |
| `0.5` (LPIPS quality fallback) | metrics.py:1758, 2596–2598 | LPIPS inversion: treated as 0.5 quality in composite | Partially — `deep_perceptual_device = "fallback"` flag mitigates |
| `50.0` (SSIMULACRA2 fallback) | metrics.py:2704–2763 (×7) | validation_checker: `ssimulacra2_triggered = 0.0` guard should catch it | Mostly — triggered=0.0 guard is fragile |
| `0.5` (SSIMULACRA2 per-frame failure) | ssimulacra2_metrics.py:216 | `ssimulacra2_triggered = 1.0` (NOT 0.0); validation guard misses it | NO — this is the critical bug |
| `1.0` (disposal_artifacts on exception) | metrics.py:1168 | Signals "clean disposal" for a calculation failure | NO — looks like perfect result |
| `0.0` (aggregate_metric empty list) | metrics.py:1536–1541 | All downstream consumers treat as valid measurement | NO |
| `0.0` (dither/posterization/transparency on exception) | gradient_color_artifacts.py | Diagnostic CSV only (not composite-weighted today) | NO |
| `1.0` (efficiency in Phase 6) | optimized_metrics.py:544 | Reports perfect efficiency with no file-size data | NO |
| `50.0` (ssimulacra2 in Phase 6) | optimized_metrics.py:555 | Same as main path sentinels | Partially |

---

### Cliffs in metric output range

| Function | Threshold | Discontinuity |
|---|---|---|
| `detect_structural_artifacts` | `edge_ratio = 1.2` | `score = 1.0` below; smooth penalty above |
| `ssimulacra2_metrics.normalize_score` | `raw_score = 10.0` | All scores ≤ 10 → 0.0 (collapse) |
| `should_use_ssimulacra2` | `composite_quality = 0.7` | SSIMULACRA2 run below 0.7; not run above — creates missing-key cliff for values > 0.7 (SSIMULACRA2 fallback dict used) |
| `_validate_ssimulacra2_metrics` | `ssimulacra2_mean < 0.3` → issue; `0.3–0.5` → warning; `> 0.5` → pass | Three discrete outcome bands |
| `detect_global_animation_pattern` | `avg_change_ratio > 0.7` | Selects different detection algorithm entirely |
| `_validate_quality_thresholds` | `minimum_quality_floor` | Binary fail/pass |

---

### Key-shape mismatches between paths

| Key | `metrics.py` main path | `calculate_selected_metrics` | `optimized_metrics.py` (Phase 6) |
|---|---|---|---|
| `ssimulacra2_p95` | Always present | ABSENT on exception | ABSENT |
| `ssimulacra2_min` | Always present | ABSENT on exception | ABSENT |
| `ssimulacra2_frame_count` | Always present | ABSENT on exception | ABSENT |
| `ssimulacra2_triggered` | Always present (0.0 or 1.0) | ABSENT on exception | Present (0.0) |
| `lpips_quality_mean` | Always present | Present on success; 0.5 on failure | ABSENT |
| `lpips_quality_p95` | Always present | ABSENT | ABSENT |
| `lpips_quality_max` | Always present | ABSENT | ABSENT |
| `flicker_excess` | Present when ENABLE_TEMPORAL_ARTIFACTS | Present when ENABLE_TEMPORAL_ARTIFACTS | ABSENT |
| `temporal_consistency_pre` | Always present | ABSENT | Present (= same as post — aliased) |
| `temporal_consistency_post` | Always present | ABSENT | Present (= same as pre — aliased) |
| `banding_score_p95` | Always present | ABSENT | ABSENT |
| `deltae_p95` | Always present | ABSENT | ABSENT |
| `composite_quality` formula | 11-metric weighted | 11-metric weighted | 2-metric simple average |

---

## Prioritised Top-5 (CRITICAL/HIGH by Blast Radius)

### 1. `append(0.0)` per-frame exception sentinels — `metrics.py` (systemic)
**Blast radius**: Every metric in the codebase. Affects `composite_quality`, `validation_checker`, `quality_validation`, CSV exports, ML feature extraction, gifprep selection. This is the root cause of the most downstream contamination.
**Proposed task**: `giflab-per-frame-exception-nan-sentinel`
**Existing coverage**: None — this is a new finding.

### 2. `any([nan, nan, nan])` in `validation_checker.py` — CRITICAL
**Blast radius**: All three validation methods (`_validate_temporal_artifacts`, `_validate_deep_perceptual_metrics`, `_validate_ssimulacra2_metrics`) produce silent PASSes for failed measurements. This is the "wrong thing" the task note warns about: downstream decisions flip the wrong way.
**Proposed task**: Covered by [[giflab-validation-checker-nan-aware-refactor]] (in_progress, wave 1).
**Note**: This finding confirms and extends the known bug — the `any()` pattern appears in three places, not just `_validate_ssimulacra2_metrics`.

### 3. SSIMULACRA2 per-frame sentinel `0.5` with `triggered=1.0` — CRITICAL
**Blast radius**: Directly contradicts the `triggered` flag's intended meaning. Every SSIMULACRA2 frame failure produces a silent-PASS in the validator. This is the specific bug described in [[giflab-validation-checker-nan-aware-refactor]] and [[giflab-dry-ssimulacra2-fallback-dict]].
**Existing coverage**: [[giflab-ssimulacra2-scale-inconsistency]] (in_progress, wave 1), [[giflab-dry-ssimulacra2-fallback-dict]] (open, wave 4).

### 4. `composite_quality` NaN-unawareness — `enhanced_metrics.py` — HIGH
**Blast radius**: After the sentinel→NaN fix (items 1 and 3), this becomes a critical blocker. Without this fix, NaN values injected into the composite formula produce `0.0` composite quality (via Python's `max(0.0, nan) = 0.0`), which fails every downstream threshold check. Fixes 1 and 4 must be coordinated.
**Proposed task**: `giflab-composite-quality-nan-guard`
**Existing coverage**: None — new finding.

### 5. Phase 6 key-schema mismatch — `optimized_metrics.py` — HIGH
**Blast radius**: Whenever `GIFLAB_ENABLE_PHASE6_OPTIMIZATION=true`, all downstream consumers (validation_checker, gifprep, CSV exports, ML features) receive an incompatible metric schema with a fundamentally different composite_quality formula. The consumer cannot detect which path produced the result.
**Existing coverage**: [[giflab-phase6-optimized-path-metric-alias-parity]] (in_progress, wave 1).

---

## New Task Slugs Proposed

| Slug | Issue | Priority |
|---|---|---|
| `giflab-per-frame-exception-nan-sentinel` | Replace all `append(0.0)` exception paths in per-frame metric loops with `append(float("nan"))`, change `_aggregate_metric` to use NaN-aware aggregation | HIGH |
| `giflab-composite-quality-nan-guard` | Add NaN guards to all metric blocks in `calculate_composite_quality` and `calculate_legacy_composite_quality` so NaN inputs are skipped (weight redistributed) rather than propagated as 0.0 | HIGH |
| `giflab-lpips-fallback-nan-sentinel` | Replace all LPIPS `0.5` fallback sentinels with NaN; consolidate to a shared `_DEFAULT_LPIPS_FALLBACK` constant parallel to [[giflab-dry-ssimulacra2-fallback-dict]] | HIGH |
| `giflab-validation-checker-falsy-guards` | Replace all `if not disposal_pre`, `if disposal_pre and temporal_consistency`, and similar truthiness guards in `validation_checker.py` and `quality_validation.py` with explicit `is not None` (and NaN-aware) guards | MEDIUM |

---

*This document was produced as a static read-only audit. No code was modified. All findings reference the `main` branch HEAD at `8689736`.*
