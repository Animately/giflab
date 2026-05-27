# Metrics Audit — Top-4 Outlier Deep Dive (2026-05-26)

**Source sweep:** `audit/sweep.csv` — 368 successful (path × lossy) pairs; 293 real, 75 synthetic.
**Sweep lossy levels:** 20, 40, 60.
**Investigation method:** Read-only analysis of sweep CSV + audit report + metric source code.
**Report authored:** 2026-05-27 by subagent `agent-ad2e533b53a2d23c0` as part of rollout `giflab-rollout-2026-05-26` Wave 1.

---

## Summary

The four GIFs identified in the 2026-05-23 production sweep update as top-outlier candidates each fail for a distinct and instructive reason. Three of the four failures are genuine animately compression failures on difficult content (not metric artefacts). One (Outlier 3) is a metric artefact: a single-stream metric mis-labelled as a comparison signal flags a perfectly-compressed animation as broken.

| # | GIF | Dominant signal | Root cause |
|---|---|---|---|
| 1 | `8e172835` (Christmas stocking) | ssim z=-6.04, chist z=-6.29, lpips z=+4.65 | Photographic palette with gradients; animately's lossy colour quantisation collapses luminance/contrast while preserving coarse texture |
| 2 | `XpkfMAWfUsWg` (gstatic growthlab) | deltae z=+8.88, lpips z=+6.53, psnr z=-3.97 | Precision-colour animated data visualisation (219 frames); animately's palette reduction destroys the exact hues; frame drops (219→208) partially degrade alignment |
| 3 | `f19df0a2` (theatre performance) | temporal_consistency≈0, banding_score=100 at lossy=60 | Two separate phenomena: (a) single-stream temporal_consistency metric is not measuring compression quality but the animation's intrinsic frame-to-frame variance; (b) genuine banding introduced at lossy=60 only |
| 4 | `e926f7a2` (vintage portrait) | texture_similarity z=-10.83 (dataset maximum), edge_similarity z=-3.01 | Film-grain photographic GIF; lossy quantisation smooths grain → LBP descriptor and Canny edge map both collapse; SSIM is more forgiving because it measures luminance patches not fine texture |

---

## Outlier 1 — `8e172835-7e1d-a4d9-0445-244425430cd5.gif` (Christmas stocking)

**File:** `https___mcusercontent.com_88adb327dcabde30998542a7d_images_8e172835-...gif`
**Dimensions:** 1200×1200, 17 frames, 3907 KB original
**Content (from thumbnail):** Marketing email GIF — a festive red high-heel boot with white fur trim and "BANG" text. Photographic-quality rendering with smooth red-to-black gradients, white fur texture, and anti-aliased text.

### Metric values at lossy=40

| Metric | Value | Z-score |
|---|---|---|
| ssim | 0.116 | −5.45 |
| ms_ssim | 0.127 | −6.12 |
| psnr | 0.190 | −3.14 |
| lpips_quality_mean | 0.445 | +4.27 |
| chist | 0.504 | −5.59 |
| edge_similarity | 0.213 | −2.59 |
| texture_similarity | 0.999 | +0.38 |
| ssimulacra2_mean | 0.000 | outlier |
| deltae_mean | 25.3 | +2.87 |
| composite_quality | 0.412 | −1.49 |

### What the data shows

- **ssimulacra2_mean=0.0** at all three lossy levels confirms this is a genuine perceptual catastrophe, not a metric artefact. SSIMULACRA2 is regarded as the most perceptually accurate metric in the suite.
- **chist=0.504** (near-random, corpus mean 0.982) means the colour histogram of the compressed output bears almost no resemblance to the original. Animately's palette quantisation has reorganised the entire colour distribution.
- **texture_similarity=0.999** despite ssim=0.116 is the key diagnostic split: LBP (Local Binary Pattern) histograms capture coarse texture pattern frequencies but are insensitive to luminance and colour changes. The macro-texture of fur and fabric is preserved; the specific luminance values are not.
- **deltae_mean=25.3**, **deltae_pct_gt1=99.6%** (virtually every pixel has CIE ΔE>1), **deltae_p95=65.2** — massive colour shift.
- **flat_region_count=13777** (huge flat areas), **gradient_region_count=69** — the image contains exactly the combination animately struggles with: large smooth gradients (red boot body) that require many shades to render faithfully, rendered in a 256-colour GIF palette.
- **flicker_frame_ratio=1.0** (every frame tagged as flickering) suggests the animation uses per-frame dithering or the rapid frame changes look like flicker to the metric.
- Compression ratio escalates from 1.25× at lossy=20 to 1.98× at lossy=40 to 2.57× at lossy=60 — animately is aggressively quantising to hit these ratios.

### Root cause hypothesis

The GIF is a photographic-quality rendering in a 177-colour original palette with smooth red gradients. At lossy=40, animately's Floyd-Steinberg or ordered dither with reduced colour count causes posterisation — broad areas that should transition smoothly through many shades collapse to a small number of discrete colour bands. SSIM, PSNR, LPIPS, and SSIMULACRA2 all detect this as catastrophic quality loss because they measure luminance/colour fidelity. LBP texture survives because the coarse spatial pattern (boot shape, fur outline) is preserved; only the per-pixel colours are wrong.

### Metric behaviour verdict

**Metrics are correct.** SSIM, PSNR, LPIPS, and SSIMULACRA2 are all correctly flagging genuine perceptual degradation. The apparent anomaly (texture_similarity=0.999 while ssim=0.116) is not a metric bug — it's the expected behaviour of LBP vs luminance-sensitive metrics, and it correctly characterises the nature of the failure (spatial structure preserved, colour destroyed).

### Actionable findings

1. **GifLab finding — content type detection gap:** GifLab does not currently classify GIFs as "photographic gradient" vs "flat-colour animation". A classifier that detects photographic-gradient GIFs could warn that lossy>20 will cause visible posterisation and refuse compression or use a lower lossy ceiling. Related to: `giflab-animately-drops-frames-malthouse-investigation` (animately aggression on complex content).

2. **Animately finding:** Animately's lossy compression at levels 40+ appears to destroy photographic gradients completely on this content. This is a known GIF limitation (256-colour palette) but the magnitude of degradation (ssimulacra2=0, ssim=0.116) at a "moderate" lossy setting (40) suggests the lossy ceiling for photographic content should be documented or enforced in gifprep. → Obsidian task: `giflab-photographic-content-lossy-ceiling`.

---

## Outlier 2 — `XpkfMAWfUsWgNNUjSs9rFrQEh4lGNfiFD4VIkoTD.gif` (gstatic growthlab)

**File:** `https___gstatic.com_growthlab_api_XpkfMAWfUsWgNNUjSs9rFrQEh4lGNfiFD4VIkoTD.gif`
**Dimensions:** 720×480, 219 frames, 297 KB original
**Content (from thumbnail):** Thumbnail appears nearly blank/white — the first frame is white or near-white. This is consistent with an animated data visualisation from Google's GrowthLab API that begins with an empty/loading state and animates into showing coloured charts or graph elements.

### Metric values at lossy=40

| Metric | Value | Z-score |
|---|---|---|
| ssim | 0.228 | −4.68 |
| ms_ssim | 0.213 | −5.47 |
| psnr | 0.054 | −3.91 |
| lpips_quality_mean | 0.612 | +6.04 |
| chist | 0.529 | −5.30 |
| deltae_mean | 65.6 | +7.93 |
| deltae_pct_gt1 | 91.7% | outlier |
| deltae_pct_gt5 | 90.8% | outlier |
| composite_quality | 0.243 | −2.35 |
| mse | 36,140 | 50.7× corpus mean |
| rmse | 188.4 | (out of 255) |
| texture_similarity | 0.999 | +0.40 |
| fsim | 0.932 | +1.15 |
| temporal_consistency | 0.00012 | effectively zero |
| ssimulacra2_mean | 0.000 | outlier |

### What the data shows

- **MSE=36,140** is 50.7× the corpus mean for real GIFs at lossy=40. The minimum per-frame MSE is 18,392 — even the best frame pair has an average pixel error of 135/255 (53% of maximum). This is not a subtle quality issue; it is a complete compression failure on this content.
- **deltae_mean=65.6** (near the maximum of 100 = black vs white), **91% of pixels have ΔE>5** (visually obvious colour error on nearly every pixel). This magnitude is consistent with the entire colour palette being inverted or drastically remapped.
- **ssimulacra2_mean=0.0** and **ssim_first=0.0005** (near-zero SSIM on the first frame pair) confirms perceptual catastrophe.
- **ssim_max=0.44** means even the *best* aligned frame pair reaches only 0.44 SSIM. The failure is not an alignment artefact.
- **texture_similarity=0.999** and **fsim=0.932** survive — as with Outlier 1, coarse spatial structure is preserved while colour is destroyed.
- **compressed_frame_count=208 vs 219** — animately dropped 11 frames (5% frame drop). **alignment_accuracy=0.976**. The metrics pipeline correctly handles this via temporal alignment, but the alignment is imperfect.
- **temporal_consistency=0.00012**: The compressed animation is highly irregular (high frame-to-frame variance relative to mean). This is a separate signal from the colour quality collapse but may be related to frame drops disrupting the animation rhythm.

### Root cause hypothesis

The GIF is a 219-frame animated data visualisation from Google's GrowthLab API. Data visualisations depend on precise specific colours (categorical palette, brand colours, grid colours) to convey meaning. Animately's lossy colour quantisation at lossy=40 reduces the effective colour depth, destroying the precise palette. The result is that 91% of pixels have ΔE>5 on average — the chart colours are wrong throughout the animation.

The frame drops (219→208) are a secondary issue: animately's lossy algorithm occasionally collapses near-identical frames, causing GifLab's alignment system to note a 0.976 accuracy and `ssim_first` to potentially reflect a misaligned first frame.

The white thumbnail is diagnostic: the first frame of this GIF is essentially white (empty chart state), and when animately compresses it, this white frame may shift to off-white or pick up colour from the quantised palette of surrounding frames, causing ssim_first=0.0005.

### Metric behaviour verdict

**Metrics are correct.** The catastrophic values across ssim, psnr, deltae, lpips, ssimulacra2 all correctly reflect a genuine compression failure. The texture_similarity/fsim survival is expected (spatial structure intact, colour wrong). This is the single worst-quality compression output in the entire corpus by MSE (50× mean).

### Actionable findings

1. **Animately finding — precision-colour animation failure:** Animated data visualisations with specific categorical palettes are extremely sensitive to colour quantisation. GifLab or gifprep should detect this content type (low colour count, high frame count, flat regions with specific hues) and apply a lower lossy ceiling or reject compression if the quality gate threshold cannot be met. → Obsidian task: `giflab-data-viz-animation-lossy-guard`.

2. **GifLab finding — frame drop and alignment:** The 219→208 frame drop at lossy=40 should be flagged with a warning or error in the metrics output when `alignment_accuracy < 0.98`. The current CSV has alignment=0.976 silently. → Obsidian task (lower priority): part of existing `giflab-validation-checker-nan-aware-refactor` scope or a new alignment-warning task.

---

## Outlier 3 — `f19df0a2-38b1-de19-799b-da8e75ea341e (1).gif` (theatre performance)

**File:** `https___mcusercontent.com_02daa6a2f0aa90cc9c88c656e_images_f19df0a2-...gif`
**Dimensions:** 1000×532, 120 frames, 890 KB original
**Content (from thumbnail):** A high-motion theatre performance photograph — two performers in dramatic motion against a dark atmospheric stage background with blue/teal lighting.

### Metric values at lossy=40 and lossy=60

| Metric | Lossy=20 | Lossy=40 | Lossy=60 |
|---|---|---|---|
| ssim | 0.99999 | 0.99999 | 0.99999 |
| psnr | 1.000 | 1.000 | 1.000 |
| lpips_quality_mean | ~0 | ~0 | ~0 |
| ssimulacra2_mean | 1.000 | 1.000 | 0.319 |
| temporal_consistency | 2.28e-8 | 2.28e-8 | 2.32e-8 |
| banding_score_mean | 0.0 | 0.0 | 100.0 |
| edge_similarity | 0.999 | 0.999 | 0.451 |
| compression_ratio | 1.062 | 1.062 | 1.448 |

### Finding A — temporal_consistency is a single-stream artefact on this content

**temporal_consistency=2.28e-8** at all three lossy levels. This is NOT a metric reporting "compression changed the temporal behaviour" — it is measuring the COMPRESSED animation in isolation and reporting that it is highly irregular.

The `calculate_temporal_consistency` function ([`src/giflab/metrics.py:655`](cursor://file/Users/lachlants/repos/animately/giflab/src/giflab/metrics.py)) computes:

```python
relative_variance = variance_diff / (mean_diff ** 2)
consistency = np.exp(-relative_variance * 2.0)
```

For a 120-frame high-motion theatre performance (100fps playback at 10ms/frame), frame-to-frame pixel differences are large AND highly variable (different movements, lighting, poses in each frame). This produces a very high `relative_variance`, collapsing `consistency` to near-zero.

**Crucially:** The metric reports temporal_consistency=2.28e-8 at lossy=20, where compression_ratio=1.062 (animately barely touched the file) and ssim=0.99999 (the compressed frames are bit-identical to the original). The near-zero temporal_consistency is a property of the SOURCE CONTENT, not of animately's compression. The metric is correctly computing that the animation is erratic — but calling it "temporal_consistency" implies it measures compression quality, not content character.

This is the same class of bug flagged in PR #14 (`giflab-single-stream-metrics-misuse-audit`): a single-stream metric named and used as if it were a pair-comparison metric. When `composite_quality` incorporates this value, it will penalise all high-motion content regardless of compression quality.

### Finding B — banding at lossy=60 is a real finding

At lossy=60, the banding_score jumps from 0 → 100 (z=+5.95, the most extreme banding score in the corpus). ssimulacra2_mean drops from 1.000 to 0.319. edge_similarity collapses from 0.999 to 0.451.

The original GIF at lossy=40 has a compression_ratio of 1.062 — animately barely compressed it. At lossy=60, it achieves 1.448, meaning the encoder is now doing meaningful quantisation. The 120 frames contain many dark gradient regions (the stage background); strong lossy quantisation introduces visible banding in these regions.

This is a genuine quality failure at lossy=60 — not a metric artefact.

### Metric behaviour verdicts

- **temporal_consistency: METRIC MISFIRES.** This metric should be excluded from `composite_quality` for high-motion content, OR the reported value should be temporal_consistency_delta (the change from original to compressed) rather than temporal_consistency_post (absolute value of compressed stream). The key: `temporal_consistency_pre=2.28e-8` and `temporal_consistency_post=2.28e-8` — they are identical, meaning compression made zero difference to the animation rhythm. The delta is essentially zero.
- **banding_score at lossy=60: METRIC IS CORRECT.** The banding is real.

### Actionable findings

1. **Metric fix — temporal_consistency in composite_quality:** The `composite_quality` formula should use `temporal_consistency_delta` (or `1 - temporal_consistency_delta`) rather than raw `temporal_consistency_post` to avoid penalising high-motion content. Alternatively, weight `temporal_consistency` to zero for content where the original already has near-zero temporal_consistency. → Obsidian task: `giflab-temporal-consistency-composite-quality-fix`.

2. **Content-aware note — high-motion threshold:** The temporal_consistency metric needs a content-characterisation companion: "did the original have low temporal_consistency too?" If `temporal_consistency_pre ≈ temporal_consistency_post`, the compression did not degrade temporal behaviour — the low score is intrinsic to the content. This information should be surfaced in the audit report as a separate column.

---

## Outlier 4 — `e926f7a2-b679-1f8b-c3ba-612a32744fd5.gif` (vintage portrait)

**File:** `https___mcusercontent.com_cdfc29de68ae93588ec6b0a8f_images_e926f7a2-...gif`
**Dimensions:** 1920×1440, 5 frames, 7544 KB original
**Content (from thumbnail):** A vintage/film-photograph style portrait of a woman in profile. The image has the soft focus, grain, and muted colours typical of 1970s film photography. The 5 frames suggest minimal animation (perhaps a slow fade or subtle motion).

### Metric values across lossy levels

| Metric | Lossy=20 | Lossy=40 | Lossy=60 |
|---|---|---|---|
| ssim | ~0.89 | 0.843 | 0.760 |
| ms_ssim | ~0.96 | 0.921 | 0.873 |
| psnr | ~0.73 | 0.665 | 0.621 |
| fsim | ~0.45 | 0.395 | 0.352 |
| edge_similarity | 0.344 | 0.179 | 0.120 |
| texture_similarity | 0.903 | 0.741 | 0.677 |
| ssimulacra2_mean | ~0.5 | 0.417 | 0.103 |
| banding_score_mean | 24.3 | 34.9 | 42.4 |
| gradient_region_count | 99 | 99 | 99 |
| color_count_original | 255.8 | 255.8 | 255.8 |
| color_count_compressed | 267.4 | 267.4 | 265.0 |

### What the data shows

- **texture_similarity z=-10.83 at lossy=60** — the most extreme z-score in the entire 368-row dataset. Value 0.677 vs corpus mean ~0.988.
- **edge_similarity z=-3.01 at lossy=60**. Both structural metrics are simultaneously nuked — the canonical failure pattern flagged in the task note.
- **gradient_region_count=99** (at maximum) — this 1920×1440 image is almost entirely composed of smooth gradients (skin tones, background, hair), which is the hardest content type for GIF palette quantisation.
- **color_count_compressed > color_count_original** (267 vs 256) — animately's dithering is creating MORE apparent colours (dither patterns) than the original had discrete values. This is a sign that Floyd-Steinberg dithering is introducing noise/grain at the sub-pixel level.
- **dither_ratio_mean=1.09** (>1.0 means more dithering in compressed than original) — confirms the above.
- **ssimulacra2_mean collapses from 0.5 to 0.103** at lossy=60 — confirming perceptual degradation is real and significant.
- **fsim=0.352-0.453** — FSIM uses phase congruency + gradient magnitude, which are extremely sensitive to film grain. When grain is smoothed, FSIM collapses.

### Root cause hypothesis

Film grain is the critical content property. A photographic emulsion creates thousands of tiny irregular grain particles that appear as high-frequency spatial variation in every channel. LBP (used by `texture_similarity`) is specifically designed to capture this kind of fine micro-texture — it encodes the relationship between each pixel and its 8 neighbours in a binary pattern. Film grain creates very specific LBP signatures (granular, irregular).

When animately's lossy quantisation smooths the grain (treating it as spatial noise to remove), the LBP histogram changes dramatically. Similarly, Canny edge detection on a film-grain image finds thousands of micro-edges (grain boundaries) — lossy compression eliminates these, collapsing the Jaccard similarity of edge maps.

SSIM is more forgiving because it measures patch-level luminance/contrast/structure across large windows (default 11×11 pixels) — at this scale, the overall tonal rendering of the portrait is largely preserved even after grain smoothing. SSIM sees "the skin tones are still in the right regions"; LBP sees "the fine micro-texture pattern has changed".

The 5-frame minimal animation with 7544 KB → 2699 KB at lossy=40 (2.8× compression) is a very aggressive compression of the largest GIF in the corpus. The content is effectively a near-still photograph — a domain where GIF format (256-colour palette) is fundamentally ill-suited.

### Metric behaviour verdict

**Metrics are correct.** Texture_similarity and edge_similarity are correctly flagging that lossy compression has destroyed the fine-grain detail that characterises this photographic content. SSIM being more forgiving is also correct (it measures coarser structure). The split between LBP/edge and SSIM is informative: it tells us the macro-structure is preserved but the micro-texture (grain) is gone.

**Secondary concern:** `composite_quality=0.678` at lossy=40 (near the corpus mean of 0.70) despite ssimulacra2=0.417 and texture_similarity=0.741 suggests the composite_quality formula may be under-weighting these structural metrics on large-resolution, nearly-still content.

### Actionable findings

1. **GifLab finding — photographic grain content type:** The suite lacks a "film grain" or "photographic noise" content classifier. Detecting high-frequency spatial noise (grain) in the original would allow gifprep to warn that lossy compression will smooth it, or to use a lower lossy ceiling. → Obsidian task: `giflab-photographic-content-lossy-ceiling` (same task as Outlier 1 finding, adding grain as a trigger).

2. **Metric note — composite_quality weighting for still-ish photographic content:** When frame count is very low (≤10) and the content is photographic (high gradient_region_count), composite_quality may need to weight ssimulacra2 and texture_similarity more heavily. This is a future recalibration item, not an immediate fix.

---

## Cross-cutting patterns

### Pattern 1 — texture_similarity (LBP) systematically survives colour quantisation

In both Outlier 1 and Outlier 2, texture_similarity ≈ 0.999 while all colour-sensitive metrics collapse. LBP is colour-blind (operates on greyscale) and captures only coarse spatial patterns. It is not a useful proxy for perceptual quality on colour-sensitive content — but it IS informative as a "structure vs colour" separator.

**Implication:** texture_similarity should be presented in reports alongside colour-sensitive metrics with a note that it measures spatial texture only. Its presence in composite_quality should be weighted accordingly — it must not offset colour quality failures.

### Pattern 2 — temporal_consistency is a single-stream metric posing as a comparison metric

Outlier 3 demonstrates this concretely: temporal_consistency_pre=temporal_consistency_post=2.28e-8. The delta is zero. Compression did not change the animation's temporal character. But the absolute value (near-zero) drags composite_quality down.

This is the pattern PR #14 (`giflab-single-stream-metrics-misuse-audit`) was meant to address. The follow-up needed here is specifically to have composite_quality use the delta signal, not the absolute post-compression value.

### Pattern 3 — banding appears as a lossy-60 cliff

Outlier 3 shows banding_score jumping from 0 to 100 between lossy=40 and lossy=60. Outlier 4 shows it increasing monotonically (24→35→42). The lossy=60 cliff in Outlier 3 is the most extreme case in the corpus (z=+5.95).

This is consistent with the known "cliff-edge threshold" issue in banding detection — the banding detector may be using a hard threshold internally. This connects to the Wave 2 task `giflab-flat-mean-tol-recalibration`.

### Pattern 4 — frame drops in long animations cause alignment degradation

Outlier 2's 219→208 frame drop (5% loss) at lossy=40 causes alignment_accuracy=0.976. The current pipeline silently proceeds — there is no warning in the metrics output when alignment is imperfect. This should be flagged.

---

## Follow-up tasks

Five Obsidian task notes have been created from these findings:

1. `giflab-photographic-content-lossy-ceiling` — detect photographic-gradient + film-grain GIFs and enforce a lower lossy ceiling in gifprep/compress
2. `giflab-data-viz-animation-lossy-guard` — detect precision-colour animated visualisations and guard against palette quantisation destroying categorical colours
3. `giflab-temporal-consistency-composite-quality-fix` — use temporal_consistency_delta rather than temporal_consistency_post in composite_quality; add content-aware skip when pre≈post
4. `giflab-texture-similarity-composite-weight-review` — document that LBP/texture_similarity is colour-blind and review its weight in composite_quality for colour-sensitive content
5. `giflab-alignment-warning-threshold` — emit a warning (or raise quality concern) when alignment_accuracy < 0.98, i.e., when frame drops cause imperfect metric comparison

---

*Generated by read-only investigation in worktree `audit-fix/outlier-deep-dive`. No source code changes made. See rollout `giflab-rollout-2026-05-26` Wave 1 for dispatch context.*
