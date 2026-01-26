# Research: Compression Curve Prediction

**Feature**: 001-compression-curve-prediction  
**Date**: 2025-01-26  
**Status**: Complete

## Research Questions

### RQ-1: Which visual features best predict compression outcomes?

**Decision**: Use a combination of existing tagger features (25 scores) plus new compressibility-specific features.

**Rationale**: The existing `HybridCompressionTagger` already extracts 25 continuous scores including:
- Spatial: `edge_density`, `color_complexity`, `gradient_smoothness`, `contrast_score`, `text_density`
- Temporal: `motion_intensity`, `motion_smoothness`, `static_region_ratio`, `temporal_entropy`, `frame_similarity`
- Quality: `blocking_artifacts`, `ringing_artifacts`, `quantization_noise`

Additional features needed for curve prediction:
- `lossless_compression_ratio`: How well the GIF compresses with lossless methods (baseline compressibility)
- `inter_frame_mse_mean`: Average pixel difference between consecutive frames
- `color_histogram_entropy`: Distribution of color usage
- `dominant_color_ratio`: Percentage of pixels using top-N colors

**Alternatives Considered**:
- Deep learning feature extraction (ResNet, CLIP embeddings): Rejected for initial version due to complexity and inference time. Can be added later.
- Raw pixel statistics only: Rejected as insufficient for capturing semantic content differences.

---

### RQ-2: What ML model architecture for curve prediction?

**Decision**: Gradient Boosting Regressor (scikit-learn) with separate models per engine and curve type.

**Rationale**:
- Interpretable: Feature importances help understand what drives compression
- Fast inference: <10ms per prediction
- Handles mixed feature types well
- Proven for tabular regression tasks
- Already available in giflab dependencies (sklearn)

**Model Structure**:
- 4 models total: `{gifsicle, animately} × {lossy_curve, color_curve}`
- Each model predicts 7 values (lossy: 0,20,40,60,80,100,120) or 5 values (color: 256,128,64,32,16)
- Multi-output regression using `MultiOutputRegressor` wrapper

**Alternatives Considered**:
- Neural network: Rejected for initial version (overkill for <20 features, harder to debug)
- Linear regression: Rejected (compression curves are non-linear)
- Random Forest: Viable alternative, but gradient boosting typically performs better on structured data

---

### RQ-3: How to handle engine-specific behavior?

**Decision**: Train separate models per engine. Include engine as a categorical feature for future unified model.

**Rationale**: Gifsicle and Animately have different compression algorithms and respond differently to the same parameters. Separate models capture these differences without complex feature engineering.

**Alternatives Considered**:
- Single model with engine as feature: May work but harder to interpret and debug initially
- Transfer learning between engines: Premature optimization

---

### RQ-4: Training data requirements?

**Decision**: Minimum 1,000 unique GIFs with full compression sweeps (all lossy levels × all color counts).

**Rationale**:
- 15+ features requires ~100 samples per feature for stable gradient boosting
- Diverse content types needed (graphics, photos, animations, screen recordings)
- Full parameter sweeps needed to learn curve shapes

**Data Collection Strategy**:
1. Use existing giflab pipeline to run compression experiments
2. Extract features for each GIF using enhanced tagger
3. Join features with compression outcomes by `gif_sha`
4. Split 80/10/10 train/val/test by GIF (not by row) to prevent leakage

---

### RQ-5: How to measure prediction accuracy?

**Decision**: Mean Absolute Percentage Error (MAPE) per curve point, with 15% threshold for lossy and 20% for color.

**Rationale**: Percentage error is more interpretable than absolute KB error across different GIF sizes. Thresholds based on acceptable user experience for size estimation.

**Metrics**:
- `mape_per_point`: MAPE at each parameter value
- `mape_overall`: Average MAPE across all points
- `within_threshold_rate`: % of predictions within target threshold

---

## Integration Points

### Existing Code to Leverage

| Module | What to Use | How |
|--------|-------------|-----|
| `tagger.py` | `HybridCompressionTagger` | Extract 25 visual features |
| `metrics.py` | `extract_gif_frames()` | Frame extraction for temporal features |
| `lossy.py` | `compress_with_gifsicle()`, `compress_with_animately()` | Generate training outcomes |
| `config.py` | `CompressionConfig` | Standard parameter values |
| `meta.py` | `extract_gif_metadata()` | GIF SHA and basic metadata |

### New Dependencies

- `pydantic>=2.0`: Schema validation (already in pyproject.toml)
- `scikit-learn`: Already in pyproject.toml for existing ML features

No new dependencies required.

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Insufficient training data | Medium | High | Start with synthetic GIFs, expand to real data |
| Feature extraction too slow | Low | Medium | Leverage existing caching, sample frames |
| Model accuracy below threshold | Medium | Medium | Iterative feature engineering, ensemble methods |
| Engine version changes break models | Low | High | Version tag models, retrain on engine updates |
