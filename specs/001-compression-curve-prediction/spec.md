# Feature Specification: Compression Curve Prediction

**Feature Branch**: `001-compression-curve-prediction`  
**Created**: 2025-01-26  
**Status**: Draft  
**Input**: User description: "Predict compression curves and color reduction curves for GIFs based on visual features and metadata"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Predict Compression Curve for New GIF (Priority: P1)

A user uploads a new GIF to the Animately platform. Before running any compression, the system analyzes the GIF's visual characteristics and predicts the file size at each lossy compression level (0, 20, 40, 60, 80, 100, 120). This allows the user to select their desired quality/size tradeoff without waiting for multiple compression runs.

**Why this priority**: This is the core value proposition—predicting compression outcomes before expensive computation. Enables instant feedback for users choosing compression settings.

**Independent Test**: Can be tested by providing a GIF file and verifying the system returns a predicted curve with file sizes for each lossy level.

**Acceptance Scenarios**:

1. **Given** a GIF file with known visual characteristics, **When** the prediction system analyzes it, **Then** it returns predicted file sizes for lossy levels 0, 20, 40, 60, 80, 100, 120
2. **Given** a GIF file, **When** the prediction is compared to actual compression results, **Then** the predicted sizes are within 15% of actual sizes for 80% of predictions

---

### User Story 2 - Predict Color Reduction Curve (Priority: P2)

A user wants to understand how reducing the color palette affects file size for their specific GIF. The system predicts file sizes at color counts 256, 128, 64, 32, 16 without running actual compression.

**Why this priority**: Color reduction is a key compression lever. Predicting its impact enables smarter palette choices, especially for graphics vs. photographic content.

**Independent Test**: Can be tested by providing a GIF and verifying predicted file sizes for each color count.

**Acceptance Scenarios**:

1. **Given** a GIF file, **When** the color curve prediction runs, **Then** it returns predicted file sizes for color counts 256, 128, 64, 32, 16
2. **Given** a GIF with many colors (>200 unique), **When** color curve is predicted, **Then** the curve shows steeper reduction than for a GIF with few colors (<50 unique)

---

### User Story 3 - Extract Visual Features for ML Training (Priority: P1)

The system extracts a standardized set of visual features from GIFs that can be used to train compression prediction models. These features capture spatial complexity, temporal characteristics, and color distribution.

**Why this priority**: Equal to P1 because without feature extraction, no prediction model can be trained. This is the data collection foundation.

**Independent Test**: Can be tested by extracting features from a GIF and verifying all required feature fields are populated with valid numeric values.

**Acceptance Scenarios**:

1. **Given** a GIF file, **When** feature extraction runs, **Then** all defined feature fields are populated with numeric values
2. **Given** the same GIF file processed twice, **When** features are extracted, **Then** identical feature values are produced (deterministic)

---

### User Story 4 - Build Training Dataset (Priority: P2)

The system collects compression results paired with visual features to create a training dataset. Each record contains the input GIF's features and the actual compression outcomes (file sizes at various parameter combinations).

**Why this priority**: Required to train prediction models, but depends on feature extraction (P1) being complete first.

**Independent Test**: Can be tested by running compression experiments and verifying the output dataset contains both features and outcomes in the correct schema.

**Acceptance Scenarios**:

1. **Given** a set of GIFs and compression parameters, **When** the dataset builder runs, **Then** it produces records with visual features joined to compression outcomes
2. **Given** the training dataset, **When** validated against schema, **Then** all records pass schema validation

---

### Edge Cases

- What happens when a GIF has only 1 frame? (Prediction should still work, temporal features default to 0)
- What happens when a GIF has >500 frames? (Feature extraction should sample frames, not process all)
- What happens when a GIF uses transparency heavily? (Features should capture transparency ratio)
- How does the system handle corrupted or truncated GIFs? (Return error, do not produce partial features)
- What happens when predicted curve is requested for an unsupported engine? (Return error listing supported engines)

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST extract at least 15 visual features from any valid GIF file
- **FR-002**: System MUST produce identical features for the same GIF file across multiple runs (determinism)
- **FR-003**: System MUST predict compression curve (file size vs lossy level) for gifsicle and animately engines
- **FR-004**: System MUST predict color reduction curve (file size vs color count) for gifsicle and animately engines
- **FR-005**: System MUST validate all training data records against a defined schema before storage
- **FR-006**: System MUST tag all prediction outputs with model version and training dataset version
- **FR-007**: System MUST support batch feature extraction for multiple GIFs
- **FR-008**: System MUST complete feature extraction for a typical GIF (<5MB, <100 frames) within 5 seconds
- **FR-009**: System MUST provide prediction confidence scores alongside predicted values

### Key Entities

- **GifFeatures**: Visual characteristics extracted from a GIF. Includes spatial features (edge density, color complexity, gradient smoothness, entropy), temporal features (motion intensity, static region ratio, inter-frame MSE), and metadata (frame count, dimensions, color count).

- **CompressionCurve**: A mapping from compression parameter values to predicted file sizes. Contains engine identifier, parameter type (lossy or colors), parameter values, and predicted sizes in KB.

- **TrainingRecord**: A paired record of GifFeatures and actual compression outcomes. Used to train prediction models. Includes GIF SHA for deduplication, feature vector, and outcome vector.

- **PredictionModel**: A trained model that maps GifFeatures to CompressionCurve. Versioned and tagged with training dataset version.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Compression curve predictions are within 15% of actual file sizes for 80% of test GIFs
- **SC-002**: Color reduction curve predictions are within 20% of actual file sizes for 75% of test GIFs
- **SC-003**: Feature extraction completes in under 5 seconds for 95% of GIFs under 5MB
- **SC-004**: Training dataset contains at least 1,000 unique GIFs with complete feature and outcome data
- **SC-005**: Prediction latency is under 100ms per GIF (excluding feature extraction)
- **SC-006**: Model retraining with new data completes in under 1 hour for datasets up to 10,000 GIFs

## Assumptions

- The existing 11-metric quality system provides sufficient quality assessment for compression outcomes
- GIFs in the training set are representative of real-world Animately platform uploads
- Gifsicle and Animately engines are the primary prediction targets; other engines can be added later
- Feature extraction can leverage existing tagger infrastructure where applicable
- Training will initially use gradient boosting or similar interpretable models before exploring neural approaches
