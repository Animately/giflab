# Preset Types Reference Guide

This reference provides detailed technical information about all available preset types, their configurations, and specific use cases.

## Preset Classification

GifLab presets are organized into several categories based on their research purpose and complexity:

### Research Presets
Focused studies comparing specific algorithm dimensions while controlling other variables.

### Specialized Presets  
Advanced configurations for specific optimization scenarios or content types.

### Development Presets
Minimal configurations for testing, debugging, and rapid iteration.

### Baseline Presets
Comprehensive comparisons across multiple dimensions for establishing performance baselines.

---

## Research Presets

### `frame-focus`: Frame Removal Focus Study

**Purpose**: Compare all available frame reduction algorithms while controlling for color quantization and lossy compression variables.

**Research Question**: "Which frame reduction algorithm produces the best quality/size ratio for my content?"

**Configuration**:
```yaml
frame_slot:
  type: variable
  scope: ["*"]  # All 5 frame algorithms
  parameters: 
    ratios: [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]

color_slot:
  type: locked
  implementation: ffmpeg-color
  parameters:
    colors: 32

lossy_slot:
  type: locked  
  implementation: animately-advanced
  parameters:
    level: 40
```

**Pipeline Generation**:
- **Count**: ~5 pipelines (one per frame algorithm)
- **Algorithms Tested**: animately-frame, ffmpeg-frame, gifsicle-frame, imagemagick-frame, none-frame

**Use Cases**:
- Content-specific frame algorithm optimization
- Performance comparison across different frame reduction approaches
- Validating frame algorithm choice for specific video content types
- Research into frame removal effectiveness

**Expected Results**:
- Clear ranking of frame algorithms by quality/size ratio
- Identification of optimal frame removal ratios per algorithm
- Content-type-specific frame algorithm recommendations

**Example Usage**:
```bash
poetry run python -m giflab run --preset frame-focus --quality-threshold 0.05
```

---

### `color-optimization`: Color Optimization Study

**Purpose**: Compare all color reduction techniques and dithering methods across all available variants.

**Research Question**: "Which color quantization approach produces the best visual quality for my palette requirements?"

**Configuration**:
```yaml
frame_slot:
  type: locked
  implementation: animately-frame
  parameters:
    ratio: 1.0  # No frame reduction

color_slot:
  type: variable
  scope: ["*"]  # All 17 color reduction variants
  parameters:
    colors: [256, 128, 64, 32, 16, 8]

lossy_slot:
  type: locked
  implementation: none-lossy
  parameters:
    level: 0  # No lossy compression
```

**Pipeline Generation**:
- **Count**: ~17 pipelines (one per color algorithm)
- **Algorithms Tested**: 
  - Animately: 1 variant (animately-color)
  - FFmpeg: 9 variants (including floyd-steinberg, sierra2, bayer0-4 dithering)
  - Gifsicle: 1 variant (gifsicle-color)  
  - ImageMagick: 4 variants (including floyd-steinberg, riemersma dithering)
  - NoOp: 1 variant (none-color)

**Use Cases**:
- Dithering algorithm comparison for specific palette sizes
- Color quantization quality analysis
- Content-type optimization (photos vs graphics vs animations)
- Palette size vs quality tradeoff analysis

**Expected Results**:
- Optimal color algorithm per palette size
- Dithering effectiveness comparison
- Visual quality rankings for different content types
- Recommendations for color count vs algorithm combinations

**Example Usage**:
```bash
poetry run python -m giflab run --preset color-optimization --output-dir color_study
```

---

### `lossy-quality-sweep`: Lossy Quality Sweep

**Purpose**: Evaluate lossy compression effectiveness across different compression engines and quality levels.

**Research Question**: "What lossy compression settings provide the best quality/size tradeoff for my content?"

**Configuration**:
```yaml
frame_slot:
  type: locked
  implementation: none-frame
  parameters:
    ratio: 1.0  # No frame reduction

color_slot:
  type: locked
  implementation: ffmpeg-color
  parameters:
    colors: 64  # Conservative color count

lossy_slot:
  type: variable
  scope: ["*"]  # All 11 lossy compression tools
  parameters:
    levels: [0, 20, 40, 60, 80, 100, 120, 140, 160]
```

**Pipeline Generation**:
- **Count**: ~11 pipelines (one per lossy algorithm)
- **Algorithms Tested**:
  - Animately: 3 variants (animately-standard, animately-advanced, animately-hard)
  - Gifsicle: 5 variants (gifsicle-lossy with different methods)
  - FFmpeg: 1 variant (ffmpeg-lossy)
  - Gifski: 1 variant (gifski-lossy)
  - ImageMagick: 1 variant (imagemagick-lossy)
  - NoOp: 1 variant (none-lossy)

**Use Cases**:
- Lossy compression effectiveness comparison
- Quality level optimization for specific content
- Engine-specific lossy algorithm evaluation
- Size reduction vs quality loss analysis

**Expected Results**:
- Optimal lossy level per compression engine
- Quality degradation curves for each algorithm
- Size reduction effectiveness comparison
- Content-specific lossy compression recommendations

**Example Usage**:
```bash
poetry run python -m giflab run --preset lossy-quality-sweep --use-gpu
```

---

## Baseline Presets

### `tool-comparison-baseline`: Tool Comparison Baseline

**Purpose**: Fair comparison across complete toolchain engines with conservative parameter sets.

**Research Question**: "How do the major GIF creation toolchains (animately, ffmpeg, gifsicle, imagemagick) compare overall?"

**Configuration**:
```yaml
frame_slot:
  type: variable
  scope: [animately-frame, ffmpeg-frame, gifsicle-frame, imagemagick-frame]
  parameters:
    ratios: [1.0, 0.8, 0.5]

color_slot:
  type: variable  
  scope: [animately-color, ffmpeg-color, gifsicle-color, imagemagick-color]
  parameters:
    colors: [64, 32, 16]

lossy_slot:
  type: variable
  scope: [animately-advanced, ffmpeg-lossy, gifsicle-lossy, imagemagick-lossy]
  parameters:
    levels: [0, 40, 120]
```

**Pipeline Generation**:
- **Count**: ~64 pipelines (4 engines × 3 frame ratios × 3 color counts × ~1.3 lossy combinations)
- **Complete Toolchains**: animately, ffmpeg, gifsicle, imagemagick

**Use Cases**:
- Establishing performance baselines across toolchains
- Fair engine comparison with matched parameters
- Initial tool selection for new projects
- Validation testing across different tool combinations

**Expected Results**:
- Overall engine rankings by quality/size/speed
- Parameter sensitivity analysis per engine
- Toolchain-specific optimization recommendations
- Performance baseline establishment

**Example Usage**:
```bash
poetry run python -m giflab run --preset tool-comparison-baseline --quality-threshold 0.1
```

---

## Specialized Presets

### `dithering-focus`: Dithering Algorithm Focus

**Purpose**: Compare dithering algorithms specifically using FFmpeg and ImageMagick variants with different palette sizes.

**Research Question**: "Which dithering method produces the best visual quality for palette-constrained GIFs?"

**Configuration**:
```yaml
frame_slot:
  type: locked
  implementation: none-frame
  parameters:
    ratio: 1.0  # No frame reduction

color_slot:
  type: variable
  scope: [
    ffmpeg-color-floyd, ffmpeg-color-sierra2,
    ffmpeg-color-bayer0, ffmpeg-color-bayer2, 
    imagemagick-color-floyd, imagemagick-color-riemersma
  ]
  parameters:
    colors: [64, 32, 16]  # Test multiple palette sizes

lossy_slot:
  type: locked
  implementation: none-lossy
  parameters:
    level: 0  # No lossy compression
```

**Pipeline Generation**:
- **Count**: ~6 pipelines (6 dithering algorithms)
- **Dithering Methods**:
  - Floyd-Steinberg (FFmpeg + ImageMagick variants)
  - Sierra2 (FFmpeg)
  - Bayer 0/2 (FFmpeg ordered dithering)
  - Riemersma (ImageMagick space-filling curve)

**Use Cases**:
- Fine-tuning dithering for specific content types (photos, illustrations, graphics)
- Low-palette optimization (16-64 colors)
- Dithering algorithm comparison for gradient-heavy content
- Print/display output optimization

**Expected Results**:
- Best dithering method per content type
- Palette size vs dithering effectiveness analysis
- Visual quality comparison across dithering approaches
- Recommendations for gradient vs flat color content

**Example Usage**:
```bash
poetry run python -m giflab run --preset dithering-focus --output-dir dithering_analysis
```

---

### `png-optimization`: PNG Sequence Optimization

**Purpose**: Focus on gifski and animately-advanced PNG sequence workflows for high-quality GIF creation.

**Research Question**: "What's the optimal configuration for high-quality GIF creation from PNG sequences?"

**Configuration**:
```yaml
frame_slot:
  type: locked
  implementation: ffmpeg-frame  # Good PNG extraction
  parameters:
    ratio: 0.8  # Slight frame reduction

color_slot:
  type: variable
  scope: [ffmpeg-color, imagemagick-color]  # High-quality color reduction
  parameters:
    colors: [128, 64, 32]

lossy_slot:
  type: variable
  scope: [gifski-lossy, animately-advanced]  # PNG-optimized tools
  parameters:
    levels: [60, 80, 100]  # Mid-to-high quality levels
```

**Pipeline Generation**:
- **Count**: ~4 pipelines (2 color × 2 lossy algorithms)
- **Optimized Tools**: gifski-lossy, animately-advanced (PNG sequence specialists)

**Use Cases**:
- High-quality animation GIF creation
- PNG sequence workflow optimization  
- Professional/commercial GIF production
- Quality-prioritized compression (size secondary)

**Expected Results**:
- Optimal PNG-to-GIF workflow configuration
- Quality/size tradeoffs for high-end output
- Tool-specific parameter recommendations
- PNG sequence processing best practices

**Example Usage**:
```bash
poetry run python -m giflab run --preset png-optimization --quality-threshold 0.02
```

---

## Development Presets

### `quick-test`: Quick Development Test

**Purpose**: Fast preset for development and debugging with minimal pipeline combinations.

**Research Question**: "Does my code/configuration work correctly?"

**Configuration**:
```yaml
frame_slot:
  type: variable
  scope: [animately-frame, gifsicle-frame]  # Just 2 reliable algorithms
  parameters:
    ratios: [1.0, 0.5]

color_slot:
  type: locked
  implementation: ffmpeg-color
  parameters:
    colors: 32

lossy_slot:
  type: locked
  implementation: animately-advanced
  parameters:
    level: 40
```

**Pipeline Generation**:
- **Count**: ~2 pipelines (2 frame algorithms)
  
- **Fast Execution**: Minimal combinations for rapid testing

**Use Cases**:
- Development workflow testing
- Configuration validation
- Quick sanity checks
- Debugging pipeline issues
- CI/CD pipeline validation

**Expected Results**:
- Fast execution confirmation
- Basic functionality validation
- Configuration error detection
- Development workflow verification

**Example Usage**:
```bash
poetry run python -m giflab run --preset quick-test --output-dir test_run
```

---

## Preset Selection Guidelines

### By Research Goal

**Algorithm Comparison** → Use single-dimension variable presets:
- Frame algorithms: `frame-focus`
- Color algorithms: `color-optimization`  
- Lossy algorithms: `lossy-quality-sweep`

**Quality Optimization** → Use specialized presets:
- High-quality output: `png-optimization`
- Dithering optimization: `dithering-focus`

**Baseline Establishment** → Use comprehensive presets:
- General comparison: `tool-comparison-baseline`

**Development/Testing** → Use minimal presets:
- Quick validation: `quick-test`

### By Pipeline Count Budget

- **2 pipelines**: `quick-test`
- **5 pipelines**: `frame-focus`
- **11 pipelines**: `lossy-quality-sweep`  
- **17 pipelines**: `color-optimization`
- **64 pipelines**: `tool-comparison-baseline`

### By Content Type

**Photographic Content**:
- Primary: `dithering-focus` (for palette optimization)
- Secondary: `color-optimization` (for quantization method)

**Animation Content**:
- Primary: `frame-focus` (for temporal optimization)  
- Secondary: `png-optimization` (for high-quality output)

**Graphic/Logo Content**:
- Primary: `color-optimization` (for clean palette handling)
- Secondary: `lossy-quality-sweep` (for size optimization)

**Mixed Content**:
- Primary: `tool-comparison-baseline` (for general optimization)

### By Performance Priority

**Quality Priority**: `png-optimization` → `dithering-focus` → `color-optimization`

**Size Priority**: `lossy-quality-sweep` → `frame-focus` → `tool-comparison-baseline`

**Speed Priority**: `quick-test` → `frame-focus` → any single-dimension preset

**Balance Priority**: `tool-comparison-baseline` → individual dimension presets as needed

---

## Technical Implementation Details

### Slot Configuration Types

**Variable Slot**:
```python
SlotConfiguration(
    type="variable",
    scope=["*"] or ["tool1", "tool2"],  # Algorithms to test
    parameters={"param": [val1, val2]}   # Parameter ranges
)
```

**Locked Slot**:
```python
SlotConfiguration(
    type="locked", 
    implementation="specific-tool",      # Fixed algorithm
    parameters={"param": value}          # Fixed parameters
)
```

### Pipeline Generation Formula

```
Pipeline Count = Variable_Frame_Count × Variable_Color_Count × Variable_Lossy_Count
```

Where each variable count is determined by:
- `scope=["*"]` → All available tools for that variable
- `scope=["tool1", "tool2"]` → Specific tool count
- Locked slots contribute 1 to the multiplication

### Efficiency Calculation

```
Efficiency = 1 - (Targeted_Pipeline_Count / Traditional_Pipeline_Count)
```

Where Traditional_Pipeline_Count = hundreds (5 frame × 17 color × 11 lossy)

---

## Custom Preset Creation

### Template Structure

```python
custom_preset = ExperimentPreset(
    name="Descriptive Name",
    description="Clear research purpose",
    frame_slot=SlotConfiguration(...),
    color_slot=SlotConfiguration(...), 
    lossy_slot=SlotConfiguration(...),
    
    # Optional enhancements
    custom_sampling="quick|representative|factorial",
    max_combinations=100,
    tags=["category", "purpose"],
    author="Your Name"
)
```

### Validation Requirements

- At least one slot must be variable
- Variable slots must specify scope
- Locked slots must specify implementation
- Custom sampling must be valid strategy
- max_combinations must be positive

This reference provides comprehensive technical details for understanding and using all available preset types effectively.