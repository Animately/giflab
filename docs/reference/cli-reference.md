# CLI Reference: Targeted Presets

This reference provides complete documentation for all command-line options and usage patterns for the targeted presets system.

## Basic Command Structure

```bash
poetry run python -m giflab run [OPTIONS]
```

## Preset-Based Options

### `--preset, -p TEXT`

Use a predefined compression preset.

**Usage**: `--preset PRESET_ID` or `-p PRESET_ID`

**Available Presets**:
- `frame-focus` - Compare frame reduction algorithms
- `color-optimization` - Compare color reduction techniques  
- `lossy-quality-sweep` - Evaluate lossy compression effectiveness
- `tool-comparison-baseline` - Fair engine comparison
- `dithering-focus` - Compare dithering algorithms
- `png-optimization` - Optimize PNG sequence workflows
- `quick-test` - Fast development testing

**Examples**:
```bash
# Basic preset usage
poetry run python -m giflab run --preset frame-focus
poetry run python -m giflab run -p color-optimization

# Preset with additional options
poetry run python -m giflab run --preset frame-focus --quality-threshold 0.1
poetry run python -m giflab run -p quick-test --output-dir test_results
```

### `--list-presets`

List all available experiment presets with descriptions.

**Usage**: `--list-presets`

**Output Format**:
```
Available experiment presets:

frame-focus: Compare all frame reduction algorithms with locked color and lossy settings
color-optimization: Compare color reduction techniques and dithering methods across all variants
lossy-quality-sweep: Evaluate lossy compression effectiveness across different engines
...
```

**Example**:
```bash
poetry run python -m giflab run --list-presets
```

---

## Custom Slot Configuration Options

### `--variable-slot SLOT=SCOPE`

Define a variable slot that tests multiple algorithms.

**Slot Names**: `frame`, `color`, `lossy`

**Scope Options**:
- `*` - All available algorithms for this slot
- `tool1,tool2,tool3` - Comma-separated list of specific tools

**Examples**:
```bash
# Test all frame algorithms
--variable-slot frame=*

# Test specific color algorithms  
--variable-slot color=ffmpeg-color,gifsicle-color

# Multiple variable slots
--variable-slot frame=* --variable-slot color=ffmpeg-color,imagemagick-color

# Test specific frame and lossy algorithms
--variable-slot frame=animately-frame,ffmpeg-frame --variable-slot lossy=*
```

### `--lock-slot SLOT=IMPLEMENTATION`

Lock a slot to use a specific algorithm with fixed parameters.

**Slot Names**: `frame`, `color`, `lossy`

**Implementation Names**: See [Available Tool Implementations](#available-tool-implementations)

**Examples**:
```bash
# Lock frame to specific algorithm
--lock-slot frame=animately-frame

# Lock multiple slots  
--lock-slot color=ffmpeg-color --lock-slot lossy=animately-advanced

# Lock all non-variable slots
--variable-slot frame=* --lock-slot color=ffmpeg-color --lock-slot lossy=none-lossy
```

### `--slot-params SLOT=PARAM:VALUE`

Specify parameters for slot algorithms.

**Parameter Formats**:
- `param:value` - Single value (integer, float, string, boolean)
- `param:[value1,value2,value3]` - List values

**Common Parameters**:
- **Frame**: `ratios:[1.0,0.8,0.5]` - Frame reduction ratios
- **Color**: `colors:32` - Color count, `colors:[64,32,16]` - Multiple color counts
- **Lossy**: `level:40` - Lossy compression level, `levels:[0,40,80]` - Multiple levels

**Examples**:
```bash
# Single parameter values
--slot-params color=colors:32
--slot-params lossy=level:60
--slot-params frame=ratio:0.8

# List parameter values  
--slot-params frame=ratios:[1.0,0.8,0.5,0.2]
--slot-params color=colors:[128,64,32,16]
--slot-params lossy=levels:[0,40,80,120]

# Multiple parameter specifications
--slot-params color=colors:32 --slot-params lossy=level:40
```

---

## Integration Options

### Quality and Threshold Options

#### `--quality-threshold FLOAT`
Set the quality threshold for experiment analysis.

**Default**: `0.05`
**Range**: `0.0` to `1.0`

**Examples**:
```bash
--quality-threshold 0.1    # More permissive quality requirement
--quality-threshold 0.02   # Stricter quality requirement
```

#### `--use-targeted-gifs`
Use targeted synthetic GIF set instead of full set for testing.

**Examples**:
```bash
--use-targeted-gifs        # Use smaller, focused GIF set
```

### Output and Performance Options

#### `--output-dir PATH`
Specify output directory for experiment results.

**Default**: Auto-generated based on experiment type
**Format**: Directory path (created if doesn't exist)

**Examples**:
```bash
--output-dir results
--output-dir /path/to/specific/directory
--output-dir frame_study_$(date +%Y%m%d)
```

#### `--use-cache`
Enable result caching for faster repeated experiments.

**Examples**:
```bash
--use-cache               # Enable caching
```

#### `--use-gpu`
Enable GPU acceleration if available.

**Examples**:
```bash
--use-gpu                 # Enable GPU acceleration
```

---

## Complete Usage Examples

### Basic Preset Usage

```bash
# Simple preset execution
poetry run python -m giflab run --preset frame-focus

# Preset with custom output directory
poetry run python -m giflab run --preset color-optimization --output-dir color_study

# Preset with custom quality threshold
poetry run python -m giflab run --preset lossy-quality-sweep --quality-threshold 0.1

# Preset with performance options
poetry run python -m giflab run --preset tool-comparison-baseline --use-cache --use-gpu
```

### Custom Slot Configurations

```bash
# Frame algorithm comparison
poetry run python -m giflab run \
  --variable-slot frame=* \
  --lock-slot color=ffmpeg-color \
  --lock-slot lossy=animately-advanced \
  --slot-params color=colors:32 \
  --slot-params lossy=level:40

# Color algorithm comparison with specific tools
poetry run python -m giflab run \
  --variable-slot color=ffmpeg-color,gifsicle-color,imagemagick-color \
  --lock-slot frame=animately-frame \
  --lock-slot lossy=none-lossy \
  --slot-params frame=ratio:1.0

# Multi-dimensional comparison
poetry run python -m giflab run \
  --variable-slot frame=animately-frame,ffmpeg-frame \
  --variable-slot color=ffmpeg-color,gifsicle-color \
  --lock-slot lossy=none-lossy \
  --slot-params frame=ratios:[1.0,0.8] \
  --slot-params color=colors:[64,32]

# Lossy compression study
poetry run python -m giflab run \
  --variable-slot lossy=* \
  --lock-slot frame=none-frame \
  --lock-slot color=ffmpeg-color \
  --slot-params color=colors:64 \
  --slot-params lossy=levels:[0,20,40,60,80]
```

### Advanced Configurations

```bash
# High-quality study with strict threshold
poetry run python -m giflab run \
  --preset png-optimization \
  --quality-threshold 0.02 \
  --output-dir high_quality_study \
  --use-gpu

# Development testing with caching
poetry run python -m giflab run \
  --preset quick-test \
  --use-cache \
  --use-targeted-gifs \
  --output-dir dev_test

# Custom dithering study
poetry run python -m giflab run \
  --variable-slot color=ffmpeg-color-floyd,ffmpeg-color-bayer2,imagemagick-color-floyd \
  --lock-slot frame=none-frame \
  --lock-slot lossy=none-lossy \
  --slot-params color=colors:[64,32,16] \
  --output-dir dithering_comparison
```

---

## Available Tool Implementations

### Frame Reduction Tools

- `animately-frame` - Animately frame reduction algorithm
- `ffmpeg-frame` - FFmpeg frame reduction algorithm  
- `gifsicle-frame` - Gifsicle frame reduction algorithm
- `imagemagick-frame` - ImageMagick frame reduction algorithm
- `none-frame` - No frame reduction (pass-through)

### Color Reduction Tools

- `animately-color` - Animately color quantization
- `ffmpeg-color` - FFmpeg standard color quantization
- `ffmpeg-color-floyd` - FFmpeg with Floyd-Steinberg dithering
- `ffmpeg-color-sierra2` - FFmpeg with Sierra2 dithering
- `ffmpeg-color-bayer0` - FFmpeg with Bayer 0 ordered dithering
- `ffmpeg-color-bayer1` - FFmpeg with Bayer 1 ordered dithering
- `ffmpeg-color-bayer2` - FFmpeg with Bayer 2 ordered dithering
- `ffmpeg-color-bayer3` - FFmpeg with Bayer 3 ordered dithering
- `ffmpeg-color-bayer4` - FFmpeg with Bayer 4 ordered dithering
- `gifsicle-color` - Gifsicle color quantization
- `imagemagick-color` - ImageMagick standard color quantization
- `imagemagick-color-floyd` - ImageMagick with Floyd-Steinberg dithering
- `imagemagick-color-riemersma` - ImageMagick with Riemersma dithering
- `none-color` - No color reduction (pass-through)

### Lossy Compression Tools

- `animately-standard` - Animately standard lossy compression
- `animately-hard` - Animately hard lossy compression
- `animately-advanced` - Animately advanced lossy compression
- `ffmpeg-lossy` - FFmpeg lossy compression
- `gifski-lossy` - Gifski lossy compression
- `gifsicle-lossy` - Gifsicle standard lossy compression
- `gifsicle-lossy-o1` - Gifsicle lossy compression level 1
- `gifsicle-lossy-o2` - Gifsicle lossy compression level 2
- `gifsicle-lossy-o3` - Gifsicle lossy compression level 3
- `gifsicle-lossy-colors` - Gifsicle color-based lossy compression
- `imagemagick-lossy` - ImageMagick lossy compression
- `none-lossy` - No lossy compression (lossless)

---

## Error Messages and Troubleshooting

### Common Error Messages

#### `Error: Unknown preset: 'preset-name'`
**Cause**: Invalid preset name provided.
**Solution**: Use `--list-presets` to see available presets.

```bash
poetry run python -m giflab run --list-presets
```

#### `Error: Invalid slot name: 'slot-name'`
**Cause**: Invalid slot name in `--variable-slot` or `--lock-slot`.
**Solution**: Use only `frame`, `color`, or `lossy`.

#### `Error: At least one slot must be variable`
**Cause**: All slots are locked, no comparison possible.
**Solution**: Make at least one slot variable.

```bash
# Wrong - all slots locked
--lock-slot frame=animately-frame --lock-slot color=ffmpeg-color --lock-slot lossy=none-lossy

# Right - at least one variable
--variable-slot frame=* --lock-slot color=ffmpeg-color --lock-slot lossy=none-lossy
```

#### `Error: Cannot mix --preset with custom slot options`
**Cause**: Using both `--preset` and custom slot configuration.
**Solution**: Use either presets OR custom configuration, not both.

```bash
# Wrong - mixing preset with custom slots
--preset frame-focus --variable-slot color=*

# Right - use preset only
--preset frame-focus

# Right - use custom slots only  
--variable-slot frame=* --lock-slot color=ffmpeg-color --lock-slot lossy=none-lossy
```

#### `Error: Implementation 'tool-name' not found for slot 'slot-name'`
**Cause**: Invalid tool name in `--lock-slot`.
**Solution**: Check available tool names in the reference above.

#### `Error: Invalid parameter format: 'param-spec'`
**Cause**: Incorrect format in `--slot-params`.
**Solution**: Use `param:value` or `param:[val1,val2,val3]` format.

```bash
# Wrong formats
--slot-params color=colors_32        # Missing colon
--slot-params frame=ratios:1.0,0.8   # List without brackets

# Right formats
--slot-params color=colors:32
--slot-params frame=ratios:[1.0,0.8]
```

### Validation Commands

#### Check Tool Availability
```bash
# List available tools (programmatically)
poetry run python -c "
from giflab.capability_registry import tools_for
print('Frame tools:', [t.NAME for t in tools_for('frame_reduction')])
print('Color tools:', [t.NAME for t in tools_for('color_reduction')])  
print('Lossy tools:', [t.NAME for t in tools_for('lossy_compression')])
"
```

#### Validate Configuration
```bash
# Test configuration with quick-test preset first
poetry run python -m giflab run --preset quick-test --output-dir validation_test
```

#### Debug Mode
```bash
# Enable verbose logging for debugging
poetry run python -m giflab run --preset frame-focus --output-dir debug_test -v
```

---

## Performance Guidelines

### Pipeline Count Optimization

**Small Studies (< 10 pipelines)**:
```bash
--preset quick-test                    # 2 pipelines
--preset frame-focus                   # ~5 pipelines
--preset dithering-focus              # ~6 pipelines
```

**Medium Studies (10-50 pipelines)**:
```bash
--preset color-optimization           # ~17 pipelines
--preset lossy-quality-sweep         # ~11 pipelines
--preset png-optimization            # ~4 pipelines
```

**Large Studies (50+ pipelines)**:
```bash
--preset tool-comparison-baseline    # ~64 pipelines

# Custom large studies
--variable-slot frame=* --variable-slot color=* --lock-slot lossy=none-lossy  # ~85 pipelines
```

### Memory Usage Optimization

**Low Memory**:
- Use single-dimension presets (`frame-focus`, `color-optimization`)
- Enable caching: `--use-cache`
- Use targeted GIFs: `--use-targeted-gifs`

**High Memory Available**:
- Use multi-dimension presets (`tool-comparison-baseline`)
- Custom multi-variable configurations
- Disable caching if storage is limited

### Speed Optimization

**Fast Execution**:
- Use `--preset quick-test` for validation
- Enable GPU: `--use-gpu`
- Use caching: `--use-cache`
- Use targeted GIFs: `--use-targeted-gifs`

**Quality Priority**:
- Use specialized presets (`png-optimization`, `dithering-focus`)
- Lower quality threshold: `--quality-threshold 0.02`
- Full GIF set testing (no `--use-targeted-gifs`)

---

## Integration with Other Tools

### Scripting Examples

#### Bash Script Automation
```bash
#!/bin/bash

# Run multiple preset studies
PRESETS=("frame-focus" "color-optimization" "lossy-quality-sweep")
BASE_DIR="results_$(date +%Y%m%d)"

for preset in "${PRESETS[@]}"; do
    echo "Running $preset experiment..."
    poetry run python -m giflab run \
        --preset "$preset" \
        --output-dir "$BASE_DIR/$preset" \
        --use-cache \
        --quality-threshold 0.05
done
```

#### Python Integration
```python
import subprocess
import sys
from pathlib import Path

def run_preset_experiment(preset_id, output_dir, quality_threshold=0.05):
    """Run a preset experiment via CLI."""
    cmd = [
        sys.executable, "-m", "giflab", "experiment",
        "--preset", preset_id,
        "--output-dir", str(output_dir),
        "--quality-threshold", str(quality_threshold),
        "--use-cache"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0, result.stdout, result.stderr

# Run experiments
presets = ["frame-focus", "color-optimization", "quick-test"]
for preset in presets:
    success, stdout, stderr = run_preset_experiment(
        preset, 
        Path(f"results/{preset}"),
        0.1
    )
    print(f"{preset}: {'✓' if success else '✗'}")
```

### CI/CD Integration

#### GitHub Actions Example
```yaml
name: GIF Compression Analysis
on: [push, pull_request]

jobs:
  compression-test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Setup Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        pip install poetry
        poetry install
    
    - name: Run quick validation
      run: |
        poetry run python -m giflab run --preset quick-test --output-dir ci_test
    
    - name: Archive results
      uses: actions/upload-artifact@v2
      with:
        name: compression-results
        path: ci_test/
```

This CLI reference provides comprehensive documentation for all available options and usage patterns in the targeted experiment presets system.