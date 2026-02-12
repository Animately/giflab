# Quick Start Guide: Targeted Presets

Get started with targeted presets in under 10 minutes. This guide shows you the essential commands and concepts to run efficient GIF compression analysis.

## What Are Targeted Presets?

Targeted presets create only the specific pipeline combinations you need for focused research studies.

**Example**: To compare frame reduction algorithms, use the `frame-focus` preset which generates 5 targeted pipelines (one per algorithm).

## 5-Minute Quick Start

### Step 1: See What's Available
```bash
# List all built-in presets
poetry run python -m giflab run --list-presets
```

You'll see presets like:
- `frame-focus` - Compare frame reduction algorithms (~5 pipelines)
- `color-optimization` - Compare color reduction methods (~17 pipelines)  
- `quick-test` - Fast testing preset (~2 pipelines)

### Step 2: Run Your First Preset
```bash
# Test the system with a quick preset
poetry run python -m giflab run --preset quick-test
```

This runs a minimal experiment to verify everything works.

### Step 3: Run a Real Study
```bash
# Compare all frame reduction algorithms  
poetry run python -m giflab run --preset frame-focus --output-dir frame_study
```

### Step 4: Check Your Results
```bash
# Look at the generated results
ls frame_study/
```

**That's it!** You've run a targeted experiment that compares frame algorithms with 99.5% efficiency vs traditional approaches.

## Core Concepts (2 minutes)

### Slots
GIF compression has 3 main stages called "slots":
- **Frame**: Reduce animation frames (`animately-frame`, `ffmpeg-frame`, etc.)
- **Color**: Reduce colors/dithering (`ffmpeg-color`, `gifsicle-color`, etc.)
- **Lossy**: Lossy compression (`animately-advanced`, `gifski-lossy`, etc.)

### Variable vs Locked
- **Variable slot**: Test multiple algorithms (what you're comparing)
- **Locked slot**: Use one specific algorithm (control variables)

### Example: Frame Focus Study
- **Variable**: Frame slot (test all frame algorithms)
- **Locked**: Color slot (use `ffmpeg-color` with 32 colors)
- **Locked**: Lossy slot (use `animately-advanced` at level 40)

**Result**: Compare frame algorithms fairly while controlling other variables.

## Essential Presets

### For Beginners

#### `quick-test` (2 pipelines)
Fast validation - perfect for testing.
```bash
poetry run python -m giflab run --preset quick-test
```

#### `frame-focus` (5 pipelines)  
Compare frame reduction algorithms.
```bash
poetry run python -m giflab run --preset frame-focus
```

#### `color-optimization` (17 pipelines)
Compare color reduction and dithering methods.
```bash  
poetry run python -m giflab run --preset color-optimization
```

### Research Questions → Presets

**"Which frame algorithm works best?"** → `frame-focus`
**"Which color method works best?"** → `color-optimization`  
**"What lossy level should I use?"** → `lossy-quality-sweep`
**"How do tools compare overall?"** → `tool-comparison-baseline`

## Custom Configurations

### Basic Custom Preset
Instead of built-in presets, create your own configuration:

```bash
# Test specific frame algorithms with locked color/lossy
poetry run python -m giflab run \
  --variable-slot frame=animately-frame,ffmpeg-frame \
  --lock-slot color=ffmpeg-color \
  --lock-slot lossy=none-lossy \
  --slot-params color=colors:32
```

**This means**: Test 2 frame algorithms, use ffmpeg-color with 32 colors, no lossy compression.

### Template for Custom Configurations
```bash
poetry run python -m giflab run \
  --variable-slot SLOT=ALGORITHMS \    # What to test
  --lock-slot SLOT=ALGORITHM \         # What to control  
  --slot-params SLOT=PARAM:VALUE       # Parameter values
```

**Slot names**: `frame`, `color`, `lossy`
**Algorithm wildcard**: `*` means "all available"

## Common Patterns

### Compare All Algorithms in One Dimension
```bash
# All frame algorithms
--variable-slot frame=* --lock-slot color=ffmpeg-color --lock-slot lossy=none-lossy

# All color algorithms  
--variable-slot color=* --lock-slot frame=animately-frame --lock-slot lossy=none-lossy

# All lossy algorithms
--variable-slot lossy=* --lock-slot frame=none-frame --lock-slot color=ffmpeg-color
```

### Compare Specific Tools Only
```bash
# Just animately vs ffmpeg for frames
--variable-slot frame=animately-frame,ffmpeg-frame --lock-slot color=ffmpeg-color --lock-slot lossy=none-lossy

# Just ffmpeg color variants
--variable-slot color=ffmpeg-color,ffmpeg-color-floyd --lock-slot frame=animately-frame --lock-slot lossy=none-lossy
```

### Multi-Dimensional Comparisons
```bash
# Vary both frame and color algorithms
--variable-slot frame=* --variable-slot color=* --lock-slot lossy=none-lossy

# Test 2×2×2 = 8 combinations  
--variable-slot frame=animately-frame,ffmpeg-frame \
--variable-slot color=ffmpeg-color,gifsicle-color \
--variable-slot lossy=none-lossy,gifsicle-lossy
```

## Useful Options

### Performance Options
```bash
--use-cache          # Speed up repeated experiments
--use-gpu            # Enable GPU acceleration (if available)
--use-targeted-gifs  # Use smaller test GIF set for faster testing
```

### Quality Options  
```bash
--quality-threshold 0.05    # Default quality requirement
--quality-threshold 0.1     # More permissive (faster)
--quality-threshold 0.02    # Stricter (slower)
```

### Output Options
```bash
--output-dir my_results     # Custom output directory
```

### Complete Example
```bash
poetry run python -m giflab run \
  --preset frame-focus \
  --output-dir frame_comparison \
  --quality-threshold 0.1 \
  --use-cache \
  --use-gpu
```

## Troubleshooting

### Command Not Working?
```bash
# Test basic functionality
poetry run python -m giflab run --preset quick-test

# Check available presets
poetry run python -m giflab run --list-presets
```

### Common Mistakes
```bash
# Wrong: mixing preset with custom slots
--preset frame-focus --variable-slot color=*

# Right: use preset only
--preset frame-focus

# Right: use custom slots only
--variable-slot frame=* --lock-slot color=ffmpeg-color --lock-slot lossy=none-lossy
```

### Error: "At least one slot must be variable"
```bash  
# Wrong: all slots locked (nothing to compare)
--lock-slot frame=animately-frame --lock-slot color=ffmpeg-color --lock-slot lossy=none-lossy

# Right: at least one variable slot
--variable-slot frame=* --lock-slot color=ffmpeg-color --lock-slot lossy=none-lossy
```

## Next Steps

### Learn More
1. **User Guide**: Complete documentation with advanced examples
2. **Preset Reference**: Details on all available presets and their use cases
3. **CLI Reference**: All command-line options and parameters
4. **Troubleshooting**: Solutions to common issues

### Typical Workflow
1. **Start Simple**: Use `--preset quick-test` to verify setup
2. **Pick Research Question**: Choose appropriate preset or create custom config
3. **Run Experiment**: Execute with performance options as needed
4. **Analyze Results**: Review output in specified directory
5. **Iterate**: Refine configuration based on results

### Advanced Usage
- Create custom presets programmatically
- Integrate with CI/CD pipelines
- Combine with other GifLab tools
- Use for regression testing

## Cheat Sheet

### Most Common Commands
```bash
# List presets
poetry run python -m giflab run --list-presets

# Quick test
poetry run python -m giflab run --preset quick-test

# Frame comparison
poetry run python -m giflab run --preset frame-focus --output-dir frames

# Color comparison  
poetry run python -m giflab run --preset color-optimization --output-dir colors

# Custom frame test
poetry run python -m giflab run \
  --variable-slot frame=* \
  --lock-slot color=ffmpeg-color \
  --lock-slot lossy=none-lossy
```

### Performance Commands
```bash
# Fast testing
poetry run python -m giflab run --preset quick-test --use-cache --use-targeted-gifs

# Quality testing
poetry run python -m giflab run --preset frame-focus --quality-threshold 0.02 --use-gpu

# Large study
poetry run python -m giflab run --preset tool-comparison-baseline --use-cache
```

**You're ready to start!** Begin with `--preset quick-test`, then try `--preset frame-focus` for your first real experiment.