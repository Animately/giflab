# Testing Strategy - GIF Test Files

## File Size Guidelines

### ✅ Acceptable for Git Repository
- **Small fixtures** (<20KB total): Essential test files committed to `tests/fixtures/`
- **Generated fixtures**: Tests that create GIFs programmatically using PIL/Pillow
- **Lightweight samples**: Files that demonstrate specific edge cases

### ❌ Not for Git Repository  
- **Large test files** (>20KB each): Use `.gitignore` to exclude
- **Bulk test data**: Generate programmatically or download on-demand
- **Debug outputs**: Always exclude from commits

## Current Test Directories

| Directory | Status | Purpose | Size Limit |
|-----------|--------|---------|------------|
| `tests/smoke/` | ✅ Committed | Imports, types, pure logic | N/A |
| `tests/functional/` | ✅ Committed | Mocked engines, synthetic GIFs | N/A |
| `tests/integration/` | ✅ Committed | Real engines, real metrics | N/A |
| `tests/nightly/` | ✅ Committed | Memory, perf, stress, golden | N/A |
| `tests/fixtures/` | ✅ Committed | Essential minimal fixtures | ~24KB total |
| `test-workspace/` | ❌ Ignored | Manual testing & debugging | Variable (excluded) |

## Best Practices

### 1. Generate Test Data Programmatically

```python
# Good: Create test GIFs in-memory
def test_gif_processing(tmp_path):
    # Create test GIF programmatically
    frames = []
    for i in range(3):
        img = Image.new('RGB', (20, 20), (i * 80, i * 80, i * 80))
        frames.append(img)
    
    gif_path = tmp_path / "test.gif"
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=100,
        loop=0
    )
    
    # Run your test logic here
    result = process_gif(gif_path)
    assert result.success
```

### 2. Essential Fixtures Only

Only commit files to `tests/fixtures/` that are:
- **Unique edge cases** that can't be generated easily
- **Cross-platform consistency** requirements  
- **Regression test cases** for specific bugs
- **Under 20KB each** and rare additions

### 3. Temporary Test Files

```python
# Use tmp_path for temporary files
def test_compression(tmp_path):
    input_gif = tmp_path / "input.gif"
    output_gif = tmp_path / "output.gif"
    
    create_test_gif(input_gif)  # Helper function
    compress_gif(input_gif, output_gif)
    
    # Files automatically cleaned up
    assert output_gif.exists()
```

## File Categories

### Development/Debug Files → `.gitignore`
- Pipeline failure analysis outputs
- Performance testing samples  
- Manual testing collections
- Debug visualizations

### Test Infrastructure → Commit
- Minimal reproducible cases
- Cross-engine consistency fixtures
- Edge case demonstrations

## Migration Checklist

When you have large test files:

1. **Evaluate necessity**: Can this be generated programmatically?
2. **Check size**: Is it under 20KB and truly essential?  
3. **Consider alternatives**: URL download, fixture generation, mocking
4. **Document purpose**: Why does this specific file need to be committed?
5. **Add to gitignore**: If not essential, exclude from repository

## Examples from Codebase

- ✅ `tests/fixtures/simple_4frame.gif` (747B) - Basic functionality test
- ✅ `tests/fixtures/single_frame.gif` (306B) - Edge case testing
- ✅ `tests/fixtures/many_colors.gif` (13KB) - Palette stress test
- ❌ `test_fixes/noise_large.gif` (2.6MB) - Should be generated or downloaded
- ❌ `test_fixes/gradient_xlarge.gif` (448KB) - Can be generated programmatically

## Tools for Test Data Generation

```python
# PIL/Pillow for GIF creation
from PIL import Image

# Create gradients
def create_gradient_gif(path, size=(100, 100), frames=10):
    images = []
    for i in range(frames):
        # Generate gradient frame
        img = Image.new('RGB', size)
        # ... gradient logic
        images.append(img)
    
    images[0].save(path, save_all=True, append_images=images[1:])

# Create noise patterns  
def create_noise_gif(path, size=(100, 100), frames=5):
    images = []
    for i in range(frames):
        # Generate noise frame
        img = Image.new('RGB', size)
        # ... noise logic
        images.append(img)
    
    images[0].save(path, save_all=True, append_images=images[1:])
```

## Elimination Testing

The elimination framework systematically tests pipeline combinations on synthetic GIFs and eliminates underperforming ones:

```bash
# Run intelligent sampling elimination
giflab eliminate-pipelines --sampling-strategy representative

# Quick development test
giflab eliminate-pipelines --sampling-strategy quick

# Resume interrupted runs
giflab eliminate-pipelines --resume
```

**📖 Sampling Strategies**: The elimination pipeline supports 6 different sampling strategies (full, representative, factorial, progressive, targeted, quick) that control coverage vs speed trade-offs. See the comprehensive [Sampling Strategies Guide](sampling-strategies.md) for detailed descriptions and usage recommendations.

**📊 Results Tracking**: As of the latest update, elimination results are now saved in timestamped directories with comprehensive historical tracking. See [Elimination Results Tracking](elimination-results-tracking.md) for details on working with historical data.

Key elimination features:
- **Smart caching**: SQLite-based result caching for 2-5x speed improvements
- **Timestamped results**: Preserve all historical runs
- **Master CSV tracking**: Trends across multiple runs  
- **Latest symlink**: Easy access to most recent results
- **GPU acceleration**: CUDA support for quality metrics
- **Comprehensive metrics**: SSIM, MS-SSIM, PSNR, and 8+ additional quality measures
- **Intelligent sampling**: Reduce testing time with strategic pipeline selection
- **Cache management**: `--no-cache` and `--clear-cache` flags for control

This approach keeps the repository clean while maintaining comprehensive test coverage. 