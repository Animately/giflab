# Multi-Engine Architecture Design

**Technical documentation of GifLab's multi-engine compression architecture, testing strategies, and operational patterns.**

> **⚠️ PARTIALLY OUT OF DATE.** The engine/capability-registry architecture described here is current, but the **usage examples** below (`--sampling`, `--preset`) reference the CLI removed in `648db9a` — those commands no longer exist. See [Compression-Pipeline Leaderboard](./compression-pipeline-leaderboard.md) for the rebuild.

---

## Overview

GifLab implements a sophisticated multi-engine compression system supporting 5 external engines (gifsicle, Animately, ImageMagick, FFmpeg, gifski) through a flexible architecture that enables comprehensive testing and efficient workflows.

---

## Multi-Engine Architecture

**Architecture Decision:** GifLab provides a **single comprehensive pipeline** that supports flexible engine selection through sampling strategies and presets:

### 🔧 Flexible Pipeline (`run` command)
- **Engines**: All 5 engines available (gifsicle, Animately, ImageMagick, FFmpeg, gifski)
- **Sampling Strategies**: Control which engines and combinations to test
  - `--sampling representative`: Balanced testing across engines
  - `--sampling quick`: Fast testing with proven engines
  - `--sampling full`: Comprehensive testing of all combinations
- **Targeted Presets**: Focused testing for specific research questions
  - `--preset frame-focus`: Compare frame reduction algorithms
  - `--preset color-optimization`: Compare color quantization methods
  - `--preset quick-test`: Minimal testing for development

### Workflow Integration
1. Use targeted presets (`run --preset frame-focus`) for focused research questions
2. Use representative sampling (`run --sampling representative`) for general optimization
3. Analyze results to understand which engines perform best for your content
4. Use selected pipelines (`run --pipelines winners.yaml`) for production processing

This architecture provides comprehensive testing capabilities while enabling efficient workflows through intelligent sampling and targeted presets.

---

## Testing Strategy for External Engines

### Test Organization

1. **Smoke tests (`tests/test_engine_smoke.py`)**  
   – Fast validation that engines can be invoked and produce basic output

2. **Per-engine integration tests**  
   – For each engine/action: generate toy GIF → run wrapper → assert functional change + metadata

3. **Fail-fast on CI**  
   – Mark integration tests `@pytest.mark.external_tools` and run them in a dedicated workflow job where the binaries are pre-installed

### Functional Validation Criteria

**Color reduction tests:**
- Input: 256-color GIF → `colors=32` → Output: ≤32 colors (measured via palette extraction)
- Metadata: `engine` field matches expected value, `render_ms > 0`, `kilobytes > 0`
- Edge case: Single-color GIF should remain stable

**Frame reduction tests:**  
- Input: 20-frame GIF → `keep_ratio=0.5` → Output: ~10 frames (±1 frame tolerance)
- Timing validation: Output duration should be ~50% of input duration
- Edge case: Single-frame GIF with `keep_ratio < 1.0` should handle gracefully

**Lossy compression tests:**
- Input/output file size: Output should be smaller than input (compression achieved)
- Quality degradation: PSNR should decrease but remain above minimum threshold (e.g., >20dB)
- Metadata completeness: All required fields present and non-zero

**Cross-engine consistency:**
- Same operation on same input should produce similar file sizes across engines (±20% tolerance)
- All engines should populate identical metadata schema
- Error handling should be consistent (same exception types)

### Test Fixtures

**Minimal fixtures (≤1KB each):**
- `simple_4frame.gif`: 4 frames, 16 colors, 64x64px - basic functionality
- `single_frame.gif`: 1 frame, 8 colors, 32x32px - edge case testing
- `many_colors.gif`: 4 frames, 256 colors, 64x64px - palette stress test

**Fixture validation:**
- All fixtures verified to be well-formed GIFs
- Known properties documented (frame count, color count, dimensions)
- Stored under `tests/fixtures/` with descriptive names

### Performance Regression Detection

**Timing thresholds (smoke tests only):**
- Operations should complete within 5x the baseline gifsicle time
- No operation should exceed 30 seconds on test fixtures
- Memory usage should remain reasonable (no >100MB allocations for small inputs)

---

## Error Handling & Risk Mitigation

### Failure Modes & Responses

**Missing executables:**
- All wrappers call `discover_tool().require()` early, failing fast with clear setup instructions
- CI/local development should install all engines or skip tests with `@pytest.mark.external_tools`
- Animately binaries included in repository `bin/` directory for supported platforms

**Command execution failures:**
- `subprocess.run(..., check=True)` raises on non-zero exit codes
- Error messages include full stderr output for debugging
- 60-second timeout prevents infinite hangs on corrupted inputs

**Integration test stability:**
- Use small, well-formed GIF fixtures to minimize flakiness
- Validate functional changes (palette size, frame count) rather than exact byte-for-byte output
- Mark tests with `@pytest.mark.external_tools` for conditional execution

### Rollback Strategy

If any stage introduces regressions:
1. **Immediate:** Revert the problematic wrapper's `apply()` method to stub behavior
2. **Short-term:** Fix the underlying issue in `external_engines.*` helpers
3. **Long-term:** Add regression tests to prevent recurrence

### Testing Isolation

- Each engine's tests should be independent (no shared state)
- Use temporary directories for all intermediate files
- Clean up PNG frames and palette files automatically via `tempfile.TemporaryDirectory()`

---

## Tool Discovery Strategy

GifLab uses a hierarchical tool discovery system with graceful fallbacks:

### Discovery Priority Order
1. **Environment variable** (e.g., `$GIFLAB_ANIMATELY_PATH`)
2. **Repository binary** (`bin/<platform>/<arch>/tool`)
3. **System PATH**
4. **Graceful failure** with clear setup instructions

### Supported Environment Variables
- `GIFLAB_IMAGEMAGICK_PATH` - Custom ImageMagick path
- `GIFLAB_FFMPEG_PATH` - Custom FFmpeg path  
- `GIFLAB_FFPROBE_PATH` - Custom FFprobe path
- `GIFLAB_GIFSKI_PATH` - Custom gifski path
- `GIFLAB_GIFSICLE_PATH` - Custom gifsicle path
- `GIFLAB_ANIMATELY_PATH` - Custom Animately path

### Platform Support
- **macOS**: Homebrew + repository Animately binary
- **Linux**: apt/package managers + download required for Animately
- **Windows**: Package managers + download required for Animately

---

## Performance Characteristics

### Engine Specialization

| Engine | Strengths | Weaknesses | Best Use Cases |
|--------|-----------|------------|----------------|
| **gifsicle** | Fast, reliable, universal | Limited quality options | Production baseline, simple graphics |
| **Animately** | Complex gradients, photo-realistic | Repository-only distribution | High-quality output, gradients |
| **ImageMagick** | Universal image processing | Memory intensive | Color reduction, format conversion |
| **FFmpeg** | Video/animation expertise | Complex parameter tuning | Frame reduction, temporal processing |
| **gifski** | Highest quality output | Slow, disk-intensive | Premium quality, small datasets |

### Typical Processing Times
- **Color reduction**: ~1-5 seconds for typical GIFs
- **Frame reduction**: ~2-10 seconds (depends on frame count)
- **Lossy compression**: ~3-15 seconds (gifski slower due to PNG workflow)

### Memory Usage Patterns
- **ImageMagick**: Loads entire GIF into memory during `-coalesce`
- **FFmpeg**: Streams frames, lower memory footprint
- **gifski**: Moderate memory usage, but high temporary disk usage

---

## Integration Guidelines

### Adding New Engines

1. **Create helper module** in `src/giflab/external_engines/`
2. **Implement wrapper classes** in `src/giflab/tool_wrappers.py`
3. **Add to capability registry** for automatic discovery
4. **Create integration tests** with `@pytest.mark.external_tools`
5. **Update CI workflows** to install the new engine
6. **Document CLI recipes** and parameter mappings

### Wrapper Implementation Requirements

All engine wrappers must:
- Inherit from appropriate interface (`ColorReductionTool`, `FrameReductionTool`, `LossyCompressionTool`)
- Implement `available()` and `version()` class methods
- Return standardized metadata dict with `render_ms`, `engine`, `command`, `kilobytes`
- Handle errors gracefully with descriptive error messages
- Support timeout configuration

---

This architecture has been battle-tested through the complete multi-engine rollout and provides a robust foundation for GIF compression at scale. 