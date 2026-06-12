# FFmpeg – Engine Reference

> Official docs: <https://ffmpeg.org>  
> Formats guide: <https://ffmpeg.org/ffmpeg-formats.html>

FFmpeg can read & write GIFs and offers fine-grained control via filters.

### Operations used in GifLab

| Action | 2-pass palette workflow | Notes |
|--------|------------------------|-------|
| **Color reduction** | 2-pass palette method:<br>1) `ffmpeg -i in.gif -vf "fps={fps},palettegen" palette.png`<br>2) `ffmpeg -i in.gif -i palette.png -lavfi "fps={fps},paletteuse" out.gif` | If a preceding **frame** tool has already decided a lower FPS it can be supplied via the helper to avoid re-sampling twice. |
| **Frame reduction** | `ffmpeg -i in.gif -vf "fps={fps}" out.gif` | Keeps the full color palette. |
| **Lossy compression** | 2-pass palette method:<br>1) `ffmpeg -i in.gif -filter_complex "palettegen=max_colors={colors}" palette.png`<br>2) `ffmpeg -i in.gif -i palette.png -filter_complex "paletteuse=dither=sierra2_4a" out.gif` | `colors` is mapped from our `lossy_level` geometrically (256→16; level 0 → 256, 25 → 128, 50 → 64, 75 → 32, 100 → 16). FFmpeg has no error-bounded GIF lossy mode, so palette size + dithering **is** its engine-native lossy axis. (The previous `-q:v {qscale}` recipe was a video-DCT knob — inert for palette-based GIF output; see the 2026-06-09 calibration finding.) |

> **Combinations**: The pipeline builder may chain these wrappers — e.g. a `FFmpegFrameReducer` followed by `FFmpegColorReducer` and then `FFmpegLossyCompressor`.  Because each wrapper operates on the GIF produced by the previous step, you do **not** need a dedicated “all-in-one” command; the helpers automatically accept optional hints (such as target FPS) so we don’t duplicate expensive work, but conceptually each wrapper still owns exactly one variable.

### Wrapper strategy
1. Probe the GIF to discover original FPS (using `ffprobe`).  
2. Compute target FPS and/or color count.  
3. Run the commands above; collect runtime & stderr.  
4. Return metadata.
5. The helper automatically deletes the temporary `palette.png` created during color reduction.

### Environment variables
| Variable | Description |
|----------|-------------|
| `GIFLAB_FFMPEG_PATH` | Path to `ffmpeg` binary |
| `GIFLAB_FFPROBE_PATH` | Path to `ffprobe` (if different) |

For exhaustive options see the [formats documentation](https://ffmpeg.org/ffmpeg-formats.html). 