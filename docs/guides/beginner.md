# 🎞️ GifLab Beginner's Guide

**A step-by-step guide to generating prediction datasets from GIF compression sweeps**

## 👋 Welcome to GifLab!

GifLab builds structured datasets that map visual features to compression outcomes. You feed it GIFs; it runs them through every combination of engine, lossy level, frame ratio, and palette size; measures quality after each run; and stores the results in SQLite. The end goal is training ML models that predict compression curves from visual features alone, without having to compress anything at inference time.

This guide walks you through the full workflow, from setup to your first trained model.

**🔧 Looking for technical details?** Check out the [**README.md**](../../README.md) for developer information, performance benchmarks, and configuration options.

---

### 🎯 What GifLab Does:
- **Extracts 25 visual features** from each GIF (color complexity, motion intensity, texture, edge density, and more)
- **Runs compression sweeps across 7 engines** (gifsicle, three Animately modes, ImageMagick, FFmpeg, gifski) with varying lossy, frame, and palette parameters
- **Stores structured training data in SQLite** with per-run quality metrics (SSIM, MS-SSIM, PSNR, FSIM, GMSD, temporal consistency, and more)
- **Trains gradient boosting models** that predict compression curves from visual features, so you can estimate outcomes without compressing

### ⏱️ Time Needed:
- **Setup**: 5-10 minutes
- **Analysis**: 10-30 minutes (depending on your collection size)
- **Processing**: Varies (1-100+ hours depending on collection size)

---

## 📋 Step-by-Step Workflow

### Phase 1: Getting Started (5-10 minutes)

#### Step 1: Set Up Your Environment
```bash
# Install GifLab
git clone <your-repo-url>
cd giflab
poetry install

# Install compression engines (dual-pipeline architecture)
# Required: Production engines (gifsicle + Animately)
# On macOS:
brew install python@3.11 gifsicle
# Animately included in repository (bin/darwin/arm64/animately)

# Optional: Additional engines for experimental pipeline
brew install ffmpeg imagemagick gifski

# On Linux/Ubuntu:
sudo apt install python3.11 gifsicle
# Animately: Download from releases (see bin/linux/x86_64/PLACEHOLDER.md)
# Optional: sudo apt install ffmpeg imagemagick-6.q16; cargo install gifski

# On Windows:
choco install python gifsicle
# Animately: Download from releases (see bin/windows/x86_64/PLACEHOLDER.md)
# Optional: choco install ffmpeg imagemagick; winget install gifski
```

📖 **For detailed setup instructions, see:** [Setup Guide](setup.md)

#### Step 2: Organize Your GIFs

GifLab can automatically detect GIF sources based on directory structure, making it easy to organize and analyze GIFs from different platforms:

```bash
# Create the directory structure with automatic source detection
poetry run python -m giflab organize-directories data/raw/

# This creates organized directories:
# data/raw/tenor/          - GIFs from Tenor
# data/raw/animately/      - GIFs from Animately platform  
# data/raw/tgif_dataset/   - GIFs from TGIF research dataset
# data/raw/unknown/        - Ungrouped GIFs
```

**Two approaches for organizing:**

**Option A: Directory-based source detection (Recommended)**
```bash
# Move GIFs to appropriate directories for automatic detection
data/raw/
├── tenor/
│   ├── love/              # "love" search results
│   └── marketing/         # "marketing" search results
├── animately/             # All user uploads (flat structure)
│   ├── user_upload_1.gif
│   ├── user_upload_2.gif
│   └── user_upload_3.gif
└── tgif_dataset/          # All research data (flat structure)
    ├── research_gif_1.gif
    ├── research_gif_2.gif
    └── research_gif_3.gif
```

**Option B: Simple organization**
```bash
# Or just put all GIFs in data/raw/ for simple analysis
cp /path/to/your/gifs/*.gif data/raw/
```

📖 **For detailed organization guide, see:** [Directory-Based Source Detection Guide](directory-source-detection.md)

### Phase 2: Analysis & Planning (10-30 minutes)

#### Step 3: Explore Your Dataset
```bash
# Start Jupyter
jupyter notebook

# Open: notebooks/01_explore_dataset.ipynb
# Follow the step-by-step instructions in the notebook
```

**What you'll learn:**
- How many GIFs you have and their characteristics
- File size distributions and complexity patterns
- Optimal compression parameters for your collection
- Processing time estimates

#### Step 4: Generate Optimization Data
```bash
# Open: notebooks/02_build_seed_json.ipynb
# Follow the step-by-step instructions in the notebook
```

**What you'll get:**
- `seed/lookup_seed_metadata.json` - Metadata index
- `seed/lookup_seed_processing.json` - Optimization recommendations
- `seed/lookup_seed_resume.json` - Resume functionality data

### Phase 3: Processing (1-100+ hours)

#### Step 5: Start Compression Processing
```bash
# Basic processing (uses default settings)
poetry run python -m giflab run data/raw data/ --workers 4

# Advanced processing (uses optimization data from notebooks)
poetry run python -m giflab run data/raw data/ --workers 8 --resume --use-seed-data
```

#### Step 6: Monitor Progress
```bash
# Check logs
tail -f logs/giflab_*.log

# Check results
ls data/csv/
ls data/renders/
```

#### Step 7: Add AI Tags (Optional)
```bash
# Add AI-generated tags to your results
poetry run python -m giflab tag data/csv/results_YYYYMMDD.csv data/raw --workers 2
```

---

## 🔧 Detailed Instructions

### 📊 Using the Analysis Notebook (01_explore_dataset.ipynb)

#### What Each Section Does:

**Section 1: Setup**
- Loads all the tools needed for analysis
- Checks that everything is working
- *Just run it - don't worry about the details*

**Section 2: Explore Your Collection**
- Finds all GIF files in your `data/raw/` directory
- Checks which ones are readable vs. corrupted
- Creates a summary table of your GIFs

**Section 3: Analyze Individual Properties**
- Shows distributions of file sizes, dimensions, frame counts
- Creates charts to visualize your data
- Identifies outliers and patterns

**Section 4: Find Relationships**
- Shows how different properties relate to each other
- Creates correlation charts
- Identifies which factors affect file size most

**Section 5: Group Similar GIFs**
- Automatically categorizes your GIFs by complexity
- Shows clusters of similar GIFs
- Helps plan processing strategies

**Section 6: Get Recommendations**
- Provides specific compression settings for your GIFs
- Estimates processing times
- Suggests optimization strategies

**Section 7: Summary**
- Gives you actionable next steps
- Saves analysis results for the next notebook

#### 💡 Tips for the Analysis Notebook:

**If you have no GIFs yet:**
- The notebook will create sample data so you can see how it works
- Add some GIF files to `data/raw/` and re-run to see real analysis

**If you have many GIFs (>1000):**
- The notebook will sample 1000 GIFs for analysis by default
- You can change this in the configuration section
- This keeps analysis fast while still being representative

**If some GIFs are corrupted:**
- The notebook will identify and skip them
- You'll get a report of which files have problems
- Corrupted files won't affect the analysis of good files

### 🌱 Using the Seed Generation Notebook (02_build_seed_json.ipynb)

#### What Each Section Does:

**Section 1: Setup**
- Prepares tools for generating optimization data
- Sets up paths and configuration

**Section 2: Data Collection**
- Scans all your GIFs comprehensively
- Extracts detailed metadata
- Identifies duplicates and calculates complexity scores

**Section 3: Metadata Seed**
- Creates `lookup_seed_metadata.json`
- Indexes all your GIFs for fast lookup
- Includes statistics and summaries

**Section 4: Processing Optimization**
- Creates `lookup_seed_processing.json`
- Groups GIFs by complexity for efficient processing
- Generates parameter recommendations for each GIF

**Section 5: Resume State**
- Creates `lookup_seed_resume.json`
- Tracks what's been processed already
- Enables resume functionality for interrupted jobs

**Section 6: Validation**
- Checks that all generated files are correct
- Validates consistency between files
- Reports any issues found

**Section 7: Integration**
- Saves all files to the `seed/` directory
- Creates integration summary
- Provides next steps

#### 💡 Tips for the Seed Generation Notebook:

**First time running:**
- This will process all your GIFs, so it may take a while
- You'll see progress bars showing the work being done
- The generated files will speed up all future processing

**Re-running after adding GIFs:**
- The notebook will update the seed files with new GIFs
- It will backup your existing files before overwriting
- Resume data will be preserved

**If processing gets interrupted:**
- You can safely re-run the notebook
- It will pick up where it left off
- No data will be lost

### 🚀 Using the CLI for Processing

#### Basic Commands:

```bash
# See all available options
poetry run python -m giflab --help

# Basic processing
poetry run python -m giflab run data/raw data/

# Processing with specific settings
poetry run python -m giflab run data/raw data/ \
  --workers 8 \
  --resume \
  --csv data/csv/my_results.csv

# Add AI tags to results
poetry run python -m giflab tag data/csv/results_20240115.csv data/raw
```

#### Understanding the Options:

**`--workers N`**
- How many CPU cores to use for processing
- More workers = faster processing (up to your CPU limit)
- Start with 4-8, adjust based on your system

**`--resume`**
- Skip GIFs that have already been processed
- Essential for large collections
- Uses the seed files to track progress

**`--csv PATH`**
- Where to save the results
- Defaults to auto-generated filename with date
- Each row represents one compressed variant


**`--dry-run`**
- Shows what would be processed without actually doing it
- Great for estimating time and checking setup
- No files are created or modified

#### 💡 Processing Tips:

**For small collections (<100 GIFs):**
```bash
poetry run python -m giflab run data/raw data/ --workers 4
```

**For medium collections (100-1000 GIFs):**
```bash
# Run analysis first, then process
poetry run python -m giflab run data/raw data/ --workers 6 --resume
```

**For large collections (1000+ GIFs):**
```bash
# Definitely run notebooks first for optimization
poetry run python -m giflab run data/raw data/ --workers 8 --resume --use-seed-data
```

**To estimate processing time:**
```bash
poetry run python -m giflab run data/raw data/ --dry-run
```

#### 🎯 Pipeline-Specific Usage:

**🏭 Production Processing (gifsicle + Animately):**
```bash
# Standard production pipeline
poetry run python -m giflab run data/raw

# Production with custom settings
poetry run python -m giflab run data/raw --workers 8 --resume
```

**🧪 Experimental Testing (All 5 Engines):**
```bash
# Test all engines with comprehensive matrix
poetry run python -m giflab run --matrix

# Quick experimental test
poetry run python -m giflab run --matrix --gifs 5

# Experimental with custom sample GIFs
poetry run python -m giflab run --matrix --sample-gifs-dir my_test_gifs/
```

**📊 Workflow Recommendations:**
```bash
# 1. Find best engines for your content
poetry run python -m giflab run --matrix --gifs 10

# 2. Use production pipeline for large-scale processing
poetry run python -m giflab run data/raw --workers 8 --resume
```

---

## 📊 Understanding Your Dataset

Every compression run produces a training example. The input features are the visual characteristics of the original GIF (extracted during the `run` step), and the targets are the compression outcomes: file size, quality scores, and render time. Together these form the dataset used to train prediction models.

### CSV Output Format

Each row in your results CSV represents one training example -- a single compression configuration applied to a single GIF:

| Column | Description | Example |
|--------|-------------|---------|
| `gif_sha` | Unique ID for the original GIF | `6c54c899e2b0baf7...` |
| `orig_filename` | Original file name | `my_animation.gif` |
| `engine` | Compression tool used | `imagemagick`, `ffmpeg`, `gifski`, `gifsicle`, `animately` |
| `lossy` | Compression level (0-120) | `40` |
| `frame_keep_ratio` | Fraction of frames kept | `0.80` |
| `color_keep_count` | Number of colors in palette | `128` |
| `kilobytes` | Size of compressed file | `245.3` |
| `ssim` | Quality score (0-1, higher=better) | `0.936` |
| `efficiency` | Overall efficiency score (0-1, higher=better) | `0.785` |
| `render_ms` | Time to compress (milliseconds) | `1250` |

### Quality Metrics Explained

**SSIM (Structural Similarity Index)**
- Measures how similar the compressed GIF looks to the original
- Range: 0.0 to 1.0 (1.0 = identical, 0.0 = completely different)
- Good values: >0.8 (>0.9 for high quality needs)

**Efficiency Score** 🎯
- **Single balanced metric** combining quality and compression (50/50 weighting)
- Range: 0.0 to 1.0 (higher = better balance of quality and file size reduction)
- **0.8-1.0**: Excellent (production-ready)
- **0.7-0.8**: Very good (great for most uses) 
- **0.6-0.7**: Good (acceptable balance)
- **0.5-0.6**: Fair (room for improvement)
- **<0.5**: Poor (investigate issues)

**Compression Ratio**
- Calculate as: `orig_kilobytes / kilobytes`
- Higher = more compression
- Example: 1000KB → 250KB = 4x compression

### Engine Selection Guide

**Which engine should you use?**

| Engine | Best For | Speed | Quality | When to Use |
|--------|----------|-------|---------|-------------|
| **gifsicle** | Simple graphics, text | ⚡⚡⚡ Fastest | ⭐⭐ Good | Quick processing, broad compatibility |
| **Animately** | Photos, gradients | ⚡⚡ Fast | ⭐⭐⭐ Excellent | Complex images, photo-realistic content |
| **ImageMagick** | General purpose | ⚡⚡ Fast | ⭐⭐ Good | Versatile, when other engines aren't available |
| **FFmpeg** | Video-like content | ⚡ Moderate | ⭐⭐⭐ Excellent | High quality, frame rate changes |
| **gifski** | Maximum quality | ⚡ Slow | ⭐⭐⭐⭐ Best | When quality is paramount, small files needed |

**Quick recommendations:**
- **Production work**: Use `poetry run python -m giflab run` (gifsicle + Animately, proven reliability)
- **Engine comparison**: Use `poetry run python -m giflab run --matrix` (tests all 5 engines)
- **Photo-like GIFs**: Experimental pipeline will test gifski and Animately automatically
- **Best quality**: Experimental pipeline includes gifski (highest quality compression)
- **Large datasets**: Production pipeline optimized for scale and stability

### Exploring Your Dataset

These SQL queries help you inspect the training data stored in SQLite. Each row is one training example linking visual features (inputs) to compression outcomes (targets).

**Top-performing training examples (highest efficiency):**
```sql
SELECT * FROM compression_runs
ORDER BY efficiency DESC
LIMIT 10
```

**Best outcome per GIF (useful for labelling):**
```sql
SELECT * FROM compression_runs
WHERE efficiency = (SELECT MAX(efficiency) FROM compression_runs WHERE gif_sha = cr.gif_sha)
```

**Smallest output per GIF:**
```sql
SELECT * FROM compression_runs
WHERE kilobytes = (SELECT MIN(kilobytes) FROM compression_runs WHERE gif_sha = cr.gif_sha)
```

**High-quality examples (SSIM > 0.9, sorted by size):**
```sql
SELECT * FROM compression_runs
WHERE ssim > 0.9
ORDER BY kilobytes ASC
LIMIT 10
```

**Fastest engine runs:**
```sql
SELECT * FROM compression_runs
WHERE engine = 'gifsicle'
ORDER BY render_ms ASC
LIMIT 10
```

**Per-engine summary statistics:**
```sql
SELECT engine, AVG(ssim) as avg_quality, AVG(kilobytes) as avg_size, AVG(render_ms) as avg_time
FROM compression_runs
GROUP BY engine
ORDER BY avg_quality DESC
```

---

## 🔧 Troubleshooting

### Common Issues and Solutions

#### "No GIF files found"
**Problem:** The notebook can't find your GIF files
**Solution:**
1. Check that GIFs are in `data/raw/` directory
2. Make sure files have `.gif` extension
3. Check file permissions

#### "Import error" when running notebooks
**Problem:** Python can't find the GifLab modules
**Solution:**
1. Make sure you're running from the project root directory
2. Check that `poetry install` completed successfully
3. Try restarting Jupyter

#### "Command not found: gifsicle"
**Problem:** Required tools aren't installed
**Solution:**
```bash
# macOS
brew install gifsicle ffmpeg

# Ubuntu/WSL
sudo apt-get install gifsicle ffmpeg

# Windows
choco install gifsicle ffmpeg
```

#### Processing is very slow
**Problem:** Each GIF takes a long time to process
**Solutions:**
1. Reduce the number of workers: `--workers 2`
2. Use seed data for optimization: run the notebooks first
3. Process smaller batches of GIFs
4. Check available RAM and disk space

#### "Out of memory" errors
**Problem:** System runs out of RAM during processing
**Solutions:**
1. Reduce workers: `--workers 2`
2. Process GIFs in smaller batches
3. Close other applications
4. Consider processing on a machine with more RAM

#### Results CSV is huge
**Problem:** The output file is very large
**This is normal!** Each GIF generates 30+ variants
**Solutions:**
1. Use compression: `gzip results.csv`
2. Process results in chunks with pandas
3. Import into a database for analysis

### Getting Help

**Check the logs:**
```bash
tail -f logs/giflab_*.log
```

**Run in dry-run mode:**
```bash
poetry run python -m giflab run data/raw data/ --dry-run
```

**Test with a few GIFs first:**
```bash
# Create a test directory with just a few GIFs
mkdir data/test
cp data/raw/sample*.gif data/test/
poetry run python -m giflab run data/test data/ --workers 2
```

---

## 🎯 Next Steps After Analysis

### Immediate Actions
1. **Review your analysis results** - Understand your GIF collection characteristics
2. **Run the seed generation** - Create optimization files
3. **Start with a small test** - Process 10-20 GIFs first
4. **Monitor the first batch** - Check logs and results
5. **Scale up gradually** - Increase batch sizes as you gain confidence

### Long-term Workflow
1. **Regular analysis** - Re-run notebooks when adding new GIFs
2. **Iterative optimization** - Adjust settings based on results
3. **Quality monitoring** - Check SSIM scores and file sizes
4. **Performance tuning** - Optimize worker counts and batch sizes

### Advanced Usage
1. **Custom compression parameters** - Modify the configuration files
2. **Integration with other tools** - Use the CSV results in other workflows
3. **Automated processing** - Set up scheduled jobs for new GIFs
4. **Quality analysis** - Deep dive into compression effectiveness

---

## 📚 Additional Resources

### Understanding GIF Compression
- **Lossy compression**: Reduces quality to achieve smaller files
- **Frame reduction**: Removes animation frames to reduce size
- **Color reduction**: Uses fewer colors in the palette
- **Engine differences**: Different tools have different strengths

### Performance Optimization
- **CPU usage**: More workers = faster processing (up to CPU limit)
- **Memory usage**: Large GIFs need more RAM per worker
- **Disk I/O**: Fast storage improves processing speed
- **Batch processing**: Group similar GIFs for efficiency

### Quality Assessment
- **Visual inspection**: Always check a few results manually
- **SSIM thresholds**: Set minimum quality requirements
- **Use case specific**: Different applications have different quality needs
- **A/B testing**: Compare different settings side-by-side

---

## 🔧 Common Issues & Troubleshooting

### Pipeline Tests Running Slowly or Giving Stale Results?
**Clear the cache** to start fresh:
```bash
poetry run python -m giflab eliminate-pipelines --clear-cache --estimate-time
```
This resets all cached test results in `elimination_results/pipeline_results_cache.db`.

### Engine Installation Issues?
Make sure all required engines are properly installed:
```bash
# Check if engines are available
which gifsicle ffmpeg magick gifski
ls bin/darwin/arm64/animately  # Check Animately binary
```

### Getting Strange Error Messages?
1. **Check the logs** in `elimination_results/` for detailed error information
2. **Try clearing the cache** (command above)
3. **Run with fresh results**: Add `--no-cache` flag to bypass cache temporarily

**📚 For more advanced troubleshooting:** See [Elimination Results Tracking Guide](elimination-results-tracking.md)

---

**🎉 You're ready to start! Begin with Step 1 and work through each phase at your own pace.** 