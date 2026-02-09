# 🎞️ GifLab Setup Guide

**Complete installation guide for all platforms**

## 📋 Prerequisites Checklist

Before starting, ensure you have:
- Python 3.11+ 
- Internet connection
- At least 2GB free disk space
- Administrative privileges for package installations

## 🚀 Cross-Platform Installation

### Step 1: Install Python 3.11+

#### macOS
```bash
# Using Homebrew (recommended)
brew install python@3.11

# Verify installation
python3.11 --version
```

#### Linux/Ubuntu
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3.11 python3.11-pip python3.11-venv

# Verify installation
python3.11 --version
```

#### Windows
```powershell
# Using winget (recommended)
winget install Python.Python.3.11

# Alternative: Microsoft Store or python.org
# Verify installation
python --version
```

### Step 2: Install Poetry

All platforms:
```bash
# Official installer (recommended)
curl -sSL https://install.python-poetry.org | python3 -

# Alternative: pip install
pip install poetry

# Verify installation
poetry --version
```

### Step 3: Install Compression Engines

GifLab uses a **dual-pipeline architecture** to balance stability and innovation:

#### 🏭 **Production Pipeline** (`run` command)
- **Engines**: gifsicle + Animately (proven, reliable)
- **Purpose**: Large-scale processing, production workflows
- **Philosophy**: Use battle-tested engines for consistent results

#### 🧪 **Compression Pipeline** (`run` command)
- **Engines**: All 5 engines (ImageMagick, FFmpeg, gifski, gifsicle, Animately)
- **Purpose**: Testing, comparison, finding optimal engines for your content
- **Philosophy**: Experiment with all available engines to identify the best performers, then promote winners to production

**Installation Strategy**: Install core engines first, then add experimental engines as needed.

#### macOS Installation
```bash
# Required: Production engines
brew install gifsicle
# Animately: Pre-installed in repository (bin/darwin/arm64/animately)

# Optional: Additional engines for experimental pipeline
brew install ffmpeg imagemagick gifski

# Verify core installations
gifsicle --version
./bin/darwin/arm64/animately --version

# Verify experimental engines (optional)
ffmpeg -version      # if installed
magick --version     # if installed  
gifski --version     # if installed
```

#### Linux/Ubuntu Installation
```bash
# Required: Production engines
sudo apt update
sudo apt install gifsicle
# Animately: Download and place in bin/linux/x86_64/
# See bin/linux/x86_64/PLACEHOLDER.md for download instructions

# Optional: Additional engines for experimental pipeline
sudo apt install ffmpeg imagemagick-6.q16

# Optional: gifski via cargo (Rust)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env
cargo install gifski

# Verify core installations
gifsicle --version
./bin/linux/x86_64/animately --version  # if downloaded

# Verify experimental engines (optional)
ffmpeg -version      # if installed
magick --version     # if installed
gifski --version     # if installed
```

#### Windows Installation
```powershell
# Install Chocolatey first (if needed)
Set-ExecutionPolicy Bypass -Scope Process -Force
[System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))

# Required: Production engines
choco install gifsicle
# Animately: Download and place in bin/windows/x86_64/
# See bin/windows/x86_64/PLACEHOLDER.md for download instructions

# Optional: Additional engines for experimental pipeline
choco install ffmpeg imagemagick
winget install gifski  # or via cargo

# Verify core installations
gifsicle --version
bin\windows\x86_64\animately.exe --version  # if downloaded

# Verify experimental engines (optional)
ffmpeg -version      # if installed
magick --version     # if installed
gifski --version     # if installed
```

### Step 4: Install GifLab Dependencies

1. Clone or navigate to the GifLab directory:
```bash
git clone <repository-url>
cd giflab
```

2. Install Python dependencies:
```bash
poetry install
```

This installs all required packages:
- **Image Processing**: Pillow, OpenCV, scikit-image
- **Data Analysis**: NumPy, Pandas, matplotlib, seaborn  
- **Machine Learning**: PyTorch, transformers, CLIP
- **Quality Metrics**: scipy, scikit-learn
- **Testing**: pytest, pytest-cov

### Step 5: Environment Variables (Optional)

Configure custom engine paths if needed:

#### macOS/Linux
```bash
# Add to ~/.bashrc, ~/.zshrc, or ~/.profile
export GIFLAB_GIFSICLE_PATH=/usr/local/bin/gifsicle
export GIFLAB_ANIMATELY_PATH=/custom/path/animately
export GIFLAB_IMAGEMAGICK_PATH=/usr/local/bin/magick
export GIFLAB_FFMPEG_PATH=/usr/local/bin/ffmpeg
export GIFLAB_FFPROBE_PATH=/usr/local/bin/ffprobe
export GIFLAB_GIFSKI_PATH=/usr/local/bin/gifski
```

#### Windows
```powershell
# Add to PowerShell profile or set permanently
$env:GIFLAB_GIFSICLE_PATH = "C:\Program Files\gifsicle\gifsicle.exe"
$env:GIFLAB_ANIMATELY_PATH = "C:\custom\path\animately.exe"
$env:GIFLAB_IMAGEMAGICK_PATH = "C:\Program Files\ImageMagick\magick.exe"
$env:GIFLAB_FFMPEG_PATH = "C:\Program Files\ffmpeg\bin\ffmpeg.exe"
$env:GIFLAB_FFPROBE_PATH = "C:\Program Files\ffmpeg\bin\ffprobe.exe"
$env:GIFLAB_GIFSKI_PATH = "C:\Program Files\gifski\gifski.exe"
```

### Step 6: Verify Complete Installation

Run the comprehensive verification:

```bash
# Test engine detection and availability
poetry run python -c "
from giflab.system_tools import get_available_tools
tools = get_available_tools()
for name, info in tools.items():
    status = '✅' if info.available else '❌'
    version = f' v{info.version}' if info.version else ''
    print(f'{status} {name}: {info.name}{version}')
"

# Run smoke tests to verify functionality
poetry run pytest tests/smoke/ -v

# Test with sample processing
mkdir -p data/raw
# Add a sample GIF to data/raw/
poetry run python -m giflab run --matrix
```

Expected output:
```
✅ imagemagick: magick v7.1.2-0
✅ ffmpeg: ffmpeg v7.1.1
✅ ffprobe: ffprobe v7.1.1
✅ gifski: gifski v[OPTIONS]
✅ gifsicle: gifsicle v1.95
✅ animately: animately v1.1.20.0
```

## 🔧 Troubleshooting

### Engine Not Found
```bash
# Check which engines are detected
poetry run python -c "from giflab.system_tools import get_available_tools; print(get_available_tools())"

# Verify PATH contains engine directories
echo $PATH  # macOS/Linux
echo $env:PATH  # Windows

# Test individual engine
ffmpeg -version
gifsicle --version
magick --version
gifski --version
```

### Repository Binary Issues (Animately)
```bash
# Check if repository binary exists
ls -la bin/darwin/arm64/animately    # macOS
ls -la bin/linux/x86_64/animately    # Linux  
dir bin\windows\x86_64\animately.exe # Windows

# Test repository binary directly
./bin/darwin/arm64/animately --version  # macOS
```

### Python/Poetry Issues
```bash
# Verify Python version
python --version  # Should be 3.11+

# Reinstall Poetry if needed
curl -sSL https://install.python-poetry.org | python3 -

# Clear Poetry cache
poetry cache clear --all pypi

# Reinstall dependencies
poetry install --no-cache
```

### Permission Issues
```bash
# macOS/Linux: Fix file permissions
chmod +x bin/darwin/arm64/animately    # macOS
chmod +x bin/linux/x86_64/animately    # Linux

# Windows: Run PowerShell as Administrator
# Right-click PowerShell → "Run as Administrator"
```

## 🎯 Engine-Specific Notes

### ImageMagick
- **Version**: Requires ImageMagick 7.x for `magick` command
- **Legacy**: ImageMagick 6.x uses `convert` command (configure via environment variable)
- **Formats**: Supports widest range of image formats

### FFmpeg  
- **Quality**: Highest quality video-based processing
- **Dependencies**: Includes ffprobe for metadata extraction
- **Performance**: Excellent for frame rate manipulation

### gifski
- **Specialization**: Lossy compression only (highest quality)
- **Installation**: Via cargo (Rust) or platform packages
- **Performance**: Slower but produces smallest high-quality files

### gifsicle
- **Compatibility**: Most widely supported engine
- **Speed**: Fastest processing
- **Optimization**: Built-in GIF optimization algorithms

### Animately
- **Distribution**: Repository binaries (not publicly available)
- **Platforms**: macOS ARM64 (included), Windows/Linux (download required)
- **Strengths**: Complex gradients, photo-realistic content

## 🎉 Quick Start After Setup

1. **Add sample GIFs**:
```bash
mkdir -p data/raw
cp /path/to/your/gifs/*.gif data/raw/
```

2. **Experimental testing** (All 5 engines):
```bash
# Test all engines with small dataset
poetry run python -m giflab run --matrix

# Analyze results
poetry run jupyter notebook notebooks/01_explore_dataset.ipynb
```

3. **Production processing** (gifsicle + Animately):
```bash
# Large-scale processing with proven engines
poetry run python -m giflab run data/raw

# Production with custom settings
poetry run python -m giflab run data/raw --workers 8 --resume
```

## 📚 Next Steps

- **Beginner Guide**: [docs/guides/beginner.md](beginner.md)
- **Engine Documentation**: [docs/engines/](../engines/)
- **CI Setup**: [docs/technical/ci-setup.md](../technical/ci-setup.md)
- **Project Overview**: [README.md](../../README.md)

## 💡 Pro Tips

- **Dual-pipeline workflow**: Use `experiment` to find best engines, then `run` for production
- **Repository binaries**: Animately auto-detected from `bin/` directory  
- **Minimal setup**: Only gifsicle + Animately needed for production pipeline
- **Engine comparison**: `experiment --matrix` tests all 5 engines automatically
- **Performance**: Production pipeline optimized for large-scale processing
- **Innovation**: Experimental pipeline tests newer engines (ImageMagick, FFmpeg, gifski) 