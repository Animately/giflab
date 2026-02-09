# 🔧 Adding New GIF Compression Tools

This guide explains how to add new GIF compression tools to GifLab's compression framework.

---

## 📋 Prerequisites

Before adding a new tool, ensure you have:
- Tool installed and accessible from command line
- Basic understanding of the tool's parameters
- Sample GIFs for testing
- Performance expectations

---

## 🏗️ Implementation Steps

### Step 1: Create Tool Wrapper

Create a new wrapper class in `src/giflab/tools/` directory:

```python
# src/giflab/tools/your_tool_wrapper.py
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional
from ..base_tool import BaseTool

class YourToolWrapper(BaseTool):
    """Wrapper for YourTool GIF compression."""
    
    def __init__(self):
        super().__init__()
        self.name = "your_tool"
        self.version = self._get_version()
    
    def _get_version(self) -> str:
        """Get tool version."""
        try:
            result = subprocess.run(
                ["your_tool", "--version"],
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except Exception:
            return "unknown"
    
    def compress(
        self,
        input_path: Path,
        output_path: Path,
        quality: float = 0.8,
        **kwargs
    ) -> Dict[str, Any]:
        """Compress GIF using YourTool."""
        
        # Build command
        cmd = [
            "your_tool",
            str(input_path),
            str(output_path),
            f"--quality={quality}",
        ]
        
        # Add additional parameters
        if kwargs.get("optimize", True):
            cmd.append("--optimize")
        
        # Execute compression
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        end_time = time.time()
        
        if result.returncode != 0:
            raise RuntimeError(f"YourTool failed: {result.stderr}")
        
        return {
            "success": True,
            "processing_time": end_time - start_time,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "parameters": kwargs
        }
    
    def get_supported_parameters(self) -> Dict[str, Any]:
        """Get supported parameters for this tool."""
        return {
            "quality": {
                "type": "float",
                "range": [0.0, 1.0],
                "default": 0.8,
                "description": "Quality level"
            },
            "optimize": {
                "type": "bool",
                "default": True,
                "description": "Enable optimization"
            }
        }
```

### Step 2: Create Base Tool Interface

Ensure your tool inherits from `BaseTool`:

```python
# src/giflab/base_tool.py
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any

class BaseTool(ABC):
    """Base class for all GIF compression tools."""
    
    def __init__(self):
        self.name = "base_tool"
        self.version = "unknown"
    
    @abstractmethod
    def compress(
        self,
        input_path: Path,
        output_path: Path,
        **kwargs
    ) -> Dict[str, Any]:
        """Compress GIF file."""
        pass
    
    @abstractmethod
    def get_supported_parameters(self) -> Dict[str, Any]:
        """Get supported parameters."""
        pass
    
    def validate_input(self, input_path: Path) -> bool:
        """Validate input file."""
        return (
            input_path.exists() and 
            input_path.suffix.lower() == '.gif'
        )
```

### Step 3: Add Tool to Registry

Register your tool in the tool registry:

```python
# src/giflab/tools/__init__.py
from .gifsicle_wrapper import GifsicleWrapper
from .animately_wrapper import AnimatelyWrapper
from .your_tool_wrapper import YourToolWrapper

AVAILABLE_TOOLS = {
    "gifsicle": GifsicleWrapper,
    "animately": AnimatelyWrapper,
    "your_tool": YourToolWrapper,
}

def get_tool(name: str):
    """Get tool by name."""
    if name not in AVAILABLE_TOOLS:
        raise ValueError(f"Unknown tool: {name}")
    return AVAILABLE_TOOLS[name]()
```

### Step 4: Create Experimental Strategy

Add a new strategy to the experimental framework:

```python
# src/giflab/experimental_strategies.py
from .strategies import ExperimentalStrategy

class YourToolStrategy(ExperimentalStrategy):
    """Strategy using YourTool."""
    
    def __init__(self):
        super().__init__()
        self.name = "your_tool"
        self.description = "Pure YourTool compression"
    
    def execute(self, input_path: Path, output_path: Path) -> Dict[str, Any]:
        """Execute YourTool compression."""
        tool = get_tool("your_tool")
        
        return tool.compress(
            input_path=input_path,
            output_path=output_path,
            quality=0.8,
            optimize=True
        )
    
    def get_parameter_space(self) -> Dict[str, Any]:
        """Get parameter space for optimization."""
        return {
            "quality": [0.6, 0.7, 0.8, 0.9],
            "optimize": [True, False]
        }

# Add to strategy registry
EXPERIMENTAL_STRATEGIES = {
    "pure_gifsicle": PureGifsicleStrategy(),
    "pure_animately": PureAnimatelyStrategy(),
    "animately_then_gifsicle": HybridStrategy(),
    "your_tool": YourToolStrategy(),  # Add here
}
```

### Step 5: Add Configuration

Update configuration to include your tool:

```python
# src/giflab/config.py
DEFAULT_TOOLS = {
    "gifsicle": {
        "enabled": True,
        "path": "gifsicle",
        "timeout": 300
    },
    "animately": {
        "enabled": True,
        "path": "animately",
        "timeout": 300
    },
    "your_tool": {
        "enabled": True,
        "path": "your_tool",
        "timeout": 300
    }
}
```

### Step 6: Add Tests

Create comprehensive tests:

```python
# tests/functional/test_your_tool.py
import pytest
from pathlib import Path
from src.giflab.tools.your_tool_wrapper import YourToolWrapper

class TestYourTool:
    def test_tool_initialization(self):
        tool = YourToolWrapper()
        assert tool.name == "your_tool"
        assert tool.version is not None
    
    def test_compression_basic(self, sample_gif):
        tool = YourToolWrapper()
        output_path = Path("test_output.gif")
        
        result = tool.compress(
            input_path=sample_gif,
            output_path=output_path,
            quality=0.8
        )
        
        assert result["success"] is True
        assert output_path.exists()
        assert output_path.stat().st_size > 0
    
    def test_parameter_validation(self):
        tool = YourToolWrapper()
        params = tool.get_supported_parameters()
        
        assert "quality" in params
        assert params["quality"]["type"] == "float"
        assert params["quality"]["range"] == [0.0, 1.0]
    
    def test_error_handling(self):
        tool = YourToolWrapper()
        
        with pytest.raises(RuntimeError):
            tool.compress(
                input_path=Path("nonexistent.gif"),
                output_path=Path("output.gif")
            )
```

### Step 7: Update Documentation

Add tool documentation:

```python
# docs/tools/your_tool.md
# YourTool Integration

## Overview
YourTool is a [brief description] that excels at [specific use cases].

## Installation
```bash
# Installation instructions
```

## Parameters
- **quality**: Quality level (0.0-1.0)
- **optimize**: Enable optimization (boolean)

## Usage Examples
```python
from giflab.tools import YourToolWrapper

tool = YourToolWrapper()
result = tool.compress(
    input_path=Path("input.gif"),
    output_path=Path("output.gif"),
    quality=0.8
)
```

## Performance Characteristics
- **Compression ratio**: [expected range]
- **Processing speed**: [relative to other tools]
- **Quality preservation**: [quality metrics]
- **Best for**: [content types]
```

---

## 🧪 Testing New Tools

### Manual Testing
```bash
# Test individual tool
poetry run python -m giflab test-tool your_tool --input test.gif

# Test in experimental framework
poetry run python -m giflab run --strategies your_tool
```

### Automated Testing
```bash
# Run tool-specific tests (functional layer)
poetry run pytest tests/functional/test_your_tool.py

# Run integration tests
poetry run pytest tests/integration/test_experimental_integration.py
```

### Performance Benchmarking
```bash
# Benchmark against existing tools
poetry run python -m giflab benchmark --tools gifsicle,animately,your_tool
```

---

## 📊 Evaluation Metrics

When adding a new tool, evaluate:

### Technical Metrics
- **Compression ratio**: Output size / Input size
- **Processing time**: Seconds per megabyte
- **Memory usage**: Peak memory consumption
- **Quality metrics**: SSIM, PSNR, perceptual quality

### Content-Specific Performance
- **Simple graphics**: Text, logos, simple animations
- **Complex images**: Photos, gradients, many colors
- **Animation types**: Smooth motion, rapid changes
- **Size categories**: Small, medium, large files

### Integration Metrics
- **Reliability**: Success rate across diverse content
- **Error handling**: Graceful failure modes
- **Parameter sensitivity**: Robustness to different settings
- **Compatibility**: Works with existing pipeline

---

## 🔄 Continuous Integration

### Automated Tests
```yaml
# .github/workflows/test-tools.yml
name: Test New Tools
on: [push, pull_request]

jobs:
  test-tools:
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
      - name: Install YourTool
        run: |
          # Tool installation commands
      - name: Run tests
        run: |
          poetry run pytest tests/functional/test_your_tool.py
```

### Performance Monitoring
```python
# scripts/monitor_tool_performance.py
import time
from pathlib import Path
from src.giflab.tools import get_tool

def benchmark_tool(tool_name: str, test_gifs: List[Path]):
    """Benchmark tool performance."""
    tool = get_tool(tool_name)
    results = []
    
    for gif_path in test_gifs:
        start_time = time.time()
        result = tool.compress(
            input_path=gif_path,
            output_path=Path(f"output_{tool_name}.gif")
        )
        end_time = time.time()
        
        results.append({
            "tool": tool_name,
            "input_size": gif_path.stat().st_size,
            "output_size": Path(f"output_{tool_name}.gif").stat().st_size,
            "processing_time": end_time - start_time,
            "success": result["success"]
        })
    
    return results
```

---

## 📈 Best Practices

### Code Quality
- **Follow PEP 8**: Use consistent coding style
- **Add type hints**: Improve code readability and IDE support
- **Write docstrings**: Document all public methods
- **Handle errors**: Graceful error handling and logging

### Performance
- **Optimize for common cases**: Fast path for typical usage
- **Stream processing**: Handle large files efficiently
- **Memory management**: Clean up temporary files
- **Parallel processing**: Support concurrent operations

### Testing
- **Unit tests**: Test individual components
- **Integration tests**: Test tool integration
- **Performance tests**: Benchmark against baselines
- **Edge case testing**: Handle unusual inputs

### Documentation
- **Clear examples**: Show typical usage patterns
- **Parameter documentation**: Explain all options
- **Performance characteristics**: Document expected behavior
- **Troubleshooting**: Common issues and solutions

---

## 🚀 Next Steps

After adding a new tool:

1. **Run comprehensive tests** across diverse GIF content
2. **Collect performance metrics** and compare to existing tools
3. **Update ML training data** with new tool results
4. **Consider hybrid strategies** combining your tool with others
5. **Document lessons learned** for future tool additions

---

## 💡 Tool Suggestions

### High Priority Tools to Add
1. **ImageMagick** - Universal image processing
2. **FFmpeg** - Video/animation specialist
3. **gifski** - High-quality Rust implementation
4. **Pillow** - Python ecosystem integration

### Medium Priority
1. **Sharp** - Node.js high-performance tool
2. **libgif** - Low-level C library
3. **WebP converter** - Modern format alternative
4. **AVIF converter** - Next-generation compression

### Research Interest
1. **Machine learning models** - Neural compression
2. **Custom algorithms** - Specialized optimizations
3. **Format converters** - Alternative output formats
4. **Hardware accelerated** - GPU-based compression

---

*Remember: The goal is to build a comprehensive dataset of tool performance across diverse content types to train our ML models for intelligent tool selection.* 