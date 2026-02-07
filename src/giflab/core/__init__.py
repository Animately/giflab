"""Core pipeline testing framework - modular components.

This module provides the core pipeline testing and analysis components:
- GifLabRunner: Main pipeline runner for systematic testing and optimization
- AnalysisResult: Result dataclass for pipeline analysis
- ParetoAnalyzer: Pareto frontier analysis for pipeline efficiency
- PipelineSampler: Intelligent sampling strategies
- SAMPLING_STRATEGIES: Available sampling strategy configurations

The GifLabRunner provides comprehensive pipeline testing with:
- Multi-engine compression testing
- Quality metrics analysis
- Intelligent sampling strategies
- Result caching and resume functionality

NOTE: elimination_cache has been replaced by giflab.storage (SQLite-based).
"""

# Import modular components
from ..elimination_errors import ErrorTypes

# Import synthetic GIF components for backward compatibility
from ..synthetic_gifs import SyntheticGifSpec
from .pareto import ParetoAnalyzer
from .runner import AnalysisResult, GifLabRunner
from .sampling import SAMPLING_STRATEGIES, PipelineSampler, SamplingStrategy

# Re-export key components
__all__ = [
    # Core components
    "GifLabRunner",
    "AnalysisResult",
    "ParetoAnalyzer",
    "PipelineSampler",
    "SAMPLING_STRATEGIES",
    "SamplingStrategy",
    # Error handling
    "ErrorTypes",
    # Synthetic GIF components
    "SyntheticGifSpec",
]
