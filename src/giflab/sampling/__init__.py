"""Frame sampling module for efficient GIF validation."""

from .frame_sampler import (
    FrameSampler,
    SamplingResult,
    SamplingStrategy,
)
from .strategies import (
    AdaptiveSampler,
    ProgressiveSampler,
    SceneAwareSampler,
    UniformSampler,
)

__all__ = [
    "FrameSampler",
    "SamplingResult",
    "SamplingStrategy",
    "UniformSampler",
    "AdaptiveSampler",
    "ProgressiveSampler",
    "SceneAwareSampler",
]
