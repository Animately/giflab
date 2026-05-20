"""Public API surface for external giflab consumers — see docs/public-api.md.

This module is the stable entry point for projects that depend on giflab as
a library (e.g., gifprep). Internals are deliberately kept out of the public
contract; consumers import from the top-level ``giflab`` package, never from
this module path directly.

Heavy dependencies (torch, lpips) are imported lazily inside function bodies
to preserve the lightweight-import contract enforced by
``src/giflab/__init__.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from giflab.error_handling import GifLabError

# Eagerly importable wrapper class symbols. These classes themselves are
# lightweight to import (the heavy work happens inside their apply() methods).
from giflab.tool_wrappers import (
    AnimatelyLossyCompressor,
    FFmpegLossyCompressor,
    GifsicleLossyCompressor,
    GifskiLossyCompressor,
    ImageMagickLossyCompressor,
)

# Re-bind calculate_comprehensive_metrics into this module's namespace so
# unit tests can patch it at `giflab.public_api.calculate_comprehensive_metrics`
# without paying its eager-import cost at module load. The function itself is
# already imported lazily inside measure(); this name shadow is the patch hook.
calculate_comprehensive_metrics: Any = None

__all__ = [
    "SUPPORTED_ENGINES",
    "SUPPORTED_METRICS",
    "CompressResult",
    "EngineIdentifier",
    "EngineUnavailableError",
    "MeasureResult",
    "MetricIdentifier",
    "UnknownEngineError",
    "UnknownMetricError",
    "compress",
    "measure",
]


# ---------------------------------------------------------------------------
# Identifier sets
# ---------------------------------------------------------------------------

SUPPORTED_ENGINES: tuple[str, ...] = (
    "animately",
    "gifsicle",
    "gifski",
    "imagemagick",
    "ffmpeg",
)
"""Engine strings recognised by :func:`compress`."""

EngineIdentifier = Literal["animately", "gifsicle", "gifski", "imagemagick", "ffmpeg"]

SUPPORTED_METRICS: tuple[str, ...] = (
    "ssim",
    "ms_ssim",
    "psnr",
    "lpips",
    "gmsd",
    "fsim",
    "chist",
)
"""Metric strings recognised by :func:`measure`."""

MetricIdentifier = Literal["ssim", "ms_ssim", "psnr", "lpips", "gmsd", "fsim", "chist"]

# Public metric identifier → internal result-dict key. Six metrics flow
# through metrics._aggregate_metric() and end up keyed by their bare name;
# LPIPS is the exception — its computation surfaces lpips_quality_{mean,p95,
# max} and we expose the mean as the public scalar. Module-level so we don't
# rebuild it on every measure() call.
_PUBLIC_TO_INTERNAL_METRIC_KEY: dict[str, str] = {
    "ssim": "ssim",
    "ms_ssim": "ms_ssim",
    "psnr": "psnr",
    "lpips": "lpips_quality_mean",
    "gmsd": "gmsd",
    "fsim": "fsim",
    "chist": "chist",
}


# ---------------------------------------------------------------------------
# Engine dispatch
# ---------------------------------------------------------------------------


def _get_engine_dispatch() -> dict[str, Any]:
    """Return the engine→wrapper-class mapping.

    Built fresh on each call so monkey-patching wrapper symbols in tests
    works without stale references. The dict construction is cheap.
    """
    return {
        "animately": AnimatelyLossyCompressor,
        "gifsicle": GifsicleLossyCompressor,
        "gifski": GifskiLossyCompressor,
        "imagemagick": ImageMagickLossyCompressor,
        "ffmpeg": FFmpegLossyCompressor,
    }


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class UnknownEngineError(GifLabError):
    """Raised when an unsupported engine identifier is passed to :func:`compress`."""

    def __init__(self, engine: str) -> None:
        supported = ", ".join(SUPPORTED_ENGINES)
        super().__init__(
            f"Unknown engine {engine!r}. Supported: {supported}",
            context={"engine": engine, "supported": list(SUPPORTED_ENGINES)},
        )


class EngineUnavailableError(GifLabError):
    """Raised when a known engine's binary is not available on PATH."""

    def __init__(self, engine: str) -> None:
        super().__init__(
            f"Engine {engine!r} binary not found on PATH",
            context={"engine": engine},
        )


class UnknownMetricError(GifLabError):
    """Raised when an unsupported metric identifier is passed to :func:`measure`."""

    def __init__(self, metric: str) -> None:
        supported = ", ".join(SUPPORTED_METRICS)
        super().__init__(
            f"Unknown metric {metric!r}. Supported: {supported}",
            context={"metric": metric, "supported": list(SUPPORTED_METRICS)},
        )


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CompressResult:
    """Return value of :func:`compress`.

    See ``docs/public-api.md`` for the contract this object satisfies.
    """

    output_path: Path
    output_bytes: int
    render_ms: int
    engine: str
    engine_version: str
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Store params as an immutable shallow copy so the caller can mutate
        # their own dict after the call without affecting this result.
        object.__setattr__(self, "params", dict(self.params))


@dataclass(frozen=True)
class MeasureResult:
    """Return value of :func:`measure`.

    Each field is populated iff that metric was requested in the call;
    otherwise it is ``None``.
    """

    ssim: float | None = None
    ms_ssim: float | None = None
    psnr: float | None = None
    lpips: float | None = None
    gmsd: float | None = None
    fsim: float | None = None
    chist: float | None = None


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def compress(
    input_path: Path,
    output_path: Path,
    engine: EngineIdentifier,
    params: dict[str, Any] | None = None,
) -> CompressResult:
    """Run a single compression engine on ``input_path``, writing ``output_path``.

    See ``docs/public-api.md`` for the full contract. Overwrites ``output_path``
    if a file already exists there. Does not mutate ``input_path``.

    Raises:
        UnknownEngineError: ``engine`` is not in :data:`SUPPORTED_ENGINES`.
        EngineUnavailableError: the engine binary is not available on PATH.
        FileNotFoundError: ``input_path`` does not exist.
    """
    if engine not in SUPPORTED_ENGINES:
        raise UnknownEngineError(engine)

    dispatch = _get_engine_dispatch()
    wrapper_cls = dispatch[engine]

    if not wrapper_cls.available():
        raise EngineUnavailableError(engine)

    if not input_path.exists():
        raise FileNotFoundError(f"input_path does not exist: {input_path}")

    effective_params = params if params is not None else {}
    wrapper = wrapper_cls()
    wrapper_result = wrapper.apply(input_path, output_path, params=effective_params)

    return CompressResult(
        output_path=output_path,
        output_bytes=output_path.stat().st_size,
        render_ms=int(wrapper_result.get("render_ms", 0)),
        engine=engine,
        engine_version=wrapper_cls.version(),
        params=effective_params,
    )


def measure(
    reference_path: Path,
    candidate_path: Path,
    metrics: list[MetricIdentifier],
) -> MeasureResult:
    """Compute the requested quality metrics between two GIFs.

    Returns a :class:`MeasureResult` with the requested metric fields populated.
    Non-requested fields remain ``None``. See ``docs/public-api.md``.

    Raises:
        ValueError: ``metrics`` is empty.
        UnknownMetricError: an element of ``metrics`` is not in
            :data:`SUPPORTED_METRICS`. Raised before any computation.
        FileNotFoundError: either path does not exist.
        GifLabError: a requested metric's computation failed. The requested
            metric set is on ``error.context["metrics"]``. If a single metric
            was identified as the failure, ``error.context["metric"]`` is set.
    """
    if not metrics:
        raise ValueError("measure() requires at least one metric in `metrics`")

    requested: set[str] = set()
    for m in metrics:
        if m not in SUPPORTED_METRICS:
            raise UnknownMetricError(m)
        requested.add(m)

    if not reference_path.exists():
        raise FileNotFoundError(f"reference_path does not exist: {reference_path}")
    if not candidate_path.exists():
        raise FileNotFoundError(f"candidate_path does not exist: {candidate_path}")

    # Lazy import to keep `from giflab import measure` cheap.
    global calculate_comprehensive_metrics
    if calculate_comprehensive_metrics is None:
        from giflab.metrics import calculate_comprehensive_metrics as _ccm

        calculate_comprehensive_metrics = _ccm

    from giflab.config import MetricsConfig

    # FR-009: the only individually expensive metric is LPIPS (loads a torch
    # model). The other six (ssim, ms_ssim, psnr, gmsd, fsim, chist) are
    # computed in a shared pass over frames. Gate LPIPS specifically; the
    # cheap metrics are always populated by the underlying call and we project
    # only what the caller requested.
    config = MetricsConfig()
    config.ENABLE_DEEP_PERCEPTUAL = "lpips" in requested

    try:
        full = calculate_comprehensive_metrics(
            reference_path,
            candidate_path,
            config=config,
            force_all_metrics=True,
        )
    except Exception as exc:  # noqa: BLE001 — surface with metric context
        raise GifLabError(
            f"measure() failed during metric computation: {exc}",
            cause=exc,
            context={"metrics": sorted(requested)},
        ) from exc

    def _project(name: str) -> float | None:
        if name not in requested:
            return None
        value = full.get(_PUBLIC_TO_INTERNAL_METRIC_KEY[name])
        if value is None:
            # Requested metric resolved to None — internal key drift or a
            # silently failed metric. Surface loudly rather than return a
            # field the caller can't distinguish from "not requested". The
            # internal key name is preserved in context for whoever debugs
            # giflab; we don't leak it into the user-facing message.
            raise GifLabError(
                f"measure() requested metric {name!r} but the internal "
                f"computation did not produce a value. This indicates a "
                f"giflab bug; please file an issue.",
                context={
                    "metric": name,
                    "metrics": sorted(requested),
                    "internal_key": _PUBLIC_TO_INTERNAL_METRIC_KEY[name],
                },
            )
        scalar = float(value)
        # PSNR is normalized to [0, 1] internally (frame_psnr / PSNR_MAX_DB).
        # Denormalize back to dB for the public surface — dB is the industry
        # convention and what consumers expect when reading `result.psnr`.
        if name == "psnr":
            scalar *= float(config.PSNR_MAX_DB)
        return scalar

    return MeasureResult(
        ssim=_project("ssim"),
        ms_ssim=_project("ms_ssim"),
        psnr=_project("psnr"),
        lpips=_project("lpips"),
        gmsd=_project("gmsd"),
        fsim=_project("fsim"),
        chist=_project("chist"),
    )
