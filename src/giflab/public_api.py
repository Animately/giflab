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
    "composite_quality",
)
"""Metric strings recognised by :func:`measure`."""

MetricIdentifier = Literal[
    "ssim",
    "ms_ssim",
    "psnr",
    "lpips",
    "gmsd",
    "fsim",
    "chist",
    "composite_quality",
]

# Public metric identifier → internal result-dict key. Six metrics flow
# through metrics._aggregate_metric() and end up keyed by their bare name;
# LPIPS is the exception — its computation surfaces lpips_quality_{mean,p95,
# max} and we expose the mean as the public scalar. composite_quality is the
# calibrated 11-metric weighted aggregate (see calculate_composite_quality);
# it is already keyed by its bare name on the result dict — no _mean sibling,
# so no denormalisation. Module-level so we don't rebuild it on every
# measure() call.
_PUBLIC_TO_INTERNAL_METRIC_KEY: dict[str, str] = {
    "ssim": "ssim",
    "ms_ssim": "ms_ssim",
    "psnr": "psnr",
    "lpips": "lpips_quality_mean",
    "gmsd": "gmsd",
    "fsim": "fsim",
    "chist": "chist",
    "composite_quality": "composite_quality",
}

# Public metrics that require the temporal_artifacts pipeline (which loads
# LPIPS internally for lpips_t_* computation). Empty in v0.3.0 — none of the
# seven SUPPORTED_METRICS need it. When a future version exposes a temporal
# metric (e.g. "temporal_consistency"), list it here so measure() flips
# ENABLE_TEMPORAL_ARTIFACTS on automatically.
_TEMPORAL_NEEDING_METRICS: frozenset[str] = frozenset()


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
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        # Store params as an immutable shallow copy so the caller can mutate
        # their own dict after the call without affecting this result.
        object.__setattr__(self, "params", dict(self.params))
        # Normalise warnings to an immutable tuple regardless of how it was
        # passed (list, tuple, generator).
        object.__setattr__(self, "warnings", tuple(self.warnings))


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
    composite_quality: float | None = None


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def compress(
    input_path: Path,
    output_path: Path,
    engine: EngineIdentifier,
    params: dict[str, Any] | None = None,
    *,
    apply_content_ceiling: bool = True,
) -> CompressResult:
    """Run a single compression engine on ``input_path``, writing ``output_path``.

    See ``docs/public-api.md`` for the full contract. Overwrites ``output_path``
    if a file already exists there. Does not mutate ``input_path``.

    Recognised ``params`` keys:

    - ``lossy_level`` *(required for lossy engines)* — engine-native lossy
      compression level.
    - ``timeout_s`` *(optional)* — per-call wall-clock timeout in seconds for
      the engine subprocess. Honoured by the animately and gifsicle engines.
      Precedence: ``params["timeout_s"]`` > ``GIFLAB_RUN_TIMEOUT`` env var >
      default of 10s. Surfaced for batch / audit workflows where the default
      cuts off legitimate ~10 MB+ inputs (see audit
      ``docs/metrics-audit/2026-05-22/report.md``). Must be a positive integer.

    Content-aware lossy ceiling (``apply_content_ceiling``, default ``True``):
        When the engine is ``animately`` and ``params`` carries a positive
        ``lossy_level``, the input's original frames are classified
        (data-viz / photographic / film-grain) and the requested level is
        clamped DOWN to a per-class ceiling (never raised) — flat categorical
        charts band at any lossy level and near-256-colour content posterises
        above modest levels (2026-05-26 outlier deep-dive). When a clamp
        happens, a human-readable warning is added to
        :attr:`CompressResult.warnings`. The ceilings are animately-calibrated
        only; other lossy engines skip classification entirely. Pass
        ``apply_content_ceiling=False`` to bypass the ceiling — the audit
        monotonicity / corpus sweeps MUST set this so their lossy grid is never
        silently clamped (see ``scripts/audit/_common.py``).

    Raises:
        UnknownEngineError: ``engine`` is not in :data:`SUPPORTED_ENGINES`.
        EngineUnavailableError: the engine binary is not available on PATH.
        FileNotFoundError: ``input_path`` does not exist.
        ValueError: ``params["timeout_s"]`` is not a positive integer.
    """
    if engine not in SUPPORTED_ENGINES:
        raise UnknownEngineError(engine)

    dispatch = _get_engine_dispatch()
    wrapper_cls = dispatch[engine]

    if not wrapper_cls.available():
        raise EngineUnavailableError(engine)

    if not input_path.exists():
        raise FileNotFoundError(f"input_path does not exist: {input_path}")

    # Copy the caller's params so the content-ceiling clamp never mutates the
    # caller's dict (params-no-leak contract). A shallow copy is enough — only
    # the scalar ``lossy_level`` is rewritten.
    effective_params: dict[str, Any] = dict(params) if params is not None else {}
    result_warnings: list[str] = []

    # Content-aware lossy ceiling: animately only — a data-backed scope, not a
    # placeholder. animately's re-quantising lossy cliffs on photographic /
    # gradient / data-viz content (the posterisation failure mode the ceiling
    # prevents); gifsicle's error-bounded lossy degrades gradually with no cliff
    # (2026-06-05 calibration, scripts/audit/engine_lossy_calibration.py), so it
    # needs no ceiling. Skip for lossless / colour-only calls (no positive
    # lossy_level), and when the caller opts out (audit sweeps).
    if (
        apply_content_ceiling
        and engine == "animately"
        and isinstance(effective_params.get("lossy_level"), int)
        and effective_params["lossy_level"] > 0
    ):
        clamp_warning = _maybe_clamp_lossy_level(input_path, effective_params)
        if clamp_warning is not None:
            result_warnings.append(clamp_warning)

    # Count input frames before dispatch so we can flag a frame drop without
    # needing the metrics pipeline (which compress() does not run). Cheap, and
    # avoids duplicating the alignment-accuracy threshold owned by the Wave-5
    # alignment-warning task.
    input_frame_count = _safe_frame_count(input_path)

    wrapper = wrapper_cls()
    wrapper_result = wrapper.apply(input_path, output_path, params=effective_params)

    output_frame_count = _safe_frame_count(output_path)
    if (
        input_frame_count is not None
        and output_frame_count is not None
        and output_frame_count < input_frame_count
    ):
        result_warnings.append(
            f"frame drop: output has {output_frame_count} frames vs "
            f"{input_frame_count} in the input"
        )

    return CompressResult(
        output_path=output_path,
        output_bytes=output_path.stat().st_size,
        render_ms=int(wrapper_result.get("render_ms", 0)),
        engine=engine,
        engine_version=wrapper_cls.version(),
        params=effective_params,
        warnings=tuple(result_warnings),
    )


def _maybe_clamp_lossy_level(
    input_path: Path, effective_params: dict[str, Any]
) -> str | None:
    """Clamp ``effective_params['lossy_level']`` DOWN to the content ceiling.

    Mutates ``effective_params`` in place (it is already a private copy). Returns
    a warning string when a clamp happened, else ``None``. Classification fails
    soft inside ``classify_content_from_path``, so this never raises.
    """
    # Lazy import keeps the no-torch lightweight-import contract: importing
    # content_classifier here (not at module top) means the metrics import graph
    # is only touched when a real lossy animately compress runs.
    from giflab.content_classifier import classify_content_from_path

    requested = int(effective_params["lossy_level"])
    classification = classify_content_from_path(input_path)
    ceiling = classification.lossy_max
    if ceiling is None or requested <= ceiling:
        return None

    effective_params["lossy_level"] = ceiling
    return f"lossy_level clamped {requested} → {ceiling}: {classification.reason}"


def _safe_frame_count(gif_path: Path) -> int | None:
    """Return the GIF's frame count, or ``None`` if it cannot be determined.

    Used only for the frame-drop warning; a failure here must never block a
    legitimate compress, so it fails soft.
    """
    try:
        from PIL import Image, ImageSequence

        with Image.open(gif_path) as img:
            return sum(1 for _ in ImageSequence.Iterator(img))
    except Exception:  # noqa: BLE001 — frame-drop warning is best-effort
        return None


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
    # only what the caller requested. The temporal_artifacts pipeline also
    # loads LPIPS (for lpips_t_*), so gate it too — no v0.4.0 public metric
    # needs it (see _TEMPORAL_NEEDING_METRICS).
    #
    # composite_quality determinism: composite_quality is the calibrated
    # weighted aggregate, and `lpips_quality_mean` is one of its contributors
    # (ENHANCED_LPIPS_WEIGHT=0.04). _resolve_composite_from_contributions
    # redistributes a contributor's weight across the rest when that
    # contributor resolves to NaN, so the composite is REQUEST-SET-DEPENDENT
    # unless we pin the LPIPS contributor on: measure(["composite_quality"])
    # (LPIPS gated off → lpips_quality_mean is NaN → 4% redistributed) would
    # return a DIFFERENT value than measure(["composite_quality", "lpips"])
    # (LPIPS computed → contributes) for the same file pair. Verified
    # empirically: 0.9007 vs 0.9045 on degraded content. For the metric
    # gifprep designates as its single deterministic verdict number, that is
    # unacceptable — so when composite_quality is requested we force the LPIPS
    # gate ON regardless of whether "lpips" is also requested. Then composite
    # is always computed over the full LPIPS-included dimension set and is
    # deterministic for a given (file pair, giflab version, environment).
    # (SSIMULACRA2's 3% weight is binary-gated, not request-gated, so it
    # remains an environmental caveat — see docs/public-api.md — that
    # measure() cannot force deterministic without fabricating a value.)
    config = MetricsConfig()
    config.ENABLE_DEEP_PERCEPTUAL = (
        "lpips" in requested or "composite_quality" in requested
    )
    config.ENABLE_TEMPORAL_ARTIFACTS = bool(_TEMPORAL_NEEDING_METRICS & requested)

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
        composite_quality=_project("composite_quality"),
    )
