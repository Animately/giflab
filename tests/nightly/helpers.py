"""Shared environment helpers for nightly-tier tests.

These helpers calibrate performance / memory assertions for the CI
environment (2-core GitHub runners, shared resources, no ssimulacra2
binary) without weakening them on developer hardware. They consolidate
the previously duplicated ``_get_performance_threshold_multiplier``
copies in test_vectorization_performance.py (CI factor 3.0) and
test_gradient_color_performance.py (CI factor 2.0), which had silently
drifted apart.

This module is stdlib-only (os, math) by design: its unit tests live in
the smoke layer (tests/smoke/test_nightly_helpers.py), which must not
pull in numpy/psutil.
"""

import math
import os

_CI_ENV_VARS = (
    "CI",  # Generic CI flag
    "CONTINUOUS_INTEGRATION",  # Common CI flag
    "GITHUB_ACTIONS",  # GitHub Actions
    "TRAVIS",  # Travis CI
    "JENKINS_URL",  # Jenkins
    "BUILDKITE",  # Buildkite
    "CIRCLECI",  # Circle CI
)

# Slope above which sustained monotonic growth is flagged as a leak.
# This is the value the pre-existing detector asserts against (and what
# its assert message always claimed: MB per *iteration*).
LEAK_RATE_MB_PER_ITERATION = 0.5


def is_ci() -> bool:
    """Return True when running under a recognised CI environment."""
    return any(os.getenv(var) for var in _CI_ENV_VARS)


def performance_threshold_multiplier(local: float = 1.0, ci: float = 2.0) -> float:
    """Multiplier for performance thresholds based on environment.

    CI runners have shared resources, variable load, fewer cores, and
    cold caches, so timing bounds calibrated on dev hardware need
    headroom there. Parameterised so each caller keeps its calibrated
    factor (e.g. the vectorization tests use ``ci=3.0``).

    Args:
        local: Multiplier on developer hardware (default 1.0 — strict).
        ci: Multiplier in CI environments (default 2.0).

    Returns:
        The multiplier for the current environment.
    """
    return ci if is_ci() else local


def effective_cpu_count() -> int:
    """Number of CPUs actually available to this process.

    Uses ``os.sched_getaffinity`` where available (Linux) because it is
    affinity-aware — in containers/CI the process may be restricted to
    fewer cores than the host exposes. Falls back to ``os.cpu_count()``
    on platforms without affinity support (macOS), which can return
    None, hence the ``or 1`` floor.
    """
    try:
        return len(os.sched_getaffinity(0)) or 1
    except AttributeError:
        return os.cpu_count() or 1


def nan_aware_close(
    a: float, b: float, rel_tol: float = 1e-9, abs_tol: float = 0.0
) -> bool:
    """``math.isclose`` with honest NaN semantics for metric comparison.

    Metric dicts contain honest NaNs when a metric genuinely cannot be
    computed (e.g. the ssimulacra2 binary is absent). Naive numeric
    comparison is NaN-blind: ``nan != 0`` is True and ``nan < tol`` is
    False, so both branches of the usual compare misfire.

    - both NaN  -> True  (both sides honestly failed the same way)
    - one NaN   -> False (a real divergence: one side fabricated a
      number where the other honestly could not compute)
    - otherwise -> ``math.isclose(a, b, ...)`` (equal infinities
      compare True; ``inf - inf`` never reaches a subtraction)
    """
    a = float(a)
    b = float(b)
    a_nan = math.isnan(a)
    b_nan = math.isnan(b)
    if a_nan and b_nan:
        return True
    if a_nan or b_nan:
        return False
    return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)


def analyze_memory_samples(
    samples: list[tuple[int, float]], warmup_samples: int = 2
) -> dict:
    """Analyze ``(iteration, memory_mb)`` samples for leak signals.

    The first ``warmup_samples`` samples are excluded: they capture
    one-time ramp costs (LPIPS model load, cache warm-up) that look like
    monotonic growth but are not leaks. The slope is regressed against
    the real *iteration* indices, so the returned rate is honestly MB
    per iteration regardless of sampling cadence (every-1 vs every-10
    sampling reports the same rate for the same leak).

    Args:
        samples: ``(iteration, memory_mb)`` tuples in sample order.
        warmup_samples: Leading samples to exclude from analysis.

    Returns:
        dict with keys:
            ``growth_rate_mb_per_iteration``: least-squares slope
                (MB/iteration) over the post-warm-up window; 0.0 when
                fewer than 2 points remain or all iteration indices
                coincide (degenerate regression).
            ``is_monotonic_growth``: post-warm-up memory never drops by
                more than ~10% between consecutive samples.
            ``potential_leak``: monotonic growth at a rate above
                ``LEAK_RATE_MB_PER_ITERATION``.
    """
    window = list(samples[warmup_samples:])
    if len(window) < 2:
        return {
            "growth_rate_mb_per_iteration": 0.0,
            "is_monotonic_growth": False,
            "potential_leak": False,
        }

    xs = [float(iteration) for iteration, _ in window]
    ys = [float(memory_mb) for _, memory_mb in window]
    n = len(window)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    denom = sum((x - mean_x) ** 2 for x in xs)
    if denom == 0.0:
        slope = 0.0
    else:
        slope = (
            sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
            / denom
        )

    is_monotonic = all(
        ys[i] <= ys[i + 1] * 1.1 for i in range(n - 1)  # Allow ~10% downward variation
    )

    return {
        "growth_rate_mb_per_iteration": slope,
        "is_monotonic_growth": is_monotonic,
        "potential_leak": is_monotonic and slope > LEAK_RATE_MB_PER_ITERATION,
    }
