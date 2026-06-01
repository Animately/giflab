"""Shared helpers for the audit scripts (sanity / pilot / sweep / report)."""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any

from PIL import Image

# Per-frame GIF delays are stored in centiseconds (10ms granularity).  When a
# lossy pass merges N duplicate frames, the merged frame carries the summed
# delay, but each merge can introduce up to one centisecond of rounding drift
# in the recomputed total.  We therefore tolerate up to this many ms of total
# duration difference per merged frame before calling a reduction "frame_loss"
# instead of benign "dedup".
_CENTISECOND_MS = 10
# Floor so even a single-frame reduction tolerates one centisecond of rounding.
_DURATION_TOL_FLOOR_MS = 10


def import_giflab() -> dict[str, Any]:
    """Lazy import of giflab internals. Returns a dict of named handles."""
    from giflab.config import MetricsConfig
    from giflab.metrics import calculate_comprehensive_metrics
    from giflab.public_api import compress
    from giflab.synthetic_gifs import SyntheticGifGenerator

    return {
        "MetricsConfig": MetricsConfig,
        "calculate_comprehensive_metrics": calculate_comprehensive_metrics,
        "compress": compress,
        "SyntheticGifGenerator": SyntheticGifGenerator,
    }


def run_metrics(original: Path, compressed: Path, gl: dict[str, Any]) -> dict[str, float]:
    """Run the comprehensive metrics suite and return a flat float dict.

    All optional metric subsystems are enabled (DEEP_PERCEPTUAL,
    TEMPORAL_ARTIFACTS) and force_all_metrics=True so we get every key.
    """
    config = gl["MetricsConfig"]()
    config.ENABLE_DEEP_PERCEPTUAL = True
    config.ENABLE_TEMPORAL_ARTIFACTS = True
    raw = gl["calculate_comprehensive_metrics"](
        original, compressed, config=config, force_all_metrics=True
    )
    return {k: float(v) for k, v in raw.items() if isinstance(v, (int, float))}


def compress_animately(
    input_path: Path, output_path: Path, lossy_level: int, gl: dict[str, Any]
) -> tuple[bool, str]:
    """Compress with animately. Returns (ok, error_message)."""
    try:
        gl["compress"](
            input_path, output_path, "animately", {"lossy_level": int(lossy_level)}
        )
        return True, ""
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def measure_pair(
    original: Path, compressed: Path, gl: dict[str, Any]
) -> tuple[dict[str, float], float]:
    """Measure metrics and return (metrics_dict, runtime_seconds)."""
    t0 = time.perf_counter()
    metrics = run_metrics(original, compressed, gl)
    return metrics, time.perf_counter() - t0


def read_gif_timing(path: Path) -> tuple[int, int | None]:
    """Read a GIF's frame count and total playback duration (ms) via PIL.

    Returns ``(frame_count, total_duration_ms)``.  On any read error the
    caller cannot trust the durations, so we return ``(0, None)`` — never a
    fabricated duration — so :func:`classify_frame_reduction` falls through to
    the honest ``"unknown"`` class rather than mislabelling a broken file as
    benign dedup.

    Mirrors the summing convention in ``giflab.metrics.extract_gif_frames``:
    per-frame delay is ``img.info.get("duration", 100)`` (default 100ms), and
    the total is the sum across every frame.
    """
    try:
        with Image.open(path) as img:
            n_frames = getattr(img, "n_frames", 1)
            total_duration = 0
            for i in range(n_frames):
                img.seek(i)
                total_duration += int(img.info.get("duration", 100))
        return int(n_frames), int(total_duration)
    except Exception:
        return 0, None


def classify_frame_reduction(
    *,
    orig_frames: int,
    orig_duration_ms: int | None,
    comp_frames: int,
    comp_duration_ms: int | None,
) -> dict[str, Any]:
    """Classify a frame-count change between original and compressed GIFs.

    Distinguishes benign temporal **deduplication** (fewer frames, total
    playback duration preserved) from a possible **frame_loss** bug (fewer
    frames, duration NOT preserved).  This lets sweep reports filter dedup
    events (no quality concern) from frame-loss events (needs investigation)
    instead of treating every frame-count mismatch as a failure.

    The duration comparison uses a tolerance that scales with the number of
    frames merged away — each merge can introduce up to one centisecond
    (10ms) of rounding drift in the recomputed total — with a floor so a
    single-frame reduction still tolerates one centisecond.

    Returns a dict with a stable key schema (every branch emits all keys):

    - ``frame_reduction``      ``orig_frames - comp_frames``, clamped at 0.
    - ``frame_reduction_class`` one of ``none`` / ``dedup`` / ``frame_loss`` /
      ``unknown``.
    - ``frames_deduplicated``  frames merged away when class is ``dedup``,
      else 0.
    - ``frame_dedup``          ``True`` only for the ``dedup`` class.
    - ``frame_loss``           ``True`` only for the ``frame_loss`` class.
    """
    reduction = orig_frames - comp_frames

    base: dict[str, Any] = {
        "frame_reduction": reduction if reduction > 0 else 0,
        "frame_reduction_class": "none",
        "frames_deduplicated": 0,
        "frame_dedup": False,
        "frame_loss": False,
    }

    # No reduction (equal, or compressed somehow has more frames): nothing to
    # classify.  A frame *increase* is not a dedup/loss event.
    if reduction <= 0:
        return base

    # We can only tell dedup from loss when both durations are known.  If
    # either is missing, report "unknown" — never silently call it benign.
    if orig_duration_ms is None or comp_duration_ms is None:
        base["frame_reduction_class"] = "unknown"
        return base

    duration_tol_ms = max(_DURATION_TOL_FLOOR_MS, reduction * _CENTISECOND_MS)
    if abs(orig_duration_ms - comp_duration_ms) <= duration_tol_ms:
        base["frame_reduction_class"] = "dedup"
        base["frames_deduplicated"] = reduction
        base["frame_dedup"] = True
    else:
        base["frame_reduction_class"] = "frame_loss"
        base["frame_loss"] = True
    return base


def open_csv_for_append(
    csv_path: Path, header_fields: list[str]
) -> tuple[Any, Any, bool]:
    """Open csv_path in append mode. Writes header if file is new.

    Returns (file_handle, csv_writer, header_was_just_written).
    """
    new_file = not csv_path.exists() or csv_path.stat().st_size == 0
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(csv_path, "a", newline="")
    writer = csv.DictWriter(fh, fieldnames=header_fields, extrasaction="ignore")
    if new_file:
        writer.writeheader()
        fh.flush()
    return fh, writer, new_file


def load_existing_keys(
    csv_path: Path, key_fields: tuple[str, ...]
) -> set[tuple[str, ...]]:
    """Load (key_field_values) tuples from an existing CSV to support resume."""
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return set()
    keys: set[tuple[str, ...]] = set()
    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                keys.add(tuple(row[f] for f in key_fields))
            except KeyError:
                continue
    return keys
