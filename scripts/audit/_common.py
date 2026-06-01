"""Shared helpers for the audit scripts (sanity / pilot / sweep / report)."""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any


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
    """Compress with animately. Returns (ok, error_message).

    Bypasses the content-aware lossy ceiling (``apply_content_ceiling=False``):
    the monotonicity sweep / corpus sweeps need the requested lossy grid to run
    verbatim. Silently clamping photographic content to 20 would corrupt the
    very audit that calibrates the ceilings (see ``public_api.compress``).
    """
    try:
        gl["compress"](
            input_path,
            output_path,
            "animately",
            {"lossy_level": int(lossy_level)},
            apply_content_ceiling=False,
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
