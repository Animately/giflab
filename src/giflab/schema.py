from __future__ import annotations

"""Schemas for GifLab data exports."""


import math

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

# --------------------------------------------------------------------------- #
# Helper constants
# --------------------------------------------------------------------------- #

_BASE_METRICS: list[str] = [
    "ssim",
    "ms_ssim",
    "psnr",
    "mse",
    "rmse",
    "fsim",
    "gmsd",
    "chist",
    "edge_similarity",
    "texture_similarity",
    "sharpness_similarity",
    "temporal_consistency",
]


# --------------------------------------------------------------------------- #
# Metric record schema
# --------------------------------------------------------------------------- #


class MetricRecordV1(BaseModel):
    """Validated record for a single GIF comparison / metric extraction.

    Only a *minimal* set of core keys is declared – the model accepts arbitrary
    additional numeric keys (e.g. per-metric stats, positional samples, raw
    metrics) thanks to *extra = "allow"*. This keeps the schema flexible while
    still enforcing basic sanity constraints for the most critical fields.
    """

    render_ms: int = Field(ge=0, description="Time taken to compute metrics (ms)")
    kilobytes: float = Field(ge=0, description="Size of compressed GIF in KB")
    # NOTE: no ``ge=0.0, le=1.0`` Field constraint here — pydantic's bound checks
    # categorically REJECT NaN (``nan <= 1`` is False), and as of the
    # composite-quality NaN guard ``calculate_composite_quality`` legitimately
    # returns ``float("nan")`` when the composite is majority-unmeasurable. The
    # CSV/metrics export must round-trip that NaN (per CLAUDE.md "CSV
    # serialisation must round-trip NaN") rather than crash the whole export.
    # The ``_check_composite_quality`` validator below permits NaN while still
    # rejecting finite out-of-range values.
    composite_quality: float = Field(
        description="Weighted composite quality score (0-1, or NaN if unmeasurable)"
    )

    model_config = ConfigDict(extra="allow")

    @field_validator("composite_quality")
    @classmethod
    def _check_composite_quality(cls, value: float) -> float:
        """Allow NaN (unmeasurable composite) through; enforce [0, 1] otherwise.

        A NaN composite is the honest signal that quality could not be measured
        (see ``calculate_composite_quality``); it must serialise and round-trip
        rather than fail validation. Finite values are still bounded to [0, 1].
        """
        if isinstance(value, float) and math.isnan(value):
            return value
        if not (0.0 <= value <= 1.0):
            raise ValueError(
                "composite_quality must be in [0.0, 1.0] or NaN, " f"got {value!r}"
            )
        return value


# --------------------------------------------------------------------------- #
# Convenience helpers
# --------------------------------------------------------------------------- #


def validate_metric_record(data: dict) -> MetricRecordV1:
    """Validate *data* against :class:`MetricRecordV1`.

    Raises ``pydantic.ValidationError`` if the record is invalid.
    Returns the parsed model instance otherwise.
    """
    return MetricRecordV1.model_validate(data)


def is_valid_record(data: dict) -> bool:
    """Return *True* if *data* passes :class:`MetricRecordV1` validation."""
    try:
        validate_metric_record(data)
        return True
    except ValidationError:
        return False
