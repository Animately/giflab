"""Regression test for Phase 6 optimized path schema contract.

Locks the key schema emitted by ``calculate_optimized_comprehensive_metrics``
so that a future change cannot silently add alias keys that would lie to
consumers about what was actually measured.

Background
----------
The main paths (``calculate_comprehensive_metrics`` /
``calculate_comprehensive_metrics_from_frames`` in ``giflab.metrics``) emit
``temporal_consistency_compressed`` and ``temporal_consistency_original`` keys
that describe independent measurements on the compressed and original streams
respectively.

Phase 6 measures temporal consistency on the compressed stream only and cannot
separate the two values. Wave 7 removed the bare ``temporal_consistency`` key
everywhere; Phase 6 now emits the honestly-labelled ``_compressed`` value (it
IS what was measured) but still must NOT emit ``_original`` (never measured —
fabrication). Per CLAUDE.md "Same key shape across paths":

    *If the optimized path can't compute ``_pre``/``_post`` separately,
    document that prominently rather than silently aliasing — silent
    equivalence lies to the consumer about what was actually measured.*

These tests enforce:
1. The Phase 6 result contains the known base keys that it genuinely computes
   (including ``temporal_consistency_compressed``).
2. The Phase 6 result does NOT contain the bare ``temporal_consistency`` key
   (removed) nor ``_original`` aliases that would misrepresent what was
   measured (single-stream values dressed up as a pair signal).

If a future PR adds silent equivalence aliases, this test will fail — which is
intentional.  The fix is either to actually compute both streams separately
(acceptable) or to keep the keys absent and update the module docstring
accordingly.
"""

from __future__ import annotations

import numpy as np
import pytest
from giflab.optimized_metrics import calculate_optimized_comprehensive_metrics

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _solid_frames(n: int, value: int = 128, size: tuple = (16, 16)) -> list[np.ndarray]:
    """Return ``n`` identical solid-colour frames of shape (size[0], size[1], 3)."""
    frame = np.full((size[0], size[1], 3), value, dtype=np.uint8)
    return [frame.copy() for _ in range(n)]


def _gradient_frames(n: int, size: tuple = (16, 16)) -> list[np.ndarray]:
    """Return ``n`` frames with a varying horizontal gradient."""
    frames = []
    for i in range(n):
        frame = np.zeros((size[0], size[1], 3), dtype=np.uint8)
        frame[:, :, 0] = np.linspace(0, 255, size[1], dtype=np.uint8)
        frame[:, :, 1] = (i * 30) % 256
        frames.append(frame)
    return frames


# ---------------------------------------------------------------------------
# Known-present keys in Phase 6 output
# ---------------------------------------------------------------------------

# These keys MUST be present in every Phase 6 result dict.
PHASE6_REQUIRED_KEYS = {
    "ssim_mean",
    "ssim_std",
    "ssim_min",
    "ssim_max",
    "ssim",
    "mse_mean",
    "mse_std",
    "mse_min",
    "mse_max",
    "mse",
    "psnr_mean",
    "psnr_std",
    "psnr_min",
    "psnr_max",
    "psnr",
    # Wave 7: bare ``temporal_consistency`` removed; Phase 6 emits the honest
    # compressed-stream value under ``_compressed`` (still required here — it
    # IS measured, correctly labelled). ``_original`` stays forbidden (never
    # measured by Phase 6).
    "temporal_consistency_compressed",
    "temporal_consistency_pre",
    "temporal_consistency_post",
    "temporal_consistency_delta",
    "frame_count",
    "compressed_frame_count",
    "render_ms",
    "_optimization_applied",
}

# These alias keys are emitted by the MAIN paths but must NOT appear in the
# Phase 6 output — they would silently lie about independent stream
# measurements. Note: ``temporal_consistency_compressed`` is NO LONGER
# forbidden (Wave 7) — Phase 6 emits it because it IS the honestly-labelled
# compressed-stream value it actually measures. Only ``_original`` (and the
# other-stream pair aliases) remain forbidden because Phase 6 never measures
# the original stream and would have to fabricate them.
PHASE6_FORBIDDEN_ALIAS_KEYS = {
    # temporal_consistency aliases — _compressed is now PRESENT (required); the
    # bare key is removed entirely; _original would be a fabrication.
    "temporal_consistency",
    "temporal_consistency_original",
    # disposal artifact aliases
    "disposal_artifacts_compressed",
    "disposal_artifacts_original",
    # enhanced temporal metric _compressed aliases
    "flicker_excess_compressed",
    "flicker_frame_ratio_compressed",
    "flat_flicker_ratio_compressed",
    "flat_region_count_compressed",
    "temporal_pumping_score_compressed",
    "quality_oscillation_frequency_compressed",
    "lpips_t_mean_compressed",
    "lpips_t_p95_compressed",
    "lpips_t_max_compressed",
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPhase6SchemaContract:
    """Ensure calculate_optimized_comprehensive_metrics emits the documented schema."""

    def _run(self, orig_frames, comp_frames):
        return calculate_optimized_comprehensive_metrics(orig_frames, comp_frames)

    def test_required_keys_present(self):
        """All PHASE6_REQUIRED_KEYS must exist in the result."""
        orig = _gradient_frames(4)
        comp = _gradient_frames(4)
        result = self._run(orig, comp)
        missing = PHASE6_REQUIRED_KEYS - result.keys()
        assert (
            not missing
        ), f"Phase 6 result is missing required keys: {sorted(missing)}"

    def test_optimization_flag_is_true(self):
        """The _optimization_applied sentinel must be True."""
        orig = _gradient_frames(4)
        comp = _gradient_frames(4)
        result = self._run(orig, comp)
        assert result.get("_optimization_applied") is True

    def test_forbidden_alias_keys_absent(self):
        """Alias keys that would lie about independent stream measurements must be absent.

        If this test fails it means a change added silent equivalence aliases
        (e.g. temporal_consistency_compressed = temporal_consistency_original
        = temporal_consistency).  That violates the CLAUDE.md accuracy rule.
        Either compute the streams separately and remove this assertion, or
        keep the keys absent and revert the aliasing change.
        """
        orig = _gradient_frames(4)
        comp = _gradient_frames(4)
        result = self._run(orig, comp)
        present_forbidden = PHASE6_FORBIDDEN_ALIAS_KEYS & result.keys()
        assert not present_forbidden, (
            "Phase 6 result contains alias keys that falsely imply independent "
            "stream measurements. Either compute both streams separately or "
            "remove the aliases. Forbidden keys found: "
            f"{sorted(present_forbidden)}"
        )

    def test_temporal_consistency_pre_post_equal(self):
        """_pre and _post must equal temporal_consistency_compressed (single-stream)."""
        orig = _gradient_frames(6)
        comp = _gradient_frames(6)
        result = self._run(orig, comp)
        tc = result["temporal_consistency_compressed"]
        assert result["temporal_consistency_pre"] == tc, (
            "_pre differs from temporal_consistency_compressed — Phase 6 may have "
            "changed to compute separate streams; update PHASE6_FORBIDDEN_ALIAS_KEYS."
        )
        assert result["temporal_consistency_post"] == tc, (
            "_post differs from temporal_consistency_compressed — Phase 6 may have "
            "changed to compute separate streams; update PHASE6_FORBIDDEN_ALIAS_KEYS."
        )

    def test_frame_counts_match_input(self):
        """frame_count and compressed_frame_count must reflect actual input lengths."""
        orig = _gradient_frames(5)
        comp = _gradient_frames(3)
        result = self._run(orig, comp)
        assert result["frame_count"] == 5
        assert result["compressed_frame_count"] == 3

    def test_result_values_are_finite(self):
        """Core metric values must be finite (no NaN or Inf from this path)."""
        orig = _gradient_frames(4)
        comp = _solid_frames(4, value=200)
        result = self._run(orig, comp)
        for key in (
            "ssim_mean",
            "mse_mean",
            "psnr_mean",
            "temporal_consistency_compressed",
        ):
            val = result[key]
            assert np.isfinite(val), f"Phase 6 result['{key}'] = {val} is not finite"

    def test_schema_stable_single_frame_each(self):
        """Even with 1 frame each the required keys must be present."""
        orig = _solid_frames(1)
        comp = _solid_frames(1, value=100)
        result = self._run(orig, comp)
        missing = PHASE6_REQUIRED_KEYS - result.keys()
        assert not missing, f"Single-frame run missing keys: {sorted(missing)}"

    def test_schema_stable_identical_frames(self):
        """Identical original and compressed frames must still yield valid output."""
        frames = _gradient_frames(4)
        result = self._run(frames, [f.copy() for f in frames])
        assert result["ssim_mean"] > 0.99
        missing = PHASE6_REQUIRED_KEYS - result.keys()
        assert not missing

    def test_no_aligned_pairs_fallback_emits_full_temporal_set(self):
        """The empty-aligned-pairs fallback must emit the FULL temporal key set.

        Wave 7 closed a pre-existing shape divergence: this branch previously
        emitted only the bare ``temporal_consistency`` key and none of the
        ``_compressed`` / ``_pre`` / ``_post`` / ``_delta`` siblings. It must now
        emit the same honest temporal shape as the normal path so the
        required-keys contract holds — and it must NOT emit the removed bare key
        or the forbidden ``_original`` alias.
        """
        # Two empty frame lists align to zero pairs -> fallback branch.
        result = self._run([], [])
        for key in (
            "temporal_consistency_compressed",
            "temporal_consistency_pre",
            "temporal_consistency_post",
            "temporal_consistency_delta",
        ):
            assert key in result, f"fallback missing temporal key {key}"
        # The removed bare key and the never-measured _original must be absent.
        assert "temporal_consistency" not in result
        assert "temporal_consistency_original" not in result

    def test_no_aligned_pairs_fallback_temporal_is_nan_not_fabricated_perfect(self):
        """The no-aligned-pairs FAILURE fallback must emit NaN — not fabricated
        perfect temporal preservation — for the four temporal keys.

        Audit-fix [[giflab-optimized-temporal-failure-nan]]: this fallback fires
        on the SAME "No frame pairs could be aligned" condition on which the main
        path honestly ``raise ValueError`` (``calculate_comprehensive_metrics_
        from_frames`` in ``giflab.metrics``). It previously fabricated
        ``temporal_consistency_{compressed,pre,post} = 1.0`` and
        ``temporal_consistency_delta = 0.0`` — perfect temporal preservation on a
        run that produced NO comparable frames — which silently inflated
        ``composite_quality`` (``temporal_consistency_delta`` /
        ``temporal_consistency_compressed`` feed ``calculate_composite_quality``).
        NaN propagates the loss honestly (``_is_missing`` filters it; the temporal
        weight is redistributed). This path was previously UNTESTED for VALUE —
        the sibling presence test only asserted the keys exist.
        """
        # Two empty frame lists align to zero pairs -> the failure fallback.
        result = self._run([], [])
        for key in (
            "temporal_consistency_compressed",
            "temporal_consistency_pre",
            "temporal_consistency_post",
            "temporal_consistency_delta",
        ):
            val = result[key]
            assert np.isnan(val), (
                f"{key} must be NaN on the no-aligned-pairs failure fallback "
                f"(honest signal loss, NOT fabricated-perfect 1.0/0.0); got {val!r}"
            )
        # Structural worst-case values stay 0.0 (no comparison was possible) — it
        # is specifically the temporal 1.0/0.0 that inflated composite_quality.
        assert result["ssim_mean"] == 0.0
        assert result["mse_mean"] == 0.0
        assert result["psnr_mean"] == 0.0
