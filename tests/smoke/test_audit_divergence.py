"""Smoke tests for the SSIM-vs-composite_quality divergence pass (report.py).

Context: gifprep adopted ``composite_quality`` as its benchmark's single
Pareto verdict gate, but no dedicated pass ever asked where SSIM and
composite disagree on real content. The divergence pass compares
control-relative paired deltas (each GIF's lossy=0 row is its own control):

    d_ssim = ssim(L) - ssim(0)
    d_comp = composite_quality(L) - composite_quality(0)

- **Hard direction divergence**: the two deltas point in opposite
  directions, both beyond EPS = 0.005 (identical to gifprep's
  ``EPSILON_QUALITY`` in ``bench/pareto.py`` and the EPS in gifprep's
  ``analyses/2026-06-12-corpus-local-composite-rerun/divergence.py``, so
  findings transfer 1:1 downstream).
- **Verdict divergence**: one metric held (``|d| <= EPS``) while the other
  moved beyond GUARD = 0.02 (4x EPS). The zone between EPS and GUARD is a
  deliberate *indeterminate* band that is never flagged — the cliff-edge
  lesson applied to the audit criterion itself.
- **Continuous severity**: ``divergence_score = |d_comp - d_ssim|`` is
  computed for EVERY analysable cell (not only flagged ones) — an
  absolute-scale quantity that, unlike the rank-spread table, cannot
  saturate from ties.
- **NaN honesty**: cells where either metric is NaN in either row are
  excluded AND counted, never imputed.

``composite_attribution`` decomposes a flagged cell's composite movement
into its 15 named contributor terms, verified by a round-trip check against
the real ``giflab.enhanced_metrics.calculate_composite_quality`` — rows
that fail the check are flagged non-attributable rather than silently
trusted.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# scripts/audit/ is not a Python package; load the modules directly.
_SCRIPTS_AUDIT = Path(__file__).resolve().parents[2] / "scripts" / "audit"
if str(_SCRIPTS_AUDIT) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_AUDIT))

import report  # type: ignore[import-not-found]  # noqa: E402

EPS = 0.005
GUARD = 0.02


def _df(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal sweep-shaped DataFrame from per-row dicts."""
    base = {"source": "real", "content_type": "photo"}
    return pd.DataFrame([{**base, **r} for r in rows])


def _cell(out: dict, path: str, lossy: int) -> pd.Series:
    cells = out["cells"]
    sub = cells[(cells["path"] == path) & (cells["lossy"] == lossy)]
    assert len(sub) == 1, f"expected exactly one cell for ({path}, {lossy})"
    return sub.iloc[0]


class TestDivergenceCriterion:
    def test_opposite_directions_beyond_eps_flagged(self) -> None:
        # composite up +0.01, ssim down -0.02: both beyond EPS, opposite
        # directions -> hard direction divergence.
        df = _df(
            [
                {
                    "path": "/tmp/a.gif",
                    "lossy": 0,
                    "ssim": 0.90,
                    "composite_quality": 0.80,
                },
                {
                    "path": "/tmp/a.gif",
                    "lossy": 20,
                    "ssim": 0.88,
                    "composite_quality": 0.81,
                },
            ]
        )
        out = report.ssim_composite_divergence(df, eps=EPS, guard=GUARD)
        cell = _cell(out, "/tmp/a.gif", 20)
        assert cell["flag"] == "direction"
        assert cell["d_ssim"] == pytest.approx(-0.02)
        assert cell["d_comp"] == pytest.approx(0.01)

    def test_opposite_directions_other_sign_flagged(self) -> None:
        # Mirror case: composite down, ssim up.
        df = _df(
            [
                {
                    "path": "/tmp/f.gif",
                    "lossy": 0,
                    "ssim": 0.80,
                    "composite_quality": 0.90,
                },
                {
                    "path": "/tmp/f.gif",
                    "lossy": 20,
                    "ssim": 0.81,
                    "composite_quality": 0.88,
                },
            ]
        )
        out = report.ssim_composite_divergence(df, eps=EPS, guard=GUARD)
        assert _cell(out, "/tmp/f.gif", 20)["flag"] == "direction"

    def test_same_direction_movement_not_flagged(self) -> None:
        df = _df(
            [
                {
                    "path": "/tmp/a.gif",
                    "lossy": 0,
                    "ssim": 0.90,
                    "composite_quality": 0.80,
                },
                {
                    "path": "/tmp/a.gif",
                    "lossy": 40,
                    "ssim": 0.85,
                    "composite_quality": 0.76,
                },
            ]
        )
        out = report.ssim_composite_divergence(df, eps=EPS, guard=GUARD)
        assert _cell(out, "/tmp/a.gif", 40)["flag"] == ""

    def test_held_vs_degraded_beyond_guard_flagged(self) -> None:
        # ssim held (|d| <= EPS) while composite moved beyond GUARD.
        df = _df(
            [
                {
                    "path": "/tmp/a.gif",
                    "lossy": 0,
                    "ssim": 0.900,
                    "composite_quality": 0.80,
                },
                {
                    "path": "/tmp/a.gif",
                    "lossy": 60,
                    "ssim": 0.899,
                    "composite_quality": 0.75,
                },
            ]
        )
        out = report.ssim_composite_divergence(df, eps=EPS, guard=GUARD)
        assert _cell(out, "/tmp/a.gif", 60)["flag"] == "verdict"

    def test_indeterminate_zone_never_flagged(self) -> None:
        # One metric held, the other moved into (EPS, GUARD] — the
        # deliberate indeterminate band: boundary jitter must not be
        # reported as disagreement (cliff-edge guard).
        df = _df(
            [
                {
                    "path": "/tmp/b.gif",
                    "lossy": 0,
                    "ssim": 0.900,
                    "composite_quality": 0.900,
                },
                {
                    "path": "/tmp/b.gif",
                    "lossy": 20,
                    "ssim": 0.899,
                    "composite_quality": 0.885,
                },
            ]
        )
        out = report.ssim_composite_divergence(df, eps=EPS, guard=GUARD)
        assert _cell(out, "/tmp/b.gif", 20)["flag"] == ""


class TestDivergenceScore:
    def test_score_continuous_and_present_for_unflagged_cells(self) -> None:
        df = _df(
            [
                {
                    "path": "/tmp/a.gif",
                    "lossy": 0,
                    "ssim": 0.90,
                    "composite_quality": 0.80,
                },
                # direction-flagged: score = |0.01 - (-0.02)| = 0.03
                {
                    "path": "/tmp/a.gif",
                    "lossy": 20,
                    "ssim": 0.88,
                    "composite_quality": 0.81,
                },
                # unflagged same-direction: score = |-0.04 - (-0.05)| = 0.01
                {
                    "path": "/tmp/a.gif",
                    "lossy": 40,
                    "ssim": 0.85,
                    "composite_quality": 0.76,
                },
                # verdict-flagged: score = |-0.05 - (-0.001)| = 0.049
                {
                    "path": "/tmp/a.gif",
                    "lossy": 60,
                    "ssim": 0.899,
                    "composite_quality": 0.75,
                },
            ]
        )
        out = report.ssim_composite_divergence(df, eps=EPS, guard=GUARD)
        cells = out["cells"]
        # Present for every analysable cell, flagged or not.
        assert len(cells) == 3
        assert np.isfinite(cells["divergence_score"]).all()
        assert _cell(out, "/tmp/a.gif", 40)["divergence_score"] == pytest.approx(0.01)
        assert _cell(out, "/tmp/a.gif", 20)["divergence_score"] == pytest.approx(0.03)
        assert _cell(out, "/tmp/a.gif", 60)["divergence_score"] == pytest.approx(0.049)
        # Sorted descending by divergence_score.
        scores = cells["divergence_score"].tolist()
        assert scores == sorted(scores, reverse=True)


class TestNaNHonesty:
    def test_nan_cell_excluded_and_counted_never_flagged(self) -> None:
        df = _df(
            [
                {
                    "path": "/tmp/d.gif",
                    "lossy": 0,
                    "ssim": 0.90,
                    "composite_quality": 0.80,
                },
                {
                    "path": "/tmp/d.gif",
                    "lossy": 20,
                    "ssim": 0.88,
                    "composite_quality": float("nan"),
                },
                {
                    "path": "/tmp/d.gif",
                    "lossy": 40,
                    "ssim": 0.85,
                    "composite_quality": 0.76,
                },
            ]
        )
        out = report.ssim_composite_divergence(df, eps=EPS, guard=GUARD)
        assert out["excluded"]["nan_cell"] == 1
        cells = out["cells"]
        assert len(cells) == 1  # only the lossy=40 cell survives
        assert (cells["lossy"] == 40).all()

    def test_nan_control_excludes_all_cells_of_gif(self) -> None:
        df = _df(
            [
                {
                    "path": "/tmp/c.gif",
                    "lossy": 0,
                    "ssim": float("nan"),
                    "composite_quality": 0.80,
                },
                {
                    "path": "/tmp/c.gif",
                    "lossy": 20,
                    "ssim": 0.88,
                    "composite_quality": 0.78,
                },
                {
                    "path": "/tmp/c.gif",
                    "lossy": 40,
                    "ssim": 0.85,
                    "composite_quality": 0.76,
                },
            ]
        )
        out = report.ssim_composite_divergence(df, eps=EPS, guard=GUARD)
        assert out["excluded"]["nan_control"] == 2
        assert out["cells"].empty

    def test_missing_control_row_excludes_all_cells_of_gif(self) -> None:
        df = _df(
            [
                {
                    "path": "/tmp/e.gif",
                    "lossy": 20,
                    "ssim": 0.88,
                    "composite_quality": 0.78,
                },
                {
                    "path": "/tmp/e.gif",
                    "lossy": 40,
                    "ssim": 0.85,
                    "composite_quality": 0.76,
                },
                # A healthy GIF alongside, to prove per-GIF (not global) exclusion.
                {
                    "path": "/tmp/a.gif",
                    "lossy": 0,
                    "ssim": 0.90,
                    "composite_quality": 0.80,
                },
                {
                    "path": "/tmp/a.gif",
                    "lossy": 20,
                    "ssim": 0.85,
                    "composite_quality": 0.76,
                },
            ]
        )
        out = report.ssim_composite_divergence(df, eps=EPS, guard=GUARD)
        assert out["excluded"]["no_control"] == 2
        assert len(out["cells"]) == 1
        assert (out["cells"]["path"] == "/tmp/a.gif").all()


class TestDedupe:
    def test_duplicate_path_lossy_rows_deduped_keep_first(self) -> None:
        # Sweep resume can append duplicate (path, lossy) rows; keep-first.
        df = _df(
            [
                {
                    "path": "/tmp/a.gif",
                    "lossy": 0,
                    "ssim": 0.90,
                    "composite_quality": 0.80,
                },
                {
                    "path": "/tmp/a.gif",
                    "lossy": 20,
                    "ssim": 0.88,
                    "composite_quality": 0.81,
                },
                {
                    "path": "/tmp/a.gif",
                    "lossy": 20,
                    "ssim": 0.10,
                    "composite_quality": 0.10,
                },
            ]
        )
        out = report.ssim_composite_divergence(df, eps=EPS, guard=GUARD)
        cell = _cell(out, "/tmp/a.gif", 20)
        assert cell["ssim_cell"] == pytest.approx(0.88)
        assert cell["comp_cell"] == pytest.approx(0.81)


class TestAdjacentLevelDivergence:
    def test_composite_recovers_while_ssim_falls(self) -> None:
        # The historical smooth_gradient dip-then-recover shape: over
        # 20 -> 40 composite recovers (+0.05) while ssim keeps falling
        # (-0.05). The 0 -> 20 step agrees and must not be flagged.
        df = _df(
            [
                {
                    "path": "/tmp/g.gif",
                    "lossy": 0,
                    "ssim": 0.90,
                    "composite_quality": 0.80,
                },
                {
                    "path": "/tmp/g.gif",
                    "lossy": 20,
                    "ssim": 0.85,
                    "composite_quality": 0.70,
                },
                {
                    "path": "/tmp/g.gif",
                    "lossy": 40,
                    "ssim": 0.80,
                    "composite_quality": 0.75,
                },
            ]
        )
        steps = report.adjacent_level_divergence(df, eps=EPS)
        assert len(steps) == 2
        flagged = steps[steps["sign_disagree"]]
        assert len(flagged) == 1
        step = flagged.iloc[0]
        assert (step["lossy_from"], step["lossy_to"]) == (20, 40)
        assert step["d_ssim"] == pytest.approx(-0.05)
        assert step["d_comp"] == pytest.approx(0.05)

    def test_within_eps_steps_not_flagged(self) -> None:
        df = _df(
            [
                {
                    "path": "/tmp/h.gif",
                    "lossy": 0,
                    "ssim": 0.900,
                    "composite_quality": 0.800,
                },
                {
                    "path": "/tmp/h.gif",
                    "lossy": 20,
                    "ssim": 0.899,
                    "composite_quality": 0.801,
                },
            ]
        )
        steps = report.adjacent_level_divergence(df, eps=EPS)
        assert not steps["sign_disagree"].any()


# ---------------------------------------------------------------------------
# composite_attribution
# ---------------------------------------------------------------------------


def _contributor_metrics(**overrides: float) -> dict[str, float]:
    """All 15 composite contributor keys with plausible high-quality values."""
    base = {
        "ssim_mean": 0.95,
        "ms_ssim_mean": 0.96,
        "psnr_mean": 38.0,
        "mse_mean": 10.0,
        "fsim_mean": 0.93,
        "edge_similarity_mean": 0.90,
        "gmsd_mean": 0.05,
        "chist_mean": 0.99,
        "sharpness_similarity_mean": 0.92,
        "texture_similarity_mean": 0.90,
        "temporal_consistency_delta": 0.02,
        "lpips_quality_mean": 0.05,
        "ssimulacra2_mean": 0.85,
        "banding_score_mean": 0.0,
        "deltae_mean": 1.0,
    }
    base.update(overrides)
    return base


def _row_with_composite(**overrides: float) -> dict[str, float]:
    """A sweep-row-shaped dict whose composite_quality is the REAL value
    computed by giflab.enhanced_metrics.calculate_composite_quality."""
    from giflab.enhanced_metrics import calculate_composite_quality

    metrics = _contributor_metrics(**overrides)
    row = dict(metrics)
    row["composite_quality"] = calculate_composite_quality(metrics)
    return row


class TestCompositeAttribution:
    def test_roundtrip_matches_and_drivers_ranked(self) -> None:
        control = _row_with_composite()
        cell = _row_with_composite(ssim_mean=0.90, chist_mean=0.95, deltae_mean=3.0)

        attr = report.composite_attribution(control, cell)
        assert attr["attributable"] is True

        drivers = attr["drivers"]
        by_name = {d["metric"]: d for d in drivers}
        # deltae: norm 0.9 -> 0.7, weight 0.05 -> weighted_delta -0.01.
        # ssim:   norm 0.95 -> 0.90, weight 0.15 -> weighted_delta -0.0075.
        # chist:  norm 0.99 -> 0.95, weight 0.04 -> weighted_delta -0.0016.
        assert drivers[0]["metric"] == "deltae_mean"
        assert by_name["deltae_mean"]["weighted_delta"] == pytest.approx(-0.01)
        assert by_name["ssim_mean"]["weighted_delta"] == pytest.approx(-0.0075)
        assert by_name["chist_mean"]["weighted_delta"] == pytest.approx(-0.0016)
        # Untouched contributors carry ~zero weighted delta.
        assert by_name["fsim_mean"]["weighted_delta"] == pytest.approx(0.0)

    def test_corrupted_stored_composite_flagged_non_attributable(self) -> None:
        control = _row_with_composite()
        cell = _row_with_composite(ssim_mean=0.90)
        cell["composite_quality"] = cell["composite_quality"] + 0.1  # corrupt

        attr = report.composite_attribution(control, cell)
        assert attr["attributable"] is False
        assert attr["reason"] != ""

    def test_nan_contributor_reported_as_missing_with_weights(self) -> None:
        # Post-#54 edgeless content: edge_similarity_mean NaN in the cell
        # row redistributes weight — must surface as a named missing driver
        # with differing surviving weights, not hidden.
        control = _row_with_composite()
        cell = _row_with_composite(edge_similarity_mean=float("nan"))

        attr = report.composite_attribution(control, cell)
        assert attr["attributable"] is True
        by_name = {d["metric"]: d for d in attr["drivers"]}
        edge = by_name["edge_similarity_mean"]
        assert edge["missing_cell"] is True
        assert edge["missing_control"] is False
        assert attr["surviving_weight_cell"] < attr["surviving_weight_control"]
