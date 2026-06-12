"""Smoke tests for tie-aware disagreement ranking in the audit scripts.

Context: the 2026-06-12 disagreement-table saturation investigation
(~/repos/obsidian/Work/Tasks/giflab-disagreement-table-saturation.md) found
the near-1.0 max rank spread in the audit report was mostly a tie-handling
artifact, not honest metric-family disagreement:

- ``rank_normalise`` (scripts/audit/report.py) and the pilot's disagreement
  loop (scripts/audit/pilot.py) ranked via argsort + linspace, which assigns
  arbitrary DISTINCT ranks across tie blocks. The corpus is heavily
  tie-saturated (232/371 sweep rows tied at banding_score_mean == 0.0), so a
  tie-block member could land at rank ~1.0 by argsort luck and pin the spread.
- Single-stream ``_compressed`` keys (e.g. temporal_consistency_compressed)
  don't measure fidelity to the original, so their "disagreement" with pair
  metrics is structural — they must not vote in the disagreement table
  (they stay in the outlier tables).

The fix: tie-aware average percentile ranks
(``_common.tie_average_unit_ranks``) in both scripts, plus exclusion of
non-pairwise-quality metrics from the disagreement metric sets (mirroring the
PR #54 sanity DIAGNOSTIC triage).
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

import _common  # type: ignore[import-not-found]  # noqa: E402
import pilot  # type: ignore[import-not-found]  # noqa: E402
import report  # type: ignore[import-not-found]  # noqa: E402


class TestTieAverageUnitRanks:
    def test_no_ties_matches_linspace_scheme(self) -> None:
        # Without ties this must reduce exactly to the old
        # linspace(0, 1, n)-over-argsort behaviour.
        ranks = _common.tie_average_unit_ranks(np.array([0.3, 0.1, 0.4, 0.2]))
        assert ranks == pytest.approx([2 / 3, 0.0, 1.0, 1 / 3])

    def test_tie_block_shares_average_rank(self) -> None:
        # Three tied at the top occupy 0-based positions 1..3 -> avg 2 -> 2/3.
        ranks = _common.tie_average_unit_ranks(np.array([1.0, 1.0, 1.0, 0.5]))
        assert ranks == pytest.approx([2 / 3, 2 / 3, 2 / 3, 0.0])

    def test_large_tie_block_does_not_saturate(self) -> None:
        # Regression for the disagreement-table saturation: every member of a
        # 9-row tie block gets the honest mid-block rank (positions 1..9,
        # avg 5 -> 5/9), never an arbitrary ~1.0.
        ranks = _common.tie_average_unit_ranks(np.array([1.0] * 9 + [0.0]))
        assert ranks[:9] == pytest.approx([5.0 / 9.0] * 9)
        assert ranks.max() < 0.6
        assert ranks[9] == pytest.approx(0.0)

    def test_all_tied_is_half(self) -> None:
        ranks = _common.tie_average_unit_ranks(np.array([2.0, 2.0, 2.0]))
        assert ranks == pytest.approx([0.5, 0.5, 0.5])

    def test_single_value_is_half(self) -> None:
        ranks = _common.tie_average_unit_ranks(np.array([42.0]))
        assert ranks == pytest.approx([0.5])

    def test_empty_input(self) -> None:
        assert len(_common.tie_average_unit_ranks(np.array([]))) == 0


class TestRankNormalise:
    def test_higher_better_no_ties(self) -> None:
        ranks = report.rank_normalise(np.array([0.1, 0.4, 0.2]), higher_better=True)
        assert ranks == pytest.approx([0.0, 1.0, 0.5])

    def test_lower_better_flips(self) -> None:
        ranks = report.rank_normalise(np.array([0.1, 0.4, 0.2]), higher_better=False)
        assert ranks == pytest.approx([1.0, 0.0, 0.5])

    def test_ties_share_rank(self) -> None:
        # [1.0, 1.0] occupy 0-based positions 2..3 -> avg 2.5 -> 2.5/3.
        ranks = report.rank_normalise(
            np.array([1.0, 1.0, 0.5, 0.2]), higher_better=True
        )
        assert ranks == pytest.approx([2.5 / 3, 2.5 / 3, 1 / 3, 0.0])

    def test_single_value_is_half(self) -> None:
        ranks = report.rank_normalise(np.array([3.0]), higher_better=True)
        assert ranks == pytest.approx([0.5])


def _sweep_df() -> pd.DataFrame:
    """Five sweep rows, three pairwise metrics + one single-stream key.

    Tie-aware expected unit ranks (1 = best):
    - ssim (higher better):              [1.0, 0.75, 0.5, 0.25, 0.0]
    - banding_score_mean (lower better): 4-way tie at 0.0 (positions 0..3,
      avg 1.5 -> 0.375), flipped -> [0.625]*4 + [0.0]
    - gmsd (lower better):               [0.0, 0.25, 0.5, 1.0, 0.75]
    Per-row spreads: [1.0, 0.5, 0.125, 0.75, 0.75].
    """
    return pd.DataFrame(
        {
            "path": [f"/tmp/g{i}.gif" for i in range(5)],
            "lossy": [40] * 5,
            "source": ["synthetic"] * 5,
            "content_type": ["gradient"] * 5,
            "ssim": [0.9, 0.8, 0.7, 0.6, 0.5],
            "banding_score_mean": [0.0, 0.0, 0.0, 0.0, 10.0],
            "gmsd": [0.5, 0.4, 0.3, 0.2, 0.25],
            # Single-stream key: extreme ranks that would dominate the table
            # if it were (wrongly) allowed to vote.
            "temporal_consistency_compressed": [1.0, 0.0, 1.0, 0.0, 1.0],
        }
    )


class TestDisagreementTable:
    def test_single_stream_keys_do_not_vote(self) -> None:
        out = report.disagreement_table(_sweep_df(), top_n=5, min_metrics=3)
        named = set(out["best_metric"]) | set(out["worst_metric"])
        assert "temporal_consistency_compressed" not in named

    def test_tie_aware_spreads(self) -> None:
        out = report.disagreement_table(_sweep_df(), top_n=5, min_metrics=3)
        spreads = sorted(out["rank_spread"], reverse=True)
        assert spreads == pytest.approx([1.0, 0.75, 0.75, 0.5, 0.125])

    def test_best_worst_absolute_values_reported(self) -> None:
        out = report.disagreement_table(_sweep_df(), top_n=5, min_metrics=3)
        assert {"best_value", "worst_value"} <= set(out.columns)
        top = out.iloc[0]
        # Unique max spread is row g0: ssim rank 1.0 (value 0.9) vs gmsd
        # rank 0.0 (value 0.5).
        assert top["best_metric"] == "ssim"
        assert top["best_value"] == pytest.approx(0.9)
        assert top["worst_metric"] == "gmsd"
        assert top["worst_value"] == pytest.approx(0.5)

    def test_single_stream_keys_stay_in_outlier_tables(self) -> None:
        # The exclusion is disagreement-only: _compressed keys keep their
        # outlier table (they are still useful diagnostics there).
        outliers = report.top_outliers_per_metric(_sweep_df(), top_n=3)
        assert "temporal_consistency_compressed" in outliers


class TestAnalysePilotDisagreement:
    def test_tie_aware_and_structural_exclusion(self, tmp_path: Path) -> None:
        """End-to-end analyse_pilot on a hand-computed 5-GIF pilot CSV.

        Pairwise metrics only (tie-aware):
        - metric_a [1, 1, 1, 1, 0.5]: 4-way tie at top -> rank 0.625 each,
          the 0.5 row -> 0.0.
        - metric_b [.1, .2, .3, .4, .5]: ranks [0, .25, .5, .75, 1.0].
        Per-GIF spreads [0.625, 0.375, 0.125, 0.125, 1.0] -> mean 0.45.

        The three structural distractor columns (single-stream, dispersion,
        diagnostic) would each shift that mean if they were allowed to vote.
        """
        csv_path = tmp_path / "pilot.csv"
        header = [
            "path",
            "lossy",
            "success",
            "metric_a",
            "metric_b",
            "temporal_consistency_compressed",  # single-stream
            "ssim_std",  # dispersion sibling
            "kilobytes",  # diagnostic/system
        ]
        metric_a = [1.0, 1.0, 1.0, 1.0, 0.5]
        metric_b = [0.1, 0.2, 0.3, 0.4, 0.5]
        tcc = [0.0, 1.0, 0.5, 0.2, 0.9]
        ssim_std = [0.5, 0.1, 0.3, 0.2, 0.4]
        kilobytes = [10.0, 20.0, 30.0, 40.0, 50.0]
        lines = [",".join(header)]
        for i in range(5):
            lines.append(
                f"/tmp/g{i}.gif,40,True,{metric_a[i]},{metric_b[i]},"
                f"{tcc[i]},{ssim_std[i]},{kilobytes[i]}"
            )
        csv_path.write_text("\n".join(lines) + "\n")

        decisions = pilot.analyse_pilot(csv_path, [40])

        assert decisions["disagreement_by_lossy"][40] == pytest.approx(0.45)
        assert decisions["chosen_lossy_levels"] == [40]
