"""Smoke tests for `scripts/audit/_common.py` frame-reduction classification.

When animately's `--lossy` pass collapses duplicate/near-duplicate frames
(temporal deduplication) the compressed GIF has fewer frames than the original
but the *total playback duration* is preserved — the merged frames simply carry
the summed delays.  This is benign: timing is correct, no quality issue.

A genuine encoder failure can ALSO produce fewer frames, but with the total
duration NOT preserved (frames silently dropped, animation runs short/fast).

The audit sweep CSV previously recorded both cases identically (just a
frame-count mismatch), forcing an operator to manually investigate every row.
``classify_frame_reduction`` distinguishes them by comparing total durations,
so sweep reports can filter benign dedup from possible-bug frame loss.

See ~/repos/obsidian/Work/Tasks/giflab-audit-classify-frame-dedup-events.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

# scripts/audit/ is not a Python package; load _common.py directly.
_SCRIPTS_AUDIT = Path(__file__).resolve().parents[2] / "scripts" / "audit"
if str(_SCRIPTS_AUDIT) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_AUDIT))

import _common  # type: ignore[import-not-found]  # noqa: E402


class TestClassifyFrameReduction:
    def test_no_reduction_is_none(self) -> None:
        """Equal frame counts → no reduction, regardless of duration."""
        cls = _common.classify_frame_reduction(
            orig_frames=8,
            orig_duration_ms=1600,
            comp_frames=8,
            comp_duration_ms=1600,
        )
        assert cls["frame_reduction_class"] == "none"
        assert cls["frames_deduplicated"] == 0
        assert cls["frame_dedup"] is False
        assert cls["frame_loss"] is False

    def test_dedup_when_duration_preserved(self) -> None:
        """The Malthouse case: 8 identical frames collapse to 3, duration kept."""
        cls = _common.classify_frame_reduction(
            orig_frames=8,
            orig_duration_ms=1600,
            comp_frames=3,
            comp_duration_ms=1600,
        )
        assert cls["frame_reduction_class"] == "dedup"
        assert cls["frames_deduplicated"] == 5
        assert cls["frame_dedup"] is True
        assert cls["frame_loss"] is False

    def test_frame_loss_when_duration_not_preserved(self) -> None:
        """Fewer frames AND duration shrank → genuine frame loss."""
        cls = _common.classify_frame_reduction(
            orig_frames=8,
            orig_duration_ms=1600,
            comp_frames=3,
            comp_duration_ms=600,
        )
        assert cls["frame_reduction_class"] == "frame_loss"
        assert cls["frames_deduplicated"] == 0
        assert cls["frame_dedup"] is False
        assert cls["frame_loss"] is True

    def test_centisecond_rounding_within_tolerance_is_dedup(self) -> None:
        """GIF delays round to 10ms granularity; a few ms of accumulated
        rounding when frames merge must NOT be misread as frame loss."""
        cls = _common.classify_frame_reduction(
            orig_frames=8,
            orig_duration_ms=1600,
            comp_frames=3,
            comp_duration_ms=1590,  # 10ms off — within one centisecond per merge
        )
        assert cls["frame_reduction_class"] == "dedup"
        assert cls["frame_dedup"] is True

    def test_tolerance_scales_with_merge_count(self) -> None:
        """The duration tolerance must scale with the number of merged frames,
        because each merge can introduce up to one centisecond of rounding.
        A 5-frame reduction tolerates more accumulated drift than a 1-frame one.
        """
        # 5 frames merged away → tolerance covers ~5 centiseconds of rounding.
        cls = _common.classify_frame_reduction(
            orig_frames=8,
            orig_duration_ms=1600,
            comp_frames=3,
            comp_duration_ms=1560,  # 40ms off, < 5*10ms = 50ms tolerance
        )
        assert cls["frame_reduction_class"] == "dedup"

    def test_large_duration_drop_is_frame_loss_not_dedup(self) -> None:
        """A drop far beyond rounding tolerance is frame loss even with a
        single-frame reduction."""
        cls = _common.classify_frame_reduction(
            orig_frames=4,
            orig_duration_ms=400,
            comp_frames=3,
            comp_duration_ms=300,  # 100ms off, well beyond 1*10ms tolerance
        )
        assert cls["frame_reduction_class"] == "frame_loss"
        assert cls["frame_loss"] is True

    def test_frame_increase_treated_as_none(self) -> None:
        """Compressed somehow has MORE frames — not a reduction; classify as
        'none' (frames_deduplicated never negative)."""
        cls = _common.classify_frame_reduction(
            orig_frames=3,
            orig_duration_ms=300,
            comp_frames=5,
            comp_duration_ms=300,
        )
        assert cls["frame_reduction_class"] == "none"
        assert cls["frames_deduplicated"] == 0
        assert cls["frame_dedup"] is False
        assert cls["frame_loss"] is False

    def test_missing_durations_returns_unknown(self) -> None:
        """If either duration is unavailable (None), the class is 'unknown' —
        a reduction happened but we cannot tell dedup from loss.  Never
        silently call it dedup (benign) — that would hide real frame loss."""
        cls = _common.classify_frame_reduction(
            orig_frames=8,
            orig_duration_ms=None,
            comp_frames=3,
            comp_duration_ms=1600,
        )
        assert cls["frame_reduction_class"] == "unknown"
        assert cls["frame_dedup"] is False
        assert cls["frame_loss"] is False

    def test_all_keys_present_in_every_branch(self) -> None:
        """Every classification dict must carry the same key schema so the CSV
        columns stay stable across all rows."""
        expected = {
            "frame_reduction",
            "frame_reduction_class",
            "frames_deduplicated",
            "frame_dedup",
            "frame_loss",
        }
        for kwargs in (
            dict(orig_frames=8, orig_duration_ms=1600, comp_frames=8, comp_duration_ms=1600),
            dict(orig_frames=8, orig_duration_ms=1600, comp_frames=3, comp_duration_ms=1600),
            dict(orig_frames=8, orig_duration_ms=1600, comp_frames=3, comp_duration_ms=600),
            dict(orig_frames=8, orig_duration_ms=None, comp_frames=3, comp_duration_ms=1600),
        ):
            cls = _common.classify_frame_reduction(**kwargs)  # type: ignore[arg-type]
            assert set(cls.keys()) == expected, f"key schema drift for {kwargs}: {cls.keys()}"


def _make_gif(path: Path, n_frames: int, per_frame_ms: int) -> None:
    frames = [
        Image.new("RGB", (8, 8), color=(i * 10 % 256, 0, 0)) for i in range(n_frames)
    ]
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=per_frame_ms,
        loop=0,
        optimize=False,
    )


class TestReadGifTiming:
    def test_reads_frame_count_and_total_duration(self, tmp_path: Path) -> None:
        p = tmp_path / "g.gif"
        _make_gif(p, n_frames=4, per_frame_ms=100)
        n_frames, total_ms = _common.read_gif_timing(p)
        assert n_frames == 4
        # PIL rounds delays to centiseconds; 4 * 100ms = 400ms expected.
        assert total_ms == 400

    def test_single_frame_gif(self, tmp_path: Path) -> None:
        p = tmp_path / "one.gif"
        Image.new("RGB", (8, 8), color=(1, 2, 3)).save(p)
        n_frames, total_ms = _common.read_gif_timing(p)
        assert n_frames == 1
        assert total_ms >= 0

    def test_unreadable_returns_zero_and_none(self, tmp_path: Path) -> None:
        """A non-GIF / corrupt file must not raise — return (0, None) so the
        caller classifies as 'unknown' rather than crashing the whole sweep."""
        p = tmp_path / "broken.gif"
        p.write_bytes(b"not a gif at all")
        n_frames, total_ms = _common.read_gif_timing(p)
        assert n_frames == 0
        assert total_ms is None


class TestFrameReductionSummaryReport:
    """``report.frame_reduction_summary`` is the production consumer that
    surfaces the new classification columns to operators reading the audit
    report.  These tests guard that wiring — without a consumer the columns
    would be written but never read.
    """

    def _import_report(self):
        import importlib.util

        rep_path = _SCRIPTS_AUDIT / "report.py"
        spec = importlib.util.spec_from_file_location("audit_report", rep_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_summary_counts_each_class(self) -> None:
        import pandas as pd

        report = self._import_report()
        df = pd.DataFrame(
            {
                "path": ["a.gif", "b.gif", "c.gif", "d.gif"],
                "lossy": [60, 60, 60, 60],
                "frame_reduction": [0, 5, 2, 1],
                "frame_reduction_class": ["none", "dedup", "frame_loss", "unknown"],
            }
        )
        md = "\n".join(report.frame_reduction_summary(df))
        assert "No reduction: 1" in md
        assert "Dedup (benign, duration preserved): 1" in md
        assert "Frame loss (possible bug): 1" in md
        assert "Unknown (durations unreadable): 1" in md

    def test_summary_lists_only_loss_and_unknown_rows(self) -> None:
        import pandas as pd

        report = self._import_report()
        df = pd.DataFrame(
            {
                "path": ["benign.gif", "lossy_bug.gif", "weird.gif"],
                "lossy": [60, 100, 40],
                "frame_reduction": [3, 2, 1],
                "frame_reduction_class": ["dedup", "frame_loss", "unknown"],
            }
        )
        md = "\n".join(report.frame_reduction_summary(df))
        # The benign dedup row must not appear in the investigation table.
        assert "benign.gif" not in md
        assert "lossy_bug.gif" in md
        assert "weird.gif" in md

    def test_summary_absent_column_returns_empty(self) -> None:
        import pandas as pd

        report = self._import_report()
        df = pd.DataFrame({"path": ["a.gif"], "lossy": [60]})
        assert report.frame_reduction_summary(df) == []
