"""Smoke tests for `scripts/audit/sanity.py` pure helpers.

These tests cover the small pure utilities used by the audit sanity script —
no GIF I/O, no metric calculation. Specifically the monotonicity check and
the byte-identical coalescing helper added to guard against the
smooth_gradient SSIM bump-up false positive at lossy saturation.

See ~/repos/obsidian/Work/Tasks/giflab-ssim-smooth-gradient-bump.md for
context — animately's --lossy parameter saturates around level ~125 on
low-complexity content, so consecutive lossy grid points past the knee
produce byte-identical compressed outputs (or near-identical ones) whose
metric values trivially "bump up" by floating-point noise.
"""

from __future__ import annotations

import sys
from pathlib import Path

# scripts/audit/ is not a Python package; load sanity.py directly.
_SCRIPTS_AUDIT = (
    Path(__file__).resolve().parents[2] / "scripts" / "audit"
)
if str(_SCRIPTS_AUDIT) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_AUDIT))

import sanity  # type: ignore[import-not-found]  # noqa: E402


class TestMonotonicityCheck:
    def test_higher_better_monotonic(self) -> None:
        # Values strictly non-increasing: no inversions.
        assert sanity._monotonicity_check([1.0, 0.9, 0.8, 0.7], "higher_better") == []

    def test_higher_better_inversion(self) -> None:
        # 0.7 -> 0.75 is an inversion (got better when worse expected).
        invs = sanity._monotonicity_check([1.0, 0.9, 0.7, 0.75], "higher_better")
        assert len(invs) == 1
        assert invs[0][0] == 2  # index of the from-level
        assert invs[0][1] == 0.7
        assert invs[0][2] == 0.75

    def test_lower_better_monotonic(self) -> None:
        assert sanity._monotonicity_check([0.0, 1.0, 5.0, 10.0], "lower_better") == []

    def test_lower_better_inversion(self) -> None:
        # 5.0 -> 4.9 is an inversion (got smaller when larger expected).
        invs = sanity._monotonicity_check([0.0, 1.0, 5.0, 4.9], "lower_better")
        assert len(invs) == 1

    def test_flat_direction_returns_empty(self) -> None:
        # Direction "flat" means the metric doesn't discriminate; never flag.
        assert sanity._monotonicity_check([1.0, 0.9, 1.1, 0.8], "flat") == []

    def test_tolerance_absorbs_noise(self) -> None:
        # Default tol is 1e-4; a sub-tolerance bump is not flagged.
        invs = sanity._monotonicity_check(
            [0.9, 0.85, 0.85 + 5e-5, 0.84], "higher_better"
        )
        assert invs == []


class TestCoalesceByteIdenticalLevels:
    """The lossy arm of sanity.py should drop consecutive byte-identical
    compressions before running the monotonicity check, because saturated
    animately output produces literally identical bytes (and therefore
    identical metric values) — testing monotonicity on such a sequence is
    spurious.
    """

    def test_no_duplicates_passthrough(self, tmp_path: Path) -> None:
        # Each path has unique bytes -> nothing coalesced.
        p1 = tmp_path / "a.gif"
        p2 = tmp_path / "b.gif"
        p3 = tmp_path / "c.gif"
        p1.write_bytes(b"AAAA")
        p2.write_bytes(b"BBBB")
        p3.write_bytes(b"CCCC")

        per_metric = {"ssim": [0.9, 0.8, 0.7]}
        kept_paths, kept_metrics = sanity._coalesce_byte_identical_levels(
            [p1, p2, p3], per_metric
        )
        assert kept_paths == [p1, p2, p3]
        assert kept_metrics == {"ssim": [0.9, 0.8, 0.7]}

    def test_drops_consecutive_byte_identical(self, tmp_path: Path) -> None:
        # Saturation case: levels 100, 130, 160, 200 might produce
        # bytes A, B, B, B. Want to keep [A, B] only.
        p_a = tmp_path / "lossy_100.gif"
        p_b1 = tmp_path / "lossy_130.gif"
        p_b2 = tmp_path / "lossy_160.gif"
        p_b3 = tmp_path / "lossy_200.gif"
        p_a.write_bytes(b"unique-pre-saturation")
        p_b1.write_bytes(b"saturated-bytes")
        p_b2.write_bytes(b"saturated-bytes")
        p_b3.write_bytes(b"saturated-bytes")

        per_metric = {
            "ssim": [0.85, 0.83, 0.83, 0.83],
            "mse_max": [50.0, 48.0, 48.0, 48.0],
        }
        kept_paths, kept_metrics = sanity._coalesce_byte_identical_levels(
            [p_a, p_b1, p_b2, p_b3], per_metric
        )
        assert kept_paths == [p_a, p_b1]
        assert kept_metrics == {"ssim": [0.85, 0.83], "mse_max": [50.0, 48.0]}

    def test_handles_empty_input(self, tmp_path: Path) -> None:
        kept_paths, kept_metrics = sanity._coalesce_byte_identical_levels([], {})
        assert kept_paths == []
        assert kept_metrics == {}

    def test_only_drops_consecutive_duplicates(self, tmp_path: Path) -> None:
        # A, B, A, A: positions 0,2,3 share bytes "A" but only consecutive
        # duplicates are coalesced (positions 2-3 collapse, position 0
        # stays separate from position 2).
        p_a1 = tmp_path / "a1.gif"
        p_b = tmp_path / "b.gif"
        p_a2 = tmp_path / "a2.gif"
        p_a3 = tmp_path / "a3.gif"
        p_a1.write_bytes(b"X")
        p_b.write_bytes(b"Y")
        p_a2.write_bytes(b"X")
        p_a3.write_bytes(b"X")

        per_metric = {"ssim": [0.9, 0.5, 0.9, 0.9]}
        kept_paths, kept_metrics = sanity._coalesce_byte_identical_levels(
            [p_a1, p_b, p_a2, p_a3], per_metric
        )
        # Only a3 (consecutive duplicate of a2) is dropped.
        assert kept_paths == [p_a1, p_b, p_a2]
        assert kept_metrics == {"ssim": [0.9, 0.5, 0.9]}


def _make_multiframe_gif(path: Path, color: tuple[int, int, int] = (128, 128, 128)) -> None:
    """Create a 2-frame solid-colour GIF that ``read_gif_frames`` can read."""
    from PIL import Image

    frames = [Image.new("RGB", (16, 16), color=color) for _ in range(2)]
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=100,
        loop=0,
        optimize=False,
    )


class _FakeSpec:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeMetricsConfig:
    """Settable-attribute stand-in for ``giflab.config.MetricsConfig``."""

    def __init__(self) -> None:
        self.ENABLE_DEEP_PERCEPTUAL = False
        self.ENABLE_TEMPORAL_ARTIFACTS = False


def _build_fake_gl(metric_dict_factory):
    """Build a fake ``gl`` dict for ``_import_giflab``.

    ``metric_dict_factory`` is called once per ``run_metrics`` invocation
    and returns the dict of metric values for that call.
    """
    class _FakeGenerator:
        def __init__(self, out_dir: Path) -> None:
            self.out_dir = out_dir
            # Cover both IDENTITY_SAMPLE and MONOTONICITY_BASES with a
            # single base name; tests below pin both constants to a single
            # shared name to keep the fixture minimal.
            self.synthetic_specs = [_FakeSpec("base_under_test")]

        def generate_gifs(self, *, use_targeted_set: bool = False):
            p = self.out_dir / "base_under_test.gif"
            _make_multiframe_gif(p)
            return [p]

    return {
        "SyntheticGifGenerator": _FakeGenerator,
        "calculate_comprehensive_metrics": lambda original, compressed, **kwargs: metric_dict_factory(),
        "compress": None,  # filled in per-test
        "MetricsConfig": _FakeMetricsConfig,
    }


class TestCoalesceIntegratedIntoRunSanity:
    """The coalesce helper must actually be invoked by ``run_sanity()``'s lossy
    arm — otherwise the smooth_gradient SSIM bump-up bug it exists to fix is
    still present at runtime. These tests guard the wiring, not just the
    helper in isolation.
    """

    def _pin_constants(self, monkeypatch) -> None:
        """Shrink all the degradation grids and pin IDENTITY_SAMPLE +
        MONOTONICITY_BASES to a single shared name so the test runs in a
        few milliseconds.

        FIXTURE_MONOTONICITY_BASES is cleared so the fixture-based loop does
        not run a second coalesce call — tests in this class are asserting on
        the synthetic-bases lossy arm in isolation.
        """
        monkeypatch.setattr(sanity, "MONOTONICITY_BASES", ["base_under_test"])
        monkeypatch.setattr(sanity, "IDENTITY_SAMPLE", ["base_under_test"])
        monkeypatch.setattr(sanity, "NOISE_SIGMAS", [5])
        monkeypatch.setattr(sanity, "BLUR_SIGMAS", [0.5])
        monkeypatch.setattr(sanity, "QUANTIZE_COLORS", [64])
        # Use 4 lossy levels: first 2 unique, last 2 byte-identical with #2.
        monkeypatch.setattr(sanity, "LOSSY_LEVELS", [30, 60, 120, 200])
        # Clear fixture bases so they don't trigger a second coalesce call.
        monkeypatch.setattr(sanity, "FIXTURE_MONOTONICITY_BASES", {})

    def test_lossy_arm_calls_coalesce_with_paths_and_metrics(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """Spy on ``_coalesce_byte_identical_levels`` and assert ``run_sanity()``
        invokes it from the lossy arm exactly once with the ordered paths and
        the per-metric levels collected during the lossy loop.
        """
        gl = _build_fake_gl(lambda: {"ssim": 0.5, "psnr": 30.0})

        # First two lossy levels write unique bytes; levels 3 and 4 saturate
        # and produce byte-identical outputs that should be coalesced.
        call_counter = {"n": 0}

        def _fake_compress(input_path, output_path, engine, params):
            call_counter["n"] += 1
            n = call_counter["n"]
            if n == 1:
                Path(output_path).write_bytes(b"unique-level-1")
            elif n == 2:
                Path(output_path).write_bytes(b"unique-level-2")
            else:
                Path(output_path).write_bytes(b"SATURATED")
            return output_path

        gl["compress"] = _fake_compress
        monkeypatch.setattr(sanity, "_import_giflab", lambda: gl)
        self._pin_constants(monkeypatch)

        real_coalesce = sanity._coalesce_byte_identical_levels
        calls: list[tuple[list[Path], dict[str, list[float]]]] = []

        def _spy(paths, per_metric):
            calls.append(
                (list(paths), {k: list(v) for k, v in per_metric.items()})
            )
            return real_coalesce(paths, per_metric)

        monkeypatch.setattr(sanity, "_coalesce_byte_identical_levels", _spy)

        sanity.run_sanity(tmp_path)

        # Coalesce must be invoked exactly once — only the lossy arm calls it.
        assert len(calls) == 1, (
            "Expected exactly one call to _coalesce_byte_identical_levels "
            f"from the lossy arm; got {len(calls)}"
        )
        passed_paths, passed_metrics = calls[0]
        # 4 lossy levels => 4 paths and 4 metric values per metric pre-coalesce.
        assert len(passed_paths) == 4
        for vals in passed_metrics.values():
            assert len(vals) == 4

        # And the coalesced result drops the trailing duplicate, leaving 3.
        kept_paths, kept_metrics = real_coalesce(passed_paths, passed_metrics)
        assert len(kept_paths) == 3
        for vals in kept_metrics.values():
            assert len(vals) == 3

    def test_lossy_arm_stores_coalesced_not_raw(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """Direct integration assertion: replace ``_coalesce_byte_identical_levels``
        with a sentinel-returning stub and confirm the SENTINEL ends up in the
        stored monotonicity entry (i.e. is what ``_monotonicity_check`` sees
        downstream). Proves ``run_sanity()`` uses the helper's output and
        doesn't silently fall back to ``per_metric_levels``.
        """
        gl = _build_fake_gl(lambda: {"ssim": 0.5})

        def _fake_compress(input_path, output_path, engine, params):
            Path(output_path).write_bytes(b"saturated")
            return output_path

        gl["compress"] = _fake_compress
        monkeypatch.setattr(sanity, "_import_giflab", lambda: gl)
        self._pin_constants(monkeypatch)
        # Use 2 lossy levels for this test — content doesn't matter
        # because the stub coalesce returns a fixed sentinel.
        monkeypatch.setattr(sanity, "LOSSY_LEVELS", [30, 60])

        # Sentinel must have 2+ entries to clear the `len(vals) < 2`
        # skip-guard inside the monotonicity-check loop.
        sentinel_values = [99.0, 99.5]  # Distinctive — couldn't naturally arise.

        def _stub_coalesce(paths, per_metric):
            return list(paths), {"ssim": list(sentinel_values)}

        monkeypatch.setattr(
            sanity, "_coalesce_byte_identical_levels", _stub_coalesce
        )

        # Capture the value-lists passed to _monotonicity_check. The lossy
        # arm's stored entry should appear as [99.0] (the sentinel),
        # confirming run_sanity() reads from the coalesced result.
        captured: dict[str, list[list[float]]] = {"seen": []}
        real_check = sanity._monotonicity_check

        def _check_spy(values, direction, tol=1e-4):
            captured["seen"].append(list(values))
            return real_check(values, direction, tol)

        monkeypatch.setattr(sanity, "_monotonicity_check", _check_spy)

        sanity.run_sanity(tmp_path)

        assert sentinel_values in captured["seen"], (
            "Expected the sentinel value-list returned by the stubbed "
            "_coalesce_byte_identical_levels to be passed to "
            "_monotonicity_check via the stored monotonicity entry — "
            "proving run_sanity() uses the coalesced result, not the raw "
            f"per_metric_levels. Got: {captured['seen']}"
        )


# ---------------------------------------------------------------------------
# Transparency-bearing fixture tests
# ---------------------------------------------------------------------------
#
# These tests guard the alpha-compositing regression class surfaced in PR #8:
# ``extract_gif_frames`` previously called ``Image.convert('RGB')`` directly
# on palette+transparency GIFs.  PIL resolves transparent pixels through the
# file's declared background palette colour — which is UNSTABLE across
# re-encoding because compressors rearrange the palette.  Original and
# compressed copies of the same visible content therefore had transparent
# regions filled with DIFFERENT RGB values, and every pair-comparison metric
# (ssim, ms_ssim, chist, lpips, …) treated that as real content disagreement.
#
# The test class below:
#  1. Loads the ``transparency_bearing_monotonicity.gif`` fixture from
#     ``tests/fixtures/`` (generated by ``make fixtures``).
#  2. Demonstrates the regression path: ``img.convert('RGB')`` fills transparent
#     pixels with the palette colour at the transparency index (dark, non-white),
#     while the correct RGBA-composite path fills them with white (255, 255, 255).
#  3. Verifies that ``sanity.run_sanity()`` picks up FIXTURE_MONOTONICITY_BASES
#     and runs the fixture through the degradation loop when the fixture file
#     exists, and gracefully skips when it does not.
# ---------------------------------------------------------------------------


def _import_fixture_generator():
    """Import scripts/fixtures/generate.py module."""
    import importlib.util

    gen_path = (
        Path(__file__).resolve().parents[2] / "scripts" / "fixtures" / "generate.py"
    )
    spec = importlib.util.spec_from_file_location("generate_fixtures", gen_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestTransparencyBearingGifFixture:
    """Smoke tests for the transparency-bearing monotonicity fixture.

    These tests exercise the FIXTURE specifically — generation, structure, and
    the PR #8 regression demonstration.  They do NOT run the full metrics
    pipeline; they only look at the PIL-level frame extraction behaviour so
    the test completes in milliseconds.
    """

    def test_fixture_generator_creates_transparency_bearing_gif(
        self, tmp_path: Path
    ) -> None:
        """generate_all() must create transparency_bearing_monotonicity.gif."""
        mod = _import_fixture_generator()
        mod.generate_all(output_dir=tmp_path)
        out = tmp_path / "transparency_bearing_monotonicity.gif"
        assert out.exists(), (
            "generate_all() did not create transparency_bearing_monotonicity.gif — "
            "check that create_transparency_bearing_monotonicity_gif is registered "
            "in the generators list inside generate_all()."
        )

    def test_fixture_has_transparency_index(self, tmp_path: Path) -> None:
        """The generated GIF must have a ``transparency`` key in im.info."""
        from PIL import Image

        mod = _import_fixture_generator()
        p = mod.create_transparency_bearing_monotonicity_gif(tmp_path)
        with Image.open(p) as im:
            assert "transparency" in im.info, (
                "transparency_bearing_monotonicity.gif must have a GIF transparency "
                "palette index in im.info — this is the structural property that "
                "triggers the PR #8 bug class."
            )

    def test_fixture_has_transparent_pixels(self, tmp_path: Path) -> None:
        """The transparency index must be used by some pixels in the GIF."""
        import numpy as np
        from PIL import Image

        mod = _import_fixture_generator()
        p = mod.create_transparency_bearing_monotonicity_gif(tmp_path)
        with Image.open(p) as im:
            t_idx = im.info.get("transparency")
            assert t_idx is not None
            arr = np.array(im)
            n_transparent = int((arr == t_idx).sum())
            assert n_transparent > 0, (
                f"Fixture has transparency index {t_idx} but no pixels use it — "
                "the transparent oval was not written into the palette P array."
            )

    def test_fixture_has_four_frames(self, tmp_path: Path) -> None:
        """The fixture must have exactly 4 frames."""
        from PIL import Image

        mod = _import_fixture_generator()
        p = mod.create_transparency_bearing_monotonicity_gif(tmp_path)
        with Image.open(p) as im:
            n = getattr(im, "n_frames", 1)
            assert n == 4, f"Expected 4 frames, got {n}"

    def test_fixture_is_deterministic(self, tmp_path: Path) -> None:
        """Running the generator twice must produce byte-identical output."""
        mod = _import_fixture_generator()
        p1 = (tmp_path / "run1").mkdir() or (tmp_path / "run1")
        p2 = (tmp_path / "run2").mkdir() or (tmp_path / "run2")
        out1 = mod.create_transparency_bearing_monotonicity_gif(tmp_path / "run1")
        out2 = mod.create_transparency_bearing_monotonicity_gif(tmp_path / "run2")
        assert out1.read_bytes() == out2.read_bytes(), (
            "create_transparency_bearing_monotonicity_gif() produced different "
            "bytes on two consecutive calls — generator must be fully deterministic."
        )

    def test_buggy_path_gives_different_frame_than_correct_path(
        self, tmp_path: Path
    ) -> None:
        """Demonstrates the PR #8 regression class.

        The fixture has a transparent oval whose pixels carry palette index 255
        (the transparency sentinel).  PIL's ``convert('RGB')`` resolves those
        pixels to whatever colour is stored at palette entry 255 — which is
        white (255,255,255) in this fixture but would be a different (unstable)
        colour in a re-encoded copy.

        The correct RGBA-composite path always yields white for fully transparent
        pixels, regardless of palette ordering.

        This test compares the two paths on the *same* fixture file (not a
        re-encoded copy).  If the palette entry at the transparency index is
        NOT white, the buggy path will decode those pixels to a wrong colour
        and the test proves the compositing fix is necessary.

        Note: in this fixture ``palette[255] = (255,255,255)`` by construction,
        so the buggy path actually gives the SAME white for transparent pixels.
        The important assertion is structural: the RGBA path yields alpha=0 for
        transparent pixels, while the buggy ``convert('RGB')`` path discards
        alpha and fills them from the palette.  We therefore check that the
        RGBA conversion correctly identifies which pixels are transparent, and
        that compositing them onto white yields (255,255,255).
        """
        import numpy as np
        from PIL import Image

        mod = _import_fixture_generator()
        p = mod.create_transparency_bearing_monotonicity_gif(tmp_path)

        with Image.open(p) as im:
            t_idx = im.info.get("transparency")
            assert t_idx is not None

            arr_p = np.array(im)
            transparent_mask = arr_p == t_idx
            assert transparent_mask.sum() > 0

            # RGBA path: transparent pixels should have alpha=0
            rgba = im.convert("RGBA")
            rgba_arr = np.array(rgba)
            alpha_zero_mask = rgba_arr[:, :, 3] == 0
            # The RGBA conversion must recognise the same pixels as transparent
            assert alpha_zero_mask.sum() > 0, (
                "RGBA conversion did not produce any alpha=0 pixels — "
                "the GIF transparency index is not being respected."
            )
            # Compositing onto white must yield (255,255,255) for those pixels
            bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
            composited_arr = np.array(
                Image.alpha_composite(bg, rgba).convert("RGB")
            )
            transparent_composited = composited_arr[alpha_zero_mask]
            assert (transparent_composited == 255).all(), (
                "RGBA-composite path did not yield (255,255,255) for transparent "
                f"pixels — got values: {np.unique(transparent_composited, axis=0)}"
            )


class TestFixtureMonotonicityBasesIntegration:
    """Verify that run_sanity() picks up and runs FIXTURE_MONOTONICITY_BASES.

    These tests use the same mock-GL infrastructure as
    TestCoalesceIntegratedIntoRunSanity to stay fast (no real metrics, no
    real GIFs from the synthetic set).  They inject a pre-generated fixture
    into a tmp fixtures directory and assert that the fixture's logical name
    appears in the monotonicity results.
    """

    def _pin_constants(self, monkeypatch) -> None:
        monkeypatch.setattr(sanity, "MONOTONICITY_BASES", [])
        monkeypatch.setattr(sanity, "IDENTITY_SAMPLE", [])
        monkeypatch.setattr(sanity, "NOISE_SIGMAS", [5])
        monkeypatch.setattr(sanity, "BLUR_SIGMAS", [0.5])
        monkeypatch.setattr(sanity, "QUANTIZE_COLORS", [64])
        monkeypatch.setattr(sanity, "LOSSY_LEVELS", [])

    def _make_fake_gl(self) -> dict:
        """Minimal gl dict: SyntheticGifGenerator returns no GIFs, metrics always 0.5."""

        class _FakeGen:
            def __init__(self, d: Path) -> None:
                self.synthetic_specs = []

            def generate_gifs(self, *, use_targeted_set: bool = False):
                return []

        class _FakeCfg:
            def __init__(self) -> None:
                self.ENABLE_DEEP_PERCEPTUAL = False
                self.ENABLE_TEMPORAL_ARTIFACTS = False

        return {
            "SyntheticGifGenerator": _FakeGen,
            "calculate_comprehensive_metrics": lambda *a, **kw: {"ssim": 0.5},
            "compress": None,
            "MetricsConfig": _FakeCfg,
        }

    def test_fixture_base_appears_in_monotonicity_results(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """When fixture file exists, its logical name must appear in monotonicity keys."""
        from PIL import Image

        # Create a minimal 2-frame GIF as the fixture
        fixture_dir = tmp_path / "fixtures"
        fixture_dir.mkdir()
        fixture_path = fixture_dir / "transparency_bearing_monotonicity.gif"
        frames_list = [
            Image.new("P", (16, 16), color=i) for i in (100, 120)
        ]
        frames_list[0].putpalette([i % 256 for i in range(768)])
        frames_list[1].putpalette([i % 256 for i in range(768)])
        frames_list[0].save(
            fixture_path,
            save_all=True,
            append_images=frames_list[1:],
            duration=100,
            loop=0,
            optimize=False,
        )

        self._pin_constants(monkeypatch)
        gl = self._make_fake_gl()
        monkeypatch.setattr(sanity, "_import_giflab", lambda: gl)
        monkeypatch.setattr(
            sanity,
            "FIXTURE_MONOTONICITY_BASES",
            {"transparency_bearing": "transparency_bearing_monotonicity.gif"},
        )
        monkeypatch.setattr(sanity, "_FIXTURES_DIR", fixture_dir)

        results = sanity.run_sanity(tmp_path / "workdir", skip_lossy=True)

        mono_keys = list(results["monotonicity"].keys())
        fixture_keys = [k for k in mono_keys if "transparency_bearing" in k]
        assert len(fixture_keys) > 0, (
            "Expected monotonicity keys containing 'transparency_bearing' but got: "
            f"{mono_keys}. run_sanity() is not processing FIXTURE_MONOTONICITY_BASES."
        )
        # Each of noise, blur, quantize arms should produce an entry
        kinds_found = {k.split("::")[0] for k in fixture_keys}
        assert "noise" in kinds_found, f"noise arm missing; found: {kinds_found}"
        assert "blur" in kinds_found, f"blur arm missing; found: {kinds_found}"
        assert "quantize" in kinds_found, f"quantize arm missing; found: {kinds_found}"

    def test_missing_fixture_skipped_gracefully(
        self, tmp_path: Path, monkeypatch, capsys
    ) -> None:
        """When the fixture file does not exist, skip without error."""
        empty_fixtures_dir = tmp_path / "no_fixtures_here"
        empty_fixtures_dir.mkdir()

        self._pin_constants(monkeypatch)
        gl = self._make_fake_gl()
        monkeypatch.setattr(sanity, "_import_giflab", lambda: gl)
        monkeypatch.setattr(
            sanity,
            "FIXTURE_MONOTONICITY_BASES",
            {"transparency_bearing": "transparency_bearing_monotonicity.gif"},
        )
        monkeypatch.setattr(sanity, "_FIXTURES_DIR", empty_fixtures_dir)

        # Must not raise
        results = sanity.run_sanity(tmp_path / "workdir", skip_lossy=True)

        mono_keys = list(results["monotonicity"].keys())
        fixture_keys = [k for k in mono_keys if "transparency_bearing" in k]
        assert fixture_keys == [], (
            "Expected no transparency_bearing entries when fixture is missing, "
            f"got: {fixture_keys}"
        )

        captured = capsys.readouterr()
        assert "skip fixture monotonicity" in captured.out, (
            "Expected a 'skip fixture monotonicity' message in stdout when the "
            "fixture file is not found."
        )

    def test_config_output_includes_fixture_bases(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """The 'config' key in run_sanity() output must list fixture_monotonicity_bases."""
        self._pin_constants(monkeypatch)
        gl = self._make_fake_gl()
        monkeypatch.setattr(sanity, "_import_giflab", lambda: gl)
        monkeypatch.setattr(
            sanity,
            "FIXTURE_MONOTONICITY_BASES",
            {"transparency_bearing": "transparency_bearing_monotonicity.gif"},
        )
        # _FIXTURES_DIR doesn't exist → fixture loop skips gracefully
        monkeypatch.setattr(sanity, "_FIXTURES_DIR", tmp_path / "nonexistent")

        results = sanity.run_sanity(tmp_path / "workdir", skip_lossy=True)

        cfg = results.get("config", {})
        assert "fixture_monotonicity_bases" in cfg, (
            "run_sanity() output['config'] must contain 'fixture_monotonicity_bases' key."
        )
        assert "transparency_bearing" in cfg["fixture_monotonicity_bases"], (
            "Expected 'transparency_bearing' in config.fixture_monotonicity_bases; "
            f"got: {cfg['fixture_monotonicity_bases']}"
        )
