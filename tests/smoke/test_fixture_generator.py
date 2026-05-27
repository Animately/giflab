"""Smoke tests for the fixture generator contract.

These tests lock the contract that `make fixtures` (or
`poetry run python scripts/fixtures/generate.py`) produces every
gitignored fixture deterministically.  They run without any real
compression binary and complete in well under 5 s.
"""

from pathlib import Path

import pytest
from PIL import Image

# ---------------------------------------------------------------------------
# Constants — the canonical list of fixtures the generator must produce.
# Any new fixture added to the generator MUST also appear here.
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"

#: GIFs produced by generate_fixtures.py (the original engine-testing set)
ENGINE_FIXTURES = [
    "simple_4frame.gif",
    "single_frame.gif",
    "many_colors.gif",
]

#: GIFs produced by create_test_fixtures.py (the validation / wrapper set)
VALIDATION_FIXTURES = [
    "test_10_frames.gif",
    "test_4_frames.gif",
    "test_256_colors.gif",
    "test_2_colors.gif",
    "test_30_frames.gif",
]

#: GIFs produced by generate_temporal_artifact_fixtures.py
TEMPORAL_FIXTURES = [
    "flicker_high.gif",
    "flicker_low.gif",
    "background_stable.gif",
    "background_flickering.gif",
    "pumping_yes.gif",
    "pumping_no.gif",
    "disposal_corrupted.gif",
    "disposal_clean.gif",
    "smooth_animation.gif",
    "static_with_noise.gif",
]

#: Complete set — union of all sub-generators
ALL_EXPECTED_FIXTURES: list[str] = (
    ENGINE_FIXTURES + VALIDATION_FIXTURES + TEMPORAL_FIXTURES
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _import_generator():
    """Import the canonical generator module; skip if not yet created."""
    import importlib.util

    generator_path = (
        Path(__file__).parent.parent.parent / "scripts" / "fixtures" / "generate.py"
    )
    if not generator_path.exists():
        pytest.skip(
            f"Generator script not found at {generator_path}; "
            "run `make fixtures` after merging this PR."
        )
    spec = importlib.util.spec_from_file_location("generate_fixtures", generator_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# RED-phase tests — these must fail before the generator exists
# ---------------------------------------------------------------------------


class TestGeneratorModuleContract:
    """The generator module must expose a callable `generate_all(output_dir)`."""

    def test_module_has_generate_all_function(self):
        mod = _import_generator()
        assert callable(
            getattr(mod, "generate_all", None)
        ), "scripts/fixtures/generate.py must expose a `generate_all(output_dir)` function"

    def test_generate_all_accepts_output_dir(self):
        import inspect

        mod = _import_generator()
        sig = inspect.signature(mod.generate_all)
        params = list(sig.parameters)
        assert "output_dir" in params, (
            "`generate_all` must accept an `output_dir` parameter "
            "so tests can redirect output to a tmp dir"
        )


class TestAllFixturesGenerated:
    """Running generate_all must produce every expected fixture file."""

    def test_all_expected_fixtures_created(self, tmp_path):
        mod = _import_generator()
        mod.generate_all(output_dir=tmp_path)

        missing = [f for f in ALL_EXPECTED_FIXTURES if not (tmp_path / f).exists()]
        assert missing == [], (
            f"generate_all() did not produce: {missing}\n"
            "Add the missing fixture generators to scripts/fixtures/generate.py"
        )

    def test_all_outputs_are_valid_gifs(self, tmp_path):
        """Every generated file must be a real, openable GIF."""
        mod = _import_generator()
        mod.generate_all(output_dir=tmp_path)

        bad = []
        for name in ALL_EXPECTED_FIXTURES:
            path = tmp_path / name
            try:
                with Image.open(path) as img:
                    if img.format != "GIF":
                        bad.append(f"{name}: format={img.format}")
            except Exception as exc:
                bad.append(f"{name}: {exc}")

        assert bad == [], f"Invalid GIF outputs:\n" + "\n".join(bad)


class TestIdempotency:
    """Running generate_all twice must produce the same byte-identical files.

    Byte identity guarantees the generator is deterministic — no random seeds
    that differ between runs, no timestamps embedded in headers.
    """

    def test_generate_all_is_idempotent(self, tmp_path):
        mod = _import_generator()

        out1 = tmp_path / "run1"
        out2 = tmp_path / "run2"
        out1.mkdir()
        out2.mkdir()

        mod.generate_all(output_dir=out1)
        mod.generate_all(output_dir=out2)

        mismatches = []
        for name in ALL_EXPECTED_FIXTURES:
            f1 = out1 / name
            f2 = out2 / name
            if not f1.exists() or not f2.exists():
                continue  # covered by TestAllFixturesGenerated
            if f1.read_bytes() != f2.read_bytes():
                mismatches.append(name)

        assert mismatches == [], (
            f"Non-deterministic output (files differ between runs): {mismatches}\n"
            "Ensure all PIL Image.save() calls use fixed seeds / no random state."
        )


class TestFixtureProperties:
    """Spot-check key property invariants so tests that depend on them stay valid."""

    @pytest.mark.parametrize(
        "name,expected_frames",
        [
            ("simple_4frame.gif", 4),
            ("single_frame.gif", 1),
            ("test_10_frames.gif", 10),
            ("test_4_frames.gif", 4),
            ("test_30_frames.gif", 30),
        ],
    )
    def test_frame_counts(self, tmp_path, name, expected_frames):
        mod = _import_generator()
        mod.generate_all(output_dir=tmp_path)

        with Image.open(tmp_path / name) as img:
            n_frames = getattr(img, "n_frames", 1)

        assert n_frames == expected_frames, (
            f"{name}: expected {expected_frames} frames, got {n_frames}"
        )
