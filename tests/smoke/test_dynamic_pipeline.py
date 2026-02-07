from pathlib import Path

import pytest
from giflab import tools_for_variable
from giflab.dynamic_pipeline import generate_all_pipelines
from PIL import Image

# ---------------------------------------------------------------------------
# Helper to create a tiny 2-frame GIF for quick tests
# ---------------------------------------------------------------------------


def _make_test_gif(tmp_path: Path) -> Path:
    path = tmp_path / "test.gif"
    img1 = Image.new("RGB", (20, 20), color="red")
    img2 = Image.new("RGB", (20, 20), color="blue")
    img1.save(path, save_all=True, append_images=[img2], duration=100, loop=0)
    return path


# ---------------------------------------------------------------------------
# Variable-isolation tests
# ---------------------------------------------------------------------------


@pytest.mark.fast
@pytest.mark.parametrize(
    "variable",
    [
        "frame_reduction",
        "color_reduction",
        "lossy_compression",
    ],
)
def test_at_least_one_tool_available(variable):
    wrappers = tools_for_variable(variable)
    assert wrappers, f"No wrappers found for variable={variable}"


@pytest.mark.fast
@pytest.mark.parametrize(
    "variable,params",
    [
        ("frame_reduction", {"ratio": 1.0}),
        ("color_reduction", {"colors": 256}),
        ("lossy_compression", {"lossy_level": 0}),
    ],
)
def test_noop_wrapper_apply_succeeds(variable, params, tmp_path):
    # Pick the always-available no-operation wrapper for this variable
    wrappers = [w for w in tools_for_variable(variable) if w.NAME.startswith("none")]
    assert wrappers, f"No no-op wrapper for {variable}"
    wrapper_cls = wrappers[0]
    wrapper = wrapper_cls()

    input_gif = _make_test_gif(tmp_path)
    output_gif = tmp_path / f"out_{variable}.gif"

    meta = wrapper.apply(input_gif, output_gif, params=params)
    assert output_gif.exists(), "Output GIF not created"
    # Render time metadata must be non-negative
    assert meta["render_ms"] >= 0


# ---------------------------------------------------------------------------
# Dynamic pipeline generation tests
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_generate_all_pipelines_non_empty():
    pipelines = generate_all_pipelines()
    assert pipelines, "No pipelines generated"


@pytest.mark.fast
def test_pipeline_identifiers_unique():
    pipelines = generate_all_pipelines()
    ids = [p.identifier() for p in pipelines]
    assert len(ids) == len(set(ids)), "Pipeline identifiers are not unique"


@pytest.mark.fast
def test_collapsed_steps_order_and_length():
    pipelines = generate_all_pipelines()
    assert len(pipelines) > 0, "Should generate at least one pipeline"
    for p in pipelines:
        # Steps must be in non-decreasing variable order according to spec
        order_map = {
            "frame_reduction": 0,
            "color_reduction": 1,
            "lossy_compression": 2,
        }
        indices = [order_map[s.variable] for s in p.steps]
        assert indices == sorted(indices), "Steps out of order"
        assert 1 <= len(p.steps) <= 3, "Collapsed step count invalid"
