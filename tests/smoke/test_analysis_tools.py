import pandas as pd
from giflab.analysis_tools import (
    performance_matrix,
    pipeline_to_mermaid,
    recommend_tools,
)
from giflab.dynamic_pipeline import generate_all_pipelines


def _fake_df() -> pd.DataFrame:
    data = {
        "variable": [
            "frame_reduction",
            "frame_reduction",
            "color_reduction",
            "color_reduction",
            "lossy_compression",
            "lossy_compression",
        ],
        "engine": [
            "gifsicle",
            "animately",
            "gifsicle",
            "imagemagick",
            "gifsicle",
            "ffmpeg",
        ],
        "ssim": [0.9, 0.85, 0.92, 0.88, 0.8, 0.82],
    }
    return pd.DataFrame(data)


def test_performance_matrix():
    df = _fake_df()
    mats = performance_matrix(df, metric="ssim")
    assert "color_reduction" in mats
    assert mats["color_reduction"].loc["gifsicle"] == 0.92


def test_recommend_tools():
    df = _fake_df()
    rec = recommend_tools(df, metric="ssim", top_n=1)
    assert rec["frame_reduction"] == ["gifsicle"]


def test_pipeline_to_mermaid_format():
    pipeline = generate_all_pipelines()[0]
    dsl = pipeline_to_mermaid(pipeline)
    assert "flowchart LR" in dsl
    # Ensure all steps appear in DSL
    for step in pipeline.steps:
        assert step.tool_cls.NAME in dsl
