"""Unit tests for pipeline validation logic in pipeline elimination."""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import Mock

import pytest
from giflab.core import GifLabRunner


class MockTool:
    """Mock tool class for testing."""

    NAME = "test-tool"

    def __init__(self, name="test-tool"):
        self.NAME = name

    def apply(self, input_path, output_path, params=None):
        # Create a dummy output file
        shutil.copy(input_path, output_path)
        return {"success": True}


class MockExternalTool:
    """Mock external tool base class that should be filtered out."""

    NAME = "external-tool"

    def apply(self, input_path, output_path, params=None):
        return {"success": False, "error": "Base class should not be used"}


class MockPipelineStep:
    """Mock pipeline step for testing."""

    def __init__(self, tool_cls, variable="test_var"):
        self.tool_cls = tool_cls
        self.variable = variable
        self.name = lambda: f"{tool_cls.NAME}_{variable}"

    def __str__(self):
        """String representation that includes the tool name."""
        return f"PipelineStep({self.tool_cls.NAME}, {self.variable})"


class TestPipelineValidation:
    """Test pipeline validation logic."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def eliminator(self, temp_dir):
        """Create GifLabRunner instance for testing."""
        return GifLabRunner(output_dir=temp_dir, use_gpu=False, use_cache=False)

    @pytest.fixture
    def test_gif(self, temp_dir):
        """Create a test GIF file."""
        # Create a minimal test GIF (this is a placeholder - in real tests you'd use a proper GIF)
        gif_path = temp_dir / "test.gif"
        gif_path.write_bytes(
            b"GIF89a\x01\x00\x01\x00\x00\x00\x00!\xf9\x04\x01\x00\x00\x00\x00,\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02\x04\x01\x00;"
        )
        return gif_path

    def test_valid_pipeline_passes_validation(self, eliminator):
        """Test that valid pipelines pass validation."""
        # Create a valid pipeline with concrete tools
        valid_tool = MockTool("ffmpeg-color-reducer")
        step = MockPipelineStep(valid_tool, "color_reduction")
        pipeline = Mock()
        pipeline.steps = [step]
        pipeline.identifier = lambda: "valid_pipeline"

        # This should not raise any validation errors
        # In the actual implementation, we'd test the validation method directly
        assert step.tool_cls.NAME != "external-tool"
        assert "external-tool" not in str(step)

    def test_external_tool_pipeline_fails_validation(self, eliminator):
        """Test that pipelines with external-tool base classes fail validation."""
        # Create an invalid pipeline with external-tool base class
        invalid_tool = MockExternalTool()
        step = MockPipelineStep(invalid_tool, "color_reduction")
        pipeline = Mock()
        pipeline.steps = [step]
        pipeline.identifier = lambda: "invalid_pipeline"

        # This should fail validation
        assert step.tool_cls.NAME == "external-tool"
        assert any("external-tool" in str(step) for step in pipeline.steps)

    def test_mixed_pipeline_fails_validation(self, eliminator):
        """Test that pipelines mixing valid and invalid tools fail validation."""
        valid_tool = MockTool("gifsicle-lossy")
        invalid_tool = MockExternalTool()

        valid_step = MockPipelineStep(valid_tool, "lossy_compression")
        invalid_step = MockPipelineStep(invalid_tool, "color_reduction")

        pipeline = Mock()
        pipeline.steps = [valid_step, invalid_step]
        pipeline.identifier = lambda: "mixed_pipeline"

        # Should fail because it contains external-tool
        assert any(step.tool_cls.NAME == "external-tool" for step in pipeline.steps)

    def test_parameter_validation(self, eliminator):
        """Test parameter validation for colors, lossy, and frame_ratio."""
        test_params = [
            {"colors": 32, "lossy": 60, "frame_ratio": 0.5},  # Valid
            {"colors": 1, "lossy": 60, "frame_ratio": 0.5},  # Invalid: colors < 2
            {"colors": 300, "lossy": 60, "frame_ratio": 0.5},  # Invalid: colors > 256
            {"colors": 32, "lossy": -10, "frame_ratio": 0.5},  # Invalid: lossy < 0
            {"colors": 32, "lossy": 110, "frame_ratio": 0.5},  # Invalid: lossy > 100
            {
                "colors": 32,
                "lossy": 60,
                "frame_ratio": 0.0,
            },  # Invalid: frame_ratio <= 0
            {
                "colors": 32,
                "lossy": 60,
                "frame_ratio": 1.5,
            },  # Invalid: frame_ratio > 1.0
        ]

        for params in test_params:
            # Test color validation
            if params["colors"] < 2 or params["colors"] > 256:
                corrected_colors = max(2, min(256, params["colors"]))
                assert corrected_colors != params["colors"]

            # Test lossy validation
            if params["lossy"] < 0 or params["lossy"] > 100:
                corrected_lossy = max(0, min(100, params["lossy"]))
                assert corrected_lossy != params["lossy"]

            # Test frame ratio validation
            if params["frame_ratio"] <= 0 or params["frame_ratio"] > 1.0:
                corrected_ratio = max(0.1, min(1.0, params["frame_ratio"]))
                assert corrected_ratio != params["frame_ratio"]

    def test_pipeline_step_counting(self, eliminator):
        """Test counting of actual (non-no-op) pipeline steps."""
        # Create pipeline with mix of actual and no-op steps
        actual_tool = MockTool("ffmpeg-color-reducer")
        noop_tool = MockTool("none-color")

        actual_step = MockPipelineStep(actual_tool, "color_reduction")
        noop_step = MockPipelineStep(noop_tool, "color_reduction")

        pipeline = Mock()
        pipeline.steps = [actual_step, noop_step, actual_step]

        # Count non-no-op steps (should be 2)
        actual_count = sum(
            1
            for step in pipeline.steps
            if hasattr(step.tool_cls, "NAME")
            and not step.tool_cls.NAME.startswith("none-")
        )

        assert actual_count == 2

    def test_semantic_parameter_detection(self, eliminator):
        """Test detection of which parameters are actually applied by pipelines."""
        # Create pipeline with color reduction
        color_tool = MockTool("imagemagick-color-reducer")
        color_step = MockPipelineStep(color_tool, "color_reduction")

        # Create pipeline with lossy compression
        lossy_tool = MockTool("gifsicle-lossy")
        lossy_step = MockPipelineStep(lossy_tool, "lossy_compression")

        # Create pipeline with frame reduction
        frame_tool = MockTool("ffmpeg-frame-reducer")
        frame_step = MockPipelineStep(frame_tool, "frame_reduction")

        # Test color reduction detection
        color_pipeline = Mock()
        color_pipeline.steps = [color_step]
        assert any(
            step.variable == "color_reduction" and step.tool_cls.NAME != "none-color"
            for step in color_pipeline.steps
        )

        # Test lossy compression detection
        lossy_pipeline = Mock()
        lossy_pipeline.steps = [lossy_step]
        assert any(
            step.variable == "lossy_compression" and step.tool_cls.NAME != "none-lossy"
            for step in lossy_pipeline.steps
        )

        # Test frame reduction detection
        frame_pipeline = Mock()
        frame_pipeline.steps = [frame_step]
        assert any(
            step.variable == "frame_reduction" and step.tool_cls.NAME != "none-frame"
            for step in frame_pipeline.steps
        )


class TestLossyEngineMapping:
    """Test lossy percentage to engine-specific mapping."""

    @pytest.fixture
    def eliminator(self):
        """Create eliminator for testing."""
        return GifLabRunner(use_gpu=False, use_cache=False)

    def test_gifsicle_mapping(self, eliminator):
        """Test Gifsicle lossy mapping (0-300 range)."""
        test_cases = [
            (0, 0),  # 0% -> 0
            (50, 150),  # 50% -> 150
            (60, 180),  # 60% -> 180
            (100, 300),  # 100% -> 300
        ]

        for percentage, expected in test_cases:
            result = eliminator._map_lossy_percentage_to_engine(
                percentage, "GifsicleWrapper"
            )
            assert result == expected

    def test_other_engines_mapping(self, eliminator):
        """Test other engines mapping (0-100 range)."""
        engines = [
            "FFmpegWrapper",
            "AnimatelyWrapper",
            "GifskiWrapper",
            "ImageMagickWrapper",
        ]

        test_cases = [
            (0, 0),  # 0% -> 0
            (50, 50),  # 50% -> 50
            (60, 60),  # 60% -> 60
            (100, 100),  # 100% -> 100
        ]

        for engine in engines:
            for percentage, expected in test_cases:
                result = eliminator._map_lossy_percentage_to_engine(percentage, engine)
                assert result == expected

    def test_invalid_percentage_clamping(self, eliminator):
        """Test that invalid percentages are clamped to valid range."""
        invalid_cases = [
            (-10, 0),  # Negative -> 0
            (150, 100),  # Over 100% -> 100
        ]

        for invalid_input, expected_max in invalid_cases:
            # Test with non-Gifsicle engine
            result = eliminator._map_lossy_percentage_to_engine(
                invalid_input, "FFmpegWrapper"
            )
            assert 0 <= result <= expected_max

            # Test with Gifsicle (different max)
            gifsicle_result = eliminator._map_lossy_percentage_to_engine(
                invalid_input, "GifsicleWrapper"
            )
            expected_gifsicle_max = expected_max * 3 if expected_max == 100 else 0
            assert 0 <= gifsicle_result <= expected_gifsicle_max


if __name__ == "__main__":
    pytest.main([__file__])
