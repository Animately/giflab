"""Integration tests for targeted experiment system.

This module tests the end-to-end functionality of the targeted preset system
including CLI integration, real pipeline execution, and performance comparisons.
"""

import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from giflab.core.runner import GifLabRunner
from giflab.core.targeted_generator import TargetedPipelineGenerator
from giflab.core.targeted_presets import (
    PRESET_REGISTRY,
    ExperimentPreset,
    SlotConfiguration,
)
from giflab.core.builtin_presets import register_builtin_presets  # noqa: F401 - triggers registration
from giflab.dynamic_pipeline import generate_all_pipelines


class TestEndToEndPresetExecution:
    """Test complete preset experiment execution workflow."""

    def test_experimental_runner_preset_integration(self):
        """Test GifLabRunner integration with presets."""

        with tempfile.TemporaryDirectory() as temp_dir:
            runner = GifLabRunner(output_dir=Path(temp_dir), use_cache=False)

            # Test preset listing
            presets = runner.list_available_presets()
            assert len(presets) > 0
            assert "frame-focus" in presets

            # Test preset pipeline generation
            pipelines = runner.generate_targeted_pipelines("frame-focus")
            assert len(pipelines) > 0
            assert len(pipelines) < 50  # Should be much smaller than full generation

    def test_run_targeted_experiment_mock_execution(self):
        """Test targeted experiment execution with mocked pipeline runs."""

        with tempfile.TemporaryDirectory() as temp_dir:
            runner = GifLabRunner(output_dir=Path(temp_dir), use_cache=False)

            # Mock the actual pipeline execution to avoid long test times
            with patch.object(runner, "run_analysis") as mock_analysis:
                mock_result = MagicMock()
                mock_result.total_jobs_run = 5
                mock_result.eliminated_pipelines = set()
                mock_result.retained_pipelines = {"test_pipeline"}
                mock_analysis.return_value = mock_result

                # Test targeted experiment execution
                result = runner.run_targeted_experiment(
                    "frame-focus", quality_threshold=0.1
                )

                # Verify the method was called with targeted pipelines
                mock_analysis.assert_called_once()
                call_args = mock_analysis.call_args
                assert "test_pipelines" in call_args.kwargs
                assert len(call_args.kwargs["test_pipelines"]) > 0
                assert result.total_jobs_run > 0

    def test_all_builtin_presets_execution(self):
        """Test that all built-in presets can execute without errors."""

        with tempfile.TemporaryDirectory() as temp_dir:
            runner = GifLabRunner(output_dir=Path(temp_dir), use_cache=False)

            # Mock execution to speed up tests
            with patch.object(runner, "run_analysis") as mock_analysis:
                mock_result = MagicMock()
                mock_result.total_jobs_run = 1
                mock_result.eliminated_pipelines = set()
                mock_result.retained_pipelines = {"test"}
                mock_analysis.return_value = mock_result

                # Test each preset
                failed_presets = []
                for preset_id in runner.list_available_presets().keys():
                    try:
                        pipelines = runner.generate_targeted_pipelines(preset_id)
                        assert (
                            len(pipelines) > 0
                        ), f"Preset {preset_id} generated no pipelines"

                        # Test execution (mocked)
                        result = runner.run_targeted_experiment(
                            preset_id, quality_threshold=0.1
                        )
                        assert result.total_jobs_run >= 0

                    except Exception as e:
                        failed_presets.append(f"{preset_id}: {str(e)}")

                assert not failed_presets, f"Failed presets: {failed_presets}"


class TestPresetTypeValidation:
    """Test validation of different preset types with real pipeline data."""

    def test_single_variable_presets(self):
        """Test presets that vary only one dimension."""

        generator = TargetedPipelineGenerator()

        # Test frame-focus preset
        if "frame-focus" in PRESET_REGISTRY.list_presets():
            preset = PRESET_REGISTRY.get("frame-focus")
            pipelines = generator.generate_targeted_pipelines(preset)

            # Should vary only frame tools
            frame_tools = set()
            color_tools = set()
            lossy_tools = set()

            for pipeline in pipelines:
                for step in pipeline.steps:
                    if step.variable == "frame_reduction":
                        frame_tools.add(step.tool_cls.NAME)
                    elif step.variable == "color_reduction":
                        color_tools.add(step.tool_cls.NAME)
                    elif step.variable == "lossy_compression":
                        lossy_tools.add(step.tool_cls.NAME)

            assert len(frame_tools) > 1, "Should vary frame tools"
            assert len(color_tools) == 1, "Should use single color tool"
            assert len(lossy_tools) == 1, "Should use single lossy tool"

    def test_multi_variable_presets(self):
        """Test presets that vary multiple dimensions."""

        generator = TargetedPipelineGenerator()

        # Test tool-comparison-baseline preset
        if "tool-comparison-baseline" in PRESET_REGISTRY.list_presets():
            preset = PRESET_REGISTRY.get("tool-comparison-baseline")
            pipelines = generator.generate_targeted_pipelines(preset)

            # Should vary multiple tools
            frame_tools = set()
            color_tools = set()
            lossy_tools = set()

            for pipeline in pipelines:
                for step in pipeline.steps:
                    if step.variable == "frame_reduction":
                        frame_tools.add(step.tool_cls.NAME)
                    elif step.variable == "color_reduction":
                        color_tools.add(step.tool_cls.NAME)
                    elif step.variable == "lossy_compression":
                        lossy_tools.add(step.tool_cls.NAME)

            # Should vary at least 2 dimensions
            varied_dimensions = sum(
                [len(frame_tools) > 1, len(color_tools) > 1, len(lossy_tools) > 1]
            )
            assert varied_dimensions >= 2, "Should vary at least 2 dimensions"

    def test_specialized_presets(self):
        """Test specialized presets like dithering-focus."""

        generator = TargetedPipelineGenerator()

        # Test dithering-focus preset
        if "dithering-focus" in PRESET_REGISTRY.list_presets():
            preset = PRESET_REGISTRY.get("dithering-focus")
            pipelines = generator.generate_targeted_pipelines(preset)

            # Should focus on specific dithering methods
            color_tools = set()
            for pipeline in pipelines:
                for step in pipeline.steps:
                    if step.variable == "color_reduction":
                        color_tools.add(step.tool_cls.NAME)

            # Should have specific dithering-related tools
            dithering_tools = [
                name
                for name in color_tools
                if any(
                    keyword in name.lower()
                    for keyword in ["floyd", "bayer", "riemersma"]
                )
            ]
            assert len(dithering_tools) > 0, "Should include dithering-specific tools"

    def test_development_presets(self):
        """Test development/debug presets like quick-test."""

        generator = TargetedPipelineGenerator()

        # Test quick-test preset
        if "quick-test" in PRESET_REGISTRY.list_presets():
            preset = PRESET_REGISTRY.get("quick-test")
            pipelines = generator.generate_targeted_pipelines(preset)

            # Should generate very few pipelines for quick testing
            assert len(pipelines) <= 5, "Quick test should have very few pipelines"
            assert len(pipelines) > 0, "Should still generate at least one pipeline"


# TestCLIIntegration class removed - custom preset CLI functionality
# was lost during pipeline consolidation refactor


class TestPerformanceComparison:
    """Test performance comparison between targeted and traditional approaches."""

    def test_generation_time_comparison(self):
        """Compare pipeline generation time between approaches."""

        runner = GifLabRunner(use_cache=False)

        # Time traditional approach
        start_time = time.time()
        all_pipelines = generate_all_pipelines()
        runner.select_pipelines_intelligently(all_pipelines, "quick")
        traditional_time = time.time() - start_time

        # Time targeted approach
        start_time = time.time()
        runner.generate_targeted_pipelines("frame-focus")
        targeted_time = time.time() - start_time

        # Targeted should be faster (though this may vary by system)
        print(
            f"Traditional time: {traditional_time:.3f}s, Targeted time: {targeted_time:.3f}s"
        )

        # Main assertion: both should complete in reasonable time
        assert traditional_time < 10.0, "Traditional approach taking too long"
        assert targeted_time < 5.0, "Targeted approach taking too long"

    def test_memory_usage_estimation(self):
        """Test memory usage estimation for different approaches."""

        runner = GifLabRunner(use_cache=False)

        # Traditional approach creates all pipelines first
        all_pipelines = generate_all_pipelines()
        traditional_memory = len(all_pipelines)

        # Targeted approach creates only needed pipelines
        targeted_pipelines = runner.generate_targeted_pipelines("frame-focus")
        targeted_memory = len(targeted_pipelines)

        # Targeted should use much less memory
        memory_ratio = targeted_memory / traditional_memory
        assert (
            memory_ratio <= 0.2
        ), f"Targeted approach should use ≤20% memory (actual: {memory_ratio:.1%})"
        print(f"Memory efficiency: {memory_ratio:.1%} of traditional approach")

    def test_efficiency_gains_across_presets(self):
        """Test efficiency gains across all built-in presets."""

        runner = GifLabRunner(use_cache=False)
        generator = TargetedPipelineGenerator()
        len(generate_all_pipelines())

        efficiency_gains = {}

        for preset_id in runner.list_available_presets().keys():
            try:
                preset = PRESET_REGISTRY.get(preset_id)
                validation = generator.validate_preset_feasibility(preset)

                if validation["valid"]:
                    efficiency_gains[preset_id] = validation["efficiency_gain"]
            except Exception as e:
                print(f"Warning: Could not test efficiency for {preset_id}: {e}")

        # Most presets should achieve significant efficiency gains
        high_efficiency_presets = [
            pid for pid, gain in efficiency_gains.items() if gain > 0.8
        ]
        assert (
            len(high_efficiency_presets) >= len(efficiency_gains) // 2
        ), "At least half of presets should have >80% efficiency gain"

        # Print results for inspection
        print("\nEfficiency gains by preset:")
        for preset_id, gain in sorted(
            efficiency_gains.items(), key=lambda x: x[1], reverse=True
        ):
            print(f"  {preset_id}: {gain:.1%}")


class TestSamplingStrategyIntegration:
    """Test integration with existing sampling strategies."""


class TestErrorHandling:
    """Test error handling in integration scenarios."""

    def test_invalid_preset_id_handling(self):
        """Test handling of invalid preset IDs."""
        runner = GifLabRunner(use_cache=False)

        with pytest.raises(ValueError, match="Unknown preset"):
            runner.generate_targeted_pipelines("nonexistent-preset")

    def test_tool_unavailability_handling(self):
        """Test handling when required tools are not available."""
        # Create preset with nonexistent tools
        invalid_preset = ExperimentPreset(
            name="Invalid Tools",
            description="Test invalid tools",
            frame_slot=SlotConfiguration(type="variable", scope=["nonexistent-tool"]),
            color_slot=SlotConfiguration(type="locked", implementation="ffmpeg-color"),
            lossy_slot=SlotConfiguration(type="locked", implementation="none-lossy"),
        )

        generator = TargetedPipelineGenerator()

        # Should fail validation
        validation = generator.validate_preset_feasibility(invalid_preset)
        assert not validation["valid"]
        assert len(validation["errors"]) > 0

        # Should fail generation
        with pytest.raises(RuntimeError):
            generator.generate_targeted_pipelines(invalid_preset)

    def test_empty_pipeline_result_handling(self):
        """Test handling when preset results in no valid pipelines."""
        # This is tested indirectly through tool unavailability
        # since empty results typically occur due to invalid tool configurations
        pass

    # test_malformed_cli_arguments_handling removed - custom preset CLI functionality
    # was lost during pipeline consolidation refactor
    pass


class TestRegressionAndCompatibility:
    """Test regression and compatibility with existing systems."""

    def test_pipeline_structure_compatibility(self):
        """Test that generated pipelines are compatible with existing systems."""

        runner = GifLabRunner(use_cache=False)

        # Generate both traditional and targeted pipelines
        traditional = generate_all_pipelines()
        targeted = runner.generate_targeted_pipelines("frame-focus")

        # Both should have same structure
        if traditional and targeted:
            trad_pipeline = traditional[0]
            targ_pipeline = targeted[0]

            # Same attributes
            assert hasattr(trad_pipeline, "steps")
            assert hasattr(targ_pipeline, "steps")
            assert hasattr(trad_pipeline, "identifier")
            assert hasattr(targ_pipeline, "identifier")

            # Same step structure
            assert len(trad_pipeline.steps) == len(targ_pipeline.steps) == 3

            for trad_step, targ_step in zip(trad_pipeline.steps, targ_pipeline.steps):
                assert hasattr(trad_step, "variable")
                assert hasattr(targ_step, "variable")
                assert hasattr(trad_step, "tool_cls")
                assert hasattr(targ_step, "tool_cls")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
