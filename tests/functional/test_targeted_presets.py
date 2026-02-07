"""Unit tests for the targeted experiment preset system.

This module tests the new slot-based preset system that replaces
generate_all_pipelines() + sampling with targeted pipeline generation.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from giflab.core.targeted_generator import TargetedPipelineGenerator
from giflab.core.targeted_presets import (
    PRESET_REGISTRY,
    ExperimentPreset,
    PresetRegistry,
    SlotConfiguration,
)

# Import builtin_presets to trigger auto-registration of presets into PRESET_REGISTRY
import giflab.core.builtin_presets  # noqa: F401


class TestSlotConfiguration:
    """Test SlotConfiguration validation and functionality."""

    def test_valid_variable_slot_creation(self):
        """Test creating a valid variable slot succeeds."""
        slot = SlotConfiguration(
            type="variable", scope=["*"], parameters={"ratios": [1.0, 0.8, 0.5]}
        )
        assert slot.type == "variable"
        assert slot.scope == ["*"]
        assert slot.implementation is None
        assert slot.parameters == {"ratios": [1.0, 0.8, 0.5]}

    def test_valid_locked_slot_creation(self):
        """Test creating a valid locked slot succeeds."""
        slot = SlotConfiguration(
            type="locked", implementation="ffmpeg-color", parameters={"colors": 32}
        )
        assert slot.type == "locked"
        assert slot.implementation == "ffmpeg-color"
        assert slot.scope is None
        assert slot.parameters == {"colors": 32}

    def test_invalid_slot_type(self):
        """Test that invalid slot type raises ValueError."""
        with pytest.raises(ValueError, match="Invalid slot type: invalid"):
            SlotConfiguration(type="invalid", implementation="test")


class TestExperimentPreset:
    """Test ExperimentPreset validation and functionality."""

    def test_valid_preset_creation(self):
        """Test creating a valid preset succeeds."""
        preset = ExperimentPreset(
            name="Test Preset",
            description="Test description",
            frame_slot=SlotConfiguration(type="variable", scope=["*"]),
            color_slot=SlotConfiguration(type="locked", implementation="ffmpeg-color"),
            lossy_slot=SlotConfiguration(type="locked", implementation="none-lossy"),
        )
        assert preset.name == "Test Preset"
        assert preset.description == "Test description"
        assert preset.frame_slot.type == "variable"
        assert preset.color_slot.type == "locked"
        assert preset.lossy_slot.type == "locked"

    def test_preset_with_all_slots_locked(self):
        """Test that preset with all locked slots raises ValueError."""
        with pytest.raises(ValueError, match="At least one slot must be variable"):
            ExperimentPreset(
                name="All Locked",
                description="Invalid preset",
                frame_slot=SlotConfiguration(
                    type="locked", implementation="animately-frame"
                ),
                color_slot=SlotConfiguration(
                    type="locked", implementation="ffmpeg-color"
                ),
                lossy_slot=SlotConfiguration(
                    type="locked", implementation="none-lossy"
                ),
            )

    def test_get_variable_slots(self):
        """Test get_variable_slots method."""
        preset = ExperimentPreset(
            name="Test",
            description="Test",
            frame_slot=SlotConfiguration(type="variable", scope=["*"]),
            color_slot=SlotConfiguration(
                type="variable", scope=["ffmpeg-color", "gifsicle-color"]
            ),
            lossy_slot=SlotConfiguration(type="locked", implementation="none-lossy"),
        )
        variable_slots = preset.get_variable_slots()
        assert set(variable_slots) == {"frame", "color"}

    def test_estimate_pipeline_count_single_variable(self):
        """Test pipeline count estimation for single variable slot."""
        preset = ExperimentPreset(
            name="Frame Focus",
            description="Test frame algorithms",
            frame_slot=SlotConfiguration(type="variable", scope=["*"]),
            color_slot=SlotConfiguration(type="locked", implementation="ffmpeg-color"),
            lossy_slot=SlotConfiguration(type="locked", implementation="none-lossy"),
        )
        count = preset.estimate_pipeline_count()
        assert count == 5  # 5 frame tools x 1 color x 1 lossy

    def test_estimate_with_max_combinations_limit(self):
        """Test estimation respects max_combinations limit."""
        preset = ExperimentPreset(
            name="Limited",
            description="Test with limit",
            frame_slot=SlotConfiguration(type="variable", scope=["*"]),
            color_slot=SlotConfiguration(type="variable", scope=["*"]),
            lossy_slot=SlotConfiguration(type="variable", scope=["*"]),
            max_combinations=100,
        )
        count = preset.estimate_pipeline_count()
        assert count == 100  # Limited to 100 instead of 5x17x11=935


class TestPresetRegistry:
    """Test PresetRegistry functionality."""

    def test_register_valid_preset(self):
        """Test registering a valid preset succeeds."""
        registry = PresetRegistry()
        preset = ExperimentPreset(
            name="Test Preset",
            description="Test",
            frame_slot=SlotConfiguration(type="variable", scope=["*"]),
            color_slot=SlotConfiguration(type="locked", implementation="ffmpeg-color"),
            lossy_slot=SlotConfiguration(type="locked", implementation="none-lossy"),
        )

        registry.register("test-preset", preset)
        assert "test-preset" in registry.presets
        assert registry.presets["test-preset"] == preset

    def test_get_nonexistent_preset(self):
        """Test that getting nonexistent preset raises ValueError."""
        registry = PresetRegistry()

        with pytest.raises(ValueError, match="Unknown preset: nonexistent"):
            registry.get("nonexistent")

    def test_list_presets_with_content(self):
        """Test listing presets with content."""
        registry = PresetRegistry()
        preset1 = ExperimentPreset(
            name="First Preset",
            description="First description",
            frame_slot=SlotConfiguration(type="variable", scope=["*"]),
            color_slot=SlotConfiguration(type="locked", implementation="ffmpeg-color"),
            lossy_slot=SlotConfiguration(type="locked", implementation="none-lossy"),
        )
        preset2 = ExperimentPreset(
            name="Second Preset",
            description="Second description",
            frame_slot=SlotConfiguration(
                type="locked", implementation="animately-frame"
            ),
            color_slot=SlotConfiguration(type="variable", scope=["*"]),
            lossy_slot=SlotConfiguration(type="locked", implementation="none-lossy"),
        )

        registry.register("first", preset1)
        registry.register("second", preset2)

        presets = registry.list_presets()
        assert presets == {"first": "First description", "second": "Second description"}


class TestTargetedPipelineGenerator:
    """Test TargetedPipelineGenerator functionality."""

    def test_validate_preset_feasibility_valid(self):
        """Test validation of valid preset."""
        generator = TargetedPipelineGenerator()
        preset = ExperimentPreset(
            name="Valid Test",
            description="Valid preset for testing",
            frame_slot=SlotConfiguration(type="variable", scope=["*"]),
            color_slot=SlotConfiguration(type="locked", implementation="ffmpeg-color"),
            lossy_slot=SlotConfiguration(type="locked", implementation="none-lossy"),
        )

        validation = generator.validate_preset_feasibility(preset)

        assert validation["valid"] is True
        assert validation["errors"] == []
        assert validation["estimated_pipelines"] > 0
        assert 0.0 <= validation["efficiency_gain"] <= 1.0
        assert isinstance(validation["tool_availability"], dict)

    def test_validate_preset_feasibility_invalid_tool(self):
        """Test validation with invalid tool name."""
        generator = TargetedPipelineGenerator()
        preset = ExperimentPreset(
            name="Invalid Tool Test",
            description="Preset with invalid tool",
            frame_slot=SlotConfiguration(type="variable", scope=["*"]),
            color_slot=SlotConfiguration(
                type="locked", implementation="nonexistent-tool"
            ),
            lossy_slot=SlotConfiguration(type="locked", implementation="none-lossy"),
        )

        validation = generator.validate_preset_feasibility(preset)

        assert validation["valid"] is False
        assert len(validation["errors"]) > 0
        assert "nonexistent-tool" in str(validation["errors"])


class TestBuiltinPresets:
    """Test built-in preset definitions."""

    def test_builtin_presets_loaded(self):
        """Test that built-in presets are loaded."""
        presets = PRESET_REGISTRY.list_presets()
        assert len(presets) > 0
        assert "frame-focus" in presets
        assert "color-optimization" in presets

    def test_all_builtin_presets_valid(self):
        """Test that all built-in presets are valid."""
        generator = TargetedPipelineGenerator()

        for preset_id in PRESET_REGISTRY.list_presets().keys():
            preset = PRESET_REGISTRY.get(preset_id)
            validation = generator.validate_preset_feasibility(preset)

            assert validation[
                "valid"
            ], f"Preset {preset_id} is invalid: {validation['errors']}"
            assert validation["estimated_pipelines"] > 0

    def test_builtin_presets_generate_pipelines(self):
        """Test that built-in presets can generate pipelines."""
        generator = TargetedPipelineGenerator()

        # Test a few key presets
        test_presets = ["frame-focus", "color-optimization", "quick-test"]

        for preset_id in test_presets:
            if preset_id in PRESET_REGISTRY.list_presets():
                preset = PRESET_REGISTRY.get(preset_id)
                pipelines = generator.generate_targeted_pipelines(preset)

                assert len(pipelines) > 0, f"Preset {preset_id} generated no pipelines"
                assert all(
                    hasattr(p, "identifier") for p in pipelines
                ), f"Invalid pipeline objects from {preset_id}"


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_generator_with_no_available_tools(self):
        """Test generator behavior when no tools are available."""
        generator = TargetedPipelineGenerator()
        preset = ExperimentPreset(
            name="No Tools Test",
            description="Test with nonexistent tools",
            frame_slot=SlotConfiguration(type="variable", scope=["nonexistent-tool"]),
            color_slot=SlotConfiguration(type="locked", implementation="ffmpeg-color"),
            lossy_slot=SlotConfiguration(type="locked", implementation="none-lossy"),
        )

        with pytest.raises(RuntimeError, match="No tools found in scope"):
            generator.generate_targeted_pipelines(preset)

    def test_max_combinations_limit_applied(self):
        """Test that max_combinations limit is properly applied."""
        generator = TargetedPipelineGenerator()

        # Create a preset that would generate many pipelines but limit it
        preset = ExperimentPreset(
            name="Limited Test",
            description="Test max combinations limit",
            frame_slot=SlotConfiguration(type="variable", scope=["*"]),
            color_slot=SlotConfiguration(type="variable", scope=["*"]),
            lossy_slot=SlotConfiguration(type="variable", scope=["*"]),
            max_combinations=5,  # Limit to 5 pipelines
        )

        pipelines = generator.generate_targeted_pipelines(preset)
        assert len(pipelines) == 5  # Should be limited


class TestIntegrationWithExistingSystem:
    """Test integration with existing GifLab systems."""

    def test_pipeline_objects_have_correct_structure(self):
        """Test that generated pipelines have expected structure."""
        generator = TargetedPipelineGenerator()
        preset = PRESET_REGISTRY.get("frame-focus")
        pipelines = generator.generate_targeted_pipelines(preset)

        for pipeline in pipelines[:3]:  # Test first few
            assert hasattr(pipeline, "steps")
            assert hasattr(pipeline, "identifier")
            assert len(pipeline.steps) == 3  # frame, color, lossy

            # Check each step has required attributes
            for step in pipeline.steps:
                assert hasattr(step, "variable")
                assert hasattr(step, "tool_cls")
                assert step.variable in [
                    "frame_reduction",
                    "color_reduction",
                    "lossy_compression",
                ]
                assert hasattr(step.tool_cls, "NAME")

    def test_tool_name_validation_against_capability_registry(self):
        """Test that tool names used in presets exist in capability registry."""
        from giflab.capability_registry import tools_for

        # Get all available tools
        all_tools = {}
        for variable in ["frame_reduction", "color_reduction", "lossy_compression"]:
            all_tools[variable] = {tool.NAME for tool in tools_for(variable)}

        # Check all built-in presets use valid tool names
        for preset_id in PRESET_REGISTRY.list_presets().keys():
            preset = PRESET_REGISTRY.get(preset_id)

            # Check locked implementations
            for slot_name in ["frame", "color", "lossy"]:
                slot = getattr(preset, f"{slot_name}_slot")
                if slot.type == "locked":
                    variable = (
                        f"{slot_name}_reduction"
                        if slot_name != "lossy"
                        else "lossy_compression"
                    )
                    assert (
                        slot.implementation in all_tools[variable]
                    ), f"Preset {preset_id} uses invalid {slot_name} tool: {slot.implementation}"

            # Check variable scopes (for specific tool names, not wildcards)
            for slot_name in ["frame", "color", "lossy"]:
                slot = getattr(preset, f"{slot_name}_slot")
                if slot.type == "variable" and slot.scope and "*" not in slot.scope:
                    variable = (
                        f"{slot_name}_reduction"
                        if slot_name != "lossy"
                        else "lossy_compression"
                    )
                    for tool_name in slot.scope:
                        assert (
                            tool_name in all_tools[variable]
                        ), f"Preset {preset_id} scope includes invalid {slot_name} tool: {tool_name}"


if __name__ == "__main__":
    pytest.main([__file__])
