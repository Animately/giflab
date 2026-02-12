"""Smoke tests for capability registry tool class lookup.

Verifies that get_tool_class_by_name() resolves known tool names
to the correct wrapper classes.
"""

import pytest
from giflab.capability_registry import get_tool_class_by_name, tools_for


class TestGetToolClassByName:
    """Test get_tool_class_by_name() lookup function."""

    @pytest.mark.parametrize(
        "name",
        [
            "gifsicle-lossy",
            "gifsicle-color",
            "gifsicle-frame",
        ],
    )
    def test_gifsicle_tools_resolve(self, name: str) -> None:
        """Gifsicle tool names resolve to classes."""
        cls = get_tool_class_by_name(name)
        if cls is None:
            pytest.skip(f"Tool {name} not available on this system")
        assert cls.NAME == name

    @pytest.mark.parametrize(
        "name",
        [
            "animately-standard",
            "animately-advanced",
            "animately-hard",
            "animately-color",
            "animately-frame",
        ],
    )
    def test_animately_tools_resolve(self, name: str) -> None:
        """Animately tool names resolve to classes."""
        cls = get_tool_class_by_name(name)
        if cls is None:
            pytest.skip(f"Tool {name} not available on this system")
        assert cls.NAME == name

    @pytest.mark.parametrize(
        "name",
        [
            "imagemagick-lossy",
            "imagemagick-color",
            "imagemagick-frame",
            "ffmpeg-lossy",
            "ffmpeg-color",
            "ffmpeg-frame",
            "gifski-lossy",
        ],
    )
    def test_other_tools_resolve(self, name: str) -> None:
        """Other engine tool names resolve to classes."""
        cls = get_tool_class_by_name(name)
        if cls is None:
            pytest.skip(f"Tool {name} not available on this system")
        assert cls.NAME == name

    def test_unknown_name_returns_none(self) -> None:
        """Unknown tool name returns None."""
        assert get_tool_class_by_name("nonexistent-tool") is None

    def test_all_lossy_tools_resolvable(self) -> None:
        """Every available lossy tool can be found by name."""
        lossy_tools = tools_for("lossy_compression")
        for cls in lossy_tools:
            found = get_tool_class_by_name(cls.NAME)
            assert found is cls, f"Lookup for {cls.NAME} returned {found}"

    def test_all_color_tools_resolvable(self) -> None:
        """Every available color tool can be found by name."""
        color_tools = tools_for("color_reduction")
        for cls in color_tools:
            found = get_tool_class_by_name(cls.NAME)
            assert found is cls, f"Lookup for {cls.NAME} returned {found}"

    def test_all_frame_tools_resolvable(self) -> None:
        """Every available frame tool can be found by name."""
        frame_tools = tools_for("frame_reduction")
        for cls in frame_tools:
            found = get_tool_class_by_name(cls.NAME)
            assert found is cls, f"Lookup for {cls.NAME} returned {found}"
