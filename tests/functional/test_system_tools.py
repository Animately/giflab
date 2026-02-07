"""Tests for giflab.system_tools module."""

import os
from unittest.mock import patch

import pytest
from giflab.config import EngineConfig
from giflab.system_tools import (
    ToolInfo,
    discover_tool,
    get_available_tools,
    verify_required_tools,
)


class TestToolInfo:
    """Tests for ToolInfo class."""

    def test_tool_info_creation(self):
        """Test ToolInfo creation."""
        info = ToolInfo(name="test_tool", available=True, version="1.0.0")

        assert info.name == "test_tool"
        assert info.available is True
        assert info.version == "1.0.0"

    def test_tool_info_require_available(self):
        """Test that require() passes for available tools."""
        info = ToolInfo(name="test_tool", available=True, version="1.0.0")

        # Should not raise
        info.require()

    def test_tool_info_require_unavailable(self):
        """Test that require() raises for unavailable tools."""
        info = ToolInfo(name="test_tool", available=False, version=None)

        with pytest.raises(RuntimeError, match="Required tool 'test_tool' not found"):
            info.require()


class TestDiscoverTool:
    """Tests for discover_tool function."""

    def test_discover_tool_unknown_key(self):
        """Test that unknown tool keys raise ValueError."""
        with pytest.raises(ValueError, match="Unknown tool: unknown_tool"):
            discover_tool("unknown_tool")

    @patch("giflab.system_tools._which")
    def test_discover_tool_with_config_found(self, mock_which):
        """Test tool discovery using configuration when tool is found."""
        mock_which.return_value = "/usr/bin/gifsicle"

        config = EngineConfig(GIFSICLE_PATH="/custom/gifsicle")

        with patch("giflab.system_tools._run_version_cmd") as mock_version:
            mock_version.return_value = "1.94"

            info = discover_tool("gifsicle", config)

            assert info.name == "/custom/gifsicle"
            assert info.available is True
            assert info.version == "1.94"

            mock_which.assert_called_once_with("/custom/gifsicle")
            mock_version.assert_called_once_with(
                ["/custom/gifsicle", "-version"], r"LCDF Gifsicle (\S+)"
            )

    @patch("giflab.system_tools._which")
    def test_discover_tool_config_not_found_fallback_success(self, mock_which):
        """Test tool discovery falls back to PATH when configured path not found."""
        # First call (configured path) returns None, second call (fallback) returns path
        mock_which.side_effect = [None, "/usr/bin/gifsicle"]

        config = EngineConfig(GIFSICLE_PATH="/custom/gifsicle")

        with patch("giflab.system_tools._run_version_cmd") as mock_version:
            mock_version.return_value = "1.94"

            info = discover_tool("gifsicle", config)

            assert info.name == "gifsicle"
            assert info.available is True
            assert info.version == "1.94"

            # Should have tried both configured path and fallback
            assert mock_which.call_count == 2
            mock_which.assert_any_call("/custom/gifsicle")
            mock_which.assert_any_call("gifsicle")

    @patch("giflab.system_tools._which")
    def test_discover_tool_not_found_anywhere(self, mock_which):
        """Test tool discovery when tool is not found anywhere."""
        mock_which.return_value = None

        config = EngineConfig(GIFSICLE_PATH="/custom/gifsicle")

        info = discover_tool("gifsicle", config)

        assert info.name == "gifsicle"
        assert info.available is False
        assert info.version is None

    @patch("giflab.system_tools._which")
    def test_discover_tool_ffmpeg_special_case(self, mock_which):
        """Test that FFmpeg uses correct version command."""
        mock_which.return_value = "/usr/bin/ffmpeg"

        config = EngineConfig(FFMPEG_PATH="/custom/ffmpeg")

        with patch("giflab.system_tools._run_version_cmd") as mock_version:
            mock_version.return_value = "4.4.2"

            discover_tool("ffmpeg", config)

            mock_version.assert_called_once_with(
                ["/custom/ffmpeg", "-version"], r"ffmpeg version (\S+)"
            )

    def test_discover_tool_uses_default_config(self):
        """Test that discover_tool uses DEFAULT_ENGINE_CONFIG when no config provided."""
        with patch("giflab.system_tools._which") as mock_which:
            mock_which.return_value = None

            # Should not raise ImportError
            info = discover_tool("gifsicle")

            assert info.available is False

    def test_discover_tool_environment_variables(self):
        """Test that environment variables work through configuration."""
        env_vars = {"GIFLAB_GIFSICLE_PATH": "/env/gifsicle"}

        with patch.dict(os.environ, env_vars):
            # Create a fresh config instance to pick up the environment variable
            config = EngineConfig()
            assert config.GIFSICLE_PATH == "/env/gifsicle"

            with patch("giflab.system_tools._which") as mock_which:
                mock_which.return_value = "/env/gifsicle"

                with patch("giflab.system_tools._run_version_cmd") as mock_version:
                    mock_version.return_value = "1.94"

                    info = discover_tool("gifsicle", config)

                    assert info.name == "/env/gifsicle"
                    assert info.available is True
                    mock_which.assert_called_with("/env/gifsicle")


class TestVerifyRequiredTools:
    """Tests for verify_required_tools function."""

    @patch("giflab.system_tools.discover_tool")
    def test_verify_required_tools_all_available(self, mock_discover):
        """Test verify_required_tools when all tools are available."""
        # Mock all tools as available
        mock_discover.return_value = ToolInfo(
            name="test", available=True, version="1.0"
        )

        result = verify_required_tools()

        # Should have checked all configured tools
        expected_tools = [
            "imagemagick",
            "ffmpeg",
            "ffprobe",
            "gifski",
            "gifsicle",
            "animately",
        ]
        assert mock_discover.call_count == len(expected_tools)

        # Result should contain all tools
        assert len(result) == len(expected_tools)
        for tool in expected_tools:
            assert tool in result
            assert result[tool].available is True

    @patch("giflab.system_tools.discover_tool")
    def test_verify_required_tools_one_missing(self, mock_discover):
        """Test verify_required_tools when one tool is missing."""

        def discover_side_effect(tool_key, config=None):
            if tool_key == "gifski":
                return ToolInfo(name="gifski", available=False, version=None)
            return ToolInfo(name=tool_key, available=True, version="1.0")

        mock_discover.side_effect = discover_side_effect

        # Should raise RuntimeError for the missing tool
        with pytest.raises(RuntimeError, match="Required tool 'gifski' not found"):
            verify_required_tools()

    @patch("giflab.system_tools.discover_tool")
    def test_verify_required_tools_with_custom_config(self, mock_discover):
        """Test verify_required_tools with custom EngineConfig."""
        mock_discover.return_value = ToolInfo(
            name="test", available=True, version="1.0"
        )

        custom_config = EngineConfig(GIFSICLE_PATH="/custom/gifsicle")

        verify_required_tools(custom_config)

        # Should pass the custom config to discover_tool
        for call in mock_discover.call_args_list:
            args, kwargs = call
            # discover_tool should be called with engine_config parameter
            assert len(args) >= 1  # tool_key
            # The config should be passed as second positional arg or kwarg
            assert (len(args) > 1 and args[1] is custom_config) or kwargs.get(
                "engine_config"
            ) is custom_config


class TestGetAvailableTools:
    """Tests for get_available_tools function."""

    @patch("giflab.system_tools.discover_tool")
    def test_get_available_tools_mixed_availability(self, mock_discover):
        """Test get_available_tools with mixed tool availability."""

        def discover_side_effect(tool_key, config=None):
            if tool_key in ["gifsicle", "ffmpeg"]:
                return ToolInfo(name=tool_key, available=True, version="1.0")
            return ToolInfo(name=tool_key, available=False, version=None)

        mock_discover.side_effect = discover_side_effect

        result = get_available_tools()

        # Should return info for all tools, regardless of availability
        expected_tools = [
            "imagemagick",
            "ffmpeg",
            "ffprobe",
            "gifski",
            "gifsicle",
            "animately",
        ]
        assert len(result) == len(expected_tools)

        # Check specific availability
        assert result["gifsicle"].available is True
        assert result["ffmpeg"].available is True
        assert result["imagemagick"].available is False
        assert result["ffprobe"].available is False
        assert result["gifski"].available is False
        assert result["animately"].available is False

    @patch("giflab.system_tools.discover_tool")
    def test_get_available_tools_with_custom_config(self, mock_discover):
        """Test get_available_tools with custom EngineConfig."""
        mock_discover.return_value = ToolInfo(
            name="test", available=True, version="1.0"
        )

        custom_config = EngineConfig(IMAGEMAGICK_PATH="/custom/magick")

        get_available_tools(custom_config)

        # Should pass the custom config to discover_tool
        for call in mock_discover.call_args_list:
            args, kwargs = call
            assert len(args) >= 1  # tool_key
            assert (len(args) > 1 and args[1] is custom_config) or kwargs.get(
                "engine_config"
            ) is custom_config


class TestIntegration:
    """Integration tests combining configuration and tool discovery."""

    def test_configuration_and_discovery_integration(self):
        """Test that configuration and discovery work together correctly."""
        # This is an integration test that doesn't mock anything
        # It tests the actual interaction between config and discovery

        env_vars = {"GIFLAB_GIFSICLE_PATH": "/nonexistent/gifsicle"}

        with patch.dict(os.environ, env_vars):
            # Create config (should pick up environment variable)
            config = EngineConfig()
            assert config.GIFSICLE_PATH == "/nonexistent/gifsicle"

            # Use config in discovery (should try the configured path)
            info = discover_tool("gifsicle", config)

            # Since the path doesn't exist, it should fall back and likely still not find it
            # (unless gifsicle is installed in PATH)
            assert info.name in [
                "gifsicle",
                "/nonexistent/gifsicle",
            ]  # Could be either depending on fallback
