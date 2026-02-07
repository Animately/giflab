"""Unit tests for monitor configuration system."""

import json
import os

# Add the scripts directory to the path for imports
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

script_dir = Path(__file__).parent.parent / "scripts" / "experimental"
sys.path.insert(0, str(script_dir))

try:
    from monitor_config import MonitorConfig, create_sample_config_file

    MONITOR_CONFIG_AVAILABLE = True
except ImportError:
    MONITOR_CONFIG_AVAILABLE = False


@pytest.mark.skipif(
    not MONITOR_CONFIG_AVAILABLE, reason="Monitor config module not available"
)
class TestMonitorConfig:
    """Test monitor configuration system."""

    def test_default_configuration(self):
        """Test default configuration values."""
        config = MonitorConfig()

        # Test default values
        assert config.refresh_interval == 30
        assert config.failures_to_show == 3
        assert config.buffer_size == 25
        assert config.estimated_total_jobs == 93500
        assert config.base_time_per_job == 2.5
        assert len(config.search_locations) > 0
        assert config.min_rate_for_processing == 0.1
        assert config.accelerating_threshold == 1.2
        assert config.slowing_threshold == 0.8

    def test_config_file_loading(self):
        """Test loading configuration from JSON file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            test_config = {
                "refresh_interval": 15,
                "failures_to_show": 5,
                "buffer_size": 50,
                "estimated_total_jobs": 50000,
                "base_time_per_job": 3.0,
                "min_rate_for_processing": 0.2,
            }
            json.dump(test_config, f)
            config_path = f.name

        try:
            config = MonitorConfig(config_path)

            # Test loaded values
            assert config.refresh_interval == 15
            assert config.failures_to_show == 5
            assert config.buffer_size == 50
            assert config.estimated_total_jobs == 50000
            assert config.base_time_per_job == 3.0
            assert config.min_rate_for_processing == 0.2
        finally:
            os.unlink(config_path)

    def test_environment_variable_loading(self):
        """Test loading configuration from environment variables."""
        env_vars = {
            "MONITOR_REFRESH_INTERVAL": "20",
            "MONITOR_FAILURES_TO_SHOW": "7",
            "MONITOR_BUFFER_SIZE": "35",
            "MONITOR_ESTIMATED_TOTAL_JOBS": "75000",
            "MONITOR_BASE_TIME_PER_JOB": "1.5",
            "MONITOR_MIN_PROCESSING_RATE": "0.15",
        }

        with patch.dict(os.environ, env_vars):
            config = MonitorConfig()

            # Test environment-loaded values
            assert config.refresh_interval == 20
            assert config.failures_to_show == 7
            assert config.buffer_size == 35
            assert config.estimated_total_jobs == 75000
            assert config.base_time_per_job == 1.5
            assert config.min_rate_for_processing == 0.15

    def test_config_validation(self):
        """Test configuration validation."""
        config = MonitorConfig()

        # Test with valid config
        warnings = config.validate()
        assert len(warnings) == 0

        # Test with invalid values
        config.refresh_interval = 0
        config.failures_to_show = -1
        config.buffer_size = 0
        config.estimated_total_jobs = -100
        config.base_time_per_job = -1.0
        config.search_locations = []
        config.min_rate_for_processing = -0.5
        config.accelerating_threshold = 0.5
        config.slowing_threshold = 1.5

        warnings = config.validate()
        assert len(warnings) > 0
        assert any("Refresh interval" in warning for warning in warnings)
        assert any("failures to show" in warning for warning in warnings)
        assert any("Buffer size" in warning for warning in warnings)
        assert any("total jobs" in warning for warning in warnings)
        assert any("time per job" in warning for warning in warnings)
        assert any("search location" in warning for warning in warnings)
        assert any("processing rate" in warning for warning in warnings)
        assert any("Accelerating threshold" in warning for warning in warnings)
        assert any("Slowing threshold" in warning for warning in warnings)

    def test_trend_description(self):
        """Test trend description generation."""
        config = MonitorConfig()

        # Test accelerating (need >20% faster)
        trend = config.get_trend_description(13.0, 10.0)  # 30% faster
        assert "Accelerating" in trend

        # Test steady
        trend = config.get_trend_description(10.5, 10.0)  # 5% faster (within threshold)
        assert "Steady" in trend

        # Test slowing (but still processing, need <20% of average)
        trend = config.get_trend_description(7.0, 10.0)  # 30% slower but above min rate
        assert "Slowing" in trend

        # Test batching (below min processing rate)
        trend = config.get_trend_description(0.05, 10.0)  # Very slow, likely batching
        assert "Batching" in trend

    def test_is_actively_processing(self):
        """Test active processing detection."""
        config = MonitorConfig()

        # Test active processing
        assert config.is_actively_processing(1.0) is True
        assert config.is_actively_processing(0.2) is True

        # Test likely batching
        assert config.is_actively_processing(0.05) is False
        assert config.is_actively_processing(0.0) is False

    def test_estimate_remaining_time(self):
        """Test remaining time estimation."""
        config = MonitorConfig()

        # Test seconds
        time_str = config.estimate_remaining_time(20)
        assert "seconds" in time_str

        # Test minutes
        time_str = config.estimate_remaining_time(100)
        assert "minutes" in time_str

        # Test hours
        time_str = config.estimate_remaining_time(5000)
        assert "hours" in time_str

    def test_batching_info(self):
        """Test batching information."""
        config = MonitorConfig()

        batching_info = config.get_batching_info()
        assert "min_batch_size" in batching_info
        assert "max_batch_size" in batching_info
        assert "description" in batching_info
        assert "explanation" in batching_info

        assert batching_info["min_batch_size"] == 15
        assert batching_info["max_batch_size"] == 25
        assert "15-25" in batching_info["description"]

    def test_search_locations(self):
        """Test search locations management."""
        config = MonitorConfig()

        locations = config.get_search_locations()
        assert isinstance(locations, list)
        assert len(locations) > 0

        # Test that default locations are included
        assert any("results" in loc for loc in locations)
        assert any("streaming_results.csv" in loc for loc in locations)

    def test_config_file_save_and_load(self):
        """Test saving and loading configuration files."""
        original_config = MonitorConfig()
        original_config.refresh_interval = 45
        original_config.failures_to_show = 8
        original_config.buffer_size = 60

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            config_path = f.name

        try:
            # Save configuration
            original_config.save_to_file(config_path)

            # Load configuration
            loaded_config = MonitorConfig(config_path)

            # Test that values match
            assert loaded_config.refresh_interval == 45
            assert loaded_config.failures_to_show == 8
            assert loaded_config.buffer_size == 60
        finally:
            os.unlink(config_path)

    def test_create_sample_config_file(self):
        """Test sample configuration file creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_config.json"

            create_sample_config_file(str(config_path))

            # Test that file was created
            assert config_path.exists()

            # Test that file contains valid JSON
            with open(config_path) as f:
                config_data = json.load(f)

            assert isinstance(config_data, dict)
            assert "refresh_interval" in config_data
            assert "failures_to_show" in config_data
            assert "search_locations" in config_data

            # Test that help comments are included
            assert any(
                key.startswith("_") and key.endswith("_help")
                for key in config_data.keys()
            )


if __name__ == "__main__":
    pytest.main([__file__])
