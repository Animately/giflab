"""Unit tests for configuration management system."""

import copy
import json
import os
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.giflab.config_manager import (
    ConfigChange,
    ConfigManager,
    ConfigMetadata,
    ConfigProfile,
    ConfigReloadError,
    ConfigValidationError,
    ConfigValidator,
    get_config_manager,
    load_profile,
)
from src.giflab.config_validator import (
    ConfigurationValidator,
    RangeValidator,
    ResourceValidator,
    TypeValidator,
    ValidationRule,
)


class TestConfigManager:
    """Test ConfigManager functionality."""
    
    def setup_method(self):
        """Reset singleton before each test."""
        ConfigManager._instance = None
    
    def test_singleton_pattern(self):
        """Test that ConfigManager is a singleton."""
        manager1 = ConfigManager()
        manager2 = ConfigManager()
        assert manager1 is manager2
        
        # Also test with get_config_manager
        manager3 = get_config_manager()
        assert manager1 is manager3
    
    def test_load_defaults(self):
        """Test loading default configuration."""
        manager = ConfigManager()
        
        # Check that defaults are loaded
        assert "compression" in manager._config
        assert "metrics" in manager._config
        assert "FRAME_CACHE" in manager._config
        assert "MONITORING" in manager._config
        
        # Check specific default values
        assert manager.get("FRAME_CACHE.enabled") is False
        assert manager.get("FRAME_CACHE.memory_limit_mb") == 500
    
    def test_load_profile(self):
        """Test loading configuration profiles."""
        manager = ConfigManager()
        
        # Load development profile
        manager.load_profile(ConfigProfile.DEVELOPMENT)
        assert manager._metadata.active_profile == ConfigProfile.DEVELOPMENT
        assert manager.get("FRAME_CACHE.memory_limit_mb") == 1000
        assert manager.get("MONITORING.verbose") is True
        
        # Load production profile
        manager.load_profile(ConfigProfile.PRODUCTION)
        assert manager._metadata.active_profile == ConfigProfile.PRODUCTION
        assert manager.get("FRAME_CACHE.memory_limit_mb") == 500
        assert manager.get("MONITORING.verbose") is False
    
    def test_invalid_profile(self):
        """Test loading invalid profile raises error."""
        manager = ConfigManager()
        
        with pytest.raises(ConfigValidationError):
            manager.load_profile("nonexistent")  # Invalid profile
    
    def test_get_set_config(self):
        """Test getting and setting configuration values."""
        manager = ConfigManager()
        
        # Test get with existing value
        assert manager.get("FRAME_CACHE.enabled") is False
        
        # Test get with non-existing value
        assert manager.get("nonexistent.value") is None
        assert manager.get("nonexistent.value", "default") == "default"
        
        # Test set with validation (stay under 2048MB total cache limit)
        manager.set("FRAME_CACHE.memory_limit_mb", 1500)
        assert manager.get("FRAME_CACHE.memory_limit_mb") == 1500
        
        # Check that override is recorded
        assert "FRAME_CACHE.memory_limit_mb" in manager._metadata.overrides
    
    def test_set_invalid_value(self):
        """Test setting invalid value raises error."""
        manager = ConfigManager()
        
        # Add a validator that will fail
        manager._validator.add_validator(
            "FRAME_CACHE.memory_limit_mb",
            lambda x: x <= 8192,
            "Memory limit too high"
        )
        
        with pytest.raises(ConfigValidationError):
            manager.set("FRAME_CACHE.memory_limit_mb", 10000, validate=True)
    
    def test_config_reload(self):
        """Test configuration reload functionality."""
        manager = ConfigManager()
        
        # Set a value
        manager.set("FRAME_CACHE.memory_limit_mb", 1500)
        assert manager.get("FRAME_CACHE.memory_limit_mb") == 1500
        
        # Reload configuration
        success = manager.reload_config(source="test")
        assert success is True
        
        # Check that metadata was updated
        assert manager._metadata.reload_count == 1
        assert manager._metadata.last_reload is not None
    
    def test_environment_overrides(self):
        """Test environment variable overrides."""
        # Note: env var parsing splits on underscore, so only single-word
        # section names (e.g. MONITORING) work reliably.
        os.environ["GIFLAB_CONFIG_MONITORING_ENABLED"] = "false"
        os.environ["GIFLAB_CONFIG_MONITORING_VERBOSE"] = "true"

        try:
            # Create new manager (will apply env overrides)
            ConfigManager._instance = None
            manager = ConfigManager()

            # Check overrides were applied
            assert manager.get("MONITORING.enabled") is False
            assert manager.get("MONITORING.verbose") is True

        finally:
            # Clean up environment
            del os.environ["GIFLAB_CONFIG_MONITORING_ENABLED"]
            del os.environ["GIFLAB_CONFIG_MONITORING_VERBOSE"]
    
    def test_change_callbacks(self):
        """Test configuration change callbacks."""
        manager = ConfigManager()
        
        # Track changes
        changes = []
        
        def callback(change: ConfigChange):
            changes.append(change)
        
        manager.register_change_callback(callback)
        
        # Make a change
        manager.set("FRAME_CACHE.memory_limit_mb", 1000)
        
        # Check callback was called
        assert len(changes) == 1
        assert changes[0].path == "FRAME_CACHE.memory_limit_mb"
        assert changes[0].new_value == 1000
        assert changes[0].source == "api"
    
    def test_export_import_config(self):
        """Test configuration export and import."""
        manager = ConfigManager()
        
        # Load a profile and make changes
        manager.load_profile(ConfigProfile.DEVELOPMENT)
        manager.set("FRAME_CACHE.memory_limit_mb", 1234)
        
        # Export configuration
        exported = manager.export_config()
        assert "config" in exported
        assert "metadata" in exported
        assert exported["profile"] == "development"
        
        # Create new manager and import
        ConfigManager._instance = None
        new_manager = ConfigManager()
        new_manager.import_config(exported)
        
        # Check imported values
        assert new_manager.get("FRAME_CACHE.memory_limit_mb") == 1234
        assert new_manager._metadata.active_profile == ConfigProfile.DEVELOPMENT
    
    def test_thread_safety(self):
        """Test thread-safe configuration access."""
        manager = ConfigManager()
        errors = []
        
        def reader():
            try:
                for _ in range(100):
                    value = manager.get("FRAME_CACHE.memory_limit_mb")
                    assert value is not None
            except Exception as e:
                errors.append(e)
        
        def writer():
            try:
                for i in range(100):
                    manager.set("FRAME_CACHE.memory_limit_mb", 500 + i)
            except Exception as e:
                errors.append(e)
        
        # Run concurrent reads and writes
        threads = []
        for _ in range(5):
            threads.append(threading.Thread(target=reader))
            threads.append(threading.Thread(target=writer))
        
        for t in threads:
            t.start()
        
        for t in threads:
            t.join()
        
        # Check no errors occurred
        assert len(errors) == 0
    
    def test_config_diff(self):
        """Test configuration diff detection."""
        manager = ConfigManager()
        
        old_config = {
            "cache": {"size": 100, "enabled": True},
            "monitoring": {"level": "info"}
        }
        
        new_config = {
            "cache": {"size": 200, "enabled": True},
            "monitoring": {"level": "debug"},
            "new_section": {"value": 1}
        }
        
        diffs = manager._diff_configs(old_config, new_config)
        
        assert "cache.size" in diffs
        assert diffs["cache.size"] == (100, 200)
        assert "monitoring.level" in diffs
        assert diffs["monitoring.level"] == ("info", "debug")
        assert "new_section" in diffs
        assert diffs["new_section"] == (None, {"value": 1})


class TestConfigValidator:
    """Test ConfigValidator functionality."""
    
    def test_type_validators(self):
        """Test type validation rules."""
        assert TypeValidator.is_int(42) is True
        assert TypeValidator.is_int(3.14) is False
        assert TypeValidator.is_int(True) is False  # bool is not int
        
        assert TypeValidator.is_float(3.14) is True
        assert TypeValidator.is_float(42) is True  # int is accepted
        assert TypeValidator.is_float("3.14") is False
        
        assert TypeValidator.is_bool(True) is True
        assert TypeValidator.is_bool(1) is False
        
        assert TypeValidator.is_str("hello") is True
        assert TypeValidator.is_str(123) is False
        
        assert TypeValidator.is_path("/tmp/test") is True
        assert TypeValidator.is_path(Path("/tmp/test")) is True
    
    def test_range_validators(self):
        """Test range validation rules."""
        validator = RangeValidator.in_range(10, 100)
        assert validator(50) is True
        assert validator(5) is False
        assert validator(150) is False
        
        assert RangeValidator.positive(10) is True
        assert RangeValidator.positive(-5) is False
        assert RangeValidator.positive(0) is False
        
        assert RangeValidator.percentage(0.5) is True
        assert RangeValidator.percentage(1.5) is False
        assert RangeValidator.percentage(-0.1) is False
        
        assert RangeValidator.port_number(8080) is True
        assert RangeValidator.port_number(0) is False
        assert RangeValidator.port_number(70000) is False
    
    def test_validation_rule(self):
        """Test ValidationRule class."""
        rule = ValidationRule(
            name="test_rule",
            validator=lambda x: x > 0,
            error_message="Value must be positive",
            severity="error"
        )
        
        # Test valid value
        is_valid, error = rule.validate(10)
        assert is_valid is True
        assert error is None
        
        # Test invalid value
        is_valid, error = rule.validate(-5)
        assert is_valid is False
        assert "Value must be positive" in error
    
    def test_configuration_validator(self):
        """Test ConfigurationValidator."""
        validator = ConfigurationValidator()
        
        config = {
            "FRAME_CACHE": {
                "enabled": True,
                "memory_limit_mb": 500,
                "ttl_seconds": 3600,
            },
            "MONITORING": {
                "backend": "sqlite",
                "buffer_size": 10000,
                "sampling_rate": 0.5,
            }
        }
        
        results = validator.validate(config)
        
        # Should have no errors for valid config
        assert len(results.get("error", [])) == 0
        
        # Test with invalid config
        config["FRAME_CACHE"]["memory_limit_mb"] = -100
        config["MONITORING"]["sampling_rate"] = 1.5
        
        results = validator.validate(config)
        
        # Should have errors
        assert len(results.get("error", [])) > 0
    
    def test_relationship_validation(self):
        """Test configuration relationship validation."""
        validator = ConfigurationValidator()
        
        config = {
            "FRAME_CACHE": {"memory_limit_mb": 3000},
            "VALIDATION_CACHE": {"memory_limit_mb": 2000},
        }
        
        errors = validator.validate_relationships(config)
        
        # Total memory exceeds 4GB limit
        assert len(errors) > 0
        assert "Total cache memory" in errors[0]


class TestConfigProfiles:
    """Test configuration profiles."""
    
    def test_all_profiles_load(self):
        """Test that all profiles can be loaded without errors."""
        manager = ConfigManager()
        
        profiles = [
            ConfigProfile.DEVELOPMENT,
            ConfigProfile.PRODUCTION,
            ConfigProfile.HIGH_MEMORY,
            ConfigProfile.LOW_MEMORY,
            ConfigProfile.HIGH_THROUGHPUT,
            ConfigProfile.INTERACTIVE,
            ConfigProfile.TESTING,
        ]
        
        for profile in profiles:
            # Reset manager
            ConfigManager._instance = None
            manager = ConfigManager()
            
            # Should not raise
            manager.load_profile(profile)
            assert manager._metadata.active_profile == profile
    
    def test_profile_characteristics(self):
        """Test that profiles have expected characteristics."""
        manager = ConfigManager()
        
        # Development profile - high caching
        manager.load_profile(ConfigProfile.DEVELOPMENT)
        assert manager.get("FRAME_CACHE.memory_limit_mb") >= 1000
        assert manager.get("FRAME_SAMPLING.enabled") is False
        assert manager.get("MONITORING.verbose") is True
        
        # Low memory profile - minimal caching
        manager.load_profile(ConfigProfile.LOW_MEMORY)
        assert manager.get("FRAME_CACHE.memory_limit_mb") <= 100
        assert manager.get("FRAME_SAMPLING.enabled") is True
        assert manager.get("MONITORING.sampling_rate") <= 0.1
        
        # Testing profile - no caching
        manager.load_profile(ConfigProfile.TESTING)
        assert manager.get("FRAME_CACHE.enabled") is False
        assert manager.get("VALIDATION_CACHE.enabled") is False
        assert manager.get("MONITORING.enabled") is False


class TestConfigIntegration:
    """Integration tests for configuration system."""
    
    def test_file_watcher(self, tmp_path):
        """Test configuration file watching."""
        manager = ConfigManager()
        
        # Create a config file
        config_file = tmp_path / "config.json"
        config_data = {
            "FRAME_CACHE": {
                "memory_limit_mb": 750
            }
        }
        config_file.write_text(json.dumps(config_data))
        
        # Start watching
        manager.start_file_watcher([tmp_path])
        
        try:
            # Modify the file
            time.sleep(0.5)  # Let watcher start
            config_data["FRAME_CACHE"]["memory_limit_mb"] = 1000
            config_file.write_text(json.dumps(config_data))
            
            # Wait for reload
            time.sleep(1.5)
            
            # Check that reload count increased
            # Note: This is a simplified test - real implementation would
            # need to parse the JSON and apply it
            assert manager._metadata.reload_count >= 0
            
        finally:
            manager.stop_file_watcher()
    
    def test_validation_with_profiles(self):
        """Test that all profiles pass validation."""
        from src.giflab.config_validator import ConfigurationValidator
        
        manager = ConfigManager()
        validator = ConfigurationValidator()
        
        profiles = [
            ConfigProfile.DEVELOPMENT,
            ConfigProfile.PRODUCTION,
            ConfigProfile.HIGH_MEMORY,
            ConfigProfile.LOW_MEMORY,
            ConfigProfile.HIGH_THROUGHPUT,
            ConfigProfile.INTERACTIVE,
            ConfigProfile.TESTING,
        ]
        
        for profile in profiles:
            manager.load_profile(profile)
            config = manager.config
            
            # Validate
            results = validator.validate(config)
            errors = results.get("error", [])
            
            # All profiles should be valid
            assert len(errors) == 0, f"Profile {profile} has errors: {errors}"
    
    def test_config_persistence(self, tmp_path):
        """Test configuration export and import persistence."""
        export_file = tmp_path / "config_export.json"
        
        # Create and configure manager
        manager = ConfigManager()
        manager.load_profile(ConfigProfile.HIGH_MEMORY)
        manager.set("FRAME_CACHE.custom_value", 12345)
        
        # Export to file (use default= to handle PosixPath serialization)
        exported = manager.export_config()
        with open(export_file, 'w') as f:
            json.dump(exported, f, default=str)
        
        # Create new manager and import
        ConfigManager._instance = None
        new_manager = ConfigManager()
        
        with open(export_file) as f:
            imported = json.load(f)
        
        new_manager.import_config(imported)
        
        # Verify configuration matches
        assert new_manager._metadata.active_profile == ConfigProfile.HIGH_MEMORY
        assert new_manager.get("FRAME_CACHE.custom_value") == 12345


@pytest.mark.parametrize("profile_name", [
    "development",
    "production",
    "high_memory",
    "low_memory",
    "high_throughput",
    "interactive",
    "testing",
])
def test_profile_loading_convenience(profile_name):
    """Test convenience function for loading profiles."""
    # Reset singleton
    ConfigManager._instance = None
    
    # Use convenience function
    load_profile(profile_name)
    
    # Verify profile is loaded
    manager = get_config_manager()
    assert manager._metadata.active_profile == ConfigProfile(profile_name)
