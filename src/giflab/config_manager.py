"""Production configuration management system for GifLab.

This module provides:
- Hierarchical configuration with defaults, profiles, and overrides
- Dynamic configuration reloading without service restart
- Configuration validation and sanity checking
- Environment-specific configuration profiles
"""

import copy
import hashlib
import json
import logging
import os
import signal
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Union

# Optional file watching support
try:
    from watchdog.events import FileModifiedEvent, FileSystemEventHandler
    from watchdog.observers import Observer

    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    Observer = None
    FileSystemEventHandler = object
    FileModifiedEvent = None

from .config import (
    FRAME_CACHE,
    FRAME_SAMPLING,
    MONITORING,
    VALIDATION_CACHE,
    CompressionConfig,
    EngineConfig,
    MetricsConfig,
    PathConfig,
    ValidationConfig,
)

logger = logging.getLogger(__name__)


class ConfigProfile(Enum):
    """Available configuration profiles for different environments."""

    DEVELOPMENT = "development"
    PRODUCTION = "production"
    HIGH_MEMORY = "high_memory"
    LOW_MEMORY = "low_memory"
    HIGH_THROUGHPUT = "high_throughput"
    INTERACTIVE = "interactive"
    TESTING = "testing"


class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""

    pass


class ConfigReloadError(Exception):
    """Raised when configuration reload fails."""

    pass


@dataclass
class ConfigChange:
    """Represents a configuration change event."""

    timestamp: float
    old_value: Any
    new_value: Any
    path: str  # Dot-separated path to the config value
    profile: ConfigProfile | None = None
    source: str = "manual"  # manual, file_watch, signal, api


@dataclass
class ConfigMetadata:
    """Metadata about configuration state."""

    version: str = "1.0.0"
    last_reload: float | None = None
    reload_count: int = 0
    active_profile: ConfigProfile | None = None
    overrides: dict[str, Any] = field(default_factory=dict)
    validation_errors: list[str] = field(default_factory=list)
    checksum: str | None = None


class ConfigValidator:
    """Validates configuration values and relationships."""

    def __init__(self):
        self.validators: dict[str, list[Callable]] = {}
        self.relationship_validators: list[Callable] = []
        self._register_default_validators()

    def _register_default_validators(self):
        """Register built-in validators for known configuration sections."""

        # Frame cache validators
        self.add_validator("FRAME_CACHE.memory_limit_mb", self._validate_positive_int)
        self.add_validator(
            "FRAME_CACHE.memory_limit_mb",
            lambda x: x <= 8192,
            "Memory limit cannot exceed 8GB",
        )
        self.add_validator("FRAME_CACHE.disk_limit_mb", self._validate_positive_int)
        self.add_validator("FRAME_CACHE.ttl_seconds", self._validate_positive_int)

        # Validation cache validators
        self.add_validator(
            "VALIDATION_CACHE.memory_limit_mb", self._validate_positive_int
        )
        self.add_validator(
            "VALIDATION_CACHE.disk_limit_mb", self._validate_positive_int
        )
        self.add_validator("VALIDATION_CACHE.ttl_seconds", self._validate_positive_int)

        # Frame sampling validators
        self.add_validator(
            "FRAME_SAMPLING.min_frames_threshold",
            lambda x: 1 <= x <= 1000,
            "Frame threshold must be between 1 and 1000",
        )
        self.add_validator(
            "FRAME_SAMPLING.confidence_level",
            lambda x: 0.5 <= x <= 0.99,
            "Confidence level must be between 0.5 and 0.99",
        )

        # Monitoring validators
        self.add_validator(
            "MONITORING.buffer_size",
            lambda x: 100 <= x <= 100000,
            "Buffer size must be between 100 and 100000",
        )
        self.add_validator(
            "MONITORING.sampling_rate",
            lambda x: 0.0 < x <= 1.0,
            "Sampling rate must be between 0 and 1",
        )

        # Relationship validators
        self.add_relationship_validator(self._validate_cache_memory_relationship)
        self.add_relationship_validator(self._validate_sampling_strategies)

    def _validate_positive_int(self, value: Any) -> bool:
        """Validate that a value is a positive integer."""
        return isinstance(value, int) and value > 0

    def _validate_cache_memory_relationship(self, config: dict[str, Any]) -> str | None:
        """Validate cache memory relationships."""
        frame_cache = config.get("FRAME_CACHE", {})
        validation_cache = config.get("VALIDATION_CACHE", {})

        total_memory = (
            frame_cache.get("memory_limit_mb", 0)
            + validation_cache.get("memory_limit_mb", 0)
            + frame_cache.get("resize_cache_memory_mb", 0)
        )

        # Warn if total cache memory exceeds 2GB
        if total_memory > 2048:
            return f"Total cache memory ({total_memory}MB) exceeds recommended 2GB"

        return None

    def _validate_sampling_strategies(self, config: dict[str, Any]) -> str | None:
        """Validate frame sampling configuration."""
        sampling = config.get("FRAME_SAMPLING", {})
        if not sampling.get("enabled", False):
            return None

        strategy = sampling.get("default_strategy", "")
        valid_strategies = ["uniform", "adaptive", "progressive", "scene_aware", "full"]

        if strategy not in valid_strategies:
            return f"Invalid sampling strategy: {strategy}"

        # Validate strategy-specific config exists
        if strategy != "full" and strategy not in sampling:
            return f"Missing configuration for strategy: {strategy}"

        return None

    def add_validator(
        self, path: str, validator: Callable, error_msg: str | None = None
    ):
        """Add a validator for a specific configuration path."""
        if path not in self.validators:
            self.validators[path] = []

        if error_msg:
            # Wrap validator with custom error message
            def wrapped(value):
                if not validator(value):
                    raise ConfigValidationError(f"{path}: {error_msg}")
                return True

            self.validators[path].append(wrapped)
        else:
            self.validators[path].append(validator)

    def add_relationship_validator(self, validator: Callable):
        """Add a validator that checks relationships between config values."""
        self.relationship_validators.append(validator)

    def validate(self, config: dict[str, Any]) -> list[str]:
        """Validate a configuration dictionary.

        Returns:
            list of validation errors (empty if valid)
        """
        errors = []

        # Validate individual values
        for path, validators in self.validators.items():
            value = self._get_value_by_path(config, path)
            if value is None:
                continue

            for validator in validators:
                try:
                    validator(value)
                except ConfigValidationError as e:
                    errors.append(str(e))
                except Exception as e:
                    errors.append(f"{path}: Validation error - {e}")

        # Validate relationships
        for validator in self.relationship_validators:
            error = validator(config)
            if error:
                errors.append(error)

        return errors

    def _get_value_by_path(self, config: dict[str, Any], path: str) -> Any:
        """Get a value from nested dict using dot-separated path."""
        parts = path.split(".")
        value = config

        for part in parts:
            if isinstance(value, dict):
                value = value.get(part)
                if value is None:
                    return None
            else:
                return None

        return value


class ConfigFileWatcher(FileSystemEventHandler):
    """Watches configuration files for changes."""

    def __init__(self, config_manager: "ConfigManager"):
        self.config_manager = config_manager
        self.last_reload = 0
        self.reload_cooldown = 1.0  # Minimum seconds between reloads

    def on_modified(self, event):
        """Handle file modification events."""
        if not WATCHDOG_AVAILABLE or not isinstance(event, FileModifiedEvent):
            return

        # Check if it's a config file
        if not event.src_path.endswith((".py", ".json", ".yaml", ".yml")):
            return

        # Avoid rapid reloads
        current_time = time.time()
        if current_time - self.last_reload < self.reload_cooldown:
            return

        self.last_reload = current_time

        logger.info(f"Configuration file modified: {event.src_path}")

        # Trigger reload in config manager
        try:
            self.config_manager.reload_config(source="file_watch")
        except Exception as e:
            logger.error(f"Failed to reload configuration: {e}")


class ConfigManager:
    """Manages application configuration with profiles and dynamic reloading."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        """Ensure singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize configuration manager."""
        if hasattr(self, "_initialized"):
            return

        self._initialized = True
        self._config: dict[str, Any] = {}
        self._profiles: dict[ConfigProfile, dict[str, Any]] = {}
        self._metadata = ConfigMetadata()
        self._validator = ConfigValidator()
        self._change_history: list[ConfigChange] = []
        self._change_callbacks: list[Callable] = []
        self._observer: Observer | None = None
        self._config_lock = threading.RLock()

        # Load default configuration
        self._load_defaults()

        # Load profiles
        self._load_profiles()

        # Apply environment overrides
        self._apply_environment_overrides()

        # Set up signal handler for manual reload
        self._setup_signal_handler()

    def _load_defaults(self):
        """Load default configuration from existing config module."""
        with self._config_lock:
            # Load dataclass configs
            self._config["compression"] = asdict(CompressionConfig())
            self._config["metrics"] = asdict(MetricsConfig())
            self._config["paths"] = asdict(PathConfig())
            self._config["engine"] = asdict(EngineConfig())
            self._config["validation"] = asdict(ValidationConfig())

            # Load dictionary configs
            self._config["FRAME_CACHE"] = copy.deepcopy(FRAME_CACHE)
            self._config["FRAME_SAMPLING"] = copy.deepcopy(FRAME_SAMPLING)
            self._config["VALIDATION_CACHE"] = copy.deepcopy(VALIDATION_CACHE)
            self._config["MONITORING"] = copy.deepcopy(MONITORING)

            # Calculate checksum
            self._update_checksum()

    def _load_profiles(self):
        """Load configuration profiles for different environments."""

        # Development profile - aggressive caching, verbose logging
        self._profiles[ConfigProfile.DEVELOPMENT] = {
            "FRAME_CACHE": {
                "memory_limit_mb": 1000,
                "disk_limit_mb": 5000,
                "ttl_seconds": 3600,  # 1 hour
            },
            "VALIDATION_CACHE": {
                "memory_limit_mb": 200,
                "disk_limit_mb": 2000,
                "ttl_seconds": 3600,
            },
            "MONITORING": {
                "verbose": True,
                "sampling_rate": 1.0,
                "buffer_size": 50000,
            },
            "FRAME_SAMPLING": {
                "enabled": False,  # Full processing in dev
            },
        }

        # Production profile - balanced for stability
        self._profiles[ConfigProfile.PRODUCTION] = {
            "FRAME_CACHE": {
                "memory_limit_mb": 500,
                "disk_limit_mb": 2000,
                "ttl_seconds": 86400,  # 24 hours
            },
            "VALIDATION_CACHE": {
                "memory_limit_mb": 100,
                "disk_limit_mb": 1000,
                "ttl_seconds": 172800,  # 48 hours
            },
            "MONITORING": {
                "verbose": False,
                "sampling_rate": 0.1,  # Sample 10% in production
                "buffer_size": 10000,
            },
            "FRAME_SAMPLING": {
                "enabled": True,
                "default_strategy": "adaptive",
            },
        }

        # High memory profile - for memory-rich environments
        self._profiles[ConfigProfile.HIGH_MEMORY] = {
            "FRAME_CACHE": {
                "memory_limit_mb": 1200,
                "disk_limit_mb": 10000,
                "ttl_seconds": 172800,
                "resize_cache_memory_mb": 300,
            },
            "VALIDATION_CACHE": {
                "memory_limit_mb": 300,
                "disk_limit_mb": 5000,
            },
            "FRAME_SAMPLING": {
                "enabled": False,  # Process all frames
            },
            "MONITORING": {
                "buffer_size": 100000,
            },
        }

        # Low memory profile - for constrained environments
        self._profiles[ConfigProfile.LOW_MEMORY] = {
            "FRAME_CACHE": {
                "memory_limit_mb": 100,
                "disk_limit_mb": 500,
                "ttl_seconds": 3600,
                "resize_cache_memory_mb": 50,
            },
            "VALIDATION_CACHE": {
                "memory_limit_mb": 25,
                "disk_limit_mb": 250,
                "ttl_seconds": 3600,
            },
            "FRAME_SAMPLING": {
                "enabled": True,
                "default_strategy": "uniform",
                "uniform": {
                    "sampling_rate": 0.2,  # Sample only 20%
                },
            },
            "metrics": {
                "LPIPS_MAX_FRAMES": 50,
                "SSIM_MAX_FRAMES": 20,
            },
            "MONITORING": {
                "buffer_size": 1000,
                "sampling_rate": 0.05,
            },
        }

        # High throughput profile - optimized for batch processing
        self._profiles[ConfigProfile.HIGH_THROUGHPUT] = {
            "FRAME_CACHE": {
                "memory_limit_mb": 1000,
                "disk_limit_mb": 5000,
                "ttl_seconds": 3600,
            },
            "VALIDATION_CACHE": {
                "enabled": True,
                "memory_limit_mb": 200,
                "disk_limit_mb": 2000,
            },
            "FRAME_SAMPLING": {
                "enabled": True,
                "default_strategy": "progressive",
                "confidence_level": 0.90,  # Lower confidence for speed
            },
            "metrics": {
                "SSIM_MODE": "optimized",  # Faster SSIM
                "ENABLE_DEEP_PERCEPTUAL": False,  # Skip expensive metrics
            },
            "MONITORING": {
                "enabled": False,  # Reduce overhead
            },
        }

        # Interactive profile - low latency for real-time use
        self._profiles[ConfigProfile.INTERACTIVE] = {
            "FRAME_CACHE": {
                "memory_limit_mb": 300,
                "ttl_seconds": 300,  # 5 minutes
            },
            "VALIDATION_CACHE": {
                "memory_limit_mb": 50,
                "ttl_seconds": 600,
            },
            "FRAME_SAMPLING": {
                "enabled": True,
                "default_strategy": "uniform",
                "uniform": {
                    "sampling_rate": 0.1,  # Quick preview
                },
            },
            "metrics": {
                "SSIM_MODE": "fast",
                "SSIM_MAX_FRAMES": 10,
                "ENABLE_DEEP_PERCEPTUAL": False,
            },
            "MONITORING": {
                "enabled": False,
            },
        }

        # Testing profile - for unit tests
        self._profiles[ConfigProfile.TESTING] = {
            "FRAME_CACHE": {
                "enabled": False,  # Disable caching in tests
            },
            "VALIDATION_CACHE": {
                "enabled": False,
            },
            "FRAME_SAMPLING": {
                "enabled": False,
            },
            "MONITORING": {
                "enabled": False,
            },
            "validation": {
                "FAIL_ON_VALIDATION_ERROR": True,
            },
        }

    def _apply_environment_overrides(self):
        """Apply configuration overrides from environment variables."""
        # Format: GIFLAB_CONFIG_<SECTION>_<KEY>=value
        # Example: GIFLAB_CONFIG_FRAME_CACHE_MEMORY_LIMIT_MB=1000

        prefix = "GIFLAB_CONFIG_"
        for env_key, env_value in os.environ.items():
            if not env_key.startswith(prefix):
                continue

            # Parse the environment variable
            config_path = env_key[len(prefix) :].lower()
            parts = config_path.split("_")

            if len(parts) < 2:
                continue

            # Try to parse the value
            try:
                # Try parsing as JSON first (for complex types)
                value = json.loads(env_value)
            except json.JSONDecodeError:
                # Fall back to string
                value = env_value

                # Try to convert to appropriate type
                if env_value.lower() in ("true", "false"):
                    value = env_value.lower() == "true"
                elif env_value.isdigit():
                    value = int(env_value)
                else:
                    try:
                        value = float(env_value)
                    except ValueError:
                        pass

            # Apply the override
            self._set_value_by_parts(self._config, parts, value)

            logger.info(f"Applied environment override: {config_path} = {value}")

    def _set_value_by_parts(self, config: dict[str, Any], parts: list[str], value: Any):
        """Set a value in nested dict using parts list."""
        # Navigate to the parent
        current = config
        for part in parts[:-1]:
            # Try exact match first
            if part in current:
                current = current[part]
            # Try uppercase match for dict configs
            elif part.upper() in current:
                current = current[part.upper()]
            else:
                # Create missing intermediate dicts
                current[part] = {}
                current = current[part]

        # Set the final value
        final_key = parts[-1]
        if final_key.upper() in current:
            current[final_key.upper()] = value
        else:
            current[final_key] = value

    def _setup_signal_handler(self):
        """Set up signal handler for configuration reload."""

        def handle_reload_signal(signum, frame):
            logger.info("Received SIGHUP signal, reloading configuration")
            try:
                self.reload_config(source="signal")
            except Exception as e:
                logger.error(f"Failed to reload configuration: {e}")

        # Register SIGHUP handler (Unix only)
        if hasattr(signal, "SIGHUP"):
            signal.signal(signal.SIGHUP, handle_reload_signal)

    def _update_checksum(self):
        """Update configuration checksum."""

        def path_encoder(obj: Any) -> str:
            if isinstance(obj, Path):
                return str(obj)
            raise TypeError(f"Object of type {type(obj)} not serializable")

        config_str = json.dumps(self._config, sort_keys=True, default=path_encoder)
        self._metadata.checksum = hashlib.sha256(config_str.encode()).hexdigest()[:16]

    def start_file_watcher(self, watch_paths: list[Path] | None = None):
        """Start watching configuration files for changes."""
        if not WATCHDOG_AVAILABLE:
            logger.warning("File watching not available - install watchdog package")
            return

        if self._observer is not None:
            return

        if watch_paths is None:
            # Default to watching the config module directory
            import giflab

            watch_paths = [Path(giflab.__file__).parent]

        self._observer = Observer()
        handler = ConfigFileWatcher(self)

        for path in watch_paths:
            self._observer.schedule(handler, str(path), recursive=False)

        self._observer.start()
        logger.info(f"Started configuration file watcher for: {watch_paths}")

    def stop_file_watcher(self):
        """Stop watching configuration files."""
        if self._observer is not None:
            self._observer.stop()
            self._observer.join()
            self._observer = None
            logger.info("Stopped configuration file watcher")

    def load_profile(self, profile: ConfigProfile):
        """Load a configuration profile."""
        if profile not in self._profiles:
            raise ConfigValidationError(f"Unknown profile: {profile}")

        with self._config_lock:
            old_config = copy.deepcopy(self._config)

            # Apply profile settings
            profile_config = self._profiles[profile]
            self._merge_config(self._config, profile_config)

            # Validate the new configuration
            errors = self._validator.validate(self._config)
            if errors:
                # Rollback
                self._config = old_config
                raise ConfigValidationError(f"Profile validation failed: {errors}")

            # Update metadata
            self._metadata.active_profile = profile
            self._update_checksum()

            # Record changes
            self._record_profile_change(old_config, self._config, profile)

            logger.info(f"Loaded configuration profile: {profile.value}")

    def _merge_config(self, base: dict[str, Any], overlay: dict[str, Any]):
        """Recursively merge overlay configuration into base."""
        for key, value in overlay.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_config(base[key], value)
            else:
                base[key] = copy.deepcopy(value)

    def _record_profile_change(
        self,
        old_config: dict[str, Any],
        new_config: dict[str, Any],
        profile: ConfigProfile,
    ):
        """Record configuration changes from profile application."""
        changes = self._diff_configs(old_config, new_config)

        for path, (old_val, new_val) in changes.items():
            change = ConfigChange(
                timestamp=time.time(),
                old_value=old_val,
                new_value=new_val,
                path=path,
                profile=profile,
                source="profile",
            )
            self._change_history.append(change)

            # Notify callbacks
            self._notify_change(change)

    def _diff_configs(
        self, old: dict[str, Any], new: dict[str, Any], prefix: str = ""
    ) -> dict[str, tuple]:
        """Find differences between two configurations."""
        changes = {}

        all_keys = set(old.keys()) | set(new.keys())

        for key in all_keys:
            current_path = f"{prefix}.{key}" if prefix else key

            old_val = old.get(key)
            new_val = new.get(key)

            if old_val != new_val:
                if isinstance(old_val, dict) and isinstance(new_val, dict):
                    # Recurse into nested dicts
                    nested_changes = self._diff_configs(old_val, new_val, current_path)
                    changes.update(nested_changes)
                else:
                    changes[current_path] = (old_val, new_val)

        return changes

    def reload_config(self, source: str = "manual") -> bool:
        """Reload configuration with validation.

        Returns:
            True if reload successful, False otherwise
        """
        with self._config_lock:
            try:
                # Save current config for rollback
                old_config = copy.deepcopy(self._config)
                old_metadata = copy.deepcopy(self._metadata)

                # Reload defaults
                self._load_defaults()

                # Reapply active profile if set
                if self._metadata.active_profile:
                    profile_config = self._profiles[self._metadata.active_profile]
                    self._merge_config(self._config, profile_config)

                # Reapply overrides
                for path, value in self._metadata.overrides.items():
                    self._set_value_by_path(self._config, path, value)

                # Reapply environment overrides
                self._apply_environment_overrides()

                # Validate new configuration
                errors = self._validator.validate(self._config)
                if errors:
                    # Rollback
                    self._config = old_config
                    self._metadata = old_metadata
                    self._metadata.validation_errors = errors
                    raise ConfigReloadError(f"Validation failed: {errors}")

                # Update metadata
                self._metadata.last_reload = time.time()
                self._metadata.reload_count += 1
                self._metadata.validation_errors = []
                self._update_checksum()

                # Record changes
                changes = self._diff_configs(old_config, self._config)
                for path, (old_val, new_val) in changes.items():
                    change = ConfigChange(
                        timestamp=time.time(),
                        old_value=old_val,
                        new_value=new_val,
                        path=path,
                        source=source,
                    )
                    self._change_history.append(change)
                    self._notify_change(change)

                logger.info(f"Configuration reloaded successfully from {source}")
                return True

            except Exception as e:
                logger.error(f"Configuration reload failed: {e}")
                return False

    def _set_value_by_path(self, config: dict[str, Any], path: str, value: Any):
        """Set a value in nested dict using dot-separated path."""
        parts = path.split(".")
        current = config

        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]

        current[parts[-1]] = value

    def get(self, path: str, default: Any = None) -> Any:
        """Get a configuration value by path."""
        parts = path.split(".")
        value = self._config

        for part in parts:
            if isinstance(value, dict):
                value = value.get(part)
                if value is None:
                    return default
            else:
                return default

        return value

    def set(self, path: str, value: Any, validate: bool = True):
        """Set a configuration value with optional validation."""
        with self._config_lock:
            old_value = self.get(path)

            # Apply the change
            self._set_value_by_path(self._config, path, value)

            # Validate if requested
            if validate:
                errors = self._validator.validate(self._config)
                if errors:
                    # Rollback
                    self._set_value_by_path(self._config, path, old_value)
                    raise ConfigValidationError(f"Validation failed: {errors}")

            # Record override
            self._metadata.overrides[path] = value
            self._update_checksum()

            # Record change
            change = ConfigChange(
                timestamp=time.time(),
                old_value=old_value,
                new_value=value,
                path=path,
                source="api",
            )
            self._change_history.append(change)
            self._notify_change(change)

    def register_change_callback(self, callback: Callable[[ConfigChange], None]):
        """Register a callback for configuration changes."""
        self._change_callbacks.append(callback)

    def _notify_change(self, change: ConfigChange):
        """Notify all registered callbacks of a configuration change."""
        for callback in self._change_callbacks:
            try:
                callback(change)
            except Exception as e:
                logger.error(f"Error in config change callback: {e}")

    def get_metadata(self) -> ConfigMetadata:
        """Get configuration metadata."""
        return copy.deepcopy(self._metadata)

    def get_change_history(self, limit: int | None = None) -> list[ConfigChange]:
        """Get configuration change history."""
        history = self._change_history[-limit:] if limit else self._change_history
        return copy.deepcopy(history)

    def export_config(self, include_defaults: bool = True) -> dict[str, Any]:
        """Export current configuration."""
        with self._config_lock:
            config = copy.deepcopy(self._config)

            if not include_defaults:
                # Only include overrides and profile changes
                # This would require tracking which values are defaults
                pass

            return {
                "config": config,
                "metadata": asdict(self._metadata),
                "profile": self._metadata.active_profile.value
                if self._metadata.active_profile
                else None,
            }

    def import_config(self, config_data: dict[str, Any], validate: bool = True):
        """Import configuration from exported data."""
        with self._config_lock:
            old_config = copy.deepcopy(self._config)
            old_metadata = copy.deepcopy(self._metadata)

            try:
                # Apply imported config
                self._config = config_data.get("config", {})

                # Apply profile if specified
                profile_name = config_data.get("profile")
                if profile_name:
                    profile = ConfigProfile(profile_name)
                    if profile in self._profiles:
                        self._metadata.active_profile = profile

                # Validate if requested
                if validate:
                    errors = self._validator.validate(self._config)
                    if errors:
                        raise ConfigValidationError(
                            f"Import validation failed: {errors}"
                        )

                self._update_checksum()
                logger.info("Configuration imported successfully")

            except Exception as e:
                # Rollback
                self._config = old_config
                self._metadata = old_metadata
                raise ConfigReloadError(f"Configuration import failed: {e}")

    @property
    def config(self) -> dict[str, Any]:
        """Get the current configuration (read-only)."""
        return copy.deepcopy(self._config)


# Global instance
_config_manager: ConfigManager | None = None


def get_config_manager() -> ConfigManager:
    """Get the global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def load_profile(profile: str | ConfigProfile):
    """Convenience function to load a configuration profile."""
    if isinstance(profile, str):
        profile = ConfigProfile(profile)

    manager = get_config_manager()
    manager.load_profile(profile)


def get_config(path: str, default: Any = None) -> Any:
    """Convenience function to get a configuration value."""
    manager = get_config_manager()
    return manager.get(path, default)


def set_config(path: str, value: Any, validate: bool = True):
    """Convenience function to set a configuration value."""
    manager = get_config_manager()
    manager.set(path, value, validate)
