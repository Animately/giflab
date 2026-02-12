"""Configuration validation framework for GifLab.

This module provides comprehensive validation for configuration values,
including type checking, range validation, dependency validation, and
resource availability checks.
"""

import os
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any, Optional, Union


class ValidationRule:
    """Represents a validation rule for a configuration value."""
    
    def __init__(self,
                 name: str,
                 validator: Callable[[Any], bool],
                 error_message: str,
                 severity: str = "error"):
        """Initialize a validation rule.
        
        Args:
            name: Name of the rule
            validator: Function that returns True if valid
            error_message: Error message if validation fails
            severity: "error", "warning", or "info"
        """
        self.name = name
        self.validator = validator
        self.error_message = error_message
        self.severity = severity
    
    def validate(self, value: Any) -> tuple[bool, str | None]:
        """Validate a value against this rule.
        
        Returns:
            tuple of (is_valid, error_message)
        """
        try:
            if self.validator(value):
                return True, None
            return False, self.error_message
        except Exception as e:
            return False, f"{self.error_message}: {e}"

class TypeValidator:
    """Validators for type checking."""
    
    @staticmethod
    def is_int(value: Any) -> bool:
        """Check if value is an integer."""
        return isinstance(value, int) and not isinstance(value, bool)
    
    @staticmethod
    def is_float(value: Any) -> bool:
        """Check if value is a float."""
        return isinstance(value, int | float) and not isinstance(value, bool)
    
    @staticmethod
    def is_bool(value: Any) -> bool:
        """Check if value is a boolean."""
        return isinstance(value, bool)
    
    @staticmethod
    def is_str(value: Any) -> bool:
        """Check if value is a string."""
        return isinstance(value, str)
    
    @staticmethod
    def is_dict(value: Any) -> bool:
        """Check if value is a dictionary."""
        return isinstance(value, dict)
    
    @staticmethod
    def is_list(value: Any) -> bool:
        """Check if value is a list."""
        return isinstance(value, list)
    
    @staticmethod
    def is_path(value: Any) -> bool:
        """Check if value is a valid path string or Path object."""
        if isinstance(value, Path):
            return True
        if isinstance(value, str):
            try:
                Path(value)
                return True
            except Exception:
                return False
        return False

class RangeValidator:
    """Validators for range checking."""
    
    @staticmethod
    def in_range(min_val: float | None = None,
                 max_val: float | None = None) -> Callable:
        """Create a range validator."""
        def validator(value: Any) -> bool:
            if not isinstance(value, int | float):
                return False
            if min_val is not None and value < min_val:
                return False
            if max_val is not None and value > max_val:
                return False
            return True
        return validator
    
    @staticmethod
    def positive(value: Any) -> bool:
        """Check if value is positive."""
        return isinstance(value, int | float) and value > 0
    
    @staticmethod
    def non_negative(value: Any) -> bool:
        """Check if value is non-negative."""
        return isinstance(value, int | float) and value >= 0
    
    @staticmethod
    def percentage(value: Any) -> bool:
        """Check if value is a valid percentage (0-1)."""
        return isinstance(value, int | float) and 0 <= value <= 1
    
    @staticmethod
    def port_number(value: Any) -> bool:
        """Check if value is a valid port number."""
        return isinstance(value, int) and 1 <= value <= 65535

class ResourceValidator:
    """Validators for system resources."""
    
    @staticmethod
    def memory_available(required_mb: int) -> Callable:
        """Check if enough memory is available."""
        def validator(value: Any) -> bool:
            if not isinstance(value, int):
                return False
            
            # Check system memory
            try:
                import psutil
                available_mb = psutil.virtual_memory().available / (1024 * 1024)
                return value <= available_mb
            except ImportError:
                # If psutil not available, just check reasonable limits
                return value <= 8192  # Max 8GB
        return validator
    
    @staticmethod
    def disk_space_available(path: str | Path, required_mb: int) -> bool:
        """Check if enough disk space is available."""
        try:
            path = Path(path) if isinstance(path, str) else path
            if not path.exists():
                path = path.parent
            
            stat = shutil.disk_usage(path)
            available_mb = stat.free / (1024 * 1024)
            return available_mb >= required_mb
        except Exception:
            return False
    
    @staticmethod
    def executable_exists(name: str) -> bool:
        """Check if an executable exists in PATH."""
        return shutil.which(name) is not None
    
    @staticmethod
    def file_exists(path: str | Path) -> bool:
        """Check if a file exists."""
        try:
            path = Path(path) if isinstance(path, str) else path
            return path.exists() and path.is_file()
        except Exception:
            return False
    
    @staticmethod
    def directory_exists(path: str | Path) -> bool:
        """Check if a directory exists."""
        try:
            path = Path(path) if isinstance(path, str) else path
            return path.exists() and path.is_dir()
        except Exception:
            return False
    
    @staticmethod
    def directory_writable(path: str | Path) -> bool:
        """Check if a directory is writable."""
        try:
            path = Path(path) if isinstance(path, str) else path
            if not path.exists():
                # Check parent directory
                path = path.parent
            
            test_file = path / ".write_test"
            try:
                test_file.touch()
                test_file.unlink()
                return True
            except Exception:
                return False
        except Exception:
            return False

class DependencyValidator:
    """Validators for configuration dependencies."""
    
    @staticmethod
    def requires(other_path: str, condition: Callable) -> Callable:
        """Create a validator that depends on another config value."""
        def validator(value: Any, config: dict[str, Any]) -> bool:
            other_value = _get_value_by_path(config, other_path)
            return condition(other_value, value)
        return validator
    
    @staticmethod
    def mutually_exclusive(*paths: str) -> Callable:
        """Create a validator for mutually exclusive options."""
        def validator(value: Any, config: dict[str, Any]) -> bool:
            if not value:
                return True
            
            for path in paths:
                other_value = _get_value_by_path(config, path)
                if other_value:
                    return False
            return True
        return validator
    
    @staticmethod
    def sum_equals(target: float, *paths: str) -> Callable:
        """Create a validator that checks if values sum to target."""
        def validator(value: Any, config: dict[str, Any]) -> bool:
            total = value if isinstance(value, int | float) else 0
            
            for path in paths:
                other_value = _get_value_by_path(config, path)
                if isinstance(other_value, int | float):
                    total += other_value
            
            return abs(total - target) < 1e-6
        return validator

class ConfigurationValidator:
    """Main configuration validator with comprehensive rules."""
    
    def __init__(self):
        self.rules: dict[str, list[ValidationRule]] = {}
        self._register_default_rules()
    
    def _register_default_rules(self):
        """Register default validation rules for known configurations."""
        
        # Frame cache rules
        self.add_rule("FRAME_CACHE.enabled",
                     ValidationRule("type", TypeValidator.is_bool,
                                   "enabled must be a boolean"))
        
        self.add_rule("FRAME_CACHE.memory_limit_mb",
                     ValidationRule("type", TypeValidator.is_int,
                                   "memory_limit_mb must be an integer"))
        self.add_rule("FRAME_CACHE.memory_limit_mb",
                     ValidationRule("range", RangeValidator.in_range(10, 8192),
                                   "memory_limit_mb must be between 10 and 8192"))
        
        self.add_rule("FRAME_CACHE.disk_limit_mb",
                     ValidationRule("positive", RangeValidator.positive,
                                   "disk_limit_mb must be positive"))
        
        self.add_rule("FRAME_CACHE.ttl_seconds",
                     ValidationRule("range", RangeValidator.in_range(60, 604800),
                                   "ttl_seconds must be between 60 and 604800 (1 week)"))
        
        # Validation cache rules
        self.add_rule("VALIDATION_CACHE.memory_limit_mb",
                     ValidationRule("range", RangeValidator.in_range(10, 2048),
                                   "memory_limit_mb must be between 10 and 2048"))
        
        # Frame sampling rules
        self.add_rule("FRAME_SAMPLING.enabled",
                     ValidationRule("type", TypeValidator.is_bool,
                                   "enabled must be a boolean"))
        
        self.add_rule("FRAME_SAMPLING.min_frames_threshold",
                     ValidationRule("range", RangeValidator.in_range(1, 1000),
                                   "min_frames_threshold must be between 1 and 1000"))
        
        self.add_rule("FRAME_SAMPLING.confidence_level",
                     ValidationRule("range", RangeValidator.in_range(0.5, 0.99),
                                   "confidence_level must be between 0.5 and 0.99"))
        
        self.add_rule("FRAME_SAMPLING.default_strategy",
                     ValidationRule("enum",
                                   lambda x: x in ["uniform", "adaptive", "progressive",
                                                  "scene_aware", "full"],
                                   "default_strategy must be a valid strategy"))
        
        # Monitoring rules
        self.add_rule("MONITORING.backend",
                     ValidationRule("enum",
                                   lambda x: x in ["memory", "sqlite", "statsd"],
                                   "backend must be memory, sqlite, or statsd"))
        
        self.add_rule("MONITORING.buffer_size",
                     ValidationRule("range", RangeValidator.in_range(100, 100000),
                                   "buffer_size must be between 100 and 100000"))
        
        self.add_rule("MONITORING.sampling_rate",
                     ValidationRule("percentage", RangeValidator.percentage,
                                   "sampling_rate must be between 0 and 1"))
        
        # Engine path rules
        self.add_rule("engine.GIFSICLE_PATH",
                     ValidationRule("executable", ResourceValidator.executable_exists,
                                   "gifsicle executable not found",
                                   severity="warning"))
        
        self.add_rule("engine.ANIMATELY_PATH",
                     ValidationRule("executable", ResourceValidator.executable_exists,
                                   "animately executable not found",
                                   severity="warning"))
        
        # Metrics weight rules
        self.add_rule("metrics.SSIM_WEIGHT",
                     ValidationRule("percentage", RangeValidator.percentage,
                                   "SSIM_WEIGHT must be between 0 and 1"))
        
        # Path rules
        self.add_rule("paths.RAW_DIR",
                     ValidationRule("type", TypeValidator.is_path,
                                   "RAW_DIR must be a valid path"))
        
        self.add_rule("paths.LOGS_DIR",
                     ValidationRule("writable", ResourceValidator.directory_writable,
                                   "LOGS_DIR must be writable",
                                   severity="warning"))
    
    def add_rule(self, path: str, rule: ValidationRule):
        """Add a validation rule for a configuration path."""
        if path not in self.rules:
            self.rules[path] = []
        self.rules[path].append(rule)
    
    def validate(self, config: dict[str, Any]) -> dict[str, list[str]]:
        """Validate a configuration dictionary.
        
        Returns:
            Dictionary mapping severity to list of messages
        """
        results = {
            "error": [],
            "warning": [],
            "info": []
        }
        
        for path, rules in self.rules.items():
            value = _get_value_by_path(config, path)
            if value is None:
                continue
            
            for rule in rules:
                # Check if validator needs config context
                if hasattr(rule.validator, "__code__") and \
                   rule.validator.__code__.co_argcount > 1:
                    # Validator needs config context
                    is_valid, error_msg = rule.validate_with_context(value, config)
                else:
                    is_valid, error_msg = rule.validate(value)
                
                if not is_valid and error_msg:
                    results[rule.severity].append(f"{path}: {error_msg}")
        
        return results
    
    def validate_relationships(self, config: dict[str, Any]) -> list[str]:
        """Validate relationships between configuration values."""
        errors = []
        
        # Check total cache memory doesn't exceed reasonable limits
        total_cache_memory = 0
        cache_configs = ["FRAME_CACHE", "VALIDATION_CACHE"]
        
        for cache in cache_configs:
            if cache in config:
                memory_limit = config[cache].get("memory_limit_mb", 0)
                total_cache_memory += memory_limit
                
                # Check resize cache if in FRAME_CACHE
                if cache == "FRAME_CACHE":
                    resize_memory = config[cache].get("resize_cache_memory_mb", 0)
                    total_cache_memory += resize_memory
        
        if total_cache_memory > 4096:
            errors.append(f"Total cache memory ({total_cache_memory}MB) exceeds "
                         f"recommended limit of 4096MB")
        
        # Check disk cache paths don't conflict
        disk_paths = []
        if "FRAME_CACHE" in config and config["FRAME_CACHE"].get("disk_path"):
            disk_paths.append(config["FRAME_CACHE"]["disk_path"])
        if "VALIDATION_CACHE" in config and config["VALIDATION_CACHE"].get("disk_path"):
            disk_paths.append(config["VALIDATION_CACHE"]["disk_path"])
        
        if len(disk_paths) > 1 and len(set(disk_paths)) < len(disk_paths):
            errors.append("Cache disk paths must be unique")
        
        # Check sampling configuration consistency
        if "FRAME_SAMPLING" in config:
            sampling = config["FRAME_SAMPLING"]
            if sampling.get("enabled"):
                strategy = sampling.get("default_strategy")
                if strategy and strategy != "full" and strategy not in sampling:
                    errors.append(f"Missing configuration for sampling strategy: {strategy}")
        
        # Check metrics weights sum to 1.0
        if "metrics" in config:
            metrics = config["metrics"]
            
            # Check legacy weights
            legacy_weights = [
                metrics.get("SSIM_WEIGHT", 0),
                metrics.get("MS_SSIM_WEIGHT", 0),
                metrics.get("PSNR_WEIGHT", 0),
                metrics.get("TEMPORAL_WEIGHT", 0)
            ]
            
            if any(w > 0 for w in legacy_weights):
                total = sum(legacy_weights)
                if abs(total - 1.0) > 1e-6:
                    errors.append(f"Legacy metric weights must sum to 1.0, got {total}")
            
            # Check enhanced weights if enabled
            if metrics.get("USE_ENHANCED_COMPOSITE_QUALITY"):
                enhanced_weights = [
                    metrics.get(f"ENHANCED_{name}_WEIGHT", 0)
                    for name in ["SSIM", "MS_SSIM", "PSNR", "MSE", "FSIM",
                                "EDGE", "GMSD", "CHIST", "SHARPNESS",
                                "TEXTURE", "TEMPORAL", "LPIPS", "SSIMULACRA2"]
                ]
                
                total = sum(enhanced_weights)
                if abs(total - 1.0) > 1e-6:
                    errors.append(f"Enhanced metric weights must sum to 1.0, got {total}")
        
        return errors
    
    def check_resources(self, config: dict[str, Any]) -> list[str]:
        """Check if required system resources are available."""
        warnings = []
        
        # Check memory availability
        try:
            import psutil
            available_mb = psutil.virtual_memory().available / (1024 * 1024)
            
            total_configured = 0
            if "FRAME_CACHE" in config:
                total_configured += config["FRAME_CACHE"].get("memory_limit_mb", 0)
            if "VALIDATION_CACHE" in config:
                total_configured += config["VALIDATION_CACHE"].get("memory_limit_mb", 0)
            
            if total_configured > available_mb * 0.5:
                warnings.append(f"Configured cache memory ({total_configured}MB) exceeds "
                               f"50% of available memory ({available_mb:.0f}MB)")
        except ImportError:
            pass
        
        # Check disk space for cache directories
        cache_paths = []
        if "FRAME_CACHE" in config and config["FRAME_CACHE"].get("disk_path"):
            cache_paths.append((config["FRAME_CACHE"]["disk_path"],
                              config["FRAME_CACHE"].get("disk_limit_mb", 0)))
        
        for path, limit_mb in cache_paths:
            if not ResourceValidator.disk_space_available(path, limit_mb):
                warnings.append(f"Insufficient disk space for cache at {path} "
                               f"(needs {limit_mb}MB)")
        
        # Check for required executables
        if "engine" in config:
            for engine, path in config["engine"].items():
                if path and not ResourceValidator.executable_exists(path):
                    warnings.append(f"{engine} executable not found: {path}")
        
        return warnings

def _get_value_by_path(config: dict[str, Any], path: str) -> Any:
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

def validate_config_file(config_path: Path) -> dict[str, list[str]]:
    """Validate a configuration file.
    
    Args:
        config_path: Path to configuration file (JSON or Python module)
    
    Returns:
        Dictionary with validation results by severity
    """
    import json
    
    # Load configuration
    if config_path.suffix == ".json":
        with open(config_path) as f:
            config = json.load(f)
    elif config_path.suffix == ".py":
        # Import Python module
        import importlib.util
        spec = importlib.util.spec_from_file_location("config", config_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Extract configuration
        config = {}
        for name in dir(module):
            if not name.startswith("_"):
                value = getattr(module, name)
                if isinstance(value, dict):
                    config[name] = value
    else:
        raise ValueError(f"Unsupported config file type: {config_path.suffix}")
    
    # Validate
    validator = ConfigurationValidator()
    results = validator.validate(config)
    
    # Add relationship errors
    relationship_errors = validator.validate_relationships(config)
    results["error"].extend(relationship_errors)
    
    # Add resource warnings
    resource_warnings = validator.check_resources(config)
    results["warning"].extend(resource_warnings)
    
    return results
