#!/usr/bin/env python3
"""
CI/CD check for preset configuration consistency.

This script validates that all presets maintain the correct distinction
between algorithm comparison and parameter sweeping. It can be run as
part of CI/CD pipelines to catch configuration errors early.

Usage:
    python tests/test_preset_configuration_consistency.py
    
Exit codes:
    0: All presets are correctly configured
    1: Configuration errors found
"""

import sys
import traceback
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from giflab.core.presets import EXPERIMENT_PRESETS


class PresetConfigurationValidator:
    """Validator for preset configuration consistency."""

    def __init__(self):
        self.errors = []
        self.warnings = []

    def validate_all_presets(self) -> bool:
        """Validate all presets and return True if all are valid."""
        print("🔍 Validating Preset Configuration Consistency")
        print("=" * 60)

        for preset_name, preset in EXPERIMENT_PRESETS.items():
            self._validate_single_preset(preset_name, preset)

        self._print_results()
        return len(self.errors) == 0

    def _validate_single_preset(self, name: str, preset):
        """Validate a single preset configuration."""
        try:
            # Test basic validation
            preset.__post_init__()

            # Test algorithm vs parameter distinction
            self._validate_experiment_type(name, preset)

            # Test naming consistency
            self._validate_naming_convention(name, preset)

            print(f"✅ {name}: Valid")

        except Exception as e:
            self.errors.append(f"❌ {name}: {str(e)}")
            print(f"❌ {name}: {str(e)}")

    def _validate_experiment_type(self, name: str, preset):
        """Validate that the preset has a clear experiment type."""
        # Special case: tool-comparison-baseline is intentionally broad
        if name == "tool-comparison-baseline":
            return  # Skip validation for this special preset

        algorithm_indicators = ["algorithm", "comparison", "focus", "study"]
        parameter_indicators = ["parameter", "sweep", "optimization"]

        is_algorithm_preset = any(
            indicator in name.lower() for indicator in algorithm_indicators
        )
        is_parameter_preset = any(
            indicator in name.lower() for indicator in parameter_indicators
        )

        for dimension in preset.vary_dimensions:
            is_locked = dimension in preset.locked_params
            has_tools_filter = preset.tools_filter is not None
            has_custom_range = (
                (dimension == "frame" and preset.frame_ratios is not None)
                or (dimension == "color" and preset.color_counts is not None)
                or (dimension == "lossy" and preset.lossy_levels is not None)
            )

            # Algorithm comparison pattern: locked param + no tools filter
            algorithm_pattern = is_locked and not has_tools_filter

            # Parameter sweep pattern: no locked param + tools filter + custom range
            parameter_pattern = not is_locked and has_tools_filter and has_custom_range

            # Mixed/unclear pattern
            mixed_pattern = not algorithm_pattern and not parameter_pattern

            # Check consistency with naming
            if is_algorithm_preset and not algorithm_pattern:
                self.errors.append(
                    f"Preset '{name}' suggests algorithm comparison but is configured for parameter sweep"
                )
            elif is_parameter_preset and not parameter_pattern:
                self.errors.append(
                    f"Preset '{name}' suggests parameter sweep but is configured for algorithm comparison"
                )
            elif mixed_pattern and not name == "tool-comparison-baseline":
                # tool-comparison-baseline is intentionally broad
                self.warnings.append(
                    f"Preset '{name}' has unclear configuration - neither pure algorithm comparison nor parameter sweep"
                )

    def _validate_naming_convention(self, name: str, preset):
        """Validate that preset names clearly indicate their purpose."""
        # Check for deprecated patterns
        deprecated_patterns = ["focus", "study", "optimization", "quality-sweep"]
        if any(pattern in name.lower() for pattern in deprecated_patterns):
            if not any(clear in name.lower() for clear in ["algorithm", "parameter"]):
                self.warnings.append(
                    f"Preset '{name}' uses deprecated naming - consider renaming to include 'algorithm' or 'parameter'"
                )

    def _print_results(self):
        """Print validation results summary."""
        print("\n" + "=" * 60)
        print("📊 Validation Results:")

        if self.errors:
            print(f"❌ Errors: {len(self.errors)}")
            for error in self.errors:
                print(f"   {error}")
        else:
            print("✅ No errors found")

        if self.warnings:
            print(f"⚠️  Warnings: {len(self.warnings)}")
            for warning in self.warnings:
                print(f"   {warning}")
        else:
            print("✅ No warnings")

        print(f"\n📋 Total presets validated: {len(EXPERIMENT_PRESETS)}")

        if len(self.errors) == 0:
            print("🎯 All presets maintain correct algorithm vs parameter distinction!")
        else:
            print("🚨 Configuration errors found - please fix before merging!")


def main():
    """Main entry point for CI/CD usage."""
    try:
        validator = PresetConfigurationValidator()
        is_valid = validator.validate_all_presets()

        if is_valid:
            print("\n✅ All preset configurations are valid!")
            sys.exit(0)
        else:
            print("\n❌ Preset configuration errors found!")
            sys.exit(1)

    except Exception as e:
        print(f"\n💥 Validation failed with exception: {str(e)}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
