# 🔧 Wrapper Validation Integration Guide

This guide provides step-by-step instructions for integrating the validation system into new wrapper classes and extending existing validation functionality.

---

## 📋 Prerequisites

Before integrating validation into wrapper classes, ensure you understand:
- Basic wrapper class structure in `src/giflab/tool_wrappers.py`
- Validation system architecture in `src/giflab/wrapper_validation/`
- Configuration system in `src/giflab/config.py`

---

## 🚀 Quick Integration

### Step 1: Basic Wrapper Integration

For any existing wrapper class, add validation with minimal changes:

```python
# src/giflab/tool_wrappers.py
from .wrapper_validation.integration import validate_wrapper_apply_result

class YourNewWrapper:
    def apply(self, input_path: Path, output_path: Path, params: dict) -> dict:
        # Existing wrapper implementation
        result = your_existing_compression_logic(input_path, output_path, params)
        
        # Add validation (non-disruptive)
        return validate_wrapper_apply_result(
            wrapper_instance=self,
            input_path=input_path,
            output_path=output_path,
            params=params,
            result=result
        )
```

**That's it!** The validation system will:
- ✅ Automatically detect wrapper type (frame/color/lossy reduction)
- ✅ Run appropriate validations (frame count, color count, timing, integrity, quality)
- ✅ Add validation results to the return metadata
- ✅ Never break existing functionality (robust error handling)

---

## 🏗️ Advanced Integration Patterns

### Pattern 1: Explicit Wrapper Type

For better control, specify the wrapper type explicitly:

```python
from .wrapper_validation.integration import add_validation_to_result

class CustomFrameReducer:
    def apply(self, input_path: Path, output_path: Path, params: dict) -> dict:
        result = self._compress_frames(input_path, output_path, params)
        
        # Explicit wrapper type for precise validation
        return add_validation_to_result(
            input_path=input_path,
            output_path=output_path,
            wrapper_params=params,
            wrapper_result=result,
            wrapper_type="frame_reduction"  # Explicit type
        )
```

### Pattern 2: Custom Validation Configuration

For specific validation requirements:

```python
from .wrapper_validation import WrapperOutputValidator, ValidationConfig

class HighPrecisionWrapper:
    def __init__(self):
        # Custom validation config for stricter requirements
        self.validation_config = ValidationConfig(
            FRAME_RATIO_TOLERANCE=0.01,      # 1% tolerance (vs default 5%)
            COLOR_COUNT_TOLERANCE=0,         # No color tolerance
            FPS_TOLERANCE=0.05,              # 5% FPS tolerance (vs default 10%)
            ENABLE_WRAPPER_VALIDATION=True,
            LOG_VALIDATION_FAILURES=True
        )
    
    def apply(self, input_path: Path, output_path: Path, params: dict) -> dict:
        result = self._high_precision_compression(input_path, output_path, params)
        
        # Use custom validation config
        return add_validation_to_result(
            input_path=input_path,
            output_path=output_path,
            wrapper_params=params,
            wrapper_result=result,
            wrapper_type="color_reduction",
            config=self.validation_config  # Custom config
        )
```

### Pattern 3: Pipeline-Level Validation

For multi-stage pipeline wrappers:

```python
from .wrapper_validation.integration import validate_pipeline_execution_result
from .dynamic_pipeline import Pipeline, PipelineStep

class MultiStageWrapper:
    def apply_pipeline(self, input_path: Path, pipeline: Pipeline, params: dict) -> dict:
        # Execute multi-stage pipeline
        pipeline_result = self._execute_pipeline(input_path, pipeline, params)
        
        # Add comprehensive pipeline validation
        return validate_pipeline_execution_result(
            input_path=input_path,
            pipeline=pipeline,
            pipeline_params=params,
            pipeline_result=pipeline_result
        )
```

---

## 🎯 Validation Types & When to Use Them

### Frame Reduction Validation
**Use for:** Wrappers that reduce frame count
**Validates:** 
- Frame count matches expected reduction ratio
- Minimum frame requirements (prevents single-frame outputs)
- Frame timing preservation

```python
# Triggered automatically for wrappers with "frame" in class name
# Or explicitly with wrapper_type="frame_reduction"
params = {"ratio": 0.5}  # Expects 50% frame reduction
```

### Color Reduction Validation  
**Use for:** Wrappers that reduce color palette
**Validates:**
- Color count within target range + tolerance
- Significant color reduction occurred
- Color palette quality

```python
# Triggered automatically for wrappers with "color" in class name  
# Or explicitly with wrapper_type="color_reduction"
params = {"colors": 64}  # Expects <= 64 colors in output
```

### Lossy Compression Validation
**Use for:** Quality-reducing compression wrappers
**Validates:**
- File integrity after compression
- Quality degradation within acceptable bounds
- Timing preservation

```python
# Triggered automatically for wrappers with "lossy" in class name
# Or explicitly with wrapper_type="lossy_compression"  
params = {"lossy_level": 30}  # 30% lossy compression
```

---

## 🔧 Configuration Options

### Essential Configuration Parameters

```python
@dataclass
class ValidationConfig:
    # Enable/disable validation system
    ENABLE_WRAPPER_VALIDATION: bool = True
    
    # Frame validation
    FRAME_RATIO_TOLERANCE: float = 0.05          # 5% tolerance on frame ratios
    MIN_FRAMES_REQUIRED: int = 1                 # Minimum output frames
    
    # Color validation  
    COLOR_COUNT_TOLERANCE: int = 2               # Allow 2 extra colors
    MIN_COLOR_REDUCTION_PERCENT: float = 0.05    # Require 5% color reduction
    
    # Timing validation
    FPS_TOLERANCE: float = 0.1                   # 10% FPS tolerance
    MIN_FPS: float = 0.1                         # Minimum valid FPS
    MAX_FPS: float = 60.0                        # Maximum valid FPS
    
    # File integrity
    MIN_FILE_SIZE_BYTES: int = 100               # Minimum valid GIF size
    MAX_FILE_SIZE_MB: float = 50.0               # Maximum GIF size
    
    # Error handling
    LOG_VALIDATION_FAILURES: bool = False        # Log validation failures
    FAIL_ON_VALIDATION_ERROR: bool = False       # Never fail pipeline on validation
```

### Performance Configuration

```python
# For high-throughput scenarios
performance_config = ValidationConfig(
    ENABLE_WRAPPER_VALIDATION=True,
    LOG_VALIDATION_FAILURES=False,              # Reduce logging overhead
    # Keep essential validations only
    FRAME_RATIO_TOLERANCE=0.1,                  # Relaxed tolerance = faster
    COLOR_COUNT_TOLERANCE=5,                    # Relaxed tolerance = faster
)

# For development/debugging
debug_config = ValidationConfig(
    ENABLE_WRAPPER_VALIDATION=True,
    LOG_VALIDATION_FAILURES=True,               # Detailed logging
    FAIL_ON_VALIDATION_ERROR=False,             # Never break development
    # Strict validations for catching issues early
    FRAME_RATIO_TOLERANCE=0.01,                 # 1% tolerance
    COLOR_COUNT_TOLERANCE=0,                    # Exact color matching
)
```

---

## 📊 Understanding Validation Results

### Validation Result Structure

Every validated wrapper returns enhanced metadata:

```python
{
    # Original wrapper result
    "compression_time": 1.23,
    "file_size_reduction": 0.65,
    
    # Validation additions
    "validations": [
        {
            "is_valid": True,
            "validation_type": "frame_count", 
            "expected": {"ratio": 0.5, "frames": 5},
            "actual": {"ratio": 0.48, "frames": 5},
            "error_message": None,
            "details": {"tolerance": 0.05}
        },
        {
            "is_valid": False,
            "validation_type": "color_count",
            "expected": 32,
            "actual": 35,
            "error_message": "Color count 35 exceeds expected 32 + tolerance 2",
            "details": {"original_colors": 256, "reduction_percent": 0.86}
        }
    ],
    "validation_passed": False,              # Overall validation status
    "validation_summary": {
        "total": 4,
        "passed": 3, 
        "failed": 1,
        "types_validated": ["frame_count", "color_count", "timing_preservation", "file_integrity"]
    }
}
```

### Interpreting Validation Failures

```python
def handle_validation_results(result: dict):
    if result.get("validation_passed", True):
        print("✅ All validations passed")
        return
    
    failed_validations = [
        v for v in result.get("validations", []) 
        if not v["is_valid"]
    ]
    
    for failure in failed_validations:
        validation_type = failure["validation_type"]
        error_message = failure["error_message"]
        
        print(f"❌ {validation_type}: {error_message}")
        
        # Access detailed failure information
        if "details" in failure:
            print(f"   Details: {failure['details']}")
```

---

## 🚨 Error Handling Best Practices

### Validation Never Breaks Pipelines

The validation system is designed to be **completely non-disruptive**:

```python
# Even if validation completely fails, wrapper execution continues
def robust_wrapper_apply(self, input_path, output_path, params):
    try:
        # Core compression logic
        result = self._compress_gif(input_path, output_path, params)
        
        # Validation is attempted but never breaks execution
        return validate_wrapper_apply_result(self, input_path, output_path, params, result)
        
    except Exception as compression_error:
        # Handle compression errors (not validation errors)
        return {"success": False, "error": str(compression_error)}
    
    # Validation errors are captured in metadata, never raised as exceptions
```

### Graceful Degradation

```python
# Validation failures are informational, not blocking
def process_validation_aware_result(result: dict):
    # Always check core operation success first
    if not result.get("success", True):
        print("❌ Compression failed")
        return
    
    # Then check validation results (informational only)
    validation_passed = result.get("validation_passed")
    
    if validation_passed is None:
        print("⚠️  Validation skipped (error occurred)")
    elif validation_passed:
        print("✅ Compression successful + validation passed")  
    else:
        print("⚠️  Compression successful but validation concerns:")
        # Log concerns but continue processing
        for validation in result.get("validations", []):
            if not validation["is_valid"]:
                print(f"   - {validation['error_message']}")
```

---

## 🧪 Testing Validation Integration

### Unit Testing Pattern

```python
# tests/functional/test_your_new_wrapper.py
import pytest
from pathlib import Path
from giflab.tool_wrappers import YourNewWrapper

class TestYourNewWrapperValidation:
    def test_successful_validation(self):
        wrapper = YourNewWrapper()
        
        # Use test fixtures with known properties
        result = wrapper.apply(
            input_path=Path("tests/fixtures/test_10_frames.gif"),
            output_path=Path("/tmp/test_output.gif"),
            params={"ratio": 0.5}
        )
        
        # Test core functionality
        assert result["success"] is True
        
        # Test validation integration
        assert "validations" in result
        assert "validation_passed" in result
        assert len(result["validations"]) > 0
        
        # Test specific validations
        frame_validations = [
            v for v in result["validations"] 
            if v["validation_type"] == "frame_count"
        ]
        assert len(frame_validations) == 1
        assert frame_validations[0]["is_valid"] is True
    
    def test_validation_error_handling(self):
        wrapper = YourNewWrapper()
        
        # Test with problematic input to trigger validation failures
        result = wrapper.apply(
            input_path=Path("tests/fixtures/single_frame.gif"),
            output_path=Path("/tmp/test_output.gif"), 
            params={"ratio": 0.1}  # Very aggressive reduction
        )
        
        # Core operation should still succeed
        assert result["success"] is True
        
        # But validation may report concerns
        if result.get("validation_passed") is False:
            print("Expected validation concerns for edge case")
        
        # Validation system should never crash
        assert "validations" in result
```

---

## 💡 Common Integration Patterns

### Pattern: Legacy Wrapper Modernization

```python
# Before: Basic wrapper
class OldWrapper:
    def apply(self, input_path, output_path, params):
        # Legacy compression logic
        subprocess.run(["old_tool", str(input_path), str(output_path)])
        return {"success": True}

# After: Validation-enabled wrapper
class ModernizedWrapper:
    def apply(self, input_path, output_path, params):
        # Same compression logic
        subprocess.run(["old_tool", str(input_path), str(output_path)])
        result = {"success": True}
        
        # Add validation with zero disruption
        return validate_wrapper_apply_result(self, input_path, output_path, params, result)
```

### Pattern: Conditional Validation

```python
class ConditionalValidationWrapper:
    def __init__(self, enable_strict_validation: bool = False):
        self.strict_mode = enable_strict_validation
    
    def apply(self, input_path, output_path, params):
        result = self._compress(input_path, output_path, params)
        
        if self.strict_mode:
            # Use strict validation config
            strict_config = ValidationConfig(
                FRAME_RATIO_TOLERANCE=0.01,
                COLOR_COUNT_TOLERANCE=0,
                FPS_TOLERANCE=0.05
            )
            return add_validation_to_result(
                input_path, output_path, params, result, 
                wrapper_type="frame_reduction",
                config=strict_config
            )
        else:
            # Use default validation
            return validate_wrapper_apply_result(self, input_path, output_path, params, result)
```

---

## 🔍 Debugging Validation Issues

### Enable Detailed Logging

```python
import logging
logging.getLogger("giflab.wrapper_validation").setLevel(logging.DEBUG)

# Or use config
debug_config = ValidationConfig(
    LOG_VALIDATION_FAILURES=True,
    ENABLE_WRAPPER_VALIDATION=True
)
```

### Validation Result Inspection

```python
def debug_validation_results(result: dict):
    print("=== Validation Debug Information ===")
    
    validations = result.get("validations", [])
    print(f"Total validations: {len(validations)}")
    
    for i, validation in enumerate(validations):
        print(f"\nValidation {i+1}: {validation['validation_type']}")
        print(f"  Status: {'✅ PASS' if validation['is_valid'] else '❌ FAIL'}")
        print(f"  Expected: {validation['expected']}")
        print(f"  Actual: {validation['actual']}")
        
        if validation.get("error_message"):
            print(f"  Error: {validation['error_message']}")
        
        if validation.get("details"):
            print(f"  Details: {validation['details']}")
```

---

## ✅ Integration Checklist

When adding validation to a new wrapper:

- [ ] **Import validation functions** from `wrapper_validation.integration`
- [ ] **Add validation call** to `apply()` method return
- [ ] **Test with known fixtures** to verify validation behavior
- [ ] **Handle validation results appropriately** in downstream code  
- [ ] **Consider custom configuration** if default tolerances don't fit
- [ ] **Add unit tests** for validation integration
- [ ] **Document wrapper-specific validation behavior** if needed

---

## 📚 Related Documentation

- [Validation Configuration Reference](../reference/validation-config-reference.md)
- [Performance Optimization Guide](../technical/validation-performance-guide.md) 
- [Troubleshooting Guide](validation-troubleshooting.md)
- [Testing Best Practices](../guides/testing-best-practices.md)
- [Quality Metrics Integration](../technical/validation-quality-integration.md)

---

## 🤝 Support

For validation integration questions:
1. Check the [Troubleshooting Guide](validation-troubleshooting.md)
2. Review existing wrapper implementations in `src/giflab/tool_wrappers.py`
3. Run validation tests: `poetry run pytest tests/functional/ -k validation`