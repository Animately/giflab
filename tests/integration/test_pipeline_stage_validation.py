"""Tests for pipeline stage validation system.

This test suite validates that the pipeline stage validation system correctly
validates multi-stage compression pipelines, inter-stage consistency, and
overall pipeline integrity.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from giflab.wrapper_validation import (
    PipelineStageValidator,
    ValidationResult,
    WrapperOutputValidator,
)
from giflab.wrapper_validation.integration import (
    _extract_final_output_from_result,
    _extract_stage_metadata_from_result,
    _extract_stage_outputs_from_result,
    create_validation_report,
    validate_pipeline_execution_result,
)


class TestPipelineStageValidator:
    """Test core pipeline stage validation functionality."""

    def test_pipeline_stage_validator_initialization(self):
        """Test pipeline stage validator initializes properly."""
        validator = PipelineStageValidator()
        assert validator.wrapper_validator is not None
        assert isinstance(validator.wrapper_validator, WrapperOutputValidator)
        assert "dimension_tolerance" in validator.pipeline_thresholds

    def test_pipeline_stage_validator_with_custom_wrapper_validator(self):
        """Test pipeline stage validator with custom wrapper validator."""
        custom_wrapper_validator = Mock(spec=WrapperOutputValidator)
        validator = PipelineStageValidator(custom_wrapper_validator)
        assert validator.wrapper_validator == custom_wrapper_validator

    @patch("giflab.wrapper_validation.pipeline_validation.extract_gif_metadata")
    def test_validate_pipeline_stage_success(self, mock_extract_metadata):
        """Test successful pipeline stage validation."""
        validator = PipelineStageValidator()

        # Mock metadata for input and output
        input_metadata = Mock()
        input_metadata.orig_frames = 10
        input_metadata.orig_n_colors = 256
        input_metadata.orig_width = 100
        input_metadata.orig_height = 100
        input_metadata.orig_fps = 10.0

        output_metadata = Mock()
        output_metadata.orig_frames = 5
        output_metadata.orig_n_colors = 32
        output_metadata.orig_width = 100
        output_metadata.orig_height = 100
        output_metadata.orig_fps = 10.0

        mock_extract_metadata.side_effect = [input_metadata, output_metadata]

        # Create mock pipeline step
        pipeline_step = Mock()
        pipeline_step.variable = "frame_reduction"
        pipeline_step.tool_cls = Mock()
        pipeline_step.tool_cls.__name__ = "TestFrameReducer"

        with tempfile.NamedTemporaryFile(
            suffix=".gif"
        ) as input_file, tempfile.NamedTemporaryFile(suffix=".gif") as output_file:
            input_path = Path(input_file.name)
            output_path = Path(output_file.name)

            # Write some data to files to ensure they exist
            input_file.write(b"fake gif data")
            output_file.write(b"fake gif data")
            input_file.flush()
            output_file.flush()

            pipeline_params = {"frame_ratio": 0.5}
            stage_metadata = {"tool_class": "TestFrameReducer"}

            results = validator.validate_pipeline_stage(
                input_path=input_path,
                output_path=output_path,
                pipeline_step=pipeline_step,
                pipeline_params=pipeline_params,
                stage_metadata=stage_metadata,
                stage_index=0,
            )

            assert len(results) >= 1
            # Should have inter-stage consistency validations
            consistency_validations = [
                r for r in results if r.validation_type.endswith("consistency")
            ]
            assert len(consistency_validations) >= 1

    def test_validate_pipeline_stage_missing_output(self):
        """Test pipeline stage validation with missing output file."""
        validator = PipelineStageValidator()

        pipeline_step = Mock()
        pipeline_step.variable = "color_reduction"
        pipeline_step.tool_cls = Mock()
        pipeline_step.tool_cls.__name__ = "TestColorReducer"

        with tempfile.NamedTemporaryFile(suffix=".gif") as input_file:
            input_path = Path(input_file.name)
            output_path = Path("/nonexistent/output.gif")  # This file doesn't exist

            results = validator.validate_pipeline_stage(
                input_path=input_path,
                output_path=output_path,
                pipeline_step=pipeline_step,
                pipeline_params={},
                stage_metadata={},
                stage_index=0,
            )

            assert len(results) == 1
            assert results[0].validation_type == "pipeline_stage_output"
            assert results[0].is_valid is False
            assert "not found" in results[0].error_message

    @patch("giflab.wrapper_validation.pipeline_validation.extract_gif_metadata")
    def test_validate_frame_reduction_stage(self, mock_extract_metadata):
        """Test frame reduction stage validation."""
        validator = PipelineStageValidator()

        input_metadata = Mock()
        input_metadata.orig_frames = 20

        output_metadata = Mock()
        output_metadata.orig_frames = 10

        pipeline_params = {"frame_ratio": 0.5}
        stage_metadata = {}

        results = validator.validate_frame_reduction_stage(
            input_metadata, output_metadata, pipeline_params, stage_metadata
        )

        assert len(results) == 1
        result = results[0]
        assert result.validation_type == "pipeline_frame_reduction"
        assert result.is_valid is True  # 10 frames is expected for 50% of 20

    @patch("giflab.wrapper_validation.pipeline_validation.extract_gif_metadata")
    def test_validate_color_reduction_stage(self, mock_extract_metadata):
        """Test color reduction stage validation."""
        validator = PipelineStageValidator()

        input_metadata = Mock()
        input_metadata.orig_n_colors = 256

        output_metadata = Mock()
        output_metadata.orig_n_colors = 64

        pipeline_params = {"colors": 64}
        stage_metadata = {}

        results = validator.validate_color_reduction_stage(
            input_metadata, output_metadata, pipeline_params, stage_metadata
        )

        assert len(results) == 1
        result = results[0]
        assert result.validation_type == "pipeline_color_reduction"
        assert result.is_valid is True

    def test_validate_lossy_compression_stage(self):
        """Test lossy compression stage validation."""
        validator = PipelineStageValidator()

        input_metadata = Mock()
        output_metadata = Mock()
        pipeline_params = {"lossy": 30}
        stage_metadata = {}

        results = validator.validate_lossy_compression_stage(
            input_metadata, output_metadata, pipeline_params, stage_metadata
        )

        assert len(results) == 1
        result = results[0]
        assert result.validation_type == "pipeline_lossy_compression"
        assert result.is_valid is True

    @patch("giflab.wrapper_validation.pipeline_validation.extract_gif_metadata")
    def test_validate_inter_stage_consistency_dimensions(self, mock_extract_metadata):
        """Test inter-stage dimension consistency validation."""
        validator = PipelineStageValidator()

        input_metadata = Mock()
        input_metadata.orig_width = 100
        input_metadata.orig_height = 100
        input_metadata.orig_fps = 10.0

        output_metadata = Mock()
        output_metadata.orig_width = 200  # Different dimensions
        output_metadata.orig_height = 200
        output_metadata.orig_fps = 10.0

        pipeline_step = Mock()
        pipeline_step.variable = "color_reduction"  # Should not change dimensions

        with tempfile.NamedTemporaryFile(
            suffix=".gif"
        ) as input_file, tempfile.NamedTemporaryFile(suffix=".gif") as output_file:
            results = validator.validate_inter_stage_consistency(
                Path(input_file.name),
                Path(output_file.name),
                input_metadata,
                output_metadata,
                pipeline_step,
            )

            # Should have dimension and fps consistency validations
            dimension_validations = [
                r
                for r in results
                if r.validation_type == "pipeline_dimension_consistency"
            ]
            assert len(dimension_validations) == 1
            assert (
                dimension_validations[0].is_valid is False
            )  # Dimensions changed unexpectedly

    @patch("giflab.wrapper_validation.pipeline_validation.extract_gif_metadata")
    def test_validate_png_sequence_integrity_success(self, mock_extract_metadata):
        """Test successful PNG sequence integrity validation."""
        validator = PipelineStageValidator()

        ref_metadata = Mock()
        ref_metadata.orig_frames = 3
        mock_extract_metadata.return_value = ref_metadata

        with tempfile.TemporaryDirectory() as temp_dir, tempfile.NamedTemporaryFile(
            suffix=".gif"
        ) as ref_gif:
            png_dir = Path(temp_dir)
            ref_gif_path = Path(ref_gif.name)

            # Create mock PNG files
            for i in range(3):
                png_file = png_dir / f"frame_{i:03d}.png"
                png_file.write_bytes(b"fake png data")

            # Mock PIL Image.open for PNG validation
            with patch(
                "giflab.wrapper_validation.pipeline_validation.Image.open"
            ) as mock_image_open:
                mock_img = Mock()
                mock_img.format = "PNG"
                mock_img.size = (100, 100)
                mock_img.__enter__ = Mock(return_value=mock_img)
                mock_img.__exit__ = Mock(return_value=None)
                mock_image_open.return_value = mock_img

                result = validator.validate_png_sequence_integrity(
                    png_sequence_dir=png_dir,
                    reference_gif=ref_gif_path,
                    expected_frame_count=3,
                )

                assert result.is_valid is True
                assert result.validation_type == "png_sequence_integrity"

    def test_validate_png_sequence_integrity_missing_directory(self):
        """Test PNG sequence integrity validation with missing directory."""
        validator = PipelineStageValidator()

        with tempfile.NamedTemporaryFile(suffix=".gif") as ref_gif:
            non_existent_dir = Path("/nonexistent/png/dir")

            result = validator.validate_png_sequence_integrity(
                png_sequence_dir=non_existent_dir, reference_gif=Path(ref_gif.name)
            )

            assert result.is_valid is False
            assert result.validation_type == "png_sequence_integrity"
            assert "not found" in result.error_message


class TestPipelineValidationIntegration:
    """Test pipeline validation integration with existing systems."""

    @patch("giflab.wrapper_validation.pipeline_validation.extract_gif_metadata")
    def test_validate_pipeline_execution_result_basic(self, mock_extract_metadata):
        """Test basic pipeline execution result validation."""
        # Mock metadata extraction to avoid GIF parsing issues
        mock_metadata = Mock()
        mock_metadata.orig_frames = 4
        mock_metadata.orig_n_colors = 64
        mock_metadata.orig_width = 100
        mock_metadata.orig_height = 100
        mock_metadata.orig_fps = 10.0
        mock_extract_metadata.return_value = mock_metadata

        # Mock pipeline
        mock_pipeline = Mock()
        mock_pipeline.identifier.return_value = "test_pipeline"
        mock_pipeline.steps = []

        pipeline_params = {"colors": 64, "frame_ratio": 0.8, "lossy": 20}

        with tempfile.NamedTemporaryFile(
            suffix=".gif"
        ) as input_file, tempfile.NamedTemporaryFile(suffix=".gif") as output_file:
            input_path = Path(input_file.name)
            final_output_path = Path(output_file.name)

            # Write some data to ensure files exist
            input_file.write(b"fake gif data")
            output_file.write(b"fake gif data")
            input_file.flush()
            output_file.flush()

            pipeline_result = {
                "output_path": str(final_output_path),
                "stage_outputs": {},
                "stage_metadata": {},
            }

            enhanced_result = validate_pipeline_execution_result(
                input_path=input_path,
                pipeline=mock_pipeline,
                pipeline_params=pipeline_params,
                pipeline_result=pipeline_result,
                final_output_path=final_output_path,
            )

            assert "pipeline_validations" in enhanced_result
            assert "pipeline_validation_passed" in enhanced_result
            assert "pipeline_validation_summary" in enhanced_result

    def test_validate_pipeline_execution_result_disabled(self):
        """Test pipeline validation when disabled in config."""
        # Mock disabled config
        disabled_config = Mock()
        disabled_config.ENABLE_WRAPPER_VALIDATION = False

        mock_pipeline = Mock()
        pipeline_result = {"test": "data"}

        with patch("giflab.config.DEFAULT_VALIDATION_CONFIG", disabled_config):
            enhanced_result = validate_pipeline_execution_result(
                input_path=Path("input.gif"),
                pipeline=mock_pipeline,
                pipeline_params={},
                pipeline_result=pipeline_result,
            )

            # Should return original result unchanged
            assert enhanced_result == pipeline_result

    def test_create_validation_report_empty(self):
        """Test validation report creation with no validations."""
        report = create_validation_report([], context={"test": "context"})

        assert report["total_validations"] == 0
        assert report["overall_status"] == "no_validations"
        assert report["context"]["test"] == "context"

    def test_create_validation_report_mixed_results(self):
        """Test validation report with mixed pass/fail results."""
        validations = [
            ValidationResult(
                is_valid=True,
                validation_type="test_pass",
                expected="pass",
                actual="pass",
            ),
            ValidationResult(
                is_valid=False,
                validation_type="test_fail",
                expected="pass",
                actual="fail",
                error_message="Test failure",
            ),
            ValidationResult(
                is_valid=True,
                validation_type="test_pass",
                expected="pass",
                actual="pass",
            ),
        ]

        report = create_validation_report(validations)

        assert report["total_validations"] == 3
        assert report["passed_validations"] == 2
        assert report["failed_validations"] == 1
        assert report["overall_status"] == "failed"  # Any failure = overall failure
        assert report["success_rate"] == 2 / 3

        # Check validation type grouping
        assert "test_pass" in report["validation_types"]
        assert "test_fail" in report["validation_types"]
        assert report["validation_types"]["test_pass"]["passed"] == 2
        assert report["validation_types"]["test_fail"]["failed"] == 1

    def test_extract_stage_outputs_from_result(self):
        """Test extraction of stage outputs from pipeline result."""
        pipeline_result = {
            "stage_outputs": {
                "frame_reduction": "/tmp/stage1.gif",
                "color_reduction": Path("/tmp/stage2.gif"),
                "invalid_stage": None,
            }
        }

        stage_outputs = _extract_stage_outputs_from_result(pipeline_result)

        assert len(stage_outputs) == 2
        assert "frame_reduction" in stage_outputs
        assert isinstance(stage_outputs["frame_reduction"], Path)
        assert str(stage_outputs["frame_reduction"]) == "/tmp/stage1.gif"

    def test_extract_final_output_from_result(self):
        """Test extraction of final output path from pipeline result."""
        pipeline_result = {"output_path": "/tmp/final.gif"}
        final_output = _extract_final_output_from_result(pipeline_result)

        assert isinstance(final_output, Path)
        assert str(final_output) == "/tmp/final.gif"

        # Test alternative key
        pipeline_result = {"final_output": "/tmp/final2.gif"}
        final_output = _extract_final_output_from_result(pipeline_result)

        assert str(final_output) == "/tmp/final2.gif"

        # Test missing key
        pipeline_result = {"other_data": "value"}
        final_output = _extract_final_output_from_result(pipeline_result)

        assert final_output is None

    def test_extract_stage_metadata_from_result(self):
        """Test extraction of stage metadata from pipeline result."""
        # Mock pipeline with steps
        mock_step1 = Mock()
        mock_step1.variable = "frame_reduction"
        mock_step1.tool_cls = Mock()
        mock_step1.tool_cls.__name__ = "TestFrameReducer"

        mock_step2 = Mock()
        mock_step2.variable = "color_reduction"
        mock_step2.tool_cls = Mock()
        mock_step2.tool_cls.__name__ = "TestColorReducer"

        mock_pipeline = Mock()
        mock_pipeline.steps = [mock_step1, mock_step2]

        pipeline_result = {
            "stage_metadata": {
                "frame_reduction_TestFrameReducer": {"custom": "metadata1"},
                "color_reduction_TestColorReducer": {"custom": "metadata2"},
            },
            "general": "data",
        }

        stage_metadata = _extract_stage_metadata_from_result(
            pipeline_result, mock_pipeline
        )

        assert len(stage_metadata) == 2
        assert "frame_reduction_TestFrameReducer" in stage_metadata
        assert "color_reduction_TestColorReducer" in stage_metadata

        # Should have custom metadata
        assert (
            stage_metadata["frame_reduction_TestFrameReducer"]["custom"] == "metadata1"
        )

        # Test fallback when stage metadata missing
        pipeline_result_no_metadata = {"general": "data"}
        stage_metadata = _extract_stage_metadata_from_result(
            pipeline_result_no_metadata, mock_pipeline
        )

        assert len(stage_metadata) == 2
        # Should have fallback metadata
        assert (
            stage_metadata["frame_reduction_TestFrameReducer"]["stage_variable"]
            == "frame_reduction"
        )
        assert (
            stage_metadata["frame_reduction_TestFrameReducer"]["tool_class"]
            == "TestFrameReducer"
        )


@pytest.mark.external_tools
class TestPipelineValidationRealScenarios:
    """Test pipeline validation with real compression scenarios."""

    @pytest.fixture
    def test_gif(self):
        """Path to test GIF fixture."""
        return Path(__file__).parent / "fixtures" / "test_4_frames.gif"

    def test_pipeline_validation_with_single_stage(self, test_gif):
        """Test pipeline validation with single-stage pipeline."""
        if not test_gif.exists():
            pytest.skip("Test fixture not available")

        # Create simple single-stage pipeline mock
        mock_step = Mock()
        mock_step.variable = "color_reduction"
        mock_step.tool_cls = Mock()
        mock_step.tool_cls.__name__ = "TestColorReducer"

        mock_pipeline = Mock()
        mock_pipeline.identifier.return_value = "TestColorReducer_Color"
        mock_pipeline.steps = [mock_step]

        validator = PipelineStageValidator()

        with tempfile.NamedTemporaryFile(suffix=".gif") as output_file:
            output_path = Path(output_file.name)

            # Write some minimal GIF data
            output_file.write(b"GIF89a" + b"\x00" * 100)
            output_file.flush()

            # Mock extract_gif_metadata to return reasonable data
            with patch(
                "giflab.wrapper_validation.pipeline_validation.extract_gif_metadata"
            ) as mock_extract:
                input_metadata = Mock()
                input_metadata.orig_frames = 4
                input_metadata.orig_n_colors = 256
                input_metadata.orig_width = 100
                input_metadata.orig_height = 100
                input_metadata.orig_fps = 10.0

                output_metadata = Mock()
                output_metadata.orig_frames = 4
                output_metadata.orig_n_colors = 64
                output_metadata.orig_width = 100
                output_metadata.orig_height = 100
                output_metadata.orig_fps = 10.0

                mock_extract.side_effect = [
                    input_metadata,
                    output_metadata,
                    input_metadata,
                    output_metadata,
                ]

                stage_outputs = {"color_reduction_TestColorReducer": output_path}
                stage_metadata = {"color_reduction_TestColorReducer": {"tool": "test"}}
                pipeline_params = {"colors": 64}

                validations = validator.validate_pipeline_execution(
                    input_path=test_gif,
                    pipeline=mock_pipeline,
                    pipeline_params=pipeline_params,
                    stage_outputs=stage_outputs,
                    stage_metadata=stage_metadata,
                    final_output_path=output_path,
                )

                assert len(validations) > 0
                # Should have some validations that pass
                passed_validations = [v for v in validations if v.is_valid]
                assert len(passed_validations) > 0

    def test_pipeline_validation_error_handling(self, test_gif):
        """Test pipeline validation error handling."""
        if not test_gif.exists():
            pytest.skip("Test fixture not available")

        validator = PipelineStageValidator()

        # Create pipeline that will cause validation errors
        mock_pipeline = Mock()
        mock_pipeline.identifier.return_value = "ErrorPipeline"
        mock_pipeline.steps = []

        # Call with invalid parameters to trigger error handling
        validations = validator.validate_pipeline_execution(
            input_path=test_gif,
            pipeline=mock_pipeline,
            pipeline_params={},
            stage_outputs={},  # Empty stage outputs
            stage_metadata={},
            final_output_path=Path("/nonexistent/output.gif"),
        )

        # Should have validation results even with errors
        assert len(validations) >= 1
        # Should have at least one validation indicating the problem
        error_validations = [v for v in validations if not v.is_valid]
        assert len(error_validations) >= 1
