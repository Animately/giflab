"""Tests for giflab.tag_pipeline module - TaggingPipeline."""

import csv
from unittest.mock import Mock, patch

import pytest
from giflab.tag_pipeline import (
    TaggingPipeline,
    create_tagging_pipeline,
    validate_tagged_csv,
)
from giflab.tagger import TaggingResult


class TestTaggingPipeline:
    """Tests for TaggingPipeline class."""

    @pytest.fixture
    def mock_tagger(self):
        """Create a mock HybridCompressionTagger."""
        with patch("giflab.tag_pipeline.HybridCompressionTagger") as mock_class:
            mock_instance = Mock()
            mock_class.return_value = mock_instance
            yield mock_instance

    @pytest.fixture
    def sample_csv_data(self):
        """Sample compression results data."""
        return [
            {
                "gif_sha": "sha123",
                "orig_filename": "test1.gif",
                "engine": "original",
                "lossy": "0",
                "frame_keep_ratio": "1.00",
                "color_keep_count": "256",
                "kilobytes": "100.5",
                "ssim": "1.000",
                "timestamp": "2024-01-01T10:00:00Z",
            },
            {
                "gif_sha": "sha123",
                "orig_filename": "test1.gif",
                "engine": "gifsicle",
                "lossy": "40",
                "frame_keep_ratio": "0.80",
                "color_keep_count": "64",
                "kilobytes": "50.2",
                "ssim": "0.936",
                "timestamp": "2024-01-01T10:01:00Z",
            },
            {
                "gif_sha": "sha456",
                "orig_filename": "test2.gif",
                "engine": "original",
                "lossy": "0",
                "frame_keep_ratio": "1.00",
                "color_keep_count": "256",
                "kilobytes": "200.0",
                "ssim": "1.000",
                "timestamp": "2024-01-01T11:00:00Z",
            },
        ]

    @pytest.fixture
    def sample_csv_file(self, tmp_path, sample_csv_data):
        """Create a sample CSV file."""
        csv_path = tmp_path / "results.csv"

        if sample_csv_data:
            fieldnames = sample_csv_data[0].keys()
            with open(csv_path, "w", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(sample_csv_data)

        return csv_path

    @pytest.fixture
    def sample_gifs(self, tmp_path):
        """Create sample GIF files."""
        gif_dir = tmp_path / "gifs"
        gif_dir.mkdir()

        # Create dummy GIF files
        gif1 = gif_dir / "test1.gif"
        gif2 = gif_dir / "test2.gif"

        gif1.write_bytes(b"fake gif content 1")
        gif2.write_bytes(b"fake gif content 2")

        return gif_dir

    def test_tagging_pipeline_initialization(self, mock_tagger):
        """Test TaggingPipeline initialization."""
        with patch(
            "giflab.tag_pipeline.HybridCompressionTagger", return_value=mock_tagger
        ):
            pipeline = TaggingPipeline(workers=2)

            assert pipeline.workers == 2
            assert len(pipeline.TAGGING_COLUMNS) == 25
            assert pipeline.tagger is not None

    def test_tagging_columns_definition(self):
        """Test that all expected tagging columns are defined."""
        expected_columns = [
            # Content classification (6)
            "screen_capture_confidence",
            "vector_art_confidence",
            "photography_confidence",
            "hand_drawn_confidence",
            "3d_rendered_confidence",
            "pixel_art_confidence",
            # Quality assessment (4)
            "blocking_artifacts",
            "ringing_artifacts",
            "quantization_noise",
            "overall_quality",
            # Technical characteristics (5)
            "text_density",
            "edge_density",
            "color_complexity",
            "contrast_score",
            "gradient_smoothness",
            # Temporal motion analysis (10)
            "frame_similarity",
            "motion_intensity",
            "motion_smoothness",
            "static_region_ratio",
            "scene_change_frequency",
            "fade_transition_presence",
            "cut_sharpness",
            "temporal_entropy",
            "loop_detection_confidence",
            "motion_complexity",
        ]

        assert set(TaggingPipeline.TAGGING_COLUMNS) == set(expected_columns)
        assert len(TaggingPipeline.TAGGING_COLUMNS) == 25

    def test_load_existing_results_success(
        self, mock_tagger, sample_csv_file, sample_csv_data
    ):
        """Test successful loading of existing results."""
        pipeline = TaggingPipeline()

        with patch(
            "giflab.tag_pipeline.read_csv_as_dicts", return_value=sample_csv_data
        ):
            results = pipeline.load_existing_results(sample_csv_file)

            assert len(results) == 3
            assert results[0]["gif_sha"] == "sha123"
            assert results[2]["gif_sha"] == "sha456"

    def test_load_existing_results_missing_columns(self, mock_tagger, tmp_path):
        """Test error handling for CSV with missing required columns."""
        csv_path = tmp_path / "invalid.csv"

        # Create CSV with missing required columns
        invalid_data = [{"some_column": "value"}]
        with open(csv_path, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["some_column"])
            writer.writeheader()
            writer.writerows(invalid_data)

        pipeline = TaggingPipeline()

        with patch("giflab.tag_pipeline.read_csv_as_dicts", return_value=invalid_data):
            with pytest.raises(ValueError, match="Missing required columns"):
                pipeline.load_existing_results(csv_path)

    def test_identify_unique_gifs_original_only(self, mock_tagger, sample_csv_data):
        """Test identification of unique original GIFs only."""
        pipeline = TaggingPipeline()

        unique_gifs = pipeline.identify_unique_gifs(sample_csv_data)

        # Should only return records with engine='original'
        assert len(unique_gifs) == 2
        assert all(gif["engine"] == "original" for gif in unique_gifs)
        assert {gif["gif_sha"] for gif in unique_gifs} == {"sha123", "sha456"}

    def test_identify_unique_gifs_no_originals(self, mock_tagger):
        """Test handling when no original GIFs are found."""
        pipeline = TaggingPipeline()

        # Data with no original engine records
        data = [
            {"gif_sha": "sha123", "engine": "gifsicle", "orig_filename": "test1.gif"},
            {"gif_sha": "sha456", "engine": "animately", "orig_filename": "test2.gif"},
        ]

        unique_gifs = pipeline.identify_unique_gifs(data)

        assert len(unique_gifs) == 0

    def test_find_original_gif_path_success(self, mock_tagger, sample_gifs):
        """Test successful finding of original GIF path."""
        pipeline = TaggingPipeline()

        result_record = {"orig_filename": "test1.gif"}
        path = pipeline.find_original_gif_path(result_record, sample_gifs)

        assert path is not None
        assert path.name == "test1.gif"
        assert path.exists()

    def test_find_original_gif_path_not_found(self, mock_tagger, sample_gifs):
        """Test handling when GIF file is not found."""
        pipeline = TaggingPipeline()

        result_record = {"orig_filename": "nonexistent.gif"}
        path = pipeline.find_original_gif_path(result_record, sample_gifs)

        assert path is None

    def test_find_original_gif_path_case_insensitive(self, mock_tagger, tmp_path):
        """Test case-insensitive GIF path finding."""
        gif_dir = tmp_path / "gifs"
        gif_dir.mkdir()

        # Create GIF with specific case
        actual_gif = gif_dir / "Test_File.GIF"
        actual_gif.write_bytes(b"test content")

        pipeline = TaggingPipeline()

        # Search with different case
        result_record = {"orig_filename": "test_file.gif"}
        path = pipeline.find_original_gif_path(result_record, gif_dir)

        assert path is not None
        assert path.name == "Test_File.GIF"

    def test_tag_single_gif_success(self, mock_tagger, sample_gifs):
        """Test successful tagging of a single GIF."""
        # Mock the tagger's tag_gif method
        mock_result = TaggingResult(
            gif_sha="test_sha",
            scores={"screen_capture_confidence": 0.8, "blocking_artifacts": 0.1},
            model_version="test_v1.0",
            processing_time_ms=100,
        )
        mock_tagger.tag_gif.return_value = mock_result

        pipeline = TaggingPipeline()
        pipeline.tagger = mock_tagger

        gif_path = sample_gifs / "test1.gif"
        result = pipeline.tag_single_gif(gif_path, "test_sha")

        assert result == mock_result
        mock_tagger.tag_gif.assert_called_once_with(gif_path, gif_sha="test_sha")

    def test_tag_single_gif_failure(self, mock_tagger, sample_gifs):
        """Test error handling when tagging fails."""
        mock_tagger.tag_gif.side_effect = Exception("Tagging failed")

        pipeline = TaggingPipeline()
        pipeline.tagger = mock_tagger

        gif_path = sample_gifs / "test1.gif"

        with pytest.raises(RuntimeError, match="Tagging failed"):
            pipeline.tag_single_gif(gif_path, "test_sha")

    def test_update_results_with_tags_success(self, mock_tagger):
        """Test updating results with tagging scores."""
        pipeline = TaggingPipeline()

        # Original results
        results = [
            {"gif_sha": "sha123", "engine": "original", "lossy": "0"},
            {"gif_sha": "sha123", "engine": "gifsicle", "lossy": "40"},
            {"gif_sha": "sha456", "engine": "original", "lossy": "0"},
        ]

        # Tagging results
        tagging_results = {
            "sha123": TaggingResult(
                gif_sha="sha123",
                scores={
                    "screen_capture_confidence": 0.8,
                    "blocking_artifacts": 0.1,
                    "text_density": 0.7,
                    "frame_similarity": 0.9,
                },
                model_version="test",
                processing_time_ms=100,
            )
        }

        updated_results = pipeline.update_results_with_tags(results, tagging_results)

        assert len(updated_results) == 3

        # Check that scores are inherited by all variants of sha123
        for result in updated_results[:2]:  # First two have sha123
            assert "screen_capture_confidence" in result
            assert result["screen_capture_confidence"] == "0.800000"
            assert result["blocking_artifacts"] == "0.100000"
            assert result["text_density"] == "0.700000"
            assert result["frame_similarity"] == "0.900000"

        # Check that sha456 gets zero scores (no tagging result)
        sha456_result = updated_results[2]
        assert "screen_capture_confidence" in sha456_result
        assert sha456_result["screen_capture_confidence"] == "0.000000"

    def test_update_results_all_tagging_columns_added(self, mock_tagger):
        """Test that all 25 tagging columns are added to results."""
        pipeline = TaggingPipeline()

        results = [{"gif_sha": "sha123", "engine": "original"}]
        tagging_results = {}  # No tagging results

        updated_results = pipeline.update_results_with_tags(results, tagging_results)

        # Check that all 25 tagging columns are present
        result = updated_results[0]
        for column in pipeline.TAGGING_COLUMNS:
            assert column in result
            assert result[column] == "0.000000"

    def test_write_tagged_csv_success(self, mock_tagger, tmp_path):
        """Test successful writing of tagged CSV."""
        pipeline = TaggingPipeline()

        updated_results = [
            {
                "gif_sha": "sha123",
                "engine": "original",
                "screen_capture_confidence": "0.800000",
                "blocking_artifacts": "0.100000",
            },
            {
                "gif_sha": "sha123",
                "engine": "gifsicle",
                "screen_capture_confidence": "0.800000",
                "blocking_artifacts": "0.100000",
            },
        ]

        output_path = tmp_path / "tagged_results.csv"
        pipeline.write_tagged_csv(updated_results, output_path)

        assert output_path.exists()

        # Verify CSV content
        with open(output_path) as csvfile:
            reader = csv.DictReader(csvfile)
            rows = list(reader)

            assert len(rows) == 2
            assert "screen_capture_confidence" in rows[0]
            assert rows[0]["screen_capture_confidence"] == "0.800000"

    def test_write_tagged_csv_column_ordering(self, mock_tagger, tmp_path):
        """Test that CSV columns are ordered correctly (original columns first)."""
        pipeline = TaggingPipeline()

        updated_results = [
            {
                "gif_sha": "sha123",
                "engine": "original",
                "lossy": "0",
                "screen_capture_confidence": "0.800000",
                "blocking_artifacts": "0.100000",
            }
        ]

        output_path = tmp_path / "ordered.csv"
        pipeline.write_tagged_csv(updated_results, output_path)

        # Check column order
        with open(output_path) as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader)

            # Original columns should come first
            assert header[0] == "gif_sha"
            assert header[1] == "engine"
            assert header[2] == "lossy"

            # Tagging columns should come after
            assert "screen_capture_confidence" in header
            assert "blocking_artifacts" in header

    def test_write_tagged_csv_empty_results(self, mock_tagger, tmp_path):
        """Test error handling for empty results."""
        pipeline = TaggingPipeline()

        output_path = tmp_path / "empty.csv"

        with pytest.raises(ValueError, match="No results to write"):
            pipeline.write_tagged_csv([], output_path)

    @patch("giflab.tag_pipeline.tqdm")
    def test_run_complete_workflow_success(
        self,
        mock_tqdm,
        mock_tagger,
        sample_csv_file,
        sample_gifs,
        sample_csv_data,
        tmp_path,
    ):
        """Test the complete tagging pipeline workflow."""
        # Setup mocks
        mock_tqdm.side_effect = (
            lambda x, **kwargs: x
        )  # Pass through without progress bar

        mock_tagging_result = TaggingResult(
            gif_sha="sha123",
            scores=dict.fromkeys(TaggingPipeline.TAGGING_COLUMNS, 0.5),
            model_version="test",
            processing_time_ms=100,
        )
        mock_tagger.tag_gif.return_value = mock_tagging_result

        # Create pipeline and run
        pipeline = TaggingPipeline()
        pipeline.tagger = mock_tagger

        with patch(
            "giflab.tag_pipeline.read_csv_as_dicts", return_value=sample_csv_data
        ):
            output_path = tmp_path / "output.csv"
            result = pipeline.run(sample_csv_file, sample_gifs, output_path)

        # Verify results
        assert result["status"] == "completed"
        assert result["total_results"] == 3
        assert result["original_gifs"] == 2
        assert result["tagged_successfully"] == 2  # Both original GIFs found and tagged
        assert result["tagging_failures"] == 0
        assert result["tagging_columns_added"] == 25

        # Verify output file exists
        assert output_path.exists()

    def test_run_no_results(self, mock_tagger, tmp_path):
        """Test handling when CSV has no results."""
        csv_path = tmp_path / "empty.csv"

        pipeline = TaggingPipeline()

        with patch("giflab.tag_pipeline.read_csv_as_dicts", return_value=[]):
            result = pipeline.run(csv_path, tmp_path, None)

            assert result["status"] == "no_results"
            assert result["tagged"] == 0

    def test_run_no_original_gifs(self, mock_tagger, sample_csv_file, sample_gifs):
        """Test handling when no original GIFs are found."""
        # Data with no original engine records
        compressed_only_data = [
            {"gif_sha": "sha123", "engine": "gifsicle", "orig_filename": "test1.gif"},
            {"gif_sha": "sha456", "engine": "animately", "orig_filename": "test2.gif"},
        ]

        pipeline = TaggingPipeline()

        with patch(
            "giflab.tag_pipeline.read_csv_as_dicts", return_value=compressed_only_data
        ):
            result = pipeline.run(sample_csv_file, sample_gifs, None)

            assert result["status"] == "no_original_gifs"

    @patch("giflab.tag_pipeline.tqdm")
    def test_run_gif_files_not_found(
        self, mock_tqdm, mock_tagger, sample_csv_file, tmp_path, sample_csv_data
    ):
        """Test handling when GIF files are not found."""
        mock_tqdm.side_effect = lambda x, **kwargs: x

        # Empty directory - no GIF files
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        pipeline = TaggingPipeline()

        with patch(
            "giflab.tag_pipeline.read_csv_as_dicts", return_value=sample_csv_data
        ):
            result = pipeline.run(sample_csv_file, empty_dir, None)

            assert result["status"] == "no_successful_tags"
            assert result["tagged_successfully"] == 0
            assert result["tagging_failures"] == 2  # Both GIFs not found

    def test_run_auto_output_path(
        self, mock_tagger, sample_csv_file, sample_gifs, sample_csv_data
    ):
        """Test automatic output path generation."""
        mock_tagging_result = TaggingResult(
            gif_sha="sha123",
            scores=dict.fromkeys(TaggingPipeline.TAGGING_COLUMNS, 0.5),
            model_version="test",
            processing_time_ms=100,
        )
        mock_tagger.tag_gif.return_value = mock_tagging_result

        pipeline = TaggingPipeline()
        pipeline.tagger = mock_tagger

        with patch(
            "giflab.tag_pipeline.read_csv_as_dicts", return_value=sample_csv_data
        ), patch("giflab.tag_pipeline.tqdm", side_effect=lambda x, **kwargs: x):
            result = pipeline.run(sample_csv_file, sample_gifs, None)

            # Should generate auto-timestamped output path
            assert "output_path" in result
            assert "results_tagged_" in result["output_path"]


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_tagging_pipeline(self):
        """Test factory function for creating tagging pipeline."""
        with patch("giflab.tag_pipeline.HybridCompressionTagger"):
            pipeline = create_tagging_pipeline(workers=3)

            assert isinstance(pipeline, TaggingPipeline)
            assert pipeline.workers == 3


class TestValidateTaggedCsv:
    """Tests for CSV validation function."""

    def test_validate_tagged_csv_success(self, tmp_path):
        """Test validation of properly tagged CSV."""
        csv_path = tmp_path / "tagged.csv"

        # Create CSV with all tagging columns
        fieldnames = ["gif_sha", "engine"] + TaggingPipeline.TAGGING_COLUMNS
        with open(csv_path, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(dict.fromkeys(fieldnames, "0.5"))

        with patch("giflab.tag_pipeline.pd.read_csv") as mock_read:
            mock_df = Mock()
            mock_df.columns = fieldnames
            mock_df.__len__ = Mock(return_value=1)
            mock_read.return_value = mock_df

            result = validate_tagged_csv(csv_path)

            assert result["valid"] is True
            assert result["tagging_columns_present"] == 25
            assert result["tagging_columns_missing"] == 0

    def test_validate_tagged_csv_missing_columns(self, tmp_path):
        """Test validation with missing tagging columns."""
        csv_path = tmp_path / "incomplete.csv"

        # Create CSV with only some tagging columns
        partial_columns = ["gif_sha", "engine"] + TaggingPipeline.TAGGING_COLUMNS[:10]
        with open(csv_path, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=partial_columns)
            writer.writeheader()

        with patch("giflab.tag_pipeline.pd.read_csv") as mock_read:
            mock_df = Mock()
            mock_df.columns = partial_columns
            mock_df.__len__ = Mock(return_value=0)
            mock_read.return_value = mock_df

            result = validate_tagged_csv(csv_path)

            assert result["valid"] is False
            assert result["tagging_columns_present"] == 10
            assert result["tagging_columns_missing"] == 15
            assert len(result["missing_columns"]) == 15

    def test_validate_tagged_csv_error_handling(self, tmp_path):
        """Test validation error handling."""
        csv_path = tmp_path / "nonexistent.csv"

        result = validate_tagged_csv(csv_path)

        assert result["valid"] is False
        assert "error" in result


@pytest.mark.integration
class TestTagPipelineIntegration:
    """Integration tests for the tag pipeline."""

    @pytest.fixture
    def integration_setup(self, tmp_path):
        """Set up a complete integration test environment."""
        # Create directory structure
        gif_dir = tmp_path / "gifs"
        gif_dir.mkdir()

        # Create test GIF files
        for i, name in enumerate(["test1.gif", "test2.gif"]):
            gif_path = gif_dir / name
            gif_path.write_bytes(f"fake gif content {i}".encode())

        # Create compression results CSV
        csv_path = tmp_path / "results.csv"
        results_data = [
            {
                "gif_sha": "sha123",
                "orig_filename": "test1.gif",
                "engine": "original",
                "lossy": "0",
                "frame_keep_ratio": "1.00",
                "color_keep_count": "256",
                "kilobytes": "100.5",
                "ssim": "1.000",
            },
            {
                "gif_sha": "sha123",
                "orig_filename": "test1.gif",
                "engine": "gifsicle",
                "lossy": "40",
                "frame_keep_ratio": "0.80",
                "color_keep_count": "64",
                "kilobytes": "50.2",
                "ssim": "0.936",
            },
            {
                "gif_sha": "sha456",
                "orig_filename": "test2.gif",
                "engine": "original",
                "lossy": "0",
                "frame_keep_ratio": "1.00",
                "color_keep_count": "256",
                "kilobytes": "200.0",
                "ssim": "1.000",
            },
        ]

        fieldnames = results_data[0].keys()
        with open(csv_path, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results_data)

        return {"gif_dir": gif_dir, "csv_path": csv_path, "results_data": results_data}

    @patch("giflab.tag_pipeline.HybridCompressionTagger")
    @patch("giflab.tag_pipeline.tqdm")
    def test_end_to_end_tagging_workflow(
        self, mock_tqdm, mock_tagger_class, integration_setup
    ):
        """Test complete end-to-end tagging workflow."""
        mock_tqdm.side_effect = lambda x, **kwargs: x

        # Setup mock tagger
        mock_tagger = Mock()
        mock_tagger_class.return_value = mock_tagger

        # Create realistic tagging results
        def create_mock_result(gif_sha):
            scores = {}
            # Add all 25 required scores
            for col in TaggingPipeline.TAGGING_COLUMNS:
                if "confidence" in col:
                    # Content classification scores
                    scores[col] = 0.8 if "screen_capture" in col else 0.1
                else:
                    # Other scores
                    scores[col] = 0.3

            return TaggingResult(
                gif_sha=gif_sha,
                scores=scores,
                model_version="integration_test",
                processing_time_ms=150,
            )

        mock_tagger.tag_gif.side_effect = lambda path, gif_sha: create_mock_result(
            gif_sha
        )

        # Run pipeline
        pipeline = TaggingPipeline()
        output_path = integration_setup["csv_path"].parent / "tagged_output.csv"

        result = pipeline.run(
            integration_setup["csv_path"], integration_setup["gif_dir"], output_path
        )

        # Verify pipeline results
        assert result["status"] == "completed"
        assert result["total_results"] == 3
        assert result["original_gifs"] == 2
        assert result["tagged_successfully"] == 2
        assert result["tagging_failures"] == 0
        assert result["tagging_columns_added"] == 25

        # Verify output file
        assert output_path.exists()

        # Verify CSV structure and content
        with open(output_path) as csvfile:
            reader = csv.DictReader(csvfile)
            rows = list(reader)

            # Should have all original rows
            assert len(rows) == 3

            # Check that all rows have tagging columns
            for row in rows:
                for col in TaggingPipeline.TAGGING_COLUMNS:
                    assert col in row
                    assert row[col] != ""  # Should have values

            # Check score inheritance - both variants of sha123 should have same scores
            sha123_rows = [row for row in rows if row["gif_sha"] == "sha123"]
            assert len(sha123_rows) == 2

            for col in TaggingPipeline.TAGGING_COLUMNS:
                assert sha123_rows[0][col] == sha123_rows[1][col]

            # Check that screen_capture_confidence is dominant (0.8)
            assert sha123_rows[0]["screen_capture_confidence"] == "0.800000"
            assert sha123_rows[0]["vector_art_confidence"] == "0.100000"

    def test_real_csv_validation_workflow(self, integration_setup):
        """Test validation workflow with real CSV files."""
        # Test untagged CSV
        validation = validate_tagged_csv(integration_setup["csv_path"])
        assert validation["valid"] is False
        assert validation["tagging_columns_missing"] == 25

        # Create tagged CSV
        tagged_path = integration_setup["csv_path"].parent / "tagged.csv"
        original_data = integration_setup["results_data"]

        # Add tagging columns to data
        tagged_data = []
        for row in original_data:
            new_row = row.copy()
            for col in TaggingPipeline.TAGGING_COLUMNS:
                new_row[col] = "0.500000"
            tagged_data.append(new_row)

        # Write tagged CSV
        fieldnames = list(original_data[0].keys()) + TaggingPipeline.TAGGING_COLUMNS
        with open(tagged_path, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(tagged_data)

        # Test tagged CSV validation
        validation = validate_tagged_csv(tagged_path)
        assert validation["valid"] is True
        assert validation["tagging_columns_present"] == 25
        assert validation["tagging_columns_missing"] == 0
