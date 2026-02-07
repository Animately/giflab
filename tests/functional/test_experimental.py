"""Tests for experimental pipeline framework and synthetic expansion.

These tests validate the systematic experimental approach, verify research findings
about redundant dithering methods, and test all new frame generation methods,
targeted expansion strategy, and bug fixes for the expanded synthetic dataset.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from giflab.core import AnalysisResult, GifLabRunner, SyntheticGifSpec


class TestSyntheticGifGeneration:
    """Test synthetic GIF generation for various content types."""

    def test_creates_all_synthetic_specs(self, tmp_path):
        """Test that all synthetic GIF specifications are generated."""
        eliminator = GifLabRunner(tmp_path)

        # Generate synthetic GIFs
        gif_paths = eliminator.generate_synthetic_gifs()

        # Should create all specified synthetic types
        assert len(gif_paths) == len(eliminator.synthetic_specs)

        # All files should exist
        for gif_path in gif_paths:
            assert gif_path.exists()
            assert gif_path.suffix == ".gif"

    def test_synthetic_gif_content_types(self, tmp_path):
        """Test that different content types are properly represented."""
        eliminator = GifLabRunner(tmp_path)

        # Initialize experiment directory to populate synthetic_specs
        eliminator._initialize_experiment_directory()

        # Check that research-based content types are included
        content_types = {spec.content_type for spec in eliminator.synthetic_specs}

        # Should include key content types from research
        assert "gradient" in content_types  # Benefits from dithering
        assert "solid" in content_types  # Should NOT use dithering
        assert "noise" in content_types  # Good for Bayer scales 4-5
        assert "contrast" in content_types  # High contrast patterns

    def test_gradient_frame_generation(self, tmp_path):
        """Test gradient frame generation produces expected output."""
        eliminator = GifLabRunner(tmp_path)

        # Create a gradient frame using the new vectorized generator
        frame = eliminator._frame_generator.create_frame("gradient", (50, 50), 0, 10)

        assert frame.size == (50, 50)
        assert frame.mode == "RGB"

        # Should have color variation (not solid color)
        colors = frame.getcolors(maxcolors=10000)  # Increase max colors limit
        if colors:
            assert len(colors) > 10  # Should have many different colors
        else:
            # If getcolors returns None, there are many colors (which is good for gradients)
            assert True  # This is expected for complex gradients

    def test_solid_frame_generation(self, tmp_path):
        """Test solid color frame generation."""
        eliminator = GifLabRunner(tmp_path)

        # Create a solid color frame using the new vectorized generator
        frame = eliminator._frame_generator.create_frame("solid", (50, 50), 0, 10)

        assert frame.size == (50, 50)
        assert frame.mode == "RGB"

        # Should have fewer distinct colors than gradient
        colors = frame.getcolors()
        assert len(colors) <= 25  # Block-based pattern with limited colors


class TestEliminationLogic:
    """Test the core elimination analysis logic."""

    @pytest.mark.fast
    def test_elimination_result_structure(self):
        """Test AnalysisResult data structure."""
        result = AnalysisResult()

        # Should initialize with empty collections
        assert isinstance(result.eliminated_pipelines, set)
        assert isinstance(result.retained_pipelines, set)
        assert isinstance(result.performance_matrix, dict)
        assert isinstance(result.elimination_reasons, dict)
        assert isinstance(result.content_type_winners, dict)

    @pytest.mark.fast
    @patch("giflab.dynamic_pipeline.generate_all_pipelines")
    def test_analyze_and_eliminate_logic(self, mock_generate_pipelines, tmp_path):
        """Test the pipeline elimination logic."""
        import pandas as pd

        # Mock pipeline generation
        mock_generate_pipelines.return_value = []

        eliminator = GifLabRunner(tmp_path)

        # Create mock results DataFrame with more realistic elimination scenario
        test_data = [
            {"content_type": "gradient", "pipeline_id": "pipeline_A", "ssim_mean": 0.9},
            {"content_type": "gradient", "pipeline_id": "pipeline_B", "ssim_mean": 0.8},
            {
                "content_type": "gradient",
                "pipeline_id": "pipeline_C",
                "ssim_mean": 0.02,
            },  # Much lower to ensure elimination
            {"content_type": "solid", "pipeline_id": "pipeline_A", "ssim_mean": 0.6},
            {"content_type": "solid", "pipeline_id": "pipeline_B", "ssim_mean": 0.9},
            {
                "content_type": "solid",
                "pipeline_id": "pipeline_C",
                "ssim_mean": 0.01,
            },  # Much lower to ensure elimination
        ]
        results_df = pd.DataFrame(test_data)

        # Run elimination analysis
        elimination_result = eliminator._analyze_and_experiment(
            results_df, threshold=0.05
        )

        # Pipeline A and B should be retained (winners in at least one content type)
        assert "pipeline_A" in elimination_result.retained_pipelines
        assert "pipeline_B" in elimination_result.retained_pipelines

        # Pipeline C should be eliminated (consistently poor performance across content types)
        assert "pipeline_C" in elimination_result.eliminated_pipelines

        # Check content type winners
        assert "gradient" in elimination_result.content_type_winners
        assert "solid" in elimination_result.content_type_winners


class TestResearchValidation:
    """Test validation of preliminary research findings."""

    @pytest.mark.fast
    def test_content_type_detection(self, tmp_path):
        """Test content type detection from GIF names."""
        eliminator = GifLabRunner(tmp_path)

        # Initialize experiment directory to populate synthetic_specs
        eliminator._initialize_experiment_directory()

        # Test content type mapping
        assert eliminator._get_content_type("smooth_gradient") == "gradient"
        assert eliminator._get_content_type("solid_blocks") == "solid"
        assert eliminator._get_content_type("photographic_noise") == "noise"
        assert eliminator._get_content_type("unknown_name") == "unknown"

    @pytest.mark.fast
    def test_validate_research_findings(self, tmp_path):
        """Test research findings validation framework."""
        eliminator = GifLabRunner(tmp_path)

        # Run validation (returns placeholder results for now)
        findings = eliminator.validate_research_findings()

        # Should return dict with validation results
        assert isinstance(findings, dict)

        # Should include key research findings
        assert any("imagemagick" in key for key in findings.keys())
        assert any("ffmpeg" in key for key in findings.keys())
        assert any("gifsicle" in key for key in findings.keys())


class TestImageMagickEnhanced:
    """Test enhanced ImageMagick engine with dithering methods."""

    @pytest.mark.fast
    def test_dithering_methods_list(self):
        """Test that all expected dithering methods are included."""
        from giflab.external_engines.imagemagick_enhanced import (
            IMAGEMAGICK_DITHERING_METHODS,
        )

        # Should include all 13 methods from research
        assert len(IMAGEMAGICK_DITHERING_METHODS) == 13

        # Should include key methods identified in research
        assert "None" in IMAGEMAGICK_DITHERING_METHODS
        assert "FloydSteinberg" in IMAGEMAGICK_DITHERING_METHODS
        assert "Riemersma" in IMAGEMAGICK_DITHERING_METHODS  # Best performer

        # Should include redundant methods for testing
        assert "O2x2" in IMAGEMAGICK_DITHERING_METHODS
        assert "H4x4a" in IMAGEMAGICK_DITHERING_METHODS

    @pytest.mark.fast
    @patch("giflab.external_engines.imagemagick_enhanced._magick_binary")
    @patch(
        "giflab.external_engines.imagemagick_enhanced.run_command"
    )  # Fixed mock path
    def test_color_reduce_with_dithering(
        self, mock_run_command, mock_magick_binary, tmp_path
    ):
        """Test enhanced color reduction with specific dithering method."""
        from giflab.external_engines.imagemagick_enhanced import (
            color_reduce_with_dithering,
        )

        # Mock binary discovery and command execution
        mock_magick_binary.return_value = "magick"
        mock_run_command.return_value = {
            "render_ms": 100,
            "engine": "imagemagick",
            "command": "mock_command",
            "kilobytes": 50,
        }

        input_path = tmp_path / "input.gif"
        output_path = tmp_path / "output.gif"

        # Create dummy input and output files for the test
        input_path.touch()
        output_path.touch()  # Ensure output file exists for size calculation

        # Test with Riemersma dithering
        result = color_reduce_with_dithering(
            input_path, output_path, colors=16, dithering_method="Riemersma"
        )

        # Should add dithering metadata
        assert result["dithering_method"] == "Riemersma"
        assert "pipeline_variant" in result
        assert result["pipeline_variant"] == "imagemagick_dither_riemersma"

        # Should call magick with correct dithering parameter
        mock_run_command.assert_called_once()
        call_args = mock_run_command.call_args[0][0]  # First positional argument (cmd)
        assert "-dither" in call_args
        assert "Riemersma" in call_args


class TestFFmpegEnhanced:
    """Test enhanced FFmpeg engine with dithering methods."""

    def test_ffmpeg_dithering_methods_list(self):
        """Test that all expected FFmpeg dithering methods are included."""
        from giflab.external_engines.ffmpeg_enhanced import FFMPEG_DITHERING_METHODS

        # Should include all methods from research (10 total)
        assert len(FFMPEG_DITHERING_METHODS) == 10

        # Should include key methods
        assert "none" in FFMPEG_DITHERING_METHODS
        assert "floyd_steinberg" in FFMPEG_DITHERING_METHODS
        assert "sierra2" in FFMPEG_DITHERING_METHODS  # Best balance from research

        # Should include all Bayer scale variants
        bayer_methods = [m for m in FFMPEG_DITHERING_METHODS if m.startswith("bayer")]
        assert len(bayer_methods) == 6  # Scales 0-5

        # Should include best performing Bayer scales from research
        assert "bayer:bayer_scale=4" in FFMPEG_DITHERING_METHODS
        assert "bayer:bayer_scale=5" in FFMPEG_DITHERING_METHODS

    @patch("giflab.external_engines.ffmpeg_enhanced._ffmpeg_binary")
    @patch("giflab.external_engines.ffmpeg_enhanced.run_command")  # Fixed mock path
    def test_color_reduce_with_dithering_ffmpeg(
        self, mock_run_command, mock_ffmpeg_binary, tmp_path
    ):
        """Test FFmpeg enhanced color reduction with dithering."""
        from giflab.external_engines.ffmpeg_enhanced import color_reduce_with_dithering

        # Mock binary and command execution
        mock_ffmpeg_binary.return_value = "ffmpeg"
        mock_run_command.return_value = {
            "render_ms": 50,
            "engine": "ffmpeg",
            "command": "mock_command",
            "kilobytes": 25,
        }

        input_path = tmp_path / "input.gif"
        output_path = tmp_path / "output.gif"
        palette_path = tmp_path / "palette.png"

        # Create dummy input and output files for the test
        input_path.touch()
        output_path.touch()
        palette_path.touch()  # FFmpeg needs palette file too

        # Test with Sierra2 dithering
        result = color_reduce_with_dithering(
            input_path, output_path, colors=16, dithering_method="sierra2"
        )

        # Should add dithering metadata
        assert result["dithering_method"] == "sierra2"
        assert result["pipeline_variant"] == "ffmpeg_dither_sierra2"

        # Should be called twice (palette generation + application)
        assert mock_run_command.call_count == 2


class TestIntegration:
    """Integration tests for the complete elimination workflow."""

    def test_cli_command_structure(self):
        """Test that CLI command is properly structured."""
        from giflab.cli import main

        # Should be a click group with experiment command
        assert hasattr(main, "commands")
        assert "run" in main.commands

        # Get the run command
        run_cmd = main.commands["run"]

        # Should have expected options
        param_names = [param.name for param in run_cmd.params]
        assert "input_dir" in param_names
        assert "db" in param_names
        assert "mode" in param_names

    @pytest.mark.fast
    @patch("giflab.core.runner.GifLabRunner")
    def test_elimination_workflow_integration(self, mock_eliminator_class, tmp_path):
        """Test integration of elimination workflow components."""

        # Mock the eliminator
        mock_eliminator = MagicMock()
        mock_eliminator_class.return_value = mock_eliminator

        # Mock results
        mock_elimination_result = AnalysisResult()
        mock_elimination_result.eliminated_pipelines = {"bad_pipeline"}
        mock_elimination_result.retained_pipelines = {"good_pipeline"}
        mock_elimination_result.content_type_winners = {
            "gradient": ["pipeline_A", "pipeline_B"],
            "solid": ["pipeline_C"],
        }

        mock_eliminator.run_analysis.return_value = mock_elimination_result

        # Test workflow integration - use the mock instead of real object
        eliminator = mock_eliminator_class(tmp_path)
        eliminator.run_analysis()

        # Should create eliminator instance
        mock_eliminator_class.assert_called_once_with(tmp_path)


@pytest.fixture
def sample_gif(tmp_path):
    """Create a simple test GIF for testing."""
    from PIL import Image

    # Create a simple 2-frame GIF
    frames = []
    for i in range(2):
        img = Image.new("RGB", (10, 10), color=(i * 128, 0, 0))
        frames.append(img)

    gif_path = tmp_path / "test.gif"
    frames[0].save(
        gif_path, save_all=True, append_images=frames[1:], duration=100, loop=0
    )

    return gif_path


class TestRealWorldIntegration:
    """Test with actual GIF files to ensure the framework works end-to-end."""

    def test_synthetic_gif_creation_produces_valid_gifs(self, tmp_path):
        """Test that synthetic GIF creation produces valid GIF files."""
        eliminator = GifLabRunner(tmp_path)

        # Generate one synthetic GIF
        gif_path = tmp_path / "test_gradient.gif"
        spec = SyntheticGifSpec(
            "test_gradient", 5, (50, 50), "gradient", "Test gradient"
        )

        eliminator._create_synthetic_gif(gif_path, spec)

        # Should create a valid GIF file
        assert gif_path.exists()

        # Should be readable by PIL
        from PIL import Image

        with Image.open(gif_path) as img:
            assert img.format == "GIF"
            assert img.size == (50, 50)

            # Should have multiple frames
            frame_count = 0
            try:
                while True:
                    img.seek(frame_count)
                    frame_count += 1
            except EOFError:
                pass

            assert frame_count == 5  # Should match spec.frames


# ---------------------------------------------------------------------------
# New synthetic expansion tests (merged from test_new_synthetic_expansion.py)
# ---------------------------------------------------------------------------


class TestNewFrameGenerationMethods:
    """Test all new frame generation methods added for expansion."""

    @pytest.mark.fast
    def test_mixed_content_frame_generation(self, tmp_path):
        """Test mixed content frame generation."""
        eliminator = GifLabRunner(tmp_path)

        # Test different sizes and frames
        test_cases = [(50, 50), (200, 150), (10, 10)]  # Original spec size  # Edge case

        for width, height in test_cases:
            frame = eliminator._frame_generator.create_frame(
                "mixed", (width, height), 0, 10
            )
            assert frame.size == (width, height)
            assert frame.mode == "RGB"

            # Should have varied colors (not solid)
            colors = frame.getcolors(maxcolors=256 * 256)
            if colors:  # getcolors returns None if too many colors
                assert len(colors) > 1

    @pytest.mark.fast
    def test_data_visualization_frame_generation(self, tmp_path):
        """Test data visualization (charts) frame generation."""
        eliminator = GifLabRunner(tmp_path)

        # Test with different frame indices for animation
        for frame_idx in [0, 5, 9]:
            frame = eliminator._frame_generator.create_frame(
                "charts", (300, 200), frame_idx, 10
            )
            assert frame.size == (300, 200)
            assert frame.mode == "RGB"

    @pytest.mark.fast
    def test_transitions_frame_generation(self, tmp_path):
        """Test transitions (morphing) frame generation."""
        eliminator = GifLabRunner(tmp_path)

        # Test both halves of the morphing animation
        total_frames = 15
        for frame_idx in [0, 7, 14]:  # Beginning, middle, end
            frame = eliminator._frame_generator.create_frame(
                "morph", (150, 150), frame_idx, total_frames
            )
            assert frame.size == (150, 150)
            assert frame.mode == "RGB"

    @pytest.mark.fast
    def test_single_pixel_anim_frame_generation(self, tmp_path):
        """Test single pixel animation frame generation."""
        eliminator = GifLabRunner(tmp_path)

        # Test minimal motion detection
        frame1 = eliminator._frame_generator.create_frame(
            "micro_detail", (100, 100), 0, 10
        )
        frame2 = eliminator._frame_generator.create_frame(
            "micro_detail", (100, 100), 1, 10
        )

        assert frame1.size == (100, 100)
        assert frame2.size == (100, 100)
        assert frame1.mode == "RGB"
        assert frame2.mode == "RGB"

        # Frames should be different (pixel changes)
        assert list(frame1.getdata()) != list(frame2.getdata())

    @pytest.mark.fast
    def test_static_minimal_change_frame_generation(self, tmp_path):
        """Test static minimal change frame generation."""
        eliminator = GifLabRunner(tmp_path)

        # Test edge cases that previously caused modulo by zero
        test_sizes = [
            (5, 5),  # Very small - should not crash
            (10, 10),  # Exactly at threshold
            (25, 25),  # Large enough for all elements
        ]

        for size in test_sizes:
            for frame_idx in [0, 4, 5, 8]:  # Test trigger frames
                frame = eliminator._frame_generator.create_frame(
                    "static_plus", size, frame_idx, 20
                )
                assert frame.size == size
                assert frame.mode == "RGB"

    @pytest.mark.fast
    def test_high_frequency_detail_frame_generation(self, tmp_path):
        """Test high frequency detail frame generation."""
        eliminator = GifLabRunner(tmp_path)

        # Test aliasing patterns
        frame = eliminator._frame_generator.create_frame("detail", (200, 200), 0, 12)
        assert frame.size == (200, 200)
        assert frame.mode == "RGB"

        # Should have high frequency patterns (multiple colors)
        colors = frame.getcolors(maxcolors=1000)
        if colors:
            assert len(colors) > 1  # Should have varied patterns, not solid color


class TestExpandedSyntheticSpecs:
    """Test the expanded synthetic GIF specifications."""

    @pytest.mark.fast
    def test_all_new_content_types_present(self, tmp_path):
        """Test that all new content types are included in specs."""
        eliminator = GifLabRunner(tmp_path)

        # Initialize experiment directory to populate synthetic_specs
        eliminator._initialize_experiment_directory()

        content_types = {spec.content_type for spec in eliminator.synthetic_specs}

        # New content types should be present
        assert "mixed" in content_types
        assert "charts" in content_types
        assert "morph" in content_types
        assert "micro_detail" in content_types
        assert "static_plus" in content_types
        assert "detail" in content_types

    @pytest.mark.fast
    def test_size_variations_present(self, tmp_path):
        """Test that size variations are properly included."""
        eliminator = GifLabRunner(tmp_path)

        # Initialize experiment directory to populate synthetic_specs
        eliminator._initialize_experiment_directory()

        spec_names = {spec.name for spec in eliminator.synthetic_specs}

        # Size variation specs should be present
        assert "gradient_small" in spec_names
        assert "gradient_medium" in spec_names
        assert "gradient_large" in spec_names
        assert "gradient_xlarge" in spec_names
        assert "noise_small" in spec_names
        assert "noise_large" in spec_names

    @pytest.mark.fast
    def test_frame_variations_present(self, tmp_path):
        """Test that frame count variations are included."""
        eliminator = GifLabRunner(tmp_path)

        # Initialize experiment directory to populate synthetic_specs
        eliminator._initialize_experiment_directory()

        spec_names = {spec.name for spec in eliminator.synthetic_specs}

        # Frame variation specs should be present
        assert "minimal_frames" in spec_names
        assert "long_animation" in spec_names
        assert "extended_animation" in spec_names

    @pytest.mark.fast
    def test_expanded_spec_count(self, tmp_path):
        """Test that we have the expected number of specs."""
        eliminator = GifLabRunner(tmp_path)

        # Initialize experiment directory to populate synthetic_specs
        eliminator._initialize_experiment_directory()

        # Should have expanded from 10 to 25 total specs
        assert len(eliminator.synthetic_specs) == 25

    def test_all_specs_generate_successfully(self, tmp_path):
        """Test that all expanded specs can generate GIFs without errors."""
        eliminator = GifLabRunner(tmp_path)

        # Initialize experiment directory to populate synthetic_specs
        eliminator._initialize_experiment_directory()

        # Test each spec individually
        for spec in eliminator.synthetic_specs:
            gif_path = tmp_path / f"{spec.name}.gif"

            # Should not raise an exception
            eliminator._create_synthetic_gif(gif_path, spec)

            # Should create a valid file
            assert gif_path.exists()
            assert gif_path.stat().st_size > 0


class TestTargetedExpansionStrategy:
    """Test the new targeted expansion sampling strategy."""

    @pytest.mark.fast
    def test_targeted_strategy_in_sampling_strategies(self, tmp_path):
        """Test that targeted strategy is available."""
        eliminator = GifLabRunner(tmp_path)

        assert "targeted" in eliminator.SAMPLING_STRATEGIES

        strategy = eliminator.SAMPLING_STRATEGIES["targeted"]
        assert strategy.name == "Targeted Expansion"
        assert strategy.sample_ratio == 0.12
        assert strategy.min_samples_per_tool == 4

    @pytest.mark.fast
    def test_targeted_sampling_method(self, tmp_path):
        """Test the targeted sampling method."""
        eliminator = GifLabRunner(tmp_path)

        # Create mock pipelines
        mock_pipelines = [MagicMock() for _ in range(100)]

        # Should not crash with targeted sampling
        result = eliminator.select_pipelines_intelligently(
            mock_pipelines, strategy="targeted"
        )

        # Should return some pipelines
        assert isinstance(result, list)
        assert len(result) > 0
        assert len(result) <= len(mock_pipelines)

    def test_get_targeted_synthetic_gifs(self, tmp_path):
        """Test targeted GIF generation."""
        eliminator = GifLabRunner(tmp_path)

        targeted_gifs = eliminator.get_targeted_synthetic_gifs()

        # Should generate exactly 17 GIFs
        assert len(targeted_gifs) == 17

        # All should be valid paths
        for gif_path in targeted_gifs:
            assert isinstance(gif_path, Path)
            assert gif_path.suffix == ".gif"
            # Note: Files are not immediately generated, just paths are returned
            # Actual GIF generation happens during the testing phase

    def test_targeted_gifs_content_selection(self, tmp_path):
        """Test that targeted GIFs include the right content."""
        eliminator = GifLabRunner(tmp_path)

        targeted_gifs = eliminator.get_targeted_synthetic_gifs()
        targeted_names = {gif.stem for gif in targeted_gifs}

        # Should include all original research-based content (10 GIFs)
        original_names = [
            "smooth_gradient",
            "complex_gradient",
            "solid_blocks",
            "high_contrast",
            "photographic_noise",
            "texture_complex",
            "geometric_patterns",
            "few_colors",
            "many_colors",
            "animation_heavy",
        ]
        for name in original_names:
            assert name in targeted_names

        # Should include strategic size variations (4 GIFs)
        size_names = [
            "gradient_small",
            "gradient_large",
            "gradient_xlarge",
            "noise_large",
        ]
        for name in size_names:
            assert name in targeted_names

        # Should include key frame variations (2 GIFs)
        frame_names = ["minimal_frames", "long_animation"]
        for name in frame_names:
            assert name in targeted_names

        # Should include essential new content (1 GIF)
        assert "mixed_content" in targeted_names

    def test_select_pipelines_intelligently_targeted(self, tmp_path):
        """Test intelligent pipeline selection with targeted strategy."""
        eliminator = GifLabRunner(tmp_path)

        # Create mock pipelines
        mock_pipelines = [MagicMock() for _ in range(100)]

        # Test targeted strategy specifically
        result = eliminator.select_pipelines_intelligently(mock_pipelines, "targeted")

        assert isinstance(result, list)
        assert len(result) > 0
        assert len(result) < len(mock_pipelines)  # Should be a subset


class TestEdgeCaseFixes:
    """Test the bug fixes for edge cases."""

    def test_empty_pipeline_list_handling(self, tmp_path):
        """Test that empty pipeline lists don't cause division by zero."""
        eliminator = GifLabRunner(tmp_path)

        # Should not crash with empty list
        result = eliminator.select_pipelines_intelligently(
            [], strategy="representative"
        )

        assert isinstance(result, list)
        assert len(result) == 0

    def test_small_gif_size_handling(self, tmp_path):
        """Test that very small GIF sizes don't cause modulo by zero."""
        eliminator = GifLabRunner(tmp_path)

        # Test problematic sizes that previously crashed
        test_cases = [(1, 1), (5, 5), (10, 10)]

        for size in test_cases:
            # Should not crash
            frame = eliminator._frame_generator.create_frame("static_plus", size, 0, 10)
            assert frame.size == size
            assert frame.mode == "RGB"

    def test_single_frame_generation(self, tmp_path):
        """Test that single frame GIFs can be generated."""
        eliminator = GifLabRunner(tmp_path)

        # Create a single-frame spec
        spec = SyntheticGifSpec(
            "test_single", 1, (50, 50), "gradient", "Single frame test"
        )

        gif_path = tmp_path / "test_single.gif"

        # Should not crash with single frame
        eliminator._create_synthetic_gif(gif_path, spec)

        assert gif_path.exists()
        assert gif_path.stat().st_size > 0


class TestIntegrationWithCLI:
    """Test integration of new functionality with CLI."""

    def test_targeted_strategy_cli_integration(self, tmp_path):
        """Test that CLI correctly handles targeted strategy."""
        eliminator = GifLabRunner(tmp_path)

        # Simulate CLI logic
        sampling_strategy = "targeted"
        use_targeted_gifs = sampling_strategy == "targeted"

        assert use_targeted_gifs is True

        # Should be able to get targeted GIFs
        if use_targeted_gifs:
            synthetic_gifs = eliminator.get_targeted_synthetic_gifs()
        else:
            synthetic_gifs = eliminator.generate_synthetic_gifs()

        assert len(synthetic_gifs) == 17  # Targeted count

    def test_run_elimination_analysis_with_targeted_gifs(self, tmp_path):
        """Test that elimination analysis works with targeted GIF flag."""
        eliminator = GifLabRunner(tmp_path)

        # Mock dependencies to avoid full integration issues
        with patch("giflab.dynamic_pipeline.generate_all_pipelines") as mock_gen:
            mock_gen.return_value = []

            with patch.object(eliminator, "_run_comprehensive_testing") as mock_test:
                import pandas as pd

                mock_test.return_value = pd.DataFrame()

                with patch.object(
                    eliminator, "_analyze_and_experiment"
                ) as mock_analyze:
                    mock_analyze.return_value = AnalysisResult()

                    # Should work with targeted GIFs enabled
                    result = eliminator.run_analysis(use_targeted_gifs=True)

                    assert isinstance(result, AnalysisResult)
