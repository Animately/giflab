"""Command-line interface for GifLab."""

import multiprocessing
import sys
from datetime import datetime
from pathlib import Path

import click
import pandas as pd

from .config import (
    DEFAULT_COMPRESSION_CONFIG,
    PathConfig,
)
from .experiment import (
    ExperimentalConfig,
    ExperimentalPipeline,
)
from .pipeline import CompressionPipeline
from .utils_pipeline_yaml import read_pipelines_yaml, write_pipelines_yaml
from .validation import ValidationError, validate_raw_dir, validate_worker_count
from giflab.combiner_registry import combiner_for
from giflab.pipeline_elimination import PipelineEliminator


@click.group()
@click.version_option(version="0.1.0", prog_name="giflab")
def main():
    """🎞️ GifLab — GIF compression and analysis laboratory."""
    pass


@main.command()
@click.argument(
    "raw_dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
)
@click.option(
    "--workers",
    "-j",
    type=int,
    default=0,
    help=f"Number of worker processes (default: {multiprocessing.cpu_count()} = CPU count)",
)
@click.option(
    "--resume/--no-resume", default=True, help="Skip existing renders (default: true)"
)
@click.option(
    "--fail-dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    help="Folder for bad GIFs (default: data/bad_gifs)",
)
@click.option(
    "--csv",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Output CSV path (default: auto-date in data/csv/)",
)
@click.option("--dry-run", is_flag=True, help="List work only, don't execute")
@click.option(
    "--renders-dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    help="Directory for rendered variants (default: data/renders)",
)
@click.option(
    "--detect-source-from-directory/--no-detect-source-from-directory",
    default=True,
    help="Detect source platform from directory structure (default: true)",
)
@click.option(
    "--pipelines",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="YAML file listing pipeline identifiers to run (overrides engine grid)",
)
def run(
    raw_dir: Path,
    workers: int,
    resume: bool,
    fail_dir: Path | None,
    csv: Path | None,
    dry_run: bool,
    renders_dir: Path | None,
    detect_source_from_directory: bool,
    pipelines: Path | None,
):
    """Run compression analysis on GIFs in RAW_DIR.

    Generates a grid of compression variants for every GIF and writes
    one CSV row per variant with quality metrics and metadata.

    RAW_DIR: Directory containing original GIF files to analyze
    """
    try:
        # Validate RAW_DIR input
        try:
            validated_raw_dir = validate_raw_dir(raw_dir, require_gifs=not dry_run)
        except ValidationError as e:
            click.echo(f"❌ Invalid RAW_DIR: {e}", err=True)
            click.echo("💡 Please provide a valid directory containing GIF files", err=True)
            sys.exit(1)

        # Validate worker count
        try:
            validated_workers = validate_worker_count(workers)
        except ValidationError as e:
            click.echo(f"❌ Invalid worker count: {e}", err=True)
            sys.exit(1)

        # Create path configuration
        path_config = PathConfig()

        # Override paths if provided
        if fail_dir:
            path_config.BAD_GIFS_DIR = fail_dir
        if renders_dir:
            path_config.RENDERS_DIR = renders_dir

        selected_pipes = None
        if pipelines is not None:
            selected_pipes = read_pipelines_yaml(pipelines)

        pipeline = CompressionPipeline(
            compression_config=DEFAULT_COMPRESSION_CONFIG,
            path_config=path_config,
            workers=validated_workers,
            resume=resume,
            detect_source_from_directory=detect_source_from_directory,
            selected_pipelines=selected_pipes,
        )

        # Generate CSV path if not provided
        if csv is None:
            timestamp = datetime.now().strftime("%Y%m%d")
            csv = path_config.CSV_DIR / f"results_{timestamp}.csv"

        # Ensure CSV parent directory exists
        csv.parent.mkdir(parents=True, exist_ok=True)

        click.echo("🎞️  GifLab Compression Pipeline")
        click.echo(f"📁 Input directory: {validated_raw_dir}")
        click.echo(f"📊 Output CSV: {csv}")
        click.echo(f"🎬 Renders directory: {path_config.RENDERS_DIR}")
        click.echo(f"❌ Bad GIFs directory: {path_config.BAD_GIFS_DIR}")
        click.echo(
            f"👥 Workers: {validated_workers if validated_workers > 0 else multiprocessing.cpu_count()}"
        )
        click.echo(f"🔄 Resume: {'Yes' if resume else 'No'}")
        if pipelines:
            click.echo(f"🎛️  Selected pipelines: {len(selected_pipes)} from {pipelines}")
        click.echo(f"🗂️  Directory source detection: {'Yes' if detect_source_from_directory else 'No'}")

        if dry_run:
            click.echo("🔍 DRY RUN MODE - Analysis only")
            _run_dry_run(pipeline, validated_raw_dir, csv)
        else:
            click.echo("🚀 Starting compression pipeline...")
            _run_pipeline(pipeline, validated_raw_dir, csv)

    except KeyboardInterrupt:
        click.echo("\n⏹️  Pipeline interrupted by user", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"❌ Pipeline failed: {e}", err=True)
        sys.exit(1)


def _run_dry_run(pipeline: CompressionPipeline, raw_dir: Path, csv_path: Path):
    """Run dry-run analysis showing what work would be done."""

    # Discover GIFs
    click.echo("\n📋 Discovering GIF files...")
    gif_paths = pipeline.discover_gifs(raw_dir)

    if not gif_paths:
        click.echo(f"⚠️  No GIF files found in {raw_dir}")
        return

    click.echo(f"✅ Found {len(gif_paths)} GIF files")

    # Generate jobs
    click.echo("\n🔧 Generating compression jobs...")
    all_jobs = pipeline.generate_jobs(gif_paths)

    if not all_jobs:
        click.echo("⚠️  No valid compression jobs could be generated")
        return

    # Filter existing jobs if resume is enabled
    jobs_to_run = pipeline.filter_existing_jobs(all_jobs, csv_path)

    # Show summary
    engines = DEFAULT_COMPRESSION_CONFIG.ENGINES
    frame_ratios = DEFAULT_COMPRESSION_CONFIG.FRAME_KEEP_RATIOS
    color_counts = DEFAULT_COMPRESSION_CONFIG.COLOR_KEEP_COUNTS
    lossy_levels = DEFAULT_COMPRESSION_CONFIG.LOSSY_LEVELS

    variants_per_gif = (
        len(engines) * len(frame_ratios) * len(color_counts) * len(lossy_levels)
    )

    click.echo("\n📊 Compression Matrix:")
    click.echo(f"   • Engines: {', '.join(engines)}")
    click.echo(f"   • Frame ratios: {', '.join(f'{r:.2f}' for r in frame_ratios)}")
    click.echo(f"   • Color counts: {', '.join(str(c) for c in color_counts)}")
    click.echo(f"   • Lossy levels: {', '.join(str(level) for level in lossy_levels)}")
    click.echo(f"   • Variants per GIF: {variants_per_gif}")

    click.echo("\n📈 Job Summary:")
    click.echo(f"   • Total jobs: {len(all_jobs)}")
    click.echo(f"   • Jobs to run: {len(jobs_to_run)}")
    click.echo(f"   • Jobs to skip: {len(all_jobs) - len(jobs_to_run)}")

    if not jobs_to_run:
        click.echo("✅ All jobs already completed")
    else:
        estimated_time = len(jobs_to_run) * 2  # Rough estimate: 2 seconds per job
        estimated_hours = estimated_time / 3600
        click.echo(
            f"⏱️  Estimated runtime: ~{estimated_time}s (~{estimated_hours:.1f}h)"
        )

    # Show sample jobs
    if jobs_to_run:
        click.echo("\n📝 Sample jobs to execute:")
        for i, job in enumerate(jobs_to_run[:5]):  # Show first 5 jobs
            click.echo(f"   {i+1}. {job.metadata.orig_filename}")
            click.echo(
                f"      • {job.engine}, lossy={job.lossy}, frames={job.frame_keep_ratio:.2f}, colors={job.color_keep_count}"
            )
            click.echo(f"      • Output: {job.output_path}")

        if len(jobs_to_run) > 5:
            click.echo(f"   ... and {len(jobs_to_run) - 5} more jobs")


def _run_pipeline(pipeline: CompressionPipeline, raw_dir: Path, csv_path: Path):
    """Execute the compression pipeline."""

    result = pipeline.run(raw_dir, csv_path)

    # Report results
    status = result["status"]
    processed = result["processed"]
    failed = result["failed"]
    skipped = result["skipped"]

    click.echo("\n📊 Pipeline Results:")
    click.echo(f"   • Status: {status}")
    click.echo(f"   • Processed: {processed}")
    click.echo(f"   • Failed: {failed}")
    click.echo(f"   • Skipped: {skipped}")

    if "total_jobs" in result:
        click.echo(f"   • Total jobs: {result['total_jobs']}")

    if "csv_path" in result:
        click.echo(f"   • Results saved to: {result['csv_path']}")

    if status == "completed":
        click.echo("✅ Pipeline completed successfully!")
    elif status == "no_files":
        click.echo("⚠️  No GIF files found to process")
    elif status == "no_jobs":
        click.echo("⚠️  No valid compression jobs could be generated")
    elif status == "all_complete":
        click.echo("✅ All jobs were already completed")
    elif status == "error":
        error_msg = result.get("error", "Unknown error")
        click.echo(f"❌ Pipeline failed: {error_msg}")
        sys.exit(1)
    else:
        click.echo(f"⚠️  Pipeline completed with status: {status}")


@main.command()
@click.argument(
    "csv_file", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.argument(
    "raw_dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path)
)
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Output CSV path (default: auto-timestamped in same directory)"
)
@click.option(
    "--workers",
    "-j",
    type=int,
    default=1,
    help="Number of worker processes (default: 1, parallel tagging not yet implemented)"
)
@click.option(
    "--validate-only",
    is_flag=True,
    help="Only validate CSV structure, don't run tagging"
)
def tag(
    csv_file: Path,
    raw_dir: Path,
    output: Path | None,
    workers: int,
    validate_only: bool
):
    """Add comprehensive tagging scores to existing compression results.

    Analyzes original GIF files and adds 25 continuous scores (0.0-1.0) to compression results:
    - 6 content classification scores (CLIP)
    - 4 quality/artifact assessment scores (Classical CV)
    - 5 technical characteristic scores (Classical CV)
    - 10 temporal motion analysis scores (Classical CV)

    CRITICAL: Tagging runs ONCE on original GIFs only, scores inherited by all variants.

    CSV_FILE: Path to existing compression results CSV file
    RAW_DIR: Directory containing original GIF files
    """
    try:
        from .tag_pipeline import TaggingPipeline, validate_tagged_csv

        # Validate RAW_DIR input
        try:
            validated_raw_dir = validate_raw_dir(raw_dir, require_gifs=True)
        except ValidationError as e:
            click.echo(f"❌ Invalid RAW_DIR: {e}", err=True)
            click.echo("💡 Please provide a valid directory containing GIF files", err=True)
            sys.exit(1)

        # Validate worker count
        try:
            validated_workers = validate_worker_count(workers)
        except ValidationError as e:
            click.echo(f"❌ Invalid worker count: {e}", err=True)
            sys.exit(1)

        click.echo("🏷️  GifLab Comprehensive Tagging Pipeline")
        click.echo(f"📊 Input CSV: {csv_file}")
        click.echo(f"📁 Raw GIFs directory: {validated_raw_dir}")

        if validate_only:
            click.echo("🔍 Validation mode - checking CSV structure...")
            validation_report = validate_tagged_csv(csv_file)

            if validation_report["valid"]:
                click.echo("✅ CSV structure is valid")
                click.echo(f"   • {validation_report['tagging_columns_present']}/25 tagging columns present")
            else:
                click.echo("❌ CSV validation failed")
                if "error" in validation_report:
                    click.echo(f"   • Error: {validation_report['error']}")
                else:
                    click.echo(f"   • Missing {validation_report['tagging_columns_missing']} tagging columns")
                    if validation_report['missing_columns']:
                        click.echo(f"   • Missing: {', '.join(validation_report['missing_columns'][:5])}...")
            return

        if output:
            click.echo(f"📄 Output CSV: {output}")
        else:
            click.echo("📄 Output CSV: auto-timestamped in same directory")

        click.echo(f"👥 Workers: {validated_workers} (parallel processing not yet implemented)")
        click.echo("🎯 Will add 25 comprehensive tagging scores")

        # Initialize tagging pipeline
        click.echo("\n🔧 Initializing hybrid tagging system...")
        pipeline = TaggingPipeline(workers=validated_workers)

        # Run comprehensive tagging
        click.echo("🚀 Starting comprehensive tagging analysis...")
        result = pipeline.run(csv_file, validated_raw_dir, output)

        # Report results
        status = result["status"]

        click.echo("\n📊 Tagging Results:")
        click.echo(f"   • Status: {status}")

        if "total_results" in result:
            click.echo(f"   • Total compression results: {result['total_results']}")
        if "original_gifs" in result:
            click.echo(f"   • Original GIFs found: {result['original_gifs']}")
        if "tagged_successfully" in result:
            click.echo(f"   • Successfully tagged: {result['tagged_successfully']}")
        if "tagging_failures" in result:
            click.echo(f"   • Tagging failures: {result['tagging_failures']}")
        if "tagging_columns_added" in result:
            click.echo(f"   • Tagging columns added: {result['tagging_columns_added']}")
        if "output_path" in result:
            click.echo(f"   • Results saved to: {result['output_path']}")

        if status == "completed":
            click.echo("✅ Comprehensive tagging completed successfully!")
            click.echo("\n🎯 Added 25 continuous scores for ML-ready compression optimization:")
            click.echo("   • Content classification (CLIP): 6 scores")
            click.echo("   • Quality assessment (Classical CV): 4 scores")
            click.echo("   • Technical characteristics (Classical CV): 5 scores")
            click.echo("   • Temporal motion analysis (Classical CV): 10 scores")
        elif status == "no_results":
            click.echo("⚠️  No compression results found in CSV")
        elif status == "no_original_gifs":
            click.echo("⚠️  No original GIFs found (engine='original')")
            click.echo("   💡 Tagging requires original records from compression pipeline")
        elif status == "no_successful_tags":
            click.echo("❌ No GIFs could be successfully tagged")
        else:
            click.echo(f"⚠️  Tagging completed with status: {status}")

    except KeyboardInterrupt:
        click.echo("\n⏹️  Tagging interrupted by user", err=True)
        sys.exit(1)
    except ImportError as e:
        click.echo(f"❌ Missing dependencies for tagging: {e}", err=True)
        click.echo("💡 Run: poetry install (to install torch and open-clip-torch)")
        sys.exit(1)
    except Exception as e:
        click.echo(f"❌ Tagging failed: {e}", err=True)
        sys.exit(1)


@main.command()
@click.argument(
    "raw_dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
)
def organize_directories(raw_dir: Path):
    """Create organized directory structure for source-based GIF collection.

    Creates subdirectories in RAW_DIR for different GIF sources:
    - tenor/      - GIFs from Tenor
    - animately/  - GIFs from Animately platform
    - tgif_dataset/ - GIFs from TGIF dataset
    - unknown/    - Ungrouped GIFs

    Each directory includes a README with organization guidelines.
    """
    from .directory_source_detection import (
        create_directory_structure,
        get_directory_organization_help,
    )

    try:
        click.echo("🗂️  Creating directory structure for source organization...")
        create_directory_structure(raw_dir)

        click.echo("✅ Directory structure created successfully!")
        click.echo(f"📁 Organized directories in: {raw_dir}")
        click.echo("\n" + get_directory_organization_help())

    except Exception as e:
        click.echo(f"❌ Failed to create directory structure: {e}", err=True)
        sys.exit(1)


@main.command()
@click.option(
    "--gifs",
    "-g",
    type=int,
    default=10,
    help="Number of test GIFs to generate (default: 10)",
)
@click.option(
    "--workers",
    "-j",
    type=int,
    default=0,
    help=f"Number of worker processes (default: {multiprocessing.cpu_count()} = CPU count)",
)
@click.option(
    "--sample-gifs-dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    help="Directory containing sample GIFs to use instead of generating new ones",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    help="Directory to save experiment results (default: data/experimental/results)",
)
@click.option(
    "--strategies",
    type=click.Choice([
        "pure_gifsicle",
        "pure_animately",
        "animately_then_gifsicle",
        "gifsicle_dithered",
        "gifsicle_optimized",
        "all"
    ]),
    multiple=True,
    default=["all"],
    help="Compression strategies to test (default: all)",
)
@click.option(
    "--matrix/--no-matrix",
    default=True,
    help="Enable dynamic matrix mode with all 5 engines (default: true)",
)
@click.option(
    "--no-analysis",
    is_flag=True,
    help="Disable detailed analysis report generation",
)
def experiment(
    gifs: int,
    workers: int,
    sample_gifs_dir: Path | None,
    output_dir: Path | None,
    strategies: tuple[str, ...],
    no_analysis: bool,
    matrix: bool,
):
    """Run experimental compression testing with diverse sample GIFs.

    This command tests different compression strategies on a small set of
    diverse GIFs to validate workflows and identify optimal parameters
    before running on large datasets.
    """
    try:
        # Validate worker count
        try:
            validated_workers = validate_worker_count(workers)
        except ValidationError as e:
            click.echo(f"❌ Invalid worker count: {e}", err=True)
            sys.exit(1)

        # Expand strategy selection
        all_strategies = [
            "pure_gifsicle",
            "pure_animately",
            "animately_then_gifsicle",
            "gifsicle_dithered",
            "gifsicle_optimized"
        ]

        if "all" in strategies:
            selected_strategies = all_strategies
        else:
            selected_strategies = list(strategies)

        # Create experimental configuration
        cfg = ExperimentalConfig(
            TEST_GIFS_COUNT=gifs,
            STRATEGIES=selected_strategies,
            ENABLE_DETAILED_ANALYSIS=not no_analysis,
            ENABLE_MATRIX_MODE=matrix,
        )

        # Override paths if provided
        if sample_gifs_dir:
            cfg.SAMPLE_GIFS_PATH = sample_gifs_dir
        if output_dir:
            cfg.RESULTS_PATH = output_dir

        # Create experimental pipeline
        pipeline = ExperimentalPipeline(cfg, validated_workers)

        click.echo("🧪 GifLab Experimental Testing")
        click.echo(f"📊 Test GIFs: {gifs}")
        click.echo(f"🛠️ Strategies: {', '.join(selected_strategies)}")
        click.echo(f"📁 Sample GIFs: {cfg.SAMPLE_GIFS_PATH}")
        click.echo(f"📈 Results: {cfg.RESULTS_PATH}")
        click.echo(f"👥 Workers: {validated_workers}")
        click.echo(f"📊 Analysis: {'Enabled' if not no_analysis else 'Disabled'}")

        # Load sample GIFs
        sample_gifs = None
        if sample_gifs_dir and sample_gifs_dir.exists():
            sample_gifs = list(sample_gifs_dir.glob("*.gif"))
            if not sample_gifs:
                click.echo(f"⚠️ No GIF files found in {sample_gifs_dir}")
                click.echo("Will generate test GIFs instead")
                sample_gifs = None
            else:
                click.echo(f"📂 Found {len(sample_gifs)} sample GIFs")

        # Run experiment
        click.echo("\n🚀 Starting experimental pipeline...")
        results_path = pipeline.run_experiment(sample_gifs)

        if results_path.exists():
            click.echo("\n✅ Experiment completed successfully!")
            click.echo(f"📊 Results saved to: {results_path}")

            # Show quick summary
            if not no_analysis:
                analysis_path = results_path.parent / "analysis_report.json"
                if analysis_path.exists():
                    click.echo(f"📈 Analysis report: {analysis_path}")

        else:
            click.echo("\n❌ Experiment failed - no results generated")
            sys.exit(1)

    except Exception as e:
        click.echo(f"❌ Experiment failed: {e}", err=True)
        sys.exit(1)


# ---------------------------------------------------------------------------
# select-pipelines command
# ---------------------------------------------------------------------------


@main.command("select-pipelines")
@click.argument(
    "csv_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option("--metric", default="ssim", help="Quality metric to optimise (default: ssim)")
@click.option("--top", default=1, help="Top-N pipelines to pick (per variable)")
@click.option("--output", "-o", type=click.Path(dir_okay=False, path_type=Path), default=Path("winners.yaml"))
def select_pipelines(csv_file: Path, metric: str, top: int, output: Path):
    """Pick the best pipelines from an experiment CSV and write a YAML list."""

    click.echo("📊 Loading experiment results…")
    df = pd.read_csv(csv_file)

    if metric not in df.columns:
        click.echo(f"❌ Metric '{metric}' not found in CSV", err=True)
        raise SystemExit(1)

    click.echo(f"🔎 Selecting top {top} pipelines by {metric}…")
    grouped = df.groupby("strategy")[metric].mean().sort_values(ascending=False)
    winners = list(grouped.head(top).index)

    write_pipelines_yaml(output, winners)
    click.echo(f"✅ Wrote {len(winners)} pipelines to {output}")


@main.command()
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    default=Path("elimination_results"),
    help="Output directory for elimination results (default: elimination_results)",
)
@click.option(
    "--threshold",
    "-t", 
    type=float,
    default=0.3,
    help="Quality threshold for elimination (default: 0.3, lower = stricter)",
)
@click.option(
    "--validate-research",
    is_flag=True,
    help="Validate preliminary research findings about redundant methods",
)
@click.option(
    "--test-dithering-only",
    is_flag=True,
    help="Only test dithering methods (skip frame reduction and lossy compression)",
)
@click.option(
    "--resume",
    is_flag=True,
    help="Resume from previous incomplete run (uses progress tracking)",
)
@click.option(
    "--estimate-time",
    is_flag=True,
    help="Show time estimate and exit (no actual testing)",
)
@click.option(
    "--max-pipelines",
    type=int,
    default=0,
    help="Limit number of pipelines to test (0 = no limit, useful for quick tests)",
)
@click.option(
    "--sampling-strategy",
    type=click.Choice(['full', 'representative', 'factorial', 'progressive', 'targeted', 'quick']),
    default='representative',
    help="Intelligent sampling strategy to reduce testing time (default: representative)",
)
@click.option(
    "--use-gpu",
    is_flag=True,
    help="Enable GPU acceleration for quality metrics calculation (requires OpenCV with CUDA)",
)
def eliminate_pipelines(
    output_dir: Path,
    threshold: float,
    validate_research: bool,
    test_dithering_only: bool,
    resume: bool,
    estimate_time: bool,
    max_pipelines: int,
    sampling_strategy: str,
    use_gpu: bool,
):
    """🔬 Systematically eliminate underperforming pipeline combinations.
    
    This command creates 25 synthetic test GIFs (expanded from 10) and runs 
    competitive elimination analysis using COMPREHENSIVE QUALITY METRICS to 
    identify pipeline combinations that never outperform others.
    
    QUALITY METRICS USED (elaborate system from the repo):
    • Traditional: SSIM, MS-SSIM, PSNR, temporal consistency  
    • Extended: MSE, RMSE, FSIM, GMSD, color histogram correlation
    • Perceptual: Edge similarity, texture similarity, sharpness similarity
    • 🎯 COMPOSITE SCORE: Weighted combination (30% SSIM + 35% MS-SSIM + 25% PSNR + 10% temporal)
    
    INTELLIGENT SAMPLING STRATEGIES (reduce 935 → ~75-140 pipelines):
    • 🧠 representative: Sample from each tool category (15% of pipelines, ~2.5 hours)
    • 🧪 factorial: Statistical design of experiments (8% of pipelines, ~1.3 hours)  
    • 📈 progressive: Multi-stage elimination with refinement (25% → 8%, ~2.8 hours)
    • 🎯 targeted: Strategic expansion with high-value GIFs (12% pipelines, 17 GIFs, ~1.4 hours)
    • ⚡ quick: Fast development testing (5% of pipelines, ~50 minutes)
    
    EXPANDED SYNTHETIC DATASET (25 GIFs):
    • 📏 Size variations: 50x50 to 1000x1000 pixels (400x range)
    • 🎬 Frame variations: 2 to 100 frames (50x range) 
    • 🔄 New content: Mixed content, data visualization, transitions
    • ⚡ Edge cases: Single pixel changes, static content, high-freq details
    
    FEATURES:
    • ⏱️  Time estimation and progress tracking
    • 🔄 Resume capability for long-running experiments  
    • 📊 Comprehensive CSV output with all metrics
    • 🎯 Multi-criteria elimination (quality + compression + efficiency)
    • 🚀 GPU acceleration for quality metrics (Windows/CUDA)
    
    Examples:
    
        # Quick time estimate (no actual testing)
        giflab eliminate-pipelines --estimate-time
        
        # Fast intelligent sampling (recommended)
        giflab eliminate-pipelines --sampling-strategy representative --resume
        
        # Statistical design approach (most scientific)
        giflab eliminate-pipelines --sampling-strategy factorial --resume
        
        # Strategic expansion with high-value GIFs (RECOMMENDED for expansion testing)
        giflab eliminate-pipelines --sampling-strategy targeted --resume
        
        # Quick development test (~50 minutes)
        giflab eliminate-pipelines --sampling-strategy quick
        
        # Validate research findings about redundant methods
        giflab eliminate-pipelines --validate-research
        
        # Resume interrupted analysis
        giflab eliminate-pipelines --resume -o previous_results/
        
        # GPU-accelerated analysis (Windows with CUDA GPU)
        giflab eliminate-pipelines --sampling-strategy representative --use-gpu --resume
    """
    from giflab.external_engines.imagemagick_enhanced import (
        identify_redundant_methods,
        test_redundant_methods
    )
    from giflab.external_engines.ffmpeg_enhanced import (
        test_bayer_scale_performance,
        validate_sierra2_vs_floyd_steinberg
    )

    eliminator = PipelineEliminator(output_dir, use_gpu=use_gpu)
    
    # Get pipeline count for estimation
    from giflab.dynamic_pipeline import generate_all_pipelines
    all_pipelines = generate_all_pipelines()
    
    # Apply intelligent sampling strategy
    if sampling_strategy != 'full':
        test_pipelines = eliminator.select_pipelines_intelligently(all_pipelines, sampling_strategy)
        strategy_info = eliminator.SAMPLING_STRATEGIES[sampling_strategy]
        click.echo(f"🧠 Sampling strategy: {strategy_info.name}")
        click.echo(f"📋 {strategy_info.description}")
    elif max_pipelines > 0 and max_pipelines < len(all_pipelines):
        test_pipelines = all_pipelines[:max_pipelines]
        click.echo(f"⚠️  Limited testing: Using {max_pipelines} of {len(all_pipelines)} available pipelines")
    else:
        test_pipelines = all_pipelines
        click.echo("🔬 Full brute-force testing: Using all available pipelines")
    
    # Calculate total job estimates
    if sampling_strategy == 'targeted':
        synthetic_gifs = eliminator.get_targeted_synthetic_gifs()
    else:
        synthetic_gifs = eliminator.generate_synthetic_gifs()
    total_jobs = len(synthetic_gifs) * len(test_pipelines) * len(eliminator.test_params)
    estimated_time = eliminator._estimate_execution_time(total_jobs)
    
    click.echo("🔬 Pipeline Elimination Analysis")
    click.echo(f"📁 Output directory: {output_dir}")
    click.echo(f"🎯 Quality threshold: {threshold}")
    click.echo(f"📊 Total jobs: {total_jobs:,}")
    click.echo(f"⏱️  Estimated time: {estimated_time}")
    click.echo(f"🔄 Resume enabled: {resume}")
    click.echo(f"🚀 GPU acceleration: {use_gpu}")
    
    if estimate_time:
        click.echo("✅ Time estimation complete. Use --no-estimate-time to run actual analysis.")
        return
    
    if validate_research:
        click.echo("\n📊 Validating preliminary research findings...")

        # Generate test GIFs for validation  
        if sampling_strategy == 'targeted':
            synthetic_gifs = eliminator.get_targeted_synthetic_gifs()
        else:
            synthetic_gifs = eliminator.generate_synthetic_gifs()

        # Test ImageMagick redundant methods
        click.echo("Testing ImageMagick method redundancy...")
        redundant_groups = identify_redundant_methods(synthetic_gifs)

        if redundant_groups:
            click.echo("✅ Found redundant method groups:")
            for representative, equivalents in redundant_groups.items():
                click.echo(f"   {representative} = {equivalents}")
        else:
            click.echo("❌ No redundant method groups found")

        # Test FFmpeg Bayer scale performance on noise content
        noise_gif = None
        for gif_path in synthetic_gifs:
            if "noise" in gif_path.stem:
                noise_gif = gif_path
                break

        if noise_gif:
            click.echo("Testing FFmpeg Bayer scale performance on noisy content...")
            bayer_results = test_bayer_scale_performance(noise_gif)

            # Find best performing Bayer scales
            best_bayers = sorted(
                [(method, result.get("kilobytes", float('inf')))
                 for method, result in bayer_results.items()
                 if "error" not in result],
                key=lambda x: x[1]
            )

            if best_bayers:
                click.echo("✅ Best Bayer scales for noisy content:")
                for method, size_kb in best_bayers[:3]:
                    scale = method.split("=")[1]
                    click.echo(f"   Scale {scale}: {size_kb}KB")

        # Test Sierra2 vs Floyd-Steinberg comparison
        click.echo("Validating Sierra2 vs Floyd-Steinberg performance...")
        comparison_results = validate_sierra2_vs_floyd_steinberg(synthetic_gifs[:3])

        # Save validation results
        validation_summary = {
            "redundant_imagemagick_groups": {k: list(v) for k, v in redundant_groups.items()},
            "bayer_scale_results": bayer_results if 'bayer_results' in locals() else {},
            "sierra2_vs_floyd_comparison": comparison_results
        }

        import json
        with open(output_dir / "research_validation.json", 'w') as f:
            json.dump(validation_summary, f, indent=2)

        click.echo(f"✅ Research validation results saved to {output_dir}/research_validation.json")

    if test_dithering_only:
        click.echo("\n🎨 Testing dithering methods only...")
        # This would implement dithering-only testing
        click.echo("Dithering-only testing would be implemented here")
    else:
        click.echo("\n⚔️  Running full competitive elimination analysis...")
        
        # Check for existing resume data if requested
        if resume:
            resume_file = output_dir / "elimination_progress.json"
            if resume_file.exists():
                click.echo(f"📂 Found existing progress file: {resume_file}")
                click.echo("🔄 Will resume from previous state...")
            else:
                click.echo("🆕 No previous progress found, starting fresh...")
        
        # Run the full elimination analysis with comprehensive metrics
        use_targeted_gifs = (sampling_strategy == 'targeted')
        elimination_result = eliminator.run_elimination_analysis(
            test_pipelines=test_pipelines,
            elimination_threshold=threshold,
            use_targeted_gifs=use_targeted_gifs
        )
        
        # Display comprehensive results
        click.echo(f"\n📊 Comprehensive Elimination Results:")
        click.echo(f"   📉 Eliminated: {len(elimination_result.eliminated_pipelines)} pipelines")
        click.echo(f"   ✅ Retained: {len(elimination_result.retained_pipelines)} pipelines")
        total_pipelines = len(elimination_result.eliminated_pipelines) + len(elimination_result.retained_pipelines)
        if total_pipelines > 0:
            elimination_rate = len(elimination_result.eliminated_pipelines) / total_pipelines * 100
            click.echo(f"   📈 Elimination rate: {elimination_rate:.1f}%")
        else:
            click.echo("   📈 Elimination rate: N/A (no pipelines tested)")
        
        if elimination_result.eliminated_pipelines:
            click.echo(f"\n❌ Top eliminated pipelines:")
            # Show first 10 eliminated pipelines with reasons
            for i, pipeline in enumerate(sorted(elimination_result.eliminated_pipelines)[:10]):
                reason = elimination_result.elimination_reasons.get(pipeline, "No reason provided")
                click.echo(f"   {i+1:2d}. {pipeline}")
                click.echo(f"       └─ {reason}")
            
            if len(elimination_result.eliminated_pipelines) > 10:
                click.echo(f"   ... and {len(elimination_result.eliminated_pipelines) - 10} more (see full results in {output_dir})")
        
        if elimination_result.content_type_winners:
            click.echo("\n🏆 Winners by content type (Quality | Efficiency | Compression):")
            for content_type, winners in elimination_result.content_type_winners.items():
                # Show performance matrix info if available
                perf_info = elimination_result.performance_matrix.get(content_type, {})
                avg_quality = perf_info.get('mean_composite_quality', 0)
                avg_compression = perf_info.get('mean_compression_ratio', 0)
                
                click.echo(f"   📁 {content_type.upper()}: (avg quality: {avg_quality:.3f}, avg compression: {avg_compression:.1f}x)")
                
                # Show top 5 winners
                for i, winner in enumerate(winners[:5]):
                    click.echo(f"      {i+1}. {winner}")
                
                if len(winners) > 5:
                    click.echo(f"      ... and {len(winners) - 5} more")
    
    # Show file outputs
    click.echo(f"\n📄 Results saved to:")
    click.echo(f"   📊 CSV data: {output_dir}/elimination_test_results.csv")
    click.echo(f"   📝 Summary: {output_dir}/elimination_summary.json") 
    failed_pipelines_file = output_dir / "failed_pipelines.json"
    has_failed_pipelines = failed_pipelines_file.exists()
    if has_failed_pipelines:
        click.echo(f"   ❌ Failed pipelines: {output_dir}/failed_pipelines.json")
        click.echo(f"   📋 Failure analysis: {output_dir}/failure_analysis_report.txt")
    if resume:
        click.echo(f"   💾 Progress: {output_dir}/elimination_progress.json")
    
    click.echo(f"\n✅ Analysis complete!")
    
    # Provide next steps
    click.echo(f"\n🔄 Next steps:")
    click.echo(f"   1. Review eliminated pipelines in the summary file")
    click.echo(f"   2. Use retained pipelines for production workflows")
    click.echo(f"   3. Run: giflab run data/raw --pipelines <retained_pipelines>")
    if has_failed_pipelines:
        click.echo(f"   4. Review failed pipelines: giflab view-failures {output_dir}")
    if elimination_result.eliminated_pipelines:
        total_tested = len(elimination_result.eliminated_pipelines) + len(elimination_result.retained_pipelines)
        if total_tested > 0:
            potential_savings = len(elimination_result.eliminated_pipelines) / total_tested * 100
            click.echo(f"   💡 Potential {potential_savings:.0f}% reduction in pipeline testing time!")
        else:
            click.echo("   💡 Potential time savings: Unable to calculate")


@main.command()
@click.argument("results_dir", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--error-type",
    type=click.Choice(['all', 'gifski', 'ffmpeg', 'imagemagick', 'gifsicle', 'animately', 'command', 'timeout', 'other']),
    default='all',
    help="Filter failures by error type (default: all)",
)
@click.option(
    "--limit",
    "-n",
    type=int,
    default=10,
    help="Maximum number of failures to show (default: 10, 0 = show all)",
)
@click.option(
    "--detailed",
    "-v",
    is_flag=True,
    help="Show detailed error information including stack traces",
)
def view_failures(results_dir: Path, error_type: str, limit: int, detailed: bool):
    """
    View detailed information about failed pipelines from elimination testing.
    
    This command analyzes the failed_pipelines.json file from elimination runs
    and provides a human-readable summary of what went wrong.
    
    Examples:
    
        # View top 10 failures from latest run
        giflab view-failures elimination_results/
        
        # View all gifski failures with details
        giflab view-failures elimination_results/ --error-type gifski --limit 0 --detailed
        
        # Quick overview of command execution failures
        giflab view-failures elimination_results/ --error-type command --limit 5
    """
    import json
    from collections import Counter
    
    failed_pipelines_file = results_dir / "failed_pipelines.json"
    if not failed_pipelines_file.exists():
        click.echo(f"❌ No failed pipelines file found at: {failed_pipelines_file}")
        click.echo("   Make sure you're pointing to a directory with elimination results.")
        return
    
    try:
        with open(failed_pipelines_file, 'r') as f:
            failed_pipelines = json.load(f)
    except json.JSONDecodeError as e:
        click.echo(f"❌ Error reading failed pipelines file: {e}")
        return
    
    if not failed_pipelines:
        click.echo("✅ No failed pipelines found!")
        return
    
    # Filter by error type if specified
    if error_type != 'all':
        filtered_failures = []
        error_keywords = {
            'gifski': 'gifski',
            'ffmpeg': 'ffmpeg', 
            'imagemagick': 'imagemagick',
            'gifsicle': 'gifsicle',
            'animately': 'animately',
            'command': 'command failed',
            'timeout': 'timeout',
            'other': None  # Will be handled separately
        }
        
        keyword = error_keywords.get(error_type)
        for failure in failed_pipelines:
            error_msg = failure.get('error_message', '').lower()
            if error_type == 'other':
                # Show failures that don't match any specific tool
                if not any(tool in error_msg for tool in ['gifski', 'ffmpeg', 'imagemagick', 'gifsicle', 'animately', 'command failed', 'timeout']):
                    filtered_failures.append(failure)
            elif keyword and keyword in error_msg:
                filtered_failures.append(failure)
        
        failed_pipelines = filtered_failures
    
    # Apply limit
    if limit > 0:
        failed_pipelines = failed_pipelines[:limit]
    
    # Show summary statistics
    all_failures_file = results_dir / "failed_pipelines.json"
    with open(all_failures_file, 'r') as f:
        all_failures = json.load(f)
    
    click.echo(f"📊 Failure Analysis for {results_dir}")
    click.echo(f"   Total failures: {len(all_failures)}")
    if error_type != 'all':
        click.echo(f"   Showing {error_type} failures: {len(failed_pipelines)}")
    
    # Error type breakdown
    error_types = Counter()
    for failure in all_failures:
        error_msg = failure.get('error_message', '').lower()
        if 'gifski' in error_msg:
            error_types['gifski'] += 1
        elif 'ffmpeg' in error_msg:
            error_types['ffmpeg'] += 1
        elif 'imagemagick' in error_msg:
            error_types['imagemagick'] += 1
        elif 'gifsicle' in error_msg:
            error_types['gifsicle'] += 1
        elif 'animately' in error_msg:
            error_types['animately'] += 1
        elif 'command failed' in error_msg:
            error_types['command'] += 1
        elif 'timeout' in error_msg:
            error_types['timeout'] += 1
        else:
            error_types['other'] += 1
    
    click.echo(f"   Error type breakdown:")
    for error_type_name, count in error_types.most_common():
        click.echo(f"     {error_type_name}: {count}")
    
    if not failed_pipelines:
        click.echo(f"\n❌ No failures found matching filter: {error_type}")
        return
    
    # Show individual failures
    click.echo(f"\n🔍 Failed Pipeline Details:")
    for i, failure in enumerate(failed_pipelines, 1):
        pipeline_id = failure.get('pipeline_id', 'unknown')
        gif_name = failure.get('gif_name', 'unknown')
        error_msg = failure.get('error_message', 'No error message')
        tools = failure.get('tools_used', [])
        
        click.echo(f"\n{i:2d}. {pipeline_id}")
        click.echo(f"    GIF: {gif_name} ({failure.get('content_type', 'unknown')})")
        click.echo(f"    Tools: {', '.join(tools) if tools else 'unknown'}")
        click.echo(f"    Error: {error_msg}")
        
        if detailed:
            traceback_info = failure.get('error_traceback', '')
            if traceback_info:
                click.echo(f"    Traceback: {traceback_info}")
            
            test_params = failure.get('test_parameters', {})
            if test_params:
                click.echo(f"    Parameters: colors={test_params.get('colors')}, lossy={test_params.get('lossy')}, frame_ratio={test_params.get('frame_ratio')}")
            
            timestamp = failure.get('error_timestamp', failure.get('timestamp'))
            if timestamp:
                click.echo(f"    Time: {timestamp}")
    
    click.echo(f"\n💡 To see more details, use --detailed flag")
    click.echo(f"💡 To filter by error type, use --error-type <type>")


if __name__ == "__main__":
    main()
