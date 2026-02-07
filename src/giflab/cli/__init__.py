"""CLI module for GifLab.

Main entry point for GifLab prediction training data generation.

Usage:
    giflab [INPUT_DIR]           Generate prediction training data (default)
    giflab train                 Train prediction models
    giflab predict GIF_PATH      Predict compression curves for a GIF
    giflab export                Export trained models
"""

from pathlib import Path

import click

from giflab.prediction_runner import run_prediction_pipeline


# Default database path
DEFAULT_DB_PATH = Path("data/giflab.db")


@click.group()
@click.version_option(version="1.0.0", prog_name="giflab")
def main() -> None:
    """🎞️ GifLab — GIF compression prediction training data generator."""
    pass


@main.command(name="run")
@click.argument("input_dir", type=click.Path(exists=True))
@click.option(
    "--db",
    "-d",
    type=click.Path(),
    default=str(DEFAULT_DB_PATH),
    help="SQLite database path",
)
@click.option(
    "--mode",
    "-m",
    type=click.Choice(["single", "full"]),
    default="single",
    help="Pipeline mode: single (quick) or full (all combinations)",
)
@click.option("--force", "-f", is_flag=True, help="Re-process all GIFs")
@click.option("--upgrade", "-u", is_flag=True, help="Re-process outdated GIFs")
def run_cmd(
    input_dir: str,
    db: str,
    mode: str,
    force: bool,
    upgrade: bool,
) -> None:
    """Generate prediction training data from a directory of GIFs.

    Example:

        giflab run data/raw/

        giflab run data/raw/ --mode full --force
    """
    click.echo("🎞️ GifLab Prediction Pipeline")
    click.echo(f"   Input: {input_dir}")
    click.echo(f"   Database: {db}")
    click.echo(f"   Mode: {mode}")
    click.echo()

    result = run_prediction_pipeline(
        input_dir=Path(input_dir),
        output_db=Path(db),
        mode=mode,
        force=force,
        upgrade=upgrade,
    )

    if "error" in result:
        click.echo(f"❌ Error: {result['error']}")
        raise SystemExit(1)

    click.echo(f"✅ Processed {result.get('processed', 0)} GIFs")
    click.echo(f"   Total GIFs in DB: {result.get('total_gifs', 0)}")
    click.echo(f"   Complete: {result.get('complete_gifs', 0)}")
    click.echo(f"   Compression runs: {result.get('total_compression_runs', 0)}")
    click.echo(f"   Elapsed: {result.get('elapsed_seconds', 0):.1f}s")


@main.command()
@click.option("--db", "-d", type=click.Path(), default=str(DEFAULT_DB_PATH))
@click.option("--lossy-tool", type=str, default=None, help="Filter by lossy tool")
def train(db: str, lossy_tool: str | None) -> None:
    """Train prediction models from the database."""
    click.echo("🚀 Training prediction models...")
    click.echo(f"   Database: {db}")

    # Import training function from prediction module
    try:
        from giflab.prediction.cli import train_models

        train_models(Path(db), lossy_tool=lossy_tool)
    except ImportError:
        click.echo("❌ Training module not available")
        raise SystemExit(1)


@main.command()
@click.argument("gif_path", type=click.Path(exists=True))
@click.option("--db", "-d", type=click.Path(), default=str(DEFAULT_DB_PATH))
def predict(gif_path: str, db: str) -> None:
    """Predict compression curves for a single GIF."""
    click.echo(f"🔮 Predicting curves for: {gif_path}")

    try:
        from giflab.prediction.features import extract_gif_features

        features = extract_gif_features(Path(gif_path))
        click.echo(f"   Features extracted: {features.gif_name}")
        click.echo(f"   Size: {features.width}x{features.height}, {features.frame_count} frames")
        click.echo(f"   Entropy: {features.entropy:.3f}, Motion: {features.motion_intensity:.3f}")
    except ImportError:
        click.echo("❌ Prediction module not available")
        raise SystemExit(1)


@main.command()
@click.option("--db", "-d", type=click.Path(), default=str(DEFAULT_DB_PATH))
@click.option("--output", "-o", type=click.Path(), default="models/")
def export(db: str, output: str) -> None:
    """Export trained models as .pkl files for Animately."""
    click.echo(f"📦 Exporting models...")
    click.echo(f"   Database: {db}")
    click.echo(f"   Output: {output}")

    try:
        from giflab.prediction.cli import export_models

        export_models(Path(db), Path(output))
    except ImportError:
        click.echo("❌ Export module not available")
        raise SystemExit(1)


@main.command()
@click.option("--db", "-d", type=click.Path(), default=str(DEFAULT_DB_PATH))
def stats(db: str) -> None:
    """Show database statistics."""
    from giflab.storage import GifLabStorage

    storage = GifLabStorage(Path(db))
    stats = storage.get_statistics()

    click.echo(f"📊 Database Statistics: {db}")
    click.echo(f"   Total GIFs: {stats['total_gifs']}")
    click.echo(f"   Complete: {stats['complete_gifs']}")
    click.echo(f"   Incomplete: {stats['incomplete_gifs']}")
    click.echo(f"   Compression runs: {stats['total_compression_runs']}")
    click.echo(f"   Failures: {stats['total_failures']}")
    click.echo(f"   Pipelines: {stats['total_pipelines']}")
    click.echo(f"   Tools: {stats['total_tools']}")


__all__ = ["main"]
