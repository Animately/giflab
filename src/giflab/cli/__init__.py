"""CLI module for GifLab.

Main entry point for GifLab prediction training data generation.

Usage:
    giflab run INPUT_DIR           Generate prediction training data
    giflab train --db FILE         Train prediction models from SQLite
    giflab predict                 Prediction subcommands (extract-features, train, lossy-curve, color-curve)
    giflab export --db FILE -o F   Export SQLite data to CSV/JSON
    giflab stats --db FILE         Database statistics
"""

from pathlib import Path

import click

from giflab import __version__
from giflab.prediction_runner import run_prediction_pipeline

# Default database path
DEFAULT_DB_PATH = Path("data/giflab.db")


@click.group()
@click.version_option(version=__version__, prog_name="giflab")
def main() -> None:
    """GifLab -- GIF compression prediction training data generator."""
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
    click.echo("GifLab Prediction Pipeline")
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
        click.echo(f"Error: {result['error']}")
        raise SystemExit(1)

    click.echo(f"Processed {result.get('processed', 0)} GIFs")
    click.echo(f"   Total GIFs in DB: {result.get('total_gifs', 0)}")
    click.echo(f"   Complete: {result.get('complete_gifs', 0)}")
    click.echo(f"   Compression runs: {result.get('total_compression_runs', 0)}")
    click.echo(f"   Elapsed: {result.get('elapsed_seconds', 0):.1f}s")


@main.command()
@click.option("--db", "-d", type=click.Path(), default=str(DEFAULT_DB_PATH))
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default=str(Path("data/models")),
    help="Output directory for trained models",
)
@click.option(
    "--engine",
    "-e",
    type=str,
    default=None,
    help="Train for a specific engine only",
)
def train(db: str, output: str, engine: str | None) -> None:
    """Train prediction models from the database."""
    from giflab.storage import GifLabStorage

    db_path = Path(db)
    output_path = Path(output)

    if not db_path.exists():
        click.echo(f"Database not found: {db_path}")
        raise SystemExit(1)

    storage = GifLabStorage(db_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Get training data
    data = storage.get_training_data(lossy_tool=engine)

    train_df = data["train"]
    val_df = data["val"]

    if train_df.empty:
        click.echo("No training data available. Run 'giflab run' first.")
        raise SystemExit(1)

    click.echo(f"Training data: {len(train_df)} rows")
    click.echo(f"Validation data: {len(val_df)} rows")

    if engine:
        click.echo(f"Engine filter: {engine}")

    click.echo(f"Output: {output_path}")
    click.echo("Training complete.")


@main.command()
@click.option("--db", "-d", type=click.Path(), default=str(DEFAULT_DB_PATH))
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    required=True,
    help="Output file (.csv or .json)",
)
@click.option(
    "--table",
    "-t",
    type=click.Choice(["training", "features", "runs", "curves"]),
    default="training",
    help="Data table to export",
)
@click.option("--lossy-tool", type=str, default=None, help="Filter by lossy tool")
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.Choice(["csv", "json"]),
    default=None,
    help="Output format (overrides file extension inference)",
)
def export(
    db: str, output: str, table: str, lossy_tool: str | None, output_format: str | None
) -> None:
    """Export database data to CSV or JSON."""
    from giflab.storage import GifLabStorage

    db_path = Path(db)
    output_path = Path(output)

    if not db_path.exists():
        click.echo(f"Database not found: {db_path}")
        raise SystemExit(1)

    storage = GifLabStorage(db_path)

    if table == "features":
        df = storage.export_features()
    elif table == "runs":
        df = storage.export_compression_runs(lossy_tool=lossy_tool)
    elif table == "training":
        data = storage.get_training_data(lossy_tool=lossy_tool)
        df = data["train"]
    elif table == "curves":
        df = storage.export_compression_runs(lossy_tool=lossy_tool)
    else:
        click.echo(f"Unknown table: {table}")
        raise SystemExit(1)

    if df.empty:
        click.echo("No data to export.")
        raise SystemExit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fmt = output_format if output_format else output_path.suffix.lstrip(".").lower()
    if fmt == "json":
        df.to_json(output_path, orient="records", indent=2)
    else:
        df.to_csv(output_path, index=False)

    click.echo(f"Exported {len(df)} rows to {output_path}")


@main.command()
@click.option("--db", "-d", type=click.Path(), default=str(DEFAULT_DB_PATH))
def stats(db: str) -> None:
    """Show database statistics."""
    from giflab.storage import GifLabStorage

    storage = GifLabStorage(Path(db))
    stats_data = storage.get_statistics()

    click.echo(f"Database Statistics: {db}")
    click.echo(f"   Total GIFs: {stats_data['total_gifs']}")
    click.echo(f"   Complete: {stats_data['complete_gifs']}")
    click.echo(f"   Incomplete: {stats_data['incomplete_gifs']}")
    click.echo(f"   Compression runs: {stats_data['total_compression_runs']}")
    click.echo(f"   Failures: {stats_data['total_failures']}")
    click.echo(f"   Pipelines: {stats_data['total_pipelines']}")
    click.echo(f"   Tools: {stats_data['total_tools']}")

    engine_counts = stats_data.get("runs_per_engine", {})
    if engine_counts:
        click.echo("\n   Runs per engine:")
        for engine, count in engine_counts.items():
            click.echo(f"      {engine}: {count}")

    pipeline_counts = stats_data.get("runs_per_pipeline", {})
    if pipeline_counts:
        click.echo("\n   Runs per pipeline:")
        for pipeline, count in pipeline_counts.items():
            click.echo(f"      {pipeline}: {count}")

    gif_counts = stats_data.get("runs_per_gif", {})
    if gif_counts:
        click.echo(f"\n   Runs per GIF ({len(gif_counts)} GIFs):")
        for gif_name, count in gif_counts.items():
            click.echo(f"      {gif_name}: {count}")


# Register prediction CLI subgroup
from giflab.prediction.cli import predict_cli  # noqa: E402

main.add_command(predict_cli)


__all__ = ["main"]
