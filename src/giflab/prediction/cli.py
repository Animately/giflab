"""CLI commands for compression curve prediction.

Constitution Compliance:
- Principle III (Poetry-First): Use `poetry run python -m giflab predict ...`
- Principle VI (LLM-Optimized): Explicit patterns, type hints, docstrings
"""

import json
import logging
from pathlib import Path

import click

from giflab.prediction.features import extract_gif_features

logger = logging.getLogger(__name__)


@click.group(name="predict")
def predict_cli() -> None:
    """Compression curve prediction commands."""
    pass


@predict_cli.command(name="extract-features")
@click.argument("gif_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    help="Output JSON file (default: stdout)",
)
@click.option("--pretty", is_flag=True, help="Pretty-print JSON output")
def extract_features_cmd(
    gif_path: Path,
    output: Path | None,
    pretty: bool,
) -> None:
    """Extract visual features from a GIF for compression prediction.

    GIF_PATH: Path to the GIF file to analyze.

    Example:
        poetry run python -m giflab predict extract-features image.gif
        poetry run python -m giflab predict extract-features image.gif -o features.json
    """
    try:
        features = extract_gif_features(gif_path)

        # Convert to dict for JSON serialization
        features_dict = features.model_dump(mode="json")

        # Output
        indent = 2 if pretty else None
        json_str = json.dumps(features_dict, indent=indent, default=str)

        if output:
            output.write_text(json_str)
            click.echo(f"Features written to: {output}")
        else:
            click.echo(json_str)

    except FileNotFoundError as e:
        raise click.ClickException(str(e)) from e
    except ValueError as e:
        raise click.ClickException(f"Feature extraction failed: {e}") from e
    except Exception as e:
        logger.exception("Unexpected error during feature extraction")
        raise click.ClickException(f"Unexpected error: {e}") from e


@predict_cli.command(name="train")
@click.option(
    "--dataset",
    "-d",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Training dataset directory",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default=Path("data/models"),
    help="Output directory for models",
)
@click.option(
    "--engine",
    "-e",
    type=click.Choice(["gifsicle", "animately"]),
    default="gifsicle",
    help="Engine to train models for",
)
def train_cmd(
    dataset: Path,
    output: Path,
    engine: str,
) -> None:
    """Train prediction models from a training dataset.

    Example:
        poetry run python -m giflab predict train -d data/training/ -o data/models/
    """
    from giflab.prediction.dataset import load_training_records
    from giflab.prediction.models import CurvePredictionModel
    from giflab.prediction.schemas import CurveType, DatasetSplit, Engine

    # Find records file
    records_dir = dataset / "records"
    records_files = list(records_dir.glob("records_*.jsonl"))

    if not records_files:
        raise click.ClickException(f"No records found in {records_dir}")

    records_file = records_files[0]
    click.echo(f"Loading records from {records_file}")

    records = load_training_records(records_file)
    click.echo(f"Loaded {len(records)} records")

    # Split records
    train_records = [r for r in records if r.split == DatasetSplit.TRAIN]
    val_records = [r for r in records if r.split == DatasetSplit.VAL]

    click.echo(f"Train: {len(train_records)}, Val: {len(val_records)}")

    engine_enum = Engine.GIFSICLE if engine == "gifsicle" else Engine.ANIMATELY
    output.mkdir(parents=True, exist_ok=True)

    # Train lossy model
    click.echo(f"Training {engine} lossy model...")
    # Pair features with curves, filtering out records with None curves
    lossy_paired = [
        (
            r.features,
            r.lossy_curve_gifsicle
            if engine == "gifsicle"
            else r.lossy_curve_animately,
        )
        for r in train_records
    ]
    lossy_paired = [(f, c) for f, c in lossy_paired if c is not None]

    if lossy_paired:
        lossy_features, lossy_curves = zip(*lossy_paired)
        lossy_model = CurvePredictionModel(engine_enum, CurveType.LOSSY)
        lossy_model.train(list(lossy_features), list(lossy_curves))

        # Validate - also pair features with curves correctly
        val_lossy_paired = [
            (
                r.features,
                r.lossy_curve_gifsicle
                if engine == "gifsicle"
                else r.lossy_curve_animately,
            )
            for r in val_records
        ]
        val_lossy_paired = [
            (f, c) for f, c in val_lossy_paired if c is not None
        ]
        if val_lossy_paired:
            val_features, val_curves = zip(*val_lossy_paired)
            mape = lossy_model.validate(list(val_features), list(val_curves))
            click.echo(f"  Validation MAPE: {mape:.2f}%")

        lossy_model.save(output / f"{engine}_lossy_v1.pkl")
        click.echo(f"  Saved to {output / f'{engine}_lossy_v1.pkl'}")

    # Train color model
    click.echo(f"Training {engine} color model...")
    # Pair features with curves, filtering out records with None curves
    color_paired = [
        (
            r.features,
            r.color_curve_gifsicle
            if engine == "gifsicle"
            else r.color_curve_animately,
        )
        for r in train_records
    ]
    color_paired = [(f, c) for f, c in color_paired if c is not None]

    if color_paired:
        color_features, color_curves = zip(*color_paired)
        color_model = CurvePredictionModel(engine_enum, CurveType.COLORS)
        color_model.train(list(color_features), list(color_curves))

        # Validate - also pair features with curves correctly
        val_color_paired = [
            (
                r.features,
                r.color_curve_gifsicle
                if engine == "gifsicle"
                else r.color_curve_animately,
            )
            for r in val_records
        ]
        val_color_paired = [
            (f, c) for f, c in val_color_paired if c is not None
        ]
        if val_color_paired:
            val_color_features, val_color_curves = zip(*val_color_paired)
            mape = color_model.validate(
                list(val_color_features), list(val_color_curves)
            )
            click.echo(f"  Validation MAPE: {mape:.2f}%")

        color_model.save(output / f"{engine}_color_v1.pkl")
        click.echo(f"  Saved to {output / f'{engine}_color_v1.pkl'}")

    click.echo("Training complete!")


@predict_cli.command(name="lossy-curve")
@click.argument("gif_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--engine",
    "-e",
    type=click.Choice(["gifsicle", "animately"]),
    default="gifsicle",
    help="Compression engine",
)
@click.option(
    "--model-dir",
    "-m",
    type=click.Path(path_type=Path),
    default=Path("data/models"),
    help="Directory containing trained models",
)
def lossy_curve_cmd(
    gif_path: Path,
    engine: str,
    model_dir: Path,
) -> None:
    """Predict lossy compression curve for a GIF.

    Example:
        poetry run python -m giflab predict lossy-curve image.gif
    """
    from giflab.prediction.features import extract_gif_features
    from giflab.prediction.models import predict_lossy_curve
    from giflab.prediction.schemas import Engine

    features = extract_gif_features(gif_path)
    engine_enum = Engine.GIFSICLE if engine == "gifsicle" else Engine.ANIMATELY

    try:
        curve = predict_lossy_curve(features, engine_enum, model_dir)
    except FileNotFoundError as e:
        raise click.ClickException(str(e)) from e

    click.echo(f"Predicted lossy curve for {gif_path.name}:")
    points = curve.get_lossy_curve_points()
    for level, size in points.items():
        if size:
            click.echo(f"  lossy={level}: {size:.1f} KB")


@predict_cli.command(name="color-curve")
@click.argument("gif_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--engine",
    "-e",
    type=click.Choice(["gifsicle", "animately"]),
    default="gifsicle",
    help="Compression engine",
)
@click.option(
    "--model-dir",
    "-m",
    type=click.Path(path_type=Path),
    default=Path("data/models"),
    help="Directory containing trained models",
)
def color_curve_cmd(
    gif_path: Path,
    engine: str,
    model_dir: Path,
) -> None:
    """Predict color reduction curve for a GIF.

    Example:
        poetry run python -m giflab predict color-curve image.gif
    """
    from giflab.prediction.features import extract_gif_features
    from giflab.prediction.models import predict_color_curve
    from giflab.prediction.schemas import Engine

    features = extract_gif_features(gif_path)
    engine_enum = Engine.GIFSICLE if engine == "gifsicle" else Engine.ANIMATELY

    try:
        curve = predict_color_curve(features, engine_enum, model_dir)
    except FileNotFoundError as e:
        raise click.ClickException(str(e)) from e

    click.echo(f"Predicted color curve for {gif_path.name}:")
    points = curve.get_color_curve_points()
    for count, size in sorted(points.items(), reverse=True):
        if size:
            click.echo(f"  colors={count}: {size:.1f} KB")


@predict_cli.command(name="build-dataset")
@click.argument("input_dir", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    required=True,
    help="Output directory for training data",
)
@click.option(
    "--recursive",
    "-r",
    is_flag=True,
    help="Search for GIFs recursively",
)
@click.option(
    "--engine",
    "-e",
    type=click.Choice(["gifsicle", "animately"]),
    default="gifsicle",
    help="Compression engine to use",
)
def build_dataset_cmd(
    input_dir: Path,
    output: Path,
    recursive: bool,
    engine: str,
) -> None:
    """Build training dataset from GIF files.

    Runs compression sweeps on all GIFs and pairs with extracted features.

    INPUT_DIR: Directory containing GIF files.

    Example:
        poetry run python -m giflab predict build-dataset data/raw/ -o data/training/
    """
    from giflab.prediction.dataset import DatasetBuilder
    from giflab.prediction.schemas import Engine

    # Find GIF files
    pattern = "**/*.gif" if recursive else "*.gif"
    gif_files = list(input_dir.glob(pattern))

    if not gif_files:
        raise click.ClickException(f"No GIF files found in {input_dir}")

    click.echo(f"Found {len(gif_files)} GIF files")

    # Build dataset
    engine_enum = Engine.GIFSICLE if engine == "gifsicle" else Engine.ANIMATELY
    builder = DatasetBuilder(output)

    def progress(current: int, total: int) -> None:
        click.echo(f"\rProcessing {current}/{total}...", nl=False)

    results = builder.build_dataset(
        gif_files,
        engines=[engine_enum],
        progress_callback=progress,
    )

    click.echo()  # Newline after progress
    click.echo(f"Dataset built successfully:")
    click.echo(f"  Total: {results['total']}")
    click.echo(f"  Success: {results['success']}")
    click.echo(f"  Failed: {results['failed']}")
    click.echo(f"  Train: {results['train']}")
    click.echo(f"  Val: {results['val']}")
    click.echo(f"  Test: {results['test']}")
    click.echo(f"  Output: {output}")


@predict_cli.command(name="batch-extract")
@click.argument("input_dir", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    required=True,
    help="Output CSV file for features",
)
@click.option(
    "--recursive",
    "-r",
    is_flag=True,
    help="Search for GIFs recursively",
)
def batch_extract_cmd(
    input_dir: Path,
    output: Path,
    recursive: bool,
) -> None:
    """Extract features from all GIFs in a directory.

    INPUT_DIR: Directory containing GIF files.

    Example:
        poetry run python -m giflab predict batch-extract data/raw/ -o features.csv
    """
    import csv

    # Find GIF files
    pattern = "**/*.gif" if recursive else "*.gif"
    gif_files = list(input_dir.glob(pattern))

    if not gif_files:
        raise click.ClickException(f"No GIF files found in {input_dir}")

    click.echo(f"Found {len(gif_files)} GIF files")

    # Extract features
    results = []
    errors = []

    with click.progressbar(gif_files, label="Extracting features") as bar:
        for gif_path in bar:
            try:
                features = extract_gif_features(gif_path)
                results.append(features.model_dump(mode="json"))
            except Exception as e:
                errors.append((gif_path, str(e)))
                logger.warning(f"Failed to extract features from {gif_path}: {e}")

    # Write CSV
    if results:
        fieldnames = list(results[0].keys())
        with open(output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        click.echo(f"Wrote {len(results)} feature records to {output}")

    if errors:
        click.echo(f"Failed to process {len(errors)} files", err=True)
        for path, error in errors[:5]:
            click.echo(f"  - {path}: {error}", err=True)
        if len(errors) > 5:
            click.echo(f"  ... and {len(errors) - 5} more", err=True)
