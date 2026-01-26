"""Training dataset builder for compression curve prediction.

This module builds training datasets by:
1. Running compression sweeps on GIFs (all lossy levels, all color counts)
2. Extracting visual features for each GIF
3. Pairing features with compression outcomes
4. Splitting into train/val/test sets

Constitution Compliance:
- Principle I (Single-Pass): Uses existing compression engines
- Principle II (ML-Ready Data): Schema-validated, versioned outputs
- Principle III (Poetry-First): CLI via poetry run
"""

import csv
import logging
import random
import tempfile
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from giflab.lossy import apply_lossy_compression, LossyEngine
from giflab.prediction.features import extract_gif_features
from giflab.prediction.schemas import (
    CompressionCurveV1,
    CurveType,
    DatasetSplit,
    Engine,
    GifFeaturesV1,
    TrainingRecordV1,
)

logger = logging.getLogger(__name__)

# Standard lossy levels to test
LOSSY_LEVELS = [0, 20, 40, 60, 80, 100, 120]

# Standard color counts to test
COLOR_COUNTS = [256, 128, 64, 32, 16]

# Dataset version
DATASET_VERSION = "1.0.0"


class DatasetBuilder:
    """Builds training datasets for compression curve prediction."""

    def __init__(
        self,
        output_dir: Path,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
    ) -> None:
        """Initialize the dataset builder.

        Args:
            output_dir: Directory to store training data.
            train_ratio: Fraction of data for training (default 0.8).
            val_ratio: Fraction of data for validation (default 0.1).
            test_ratio: Fraction of data for testing (default 0.1).
            seed: Random seed for reproducible splits.
        """
        self.output_dir = output_dir
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed

        # Validate ratios
        total = train_ratio + val_ratio + test_ratio
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Ratios must sum to 1.0, got {total}")

        # Create output directories
        self.features_dir = output_dir / "features"
        self.outcomes_dir = output_dir / "outcomes"
        self.records_dir = output_dir / "records"

        for d in [self.features_dir, self.outcomes_dir, self.records_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self.random = random.Random(seed)

    def build_dataset(
        self,
        gif_paths: list[Path],
        engines: list[Engine] | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> dict[str, int]:
        """Build training dataset from a list of GIF files.

        Args:
            gif_paths: List of paths to GIF files.
            engines: Engines to test (default: gifsicle only).
            progress_callback: Optional callback(current, total) for progress.

        Returns:
            Dict with counts: {"total", "success", "failed", "train", "val", "test"}
        """
        if engines is None:
            engines = [Engine.GIFSICLE]

        # Shuffle GIFs for random split assignment
        shuffled_paths = list(gif_paths)
        self.random.shuffle(shuffled_paths)

        results = {
            "total": len(shuffled_paths),
            "success": 0,
            "failed": 0,
            "train": 0,
            "val": 0,
            "test": 0,
        }

        records = []

        for i, gif_path in enumerate(shuffled_paths):
            if progress_callback:
                progress_callback(i + 1, len(shuffled_paths))

            try:
                # Extract features
                features = extract_gif_features(gif_path)

                # Run compression sweeps
                curves = self._run_compression_sweeps(gif_path, features, engines)

                # Determine split
                split = self._assign_split(i, len(shuffled_paths))

                # Create training record
                record = self._create_training_record(features, curves, split)
                records.append(record)

                results["success"] += 1
                results[split.value] += 1

            except Exception as e:
                logger.warning(f"Failed to process {gif_path}: {e}")
                results["failed"] += 1

        # Save records
        self._save_records(records)

        return results

    def _run_compression_sweeps(
        self,
        gif_path: Path,
        features: GifFeaturesV1,
        engines: list[Engine],
    ) -> dict[str, CompressionCurveV1]:
        """Run compression sweeps for all engines and curve types.

        Returns dict with keys like "lossy_gifsicle", "color_gifsicle", etc.
        """
        curves = {}

        for engine in engines:
            # Lossy curve
            lossy_curve = self._run_lossy_sweep(gif_path, features.gif_sha, engine)
            curves[f"lossy_{engine.value}"] = lossy_curve

            # Color curve
            color_curve = self._run_color_sweep(gif_path, features.gif_sha, engine)
            curves[f"color_{engine.value}"] = color_curve

        return curves

    def _run_lossy_sweep(
        self,
        gif_path: Path,
        gif_sha: str,
        engine: Engine,
    ) -> CompressionCurveV1:
        """Run lossy compression sweep and record file sizes."""
        sizes = {}

        with tempfile.TemporaryDirectory() as tmpdir:
            for lossy_level in LOSSY_LEVELS:
                try:
                    output_path = Path(tmpdir) / f"lossy_{lossy_level}.gif"

                    # Map engine enum to LossyEngine
                    lossy_engine = (
                        LossyEngine.GIFSICLE
                        if engine == Engine.GIFSICLE
                        else LossyEngine.ANIMATELY
                    )

                    apply_lossy_compression(
                        gif_path,
                        output_path,
                        lossy_level=lossy_level,
                        engine=lossy_engine,
                    )

                    if output_path.exists():
                        size_kb = output_path.stat().st_size / 1024
                        sizes[lossy_level] = size_kb

                except Exception as e:
                    logger.debug(f"Lossy {lossy_level} failed: {e}")

        return CompressionCurveV1(
            gif_sha=gif_sha,
            engine=engine,
            curve_type=CurveType.LOSSY,
            is_predicted=False,
            created_at=datetime.now(timezone.utc),
            size_at_lossy_0=sizes.get(0),
            size_at_lossy_20=sizes.get(20),
            size_at_lossy_40=sizes.get(40),
            size_at_lossy_60=sizes.get(60),
            size_at_lossy_80=sizes.get(80),
            size_at_lossy_100=sizes.get(100),
            size_at_lossy_120=sizes.get(120),
        )

    def _run_color_sweep(
        self,
        gif_path: Path,
        gif_sha: str,
        engine: Engine,
    ) -> CompressionCurveV1:
        """Run color reduction sweep and record file sizes."""
        import subprocess

        sizes = {}

        with tempfile.TemporaryDirectory() as tmpdir:
            for color_count in COLOR_COUNTS:
                try:
                    output_path = Path(tmpdir) / f"colors_{color_count}.gif"

                    if engine == Engine.GIFSICLE:
                        # Use gifsicle with --colors flag
                        cmd = [
                            "gifsicle",
                            f"--colors={color_count}",
                            "-O3",
                            str(gif_path),
                            "-o",
                            str(output_path),
                        ]
                        subprocess.run(
                            cmd,
                            capture_output=True,
                            timeout=60,
                            check=True,
                        )
                    else:
                        # Skip animately for now - color reduction not supported
                        continue

                    if output_path.exists():
                        size_kb = output_path.stat().st_size / 1024
                        sizes[color_count] = size_kb

                except Exception as e:
                    logger.debug("Colors %d failed: %s", color_count, e)

        return CompressionCurveV1(
            gif_sha=gif_sha,
            engine=engine,
            curve_type=CurveType.COLORS,
            is_predicted=False,
            created_at=datetime.now(timezone.utc),
            size_at_colors_256=sizes.get(256),
            size_at_colors_128=sizes.get(128),
            size_at_colors_64=sizes.get(64),
            size_at_colors_32=sizes.get(32),
            size_at_colors_16=sizes.get(16),
        )

    def _assign_split(self, index: int, total: int) -> DatasetSplit:
        """Assign a GIF to train/val/test split based on position."""
        train_end = int(total * self.train_ratio)
        val_end = train_end + int(total * self.val_ratio)

        if index < train_end:
            return DatasetSplit.TRAIN
        elif index < val_end:
            return DatasetSplit.VAL
        else:
            return DatasetSplit.TEST

    def _create_training_record(
        self,
        features: GifFeaturesV1,
        curves: dict[str, CompressionCurveV1],
        split: DatasetSplit,
    ) -> TrainingRecordV1:
        """Create a training record from features and curves."""
        return TrainingRecordV1(
            record_id=str(uuid.uuid4()),
            gif_sha=features.gif_sha,
            dataset_version=DATASET_VERSION,
            split=split,
            features=features,
            lossy_curve_gifsicle=curves.get(
                "lossy_gifsicle",
                self._empty_lossy_curve(features.gif_sha, Engine.GIFSICLE),
            ),
            lossy_curve_animately=curves.get("lossy_animately"),
            color_curve_gifsicle=curves.get(
                "color_gifsicle",
                self._empty_color_curve(features.gif_sha, Engine.GIFSICLE),
            ),
            color_curve_animately=curves.get("color_animately"),
            created_at=datetime.now(timezone.utc),
        )

    def _empty_lossy_curve(
        self,
        gif_sha: str,
        engine: Engine,
    ) -> CompressionCurveV1:
        """Create an empty lossy curve as placeholder."""
        return CompressionCurveV1(
            gif_sha=gif_sha,
            engine=engine,
            curve_type=CurveType.LOSSY,
            is_predicted=False,
            created_at=datetime.now(timezone.utc),
        )

    def _empty_color_curve(
        self,
        gif_sha: str,
        engine: Engine,
    ) -> CompressionCurveV1:
        """Create an empty color curve as placeholder."""
        return CompressionCurveV1(
            gif_sha=gif_sha,
            engine=engine,
            curve_type=CurveType.COLORS,
            is_predicted=False,
            created_at=datetime.now(timezone.utc),
        )

    def _save_records(self, records: list[TrainingRecordV1]) -> None:
        """Save training records to disk."""
        if not records:
            return

        # Save as JSON lines
        records_file = self.records_dir / f"records_{DATASET_VERSION}.jsonl"
        with open(records_file, "w") as f:
            for record in records:
                f.write(record.model_dump_json() + "\n")

        logger.info(f"Saved {len(records)} records to {records_file}")

        # Also save features as CSV for easier analysis
        features_file = self.features_dir / f"features_{DATASET_VERSION}.csv"
        self._save_features_csv(records, features_file)

    def _save_features_csv(
        self,
        records: list[TrainingRecordV1],
        output_path: Path,
    ) -> None:
        """Save features to CSV for analysis."""
        if not records:
            return

        rows = []
        for record in records:
            row = record.features.model_dump(mode="json")
            row["split"] = record.split.value
            row["record_id"] = record.record_id
            rows.append(row)

        fieldnames = list(rows[0].keys())
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        logger.info(f"Saved features CSV to {output_path}")


def load_training_records(records_file: Path) -> list[TrainingRecordV1]:
    """Load training records from a JSONL file.

    Args:
        records_file: Path to the records JSONL file.

    Returns:
        List of TrainingRecordV1 objects.
    """
    records = []
    with open(records_file) as f:
        for line in f:
            if line.strip():
                record = TrainingRecordV1.model_validate_json(line)
                records.append(record)
    return records
