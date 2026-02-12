"""Training dataset utilities for compression curve prediction.

The DatasetBuilder CSV workflow has been replaced by PredictionRunner + SQLite.
Use ``giflab run`` to generate training data and ``giflab export`` to export.

This module retains ``load_training_records()`` for reading legacy JSONL datasets
used by ``giflab predict train``.
"""

import logging
from pathlib import Path

from giflab.prediction.schemas import (
    LOSSY_LEVELS,
    TrainingRecordV1,
)

logger = logging.getLogger(__name__)

# Standard color counts to test
COLOR_COUNTS = [256, 128, 64, 32, 16]

# Dataset version
DATASET_VERSION = "1.0.0"


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
