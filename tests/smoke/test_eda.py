from pathlib import Path

import numpy as np
import pandas as pd
from giflab.eda import generate_eda


def _create_fake_csv(tmp_path: Path) -> Path:
    csv_path = tmp_path / "metrics.csv"
    df = pd.DataFrame(
        {
            "ssim": np.random.rand(20),
            "psnr": np.random.uniform(20, 40, 20),
            "mse": np.random.uniform(0, 500, 20),
            "fsim": np.random.rand(20),
            "kilobytes": np.random.uniform(10, 500, 20),
        }
    )
    df.to_csv(csv_path, index=False)
    return csv_path


def test_generate_eda(tmp_path):
    csv_path = _create_fake_csv(tmp_path)
    out_dir = tmp_path / "eda"

    artefacts = generate_eda(csv_path, out_dir)

    # Expect at least histograms + heatmap + scree plot
    assert len(artefacts) >= 3
    for name, path in artefacts.items():
        assert path.exists(), f"Missing artefact {name}"


def test_filename_sanitization(tmp_path):
    """Test that special characters in column names are handled safely."""
    csv_path = tmp_path / "special.csv"
    df = pd.DataFrame(
        {
            "metric/with/slashes": [1, 2, 3],
            "metric with spaces": [4, 5, 6],
            "metric-normal": [7, 8, 9],
        }
    )
    df.to_csv(csv_path, index=False)

    out_dir = tmp_path / "eda_special"
    artefacts = generate_eda(csv_path, out_dir)

    # Check that files were created with sanitized names
    expected_files = [
        "hist_metric_with_slashes",
        "hist_metric_with_spaces",
        "hist_metric-normal",
    ]
    for expected in expected_files:
        assert any(
            expected in name for name in artefacts.keys()
        ), f"Missing sanitized file for {expected}"

    # Verify all files actually exist
    for name, path in artefacts.items():
        assert path.exists(), f"Missing artefact {name} at {path}"
