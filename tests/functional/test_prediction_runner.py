"""Tests for prediction runner GIF output validation."""

import tempfile
from pathlib import Path

import pytest
from giflab.prediction_runner import PredictionRunner


class TestValidateGifOutput:
    """Tests for PredictionRunner._validate_gif_output."""

    @pytest.fixture()
    def runner(self):
        """Create a PredictionRunner with a temporary database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            r = PredictionRunner(db_path=db_path)
            yield r

    def test_valid_gif89a(self, runner: PredictionRunner, tmp_path: Path) -> None:
        """Valid GIF89a file passes validation."""
        gif = tmp_path / "valid.gif"
        # Minimal GIF89a: 6-byte header + 7-byte logical screen descriptor
        gif.write_bytes(b"GIF89a" + b"\x01\x00\x01\x00\x00\x00\x00")
        runner._validate_gif_output(gif)  # should not raise

    def test_valid_gif87a(self, runner: PredictionRunner, tmp_path: Path) -> None:
        """Valid GIF87a file passes validation."""
        gif = tmp_path / "valid87.gif"
        gif.write_bytes(b"GIF87a" + b"\x01\x00\x01\x00\x00\x00\x00")
        runner._validate_gif_output(gif)  # should not raise

    def test_truncated_file(self, runner: PredictionRunner, tmp_path: Path) -> None:
        """Truncated file (< 13 bytes) is rejected."""
        gif = tmp_path / "truncated.gif"
        gif.write_bytes(b"GIF89a\x01\x00")  # only 8 bytes
        with pytest.raises(RuntimeError, match="too small"):
            runner._validate_gif_output(gif)

    def test_empty_file(self, runner: PredictionRunner, tmp_path: Path) -> None:
        """Empty file is rejected."""
        gif = tmp_path / "empty.gif"
        gif.write_bytes(b"")
        with pytest.raises(RuntimeError, match="too small"):
            runner._validate_gif_output(gif)

    def test_wrong_magic_bytes(self, runner: PredictionRunner, tmp_path: Path) -> None:
        """File with wrong magic bytes (e.g. PNG) is rejected."""
        gif = tmp_path / "not_a_gif.gif"
        # PNG magic bytes + padding to reach 13 bytes
        gif.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 5)
        with pytest.raises(RuntimeError, match="not a valid GIF"):
            runner._validate_gif_output(gif)

    def test_random_bytes(self, runner: PredictionRunner, tmp_path: Path) -> None:
        """Random bytes are rejected."""
        gif = tmp_path / "random.gif"
        gif.write_bytes(b"\x00" * 20)
        with pytest.raises(RuntimeError, match="not a valid GIF"):
            runner._validate_gif_output(gif)
