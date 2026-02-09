"""Tests for CLI commands using click.testing.CliRunner."""

import tempfile
from pathlib import Path

import pytest
from click.testing import CliRunner

from giflab import __version__
from giflab.cli import main


class TestMainCLI:
    """Tests for main CLI group."""

    def test_main_help(self):
        """Test main CLI help command."""
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])

        assert result.exit_code == 0
        assert "GifLab" in result.output
        assert "Commands:" in result.output

    def test_main_version(self):
        """Test main CLI version command."""
        runner = CliRunner()
        result = runner.invoke(main, ["--version"])

        assert result.exit_code == 0
        assert f"giflab, version {__version__}" in result.output

    def test_main_invalid_command(self):
        """Test main CLI with invalid command."""
        runner = CliRunner()
        result = runner.invoke(main, ["invalid-command"])

        assert result.exit_code == 2
        assert "No such command" in result.output


class TestStatsCommand:
    """Tests for stats command."""

    def test_stats_creates_db(self):
        """Test stats command creates database if needed."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            result = runner.invoke(main, ["stats", "--db", str(db_path)])

            assert result.exit_code == 0
            assert "Database Statistics" in result.output
            assert "Total GIFs:" in result.output


class TestRunCommand:
    """Tests for run command."""

    def test_run_help(self):
        """Test run command help."""
        runner = CliRunner()
        result = runner.invoke(main, ["run", "--help"])

        assert result.exit_code == 0
        assert "Generate prediction training data" in result.output

    def test_run_missing_input_dir(self):
        """Test run command with missing input directory."""
        runner = CliRunner()
        result = runner.invoke(main, ["run"])

        assert result.exit_code == 2
        assert "Missing argument" in result.output


class TestTrainCommand:
    """Tests for train command."""

    def test_train_help(self):
        """Test train command help."""
        runner = CliRunner()
        result = runner.invoke(main, ["train", "--help"])

        assert result.exit_code == 0
        assert "Train prediction models" in result.output


class TestPredictCommand:
    """Tests for predict command."""

    def test_predict_help(self):
        """Test predict command help."""
        runner = CliRunner()
        result = runner.invoke(main, ["predict", "--help"])

        assert result.exit_code == 0
        assert "Predict compression curves" in result.output


class TestExportCommand:
    """Tests for export command."""

    def test_export_help(self):
        """Test export command help."""
        runner = CliRunner()
        result = runner.invoke(main, ["export", "--help"])

        assert result.exit_code == 0
        assert "Export trained models" in result.output
