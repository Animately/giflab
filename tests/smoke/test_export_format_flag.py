"""Smoke tests for the --format flag on the export CLI command."""

from __future__ import annotations

from click.testing import CliRunner

from giflab.cli import main


def test_format_csv_is_accepted() -> None:
    """The --format csv option should be accepted without a Click UsageError."""
    runner = CliRunner()
    result = runner.invoke(main, ["export", "--output", "out.csv", "--format", "csv"])
    # We expect the command to proceed past option parsing.
    # It will fail because there is no database, but that is fine --
    # a Click UsageError (exit code 2) would mean the flag was rejected.
    assert result.exit_code != 2, f"--format csv was rejected: {result.output}"


def test_format_json_is_accepted() -> None:
    """The --format json option should be accepted without a Click UsageError."""
    runner = CliRunner()
    result = runner.invoke(main, ["export", "--output", "out.json", "--format", "json"])
    assert result.exit_code != 2, f"--format json was rejected: {result.output}"


def test_format_short_flag_is_accepted() -> None:
    """The short -f flag should work the same as --format."""
    runner = CliRunner()
    result = runner.invoke(main, ["export", "--output", "out.csv", "-f", "csv"])
    assert result.exit_code != 2, f"-f csv was rejected: {result.output}"


def test_invalid_format_is_rejected() -> None:
    """An invalid format like 'xml' should be rejected by Click (exit code 2)."""
    runner = CliRunner()
    result = runner.invoke(main, ["export", "--output", "out.xml", "--format", "xml"])
    assert result.exit_code == 2, (
        f"Expected exit code 2 for invalid format, got {result.exit_code}"
    )
    assert "Invalid value" in result.output or "invalid choice" in result.output.lower()


def test_omitting_format_is_backward_compatible() -> None:
    """Omitting --format should still work (infers from file extension)."""
    runner = CliRunner()
    result = runner.invoke(main, ["export", "--output", "out.csv"])
    # Should proceed past option parsing (no exit code 2).
    assert result.exit_code != 2, f"Missing --format caused UsageError: {result.output}"
