"""CLI module for GifLab commands.

This module re-exports all command functions to maintain compatibility
with the main CLI entry point while keeping commands organized in separate modules.
"""

import click

from .cache_cmd import cache
from .deps_cmd import deps
from .metrics_cmd import metrics
from .organize_cmd import organize_directories
from .performance_cmd import performance
from .run_cmd import run
from .select_pipelines_cmd import select_pipelines
from .tag_cmd import tag
from .validate_cmd import validate
from .view_failures_cmd import view_failures

from giflab.prediction.cli import predict_cli


@click.group()
@click.version_option(version="0.1.0", prog_name="giflab")
def main() -> None:
    """🎞️ GifLab — GIF compression and analysis laboratory."""
    pass


# Register all commands from the modular CLI structure
main.add_command(cache)
main.add_command(deps)
main.add_command(metrics)
main.add_command(performance)
main.add_command(run)
main.add_command(tag)
main.add_command(organize_directories)
main.add_command(select_pipelines)
main.add_command(validate)
main.add_command(view_failures)
main.add_command(predict_cli)

__all__ = [
    "cache",
    "deps",
    "main",
    "metrics",
    "organize_directories",
    "performance",
    "run",
    "select_pipelines",
    "tag",
    "validate",
    "view_failures",
]