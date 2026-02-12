# CLI Development Guide

This guide covers conventions and patterns for developing CLI commands in the GifLab project.

## Command Module Conventions

### File and Function Naming Pattern

**🚨 IMPORTANT**: Follow this exact pattern to avoid import errors:

```
File name:     {command}_cmd.py
Function name: {command} 
CLI command:   {command}
```

**Examples**:

| File | Function | CLI Command | Import Statement |
|------|----------|-------------|------------------|
| `run_cmd.py` | `run` | `giflab run` | `from .run_cmd import run` |
| `validate_cmd.py` | `validate` | `giflab validate` | `from .validate_cmd import validate` |
| `tag_cmd.py` | `tag` | `giflab tag` | `from .tag_cmd import tag` |
| `organize_cmd.py` | `organize_directories` | `giflab organize-directories` | `from .organize_cmd import organize_directories` |

**Rationale**: 
- `_cmd.py` suffix distinguishes command modules from utility files
- Function name matches the actual CLI command users type
- Keeps imports clean and intuitive

### Command Module Structure

**Template for new commands**:

```python
# src/giflab/cli/{command}_cmd.py

import click
from pathlib import Path

@click.command()
@click.option('--option1', help="Description")
@click.argument('argument1', required=False)
def {command}(option1: str | None, argument1: str | None) -> None:
    """Brief command description.
    
    Detailed usage information here.
    """
    # Implementation here
    pass
```

### Registration Process

**Step 1**: Create the command file following the naming convention
**Step 2**: Add to CLI module exports:

```python
# src/giflab/cli/__init__.py

# Import the command
from .{command}_cmd import {command}

# Add to main CLI group  
main.add_command({command})

# Add to __all__ list
__all__ = [
    # ... other commands
    "{command}",
]
```

## Command Implementation Guidelines

### Error Handling
```python
@click.command()
def my_command():
    try:
        # Command logic
        pass
    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        raise click.Abort()
    except Exception as e:
        click.echo(f"Unexpected error: {e}", err=True) 
        raise click.Abort()
```

### Progress Display
```python
import click

@click.command()
def long_running_command():
    items = get_items_to_process()
    
    with click.progressbar(items, label="Processing items") as bar:
        for item in bar:
            process_item(item)
```

### Configuration Integration
```python
from ..config import get_config

@click.command()
@click.option('--use-cache/--no-cache', default=None, 
              help="Enable/disable result caching (default: disabled)")
def my_command(use_cache: bool | None):
    config = get_config()
    
    # Use explicit option or fall back to config default
    cache_enabled = use_cache if use_cache is not None else config.use_cache
    
    # Use cache_enabled in logic...
```

## Testing CLI Commands

### Test Structure
```python
# tests/test_cli_commands.py

import pytest
from click.testing import CliRunner
from giflab.cli.{command}_cmd import {command}

class Test{Command}Command:
    def test_{command}_basic_functionality(self):
        runner = CliRunner()
        result = runner.invoke({command}, ['--help'])
        assert result.exit_code == 0
        
    def test_{command}_with_options(self):
        runner = CliRunner()
        result = runner.invoke({command}, ['--option1', 'value'])
        assert result.exit_code == 0
```

### Integration Testing
```python
def test_cli_integration():
    """Ensure all commands are properly registered."""
    from giflab.cli import main
    
    # Check that command is registered
    assert '{command}' in [cmd.name for cmd in main.commands.values()]
```

## Performance Considerations

### Caching Behavior

The production pipeline uses `PredictionRunner` with SQLite storage. The CLI commands (`run`, `train`, `predict`, `export`, `stats`) delegate to `PredictionRunner` for all operations.

```python
from giflab.prediction_runner import PredictionRunner

# Create runner with SQLite storage
runner = PredictionRunner(db_path="results.db")
runner.run(data_dir="data/raw", workers=4)
```

### Memory Management
```python
# For large dataset operations
@click.option('--batch-size', default=10, type=int,
              help="Number of files to process in each batch")
def batch_command(batch_size: int):
    # Process files in batches to control memory usage
    for batch in chunked(files, batch_size):
        process_batch(batch)
```

## Common Patterns

### Path Handling
```python
@click.argument('input_path', type=click.Path(exists=True, path_type=Path))
@click.option('--output-dir', type=click.Path(path_type=Path), 
              default=Path("results"), help="Output directory")
def path_command(input_path: Path, output_dir: Path):
    # All paths are properly typed as pathlib.Path objects
    pass
```

### Verbose Output
```python
@click.option('-v', '--verbose', is_flag=True, help="Verbose output")
def verbose_command(verbose: bool):
    if verbose:
        click.echo("Detailed information...")
    click.echo("Basic output")
```

This guide ensures consistent CLI development and helps avoid common naming and import issues.