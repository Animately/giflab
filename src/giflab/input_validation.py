"""Input validation utilities for GifLab.

This module provides comprehensive input validation functions to ensure security
and data integrity throughout the GifLab pipeline.
"""

import os
import re
from pathlib import Path
from typing import Any


class ValidationError(ValueError):
    """Raised when input validation fails."""

    pass


def validate_raw_dir(raw_dir: str | Path, require_gifs: bool = True) -> Path:
    """Validate RAW_DIR input for security and usability.

    Args:
        raw_dir: Directory path to validate
        require_gifs: Whether to require at least one GIF file in the directory

    Returns:
        Validated Path object

    Raises:
        ValidationError: If validation fails
    """
    if not raw_dir:
        raise ValidationError("RAW_DIR cannot be empty")

    # Convert to Path object
    raw_path = Path(raw_dir)

    # Basic path security validation
    validate_path_security(raw_path)

    # Check existence
    if not raw_path.exists():
        raise ValidationError(f"RAW_DIR does not exist: {raw_path}")

    # Check it's a directory
    if not raw_path.is_dir():
        raise ValidationError(f"RAW_DIR is not a directory: {raw_path}")

    # Check readability
    if not os.access(raw_path, os.R_OK):
        raise ValidationError(f"RAW_DIR is not readable: {raw_path}")

    # Check for GIF files if required
    if require_gifs:
        gif_files = list(raw_path.rglob("*.gif"))
        if not gif_files:
            raise ValidationError(f"RAW_DIR contains no GIF files: {raw_path}")

    return raw_path


def validate_path_security(path: str | Path) -> Path:
    """Validate path for security concerns.

    Security model
    --------------
    GifLab invokes every external tool (animately, gifsicle, ffmpeg,
    imagemagick, gifski, ...) through ``subprocess.run(cmd_list, shell=False)``
    -- see :func:`giflab.external_engines.common.run_command`. Arguments are
    handed straight to ``execve`` and are *never* re-interpreted by a shell,
    so traditional shell-meta-characters (``;``, ``&``, ``|``, ``$``,
    backticks, etc.) cannot be used for command injection in this codebase.

    Rejecting those characters here previously broke perfectly legitimate
    filenames sourced from HubSpot / generic CDN URLs (e.g.
    ``Untitled-1-2.gif_upscale=true&width=1600&name=foo.gif``) without
    providing any actual security benefit. The audit at
    ``docs/metrics-audit/2026-05-22/report.md`` lost 9 sweep attempts to
    this misfire.

    The remaining blacklist deliberately covers only characters that are
    dangerous *regardless* of how the path is later used:

    * ``\\x00`` (null byte) -- truncates C strings, can confuse libc and
      external tools
    * ``\\n`` / ``\\r`` -- corrupt log output and can be smuggled into other
      structured outputs (CSV rows, JSON, etc.) the pipeline emits

    If a future caller starts forwarding paths into ``shell=True`` or building
    shell strings by interpolation, that caller is responsible for its own
    escaping (e.g. ``shlex.quote``) -- it must not lean on this validator.

    Args:
        path: Path to validate

    Returns:
        Validated Path object

    Raises:
        ValidationError: If path contains security risks
    """
    if not path:
        raise ValidationError("Path cannot be empty")

    path_obj = Path(path)
    path_str = str(path)  # Check original string first

    # Check for null bytes before calling resolve()
    if "\x00" in path_str:
        raise ValidationError(f"Path contains null bytes: {path_str}")

    # Now resolve the path
    resolved_path_str = str(path_obj.resolve())

    # Reject control characters that corrupt logs / structured outputs.
    # Shell-meta-chars (; & | ` $) are intentionally *not* here -- see the
    # docstring above for the security model.
    control_chars = ["\n", "\r"]
    for char in control_chars:
        if char in resolved_path_str:
            raise ValidationError(
                f"Path contains potentially dangerous characters: {resolved_path_str!r}"
            )

    # Check for path traversal attempts
    if ".." in path_obj.parts:
        raise ValidationError(f"Path contains directory traversal: {resolved_path_str}")

    # Check for excessively long paths (platform-dependent but generally safe limit)
    if len(resolved_path_str) > 4096:
        raise ValidationError(
            f"Path too long ({len(resolved_path_str)} chars): {resolved_path_str[:100]}..."
        )

    return path_obj


def validate_output_path(path: str | Path, create_parent: bool = True) -> Path:
    """Validate output path for writing.

    Args:
        path: Output path to validate
        create_parent: Whether to create parent directories if they don't exist

    Returns:
        Validated Path object

    Raises:
        ValidationError: If path is invalid or not writable
    """
    path_obj = validate_path_security(path)

    # Check parent directory exists or can be created
    parent = path_obj.parent
    if not parent.exists():
        if create_parent:
            try:
                parent.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                raise ValidationError(f"Cannot create parent directory {parent}: {e}")
        else:
            raise ValidationError(f"Parent directory does not exist: {parent}")

    # Check parent is writable
    if not os.access(parent, os.W_OK):
        raise ValidationError(f"Parent directory is not writable: {parent}")

    # If file exists, check it's writable
    if path_obj.exists() and not os.access(path_obj, os.W_OK):
        raise ValidationError(f"Output file is not writable: {path_obj}")

    return path_obj


def validate_worker_count(workers: int) -> int:
    """Validate worker count parameter.

    Args:
        workers: Number of worker processes

    Returns:
        Validated worker count

    Raises:
        ValidationError: If worker count is invalid
    """
    if not isinstance(workers, int):
        raise ValidationError(f"Worker count must be an integer, got {type(workers)}")

    if workers < 0:
        raise ValidationError(f"Worker count cannot be negative: {workers}")

    # Cap at reasonable maximum (4x CPU count)
    import multiprocessing

    max_workers = multiprocessing.cpu_count() * 4
    if workers > max_workers:
        raise ValidationError(f"Worker count too high: {workers} (max: {max_workers})")

    return workers


def validate_file_extension(path: str | Path, expected_extensions: list[str]) -> Path:
    """Validate file has expected extension.

    Args:
        path: File path to validate
        expected_extensions: List of allowed extensions (with or without dots)

    Returns:
        Validated Path object

    Raises:
        ValidationError: If file extension is not allowed
    """
    path_obj = Path(path)

    # Normalize extensions (ensure they start with dot)
    normalized_exts = []
    for ext in expected_extensions:
        if not ext.startswith("."):
            ext = "." + ext
        normalized_exts.append(ext.lower())

    file_ext = path_obj.suffix.lower()
    if file_ext not in normalized_exts:
        raise ValidationError(
            f"Invalid file extension: {file_ext} (expected: {normalized_exts})"
        )

    return path_obj


def validate_config_paths(config_dict: dict[str, Any]) -> dict[str, Path]:
    """Validate configuration paths.

    Args:
        config_dict: Dictionary of configuration values

    Returns:
        Dictionary of validated Path objects

    Raises:
        ValidationError: If any path is invalid
    """
    validated_paths = {}

    for key, value in config_dict.items():
        if key.endswith("_DIR") or key.endswith("_PATH"):
            if value is not None:
                try:
                    validated_paths[key] = validate_path_security(value)
                except ValidationError as e:
                    raise ValidationError(f"Invalid {key}: {e}")

    return validated_paths


def sanitize_filename(filename: str, replacement: str = "_") -> str:
    """Sanitize filename for safe filesystem usage.

    Args:
        filename: Original filename
        replacement: Character to replace invalid characters with

    Returns:
        Sanitized filename
    """
    if not filename:
        return "unnamed"

    # Replace invalid characters for most filesystems
    # Windows: < > : " | ? * and control characters
    # Unix: / and null
    invalid_chars = r'[<>:"/|?*\x00-\x1f\x7f]'
    sanitized = re.sub(invalid_chars, replacement, filename)

    # Remove leading/trailing dots and spaces
    sanitized = sanitized.strip(". ")

    # Ensure not empty after sanitization
    if not sanitized:
        sanitized = "unnamed"

    # Limit length to reasonable maximum
    max_length = 255
    if len(sanitized) > max_length:
        name, ext = os.path.splitext(sanitized)
        available_length = max_length - len(ext)
        sanitized = name[:available_length] + ext

    return sanitized
