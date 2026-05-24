"""Functional tests for the public ``giflab.compress`` API.

Mocks the engine wrapper classes — these tests do not run real engine
subprocesses. Real-engine coverage lives in
``tests/integration/test_public_api_compress_e2e.py``.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from giflab import (
    SUPPORTED_ENGINES,
    CompressResult,
    EngineUnavailableError,
    UnknownEngineError,
    compress,
)

# Map each public engine string to the wrapper class symbol that compress()
# dispatches to. Used to patch the right symbol per test.
_ENGINE_TO_WRAPPER = {
    "animately": "AnimatelyLossyCompressor",
    "gifsicle": "GifsicleLossyCompressor",
    "gifski": "GifskiLossyCompressor",
    "imagemagick": "ImageMagickLossyCompressor",
    "ffmpeg": "FFmpegLossyCompressor",
}


def _mock_wrapper_class(render_ms: int = 42, version: str = "test-1.0"):
    """Build a Mock that quacks like a wrapper class with a working apply()."""
    instance = MagicMock()
    instance.apply.return_value = {
        "render_ms": render_ms,
        "engine": "test",
        "command": "test cmd",
        "kilobytes": 1,
    }
    cls = MagicMock(return_value=instance)
    cls.available = MagicMock(return_value=True)
    cls.version = MagicMock(return_value=version)
    return cls, instance


def _write_dummy_output(path: Path, size_bytes: int = 512) -> None:
    path.write_bytes(b"\x00" * size_bytes)


@pytest.mark.parametrize("engine", list(SUPPORTED_ENGINES))
def test_compress_happy_path_per_engine(
    tmp_path: Path, tiny_gif: Path, engine: str
) -> None:
    """compress() dispatches to the correct wrapper for each supported engine."""
    out_path = tmp_path / f"out_{engine}.gif"
    wrapper_cls, wrapper_instance = _mock_wrapper_class(
        render_ms=123, version=f"{engine}-9.9"
    )

    # Patch the symbol inside giflab.public_api so the dispatch table picks up the mock.
    with patch(f"giflab.public_api.{_ENGINE_TO_WRAPPER[engine]}", wrapper_cls):
        wrapper_instance.apply.side_effect = lambda *a, **kw: (
            _write_dummy_output(out_path, 777),
            {"render_ms": 123, "engine": engine, "command": "x", "kilobytes": 1},
        )[1]
        # Reset the dispatch cache so the patched symbol is picked up.

        result = compress(
            input_path=tiny_gif,
            output_path=out_path,
            engine=engine,
            params={"lossy_level": 40},
        )

    assert isinstance(result, CompressResult)
    assert result.output_path == out_path
    assert result.output_bytes == 777
    assert result.render_ms == 123
    assert result.engine == engine
    assert result.engine_version == f"{engine}-9.9"
    assert result.params == {"lossy_level": 40}


def test_compress_unknown_engine_raises_before_io(
    tmp_path: Path, tiny_gif: Path
) -> None:
    """UnknownEngineError raised before any file I/O happens."""
    out_path = tmp_path / "never_written.gif"

    with pytest.raises(UnknownEngineError) as exc_info:
        compress(
            input_path=tiny_gif,
            output_path=out_path,
            engine="not-a-real-engine",  # type: ignore[arg-type]
            params={},
        )

    msg = str(exc_info.value)
    assert "not-a-real-engine" in msg
    # Message must list the supported engines so the consumer can correct themselves.
    for eng in SUPPORTED_ENGINES:
        assert eng in msg
    # No output file should exist — error raised before dispatch.
    assert not out_path.exists()


def test_compress_unavailable_engine_raises_typed_error(
    tmp_path: Path, tiny_gif: Path
) -> None:
    """EngineUnavailableError raised when wrapper reports binary missing."""
    out_path = tmp_path / "unavailable.gif"
    wrapper_cls, _ = _mock_wrapper_class()
    wrapper_cls.available = MagicMock(return_value=False)

    with patch("giflab.public_api.GifskiLossyCompressor", wrapper_cls):
        with pytest.raises(EngineUnavailableError) as exc_info:
            compress(
                input_path=tiny_gif,
                output_path=out_path,
                engine="gifski",
                params={"lossy_level": 40},
            )

    assert "gifski" in str(exc_info.value)
    assert not out_path.exists()


def test_compress_missing_input_raises_filenotfounderror(tmp_path: Path) -> None:
    """compress() validates input_path before dispatching to the wrapper."""
    wrapper_cls, wrapper_instance = _mock_wrapper_class()
    with patch("giflab.public_api.GifsicleLossyCompressor", wrapper_cls):
        with pytest.raises(FileNotFoundError):
            compress(
                input_path=tmp_path / "does_not_exist.gif",
                output_path=tmp_path / "out.gif",
                engine="gifsicle",
                params={"lossy_level": 40},
            )

    # Wrapper must not have been called.
    wrapper_instance.apply.assert_not_called()


def test_compress_result_is_frozen(tmp_path: Path, tiny_gif: Path) -> None:
    """CompressResult is immutable — mutation raises FrozenInstanceError."""
    out_path = tmp_path / "frozen.gif"
    wrapper_cls, wrapper_instance = _mock_wrapper_class()
    wrapper_instance.apply.side_effect = lambda *a, **kw: (
        _write_dummy_output(out_path, 100),
        {"render_ms": 1, "engine": "x", "command": "x", "kilobytes": 1},
    )[1]

    with patch("giflab.public_api.GifsicleLossyCompressor", wrapper_cls):
        result = compress(
            tiny_gif, out_path, engine="gifsicle", params={"lossy_level": 40}
        )

    with pytest.raises(dataclasses.FrozenInstanceError):
        result.output_bytes = 999  # type: ignore[misc]


def test_compress_passes_timeout_s_param_through_to_wrapper(
    tmp_path: Path, tiny_gif: Path
) -> None:
    """``params['timeout_s']`` is forwarded verbatim to the wrapper's apply().

    Locks the public-API contract added by the audit-fix for
    ``[[giflab-animately-10s-timeout-configurable]]``: a consumer can pass
    ``compress(..., params={"lossy_level": 40, "timeout_s": 60})`` and the
    wrapper sees ``timeout_s`` in its params dict (which it then forwards to
    the underlying compress_with_* helper).
    """
    out_path = tmp_path / "with_timeout.gif"
    wrapper_cls, wrapper_instance = _mock_wrapper_class()
    wrapper_instance.apply.side_effect = lambda *a, **kw: (
        _write_dummy_output(out_path, 100),
        {"render_ms": 1, "engine": "animately", "command": "x", "kilobytes": 1},
    )[1]

    with patch("giflab.public_api.AnimatelyLossyCompressor", wrapper_cls):
        result = compress(
            tiny_gif,
            out_path,
            engine="animately",
            params={"lossy_level": 40, "timeout_s": 60},
        )

    # Wrapper received the timeout in its params kwarg.
    _, kwargs = wrapper_instance.apply.call_args
    assert kwargs["params"]["timeout_s"] == 60
    assert kwargs["params"]["lossy_level"] == 40
    # And the timeout is preserved on the result for round-tripping.
    assert result.params["timeout_s"] == 60


def test_compress_params_mutation_does_not_leak(tmp_path: Path, tiny_gif: Path) -> None:
    """Mutating caller's params dict after the call does not affect the result."""
    out_path = tmp_path / "leak.gif"
    wrapper_cls, wrapper_instance = _mock_wrapper_class()
    wrapper_instance.apply.side_effect = lambda *a, **kw: (
        _write_dummy_output(out_path, 100),
        {"render_ms": 1, "engine": "x", "command": "x", "kilobytes": 1},
    )[1]

    caller_params = {"lossy_level": 40}

    with patch("giflab.public_api.GifsicleLossyCompressor", wrapper_cls):
        result = compress(tiny_gif, out_path, engine="gifsicle", params=caller_params)

    caller_params["lossy_level"] = 99
    caller_params["new_key"] = "leaked"

    assert result.params == {"lossy_level": 40}
    assert "new_key" not in result.params
