"""Functional tests for the configurable animately subprocess timeout.

Background:
    The animately wrapper historically hard-capped subprocess.run() at a
    10s wall-clock timeout (``RUN_TIMEOUT = int(os.getenv("GIFLAB_RUN_TIMEOUT", "10"))``
    evaluated once at module import). That cutoff was sensible for interactive
    product use, but blocked the metrics audit on legitimate ~9 MB / 99-frame
    real GIFs that exceed 10s of compression time even though they would
    succeed if given more wall-clock.

    Surfaced by the 2026-05-22 metrics audit
    (``docs/metrics-audit/2026-05-22/report.md``).

Contract under test:
    1. ``compress(engine="animately", params={"timeout_s": N})`` causes the
       animately subprocess to be invoked with ``timeout=N`` (not 10).
    2. ``GIFLAB_RUN_TIMEOUT`` env var is honoured **at call time**, not just
       at module import — so a test/process can change it dynamically.
    3. Precedence is params > env var > default (10).

    All three resolve via a single helper ``_resolve_run_timeout(params)`` in
    ``giflab.lossy`` so all five animately wrappers (standard, hard, advanced,
    plus gifsicle which shares the helper) see the same value.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Resolver — the unit that decides what timeout to use
# ---------------------------------------------------------------------------


def test_resolve_run_timeout_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """No env var, no params → default of 10s (preserves existing behaviour)."""
    from giflab.lossy import _resolve_run_timeout

    monkeypatch.delenv("GIFLAB_RUN_TIMEOUT", raising=False)
    assert _resolve_run_timeout(None) == 10
    assert _resolve_run_timeout({}) == 10
    assert _resolve_run_timeout({"lossy_level": 40}) == 10


def test_resolve_run_timeout_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    """GIFLAB_RUN_TIMEOUT is read at call time (not just at module import)."""
    from giflab.lossy import _resolve_run_timeout

    monkeypatch.setenv("GIFLAB_RUN_TIMEOUT", "45")
    assert _resolve_run_timeout(None) == 45
    assert _resolve_run_timeout({}) == 45
    # params without timeout_s falls through to env var
    assert _resolve_run_timeout({"lossy_level": 40}) == 45


def test_resolve_run_timeout_params_override_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """params['timeout_s'] takes precedence over the env var."""
    from giflab.lossy import _resolve_run_timeout

    monkeypatch.setenv("GIFLAB_RUN_TIMEOUT", "45")
    assert _resolve_run_timeout({"timeout_s": 120}) == 120
    # env var is overridden even when params has other keys too
    assert _resolve_run_timeout({"lossy_level": 40, "timeout_s": 90}) == 90


def test_resolve_run_timeout_params_override_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """params['timeout_s'] wins when no env var is set."""
    from giflab.lossy import _resolve_run_timeout

    monkeypatch.delenv("GIFLAB_RUN_TIMEOUT", raising=False)
    assert _resolve_run_timeout({"timeout_s": 60}) == 60


def test_resolve_run_timeout_invalid_value_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-positive or non-integer timeout values raise ValueError early."""
    from giflab.lossy import _resolve_run_timeout

    monkeypatch.delenv("GIFLAB_RUN_TIMEOUT", raising=False)
    with pytest.raises(ValueError):
        _resolve_run_timeout({"timeout_s": 0})
    with pytest.raises(ValueError):
        _resolve_run_timeout({"timeout_s": -5})


# ---------------------------------------------------------------------------
# compress_with_animately — uses the resolver
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_gif(tmp_path: Path) -> Path:
    """Create a minimal 10x10 single-frame GIF for testing."""
    from PIL import Image

    path = tmp_path / "tiny.gif"
    Image.new("RGB", (10, 10), (128, 64, 200)).save(path, format="GIF")
    return path


def _fake_subprocess_run_success(out_path: Path):
    """Build a subprocess.run replacement that succeeds and writes out_path.

    Captures the `timeout=` kwarg of the *compression* call (identified by the
    output path appearing in the command argv). Other subprocess.run calls
    that happen incidentally inside compress_with_animately — e.g. the
    post-call get_animately_version() — are returned successfully but not
    captured, so they don't clobber the captured timeout.
    """
    captured: dict[str, Any] = {}
    out_marker = str(out_path)

    def fake_run(*args: Any, **kwargs: Any) -> Any:
        cmd = args[0] if args else kwargs.get("args", [])
        cmd_str = " ".join(map(str, cmd)) if isinstance(cmd, list) else str(cmd)
        if out_marker in cmd_str:
            captured["timeout"] = kwargs.get("timeout")
            captured["args"] = args
            # Write a non-empty output so the post-condition check passes
            out_path.write_bytes(b"GIF89a\x00\x00")
        result = MagicMock()
        result.returncode = 0
        result.stderr = ""
        result.stdout = "animately-engine 1.0.0"
        return result

    return fake_run, captured


def test_compress_with_animately_uses_default_timeout(
    monkeypatch: pytest.MonkeyPatch, tiny_gif: Path, tmp_path: Path
) -> None:
    """Without params or env var, animately subprocess.run gets timeout=10."""
    from giflab import lossy

    monkeypatch.delenv("GIFLAB_RUN_TIMEOUT", raising=False)
    out_path = tmp_path / "out.gif"
    fake_run, captured = _fake_subprocess_run_success(out_path)

    with patch.object(lossy.subprocess, "run", side_effect=fake_run):
        lossy.compress_with_animately(
            tiny_gif, out_path, lossy_level=40, frame_keep_ratio=1.0
        )

    assert captured["timeout"] == 10


def test_compress_with_animately_honours_env_var(
    monkeypatch: pytest.MonkeyPatch, tiny_gif: Path, tmp_path: Path
) -> None:
    """GIFLAB_RUN_TIMEOUT at call time controls the subprocess timeout."""
    from giflab import lossy

    monkeypatch.setenv("GIFLAB_RUN_TIMEOUT", "30")
    out_path = tmp_path / "out.gif"
    fake_run, captured = _fake_subprocess_run_success(out_path)

    with patch.object(lossy.subprocess, "run", side_effect=fake_run):
        lossy.compress_with_animately(
            tiny_gif, out_path, lossy_level=40, frame_keep_ratio=1.0
        )

    assert captured["timeout"] == 30


def test_compress_with_animately_honours_timeout_param(
    monkeypatch: pytest.MonkeyPatch, tiny_gif: Path, tmp_path: Path
) -> None:
    """timeout_s kwarg overrides default."""
    from giflab import lossy

    monkeypatch.delenv("GIFLAB_RUN_TIMEOUT", raising=False)
    out_path = tmp_path / "out.gif"
    fake_run, captured = _fake_subprocess_run_success(out_path)

    with patch.object(lossy.subprocess, "run", side_effect=fake_run):
        lossy.compress_with_animately(
            tiny_gif,
            out_path,
            lossy_level=40,
            frame_keep_ratio=1.0,
            timeout_s=60,
        )

    assert captured["timeout"] == 60


def test_compress_with_animately_param_overrides_env(
    monkeypatch: pytest.MonkeyPatch, tiny_gif: Path, tmp_path: Path
) -> None:
    """timeout_s kwarg takes precedence over GIFLAB_RUN_TIMEOUT env var."""
    from giflab import lossy

    monkeypatch.setenv("GIFLAB_RUN_TIMEOUT", "30")
    out_path = tmp_path / "out.gif"
    fake_run, captured = _fake_subprocess_run_success(out_path)

    with patch.object(lossy.subprocess, "run", side_effect=fake_run):
        lossy.compress_with_animately(
            tiny_gif,
            out_path,
            lossy_level=40,
            frame_keep_ratio=1.0,
            timeout_s=120,
        )

    assert captured["timeout"] == 120


def test_compress_with_animately_timeout_error_message_reflects_actual_value(
    monkeypatch: pytest.MonkeyPatch, tiny_gif: Path, tmp_path: Path
) -> None:
    """The 'timed out after N seconds' error mentions the actual timeout used,
    not the stale module-level RUN_TIMEOUT constant."""
    from giflab import lossy

    monkeypatch.delenv("GIFLAB_RUN_TIMEOUT", raising=False)
    out_path = tmp_path / "out.gif"

    def fake_run_timeout(*args: Any, **kwargs: Any) -> Any:
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=kwargs.get("timeout", 10))

    with patch.object(lossy.subprocess, "run", side_effect=fake_run_timeout):
        with pytest.raises(RuntimeError, match=r"timed out after 75 seconds"):
            lossy.compress_with_animately(
                tiny_gif,
                out_path,
                lossy_level=40,
                frame_keep_ratio=1.0,
                timeout_s=75,
            )


# ---------------------------------------------------------------------------
# AnimatelyLossyCompressor.apply — pipes timeout_s from params to lossy.py
# ---------------------------------------------------------------------------


def test_animately_wrapper_passes_timeout_s_to_lossy(
    monkeypatch: pytest.MonkeyPatch, tiny_gif: Path, tmp_path: Path
) -> None:
    """AnimatelyLossyCompressor.apply forwards params['timeout_s'] through."""
    from giflab.tool_wrappers import AnimatelyLossyCompressor

    captured: dict[str, Any] = {}

    def fake_compress(*args: Any, **kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        captured["args"] = args
        # Return a minimal wrapper-result dict
        return {
            "render_ms": 5,
            "engine": "animately-standard",
            "engine_version": "test",
            "lossy_level": kwargs.get("lossy_level", 0),
            "frame_keep_ratio": 1.0,
            "color_keep_count": None,
            "original_frames": 1,
            "original_colors": 16,
            "command": "fake",
            "stderr": None,
        }

    # Pre-create output so validate_wrapper_apply_result is happy
    out_path = tmp_path / "out.gif"
    out_path.write_bytes(b"GIF89a\x00\x00")

    monkeypatch.setattr(
        "giflab.tool_wrappers.compress_with_animately", fake_compress
    )

    AnimatelyLossyCompressor().apply(
        tiny_gif, out_path, params={"lossy_level": 40, "timeout_s": 90}
    )

    assert captured.get("timeout_s") == 90
