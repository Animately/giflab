# Quickstart: Using the giflab Public API

**Audience**: A developer in an external Python project (e.g., gifprep) who has never read giflab internals and wants to compress and measure GIFs in under 5 minutes.

---

## 1. Install

In the external project's `pyproject.toml`:

```toml
[tool.poetry.dependencies]
python = "^3.11"
giflab = "^0.3.0"
```

Then:

```bash
poetry install
```

You must also have the engine binaries you intend to use available on `PATH`. The library will raise `EngineUnavailableError` for any engine whose binary is missing.

| Engine | macOS install | Linux install |
|---|---|---|
| `animately` | (internal binary; see `~/bin/animately`) | (internal binary) |
| `gifsicle` | `brew install gifsicle` | `apt install gifsicle` |
| `gifski` | `brew install gifski` | `apt install gifski` |
| `imagemagick` | `brew install imagemagick` | `apt install imagemagick` |
| `ffmpeg` | `brew install ffmpeg` | `apt install ffmpeg` |

---

## 2. Compress a GIF

```python
from pathlib import Path
from giflab import compress

result = compress(
    input_path=Path("samples/source.gif"),
    output_path=Path("out/compressed.gif"),
    engine="gifsicle",
    params={"lossy_level": 40},
)

print(f"Wrote {result.output_bytes} bytes in {result.render_ms} ms")
print(f"Used {result.engine} {result.engine_version} with params {result.params}")
```

---

## 3. Measure quality between two GIFs

```python
from pathlib import Path
from giflab import measure

scores = measure(
    reference_path=Path("samples/source.gif"),
    candidate_path=Path("out/compressed.gif"),
    metrics=["ssim", "psnr"],
)

print(f"SSIM: {scores.ssim:.4f}")
print(f"PSNR: {scores.psnr:.2f} dB")
# scores.ms_ssim, scores.lpips, scores.gmsd, scores.fsim, scores.chist are all None
```

Only the requested metrics are computed. Asking for `lpips` triggers a PyTorch model load on first call within the process; subsequent calls reuse the cached model.

---

## 4. End-to-end: compress + measure (the gifprep harness shape)

```python
from pathlib import Path
from giflab import compress, measure

source = Path("samples/source.gif")

for engine in ["animately", "gifsicle"]:
    for lossy_level in [20, 40, 60, 80]:
        out = Path(f"out/{engine}_l{lossy_level}.gif")
        compressed = compress(source, out, engine=engine, params={"lossy_level": lossy_level})
        scores = measure(source, compressed.output_path, metrics=["ssim", "ms_ssim"])
        print(
            f"{engine:12} L={lossy_level:3} "
            f"size={compressed.output_bytes:>8} ssim={scores.ssim:.4f} "
            f"ms_ssim={scores.ms_ssim:.4f}"
        )
```

This is the shape of a Pareto-tradeoff sweep: vary one parameter, measure quality at matched settings, decide which engine + setting wins for the content type. gifprep's benchmark harness extends this by adding a preprocessing step before `compress`.

---

## 5. Supported engines and metrics

```python
from giflab import SUPPORTED_ENGINES, SUPPORTED_METRICS

print(SUPPORTED_ENGINES)
# ('animately', 'gifsicle', 'gifski', 'imagemagick', 'ffmpeg')

print(SUPPORTED_METRICS)
# ('ssim', 'ms_ssim', 'psnr', 'lpips', 'gmsd', 'fsim', 'chist')
```

These tuples are the authoritative source for what the public API accepts in this release.

---

## 6. Errors you will see

```python
from giflab import (
    compress,
    UnknownEngineError,
    UnknownMetricError,
    EngineUnavailableError,
)

try:
    compress(Path("a.gif"), Path("b.gif"), engine="not-a-real-engine", params={})
except UnknownEngineError as e:
    print(e)  # "Unknown engine 'not-a-real-engine'. Supported: animately, gifsicle, gifski, imagemagick, ffmpeg"

try:
    compress(Path("a.gif"), Path("b.gif"), engine="gifski", params={})
except EngineUnavailableError as e:
    print(e)  # "Engine 'gifski' binary not found on PATH"
```

All three exceptions inherit from `giflab.error_handling.GifLabError`, so a `except GifLabError:` catches them uniformly if you prefer.

---

## 7. What this API does *not* do

If you need any of the following, look elsewhere — they are intentionally out of scope for v1:

- **Preprocessing** (denoising, cleanup, generative reimagining) — see [gifprep](https://github.com/Animately/gifprep).
- **Pipeline chaining** (compress→compress, preprocess→compress as one call) — compose the functions yourself.
- **Dataset generation** at scale (the matrix benchmark, ML training data) — use giflab's CLI directly (`python -m giflab run --preset ...`), not the public API.
- **Async or streaming variants** — the public API is synchronous.
- **Batch multi-file APIs** — loop over files yourself.

The full live contract — invariants, version-pinning rules, exception details — lives at `docs/public-api.md` in the giflab repository.
