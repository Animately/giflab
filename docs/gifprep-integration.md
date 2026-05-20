# gifprep integration — public API refactor

**Status**: Implemented as of v0.3.0 — see [public-api.md](./public-api.md) for the live contract. This document is preserved as the historical proposal.
**Last updated**: 2026-04-11
**Implemented**: 2026-05-19
**Owner**: Lachy (leads both giflab and gifprep)
**Scope**: A small refactor pass on giflab to expose a stable public API (`compress`, `measure`) that gifprep can depend on as a library.

## Context

[gifprep](https://github.com/animately/gifprep) is an experimental sibling repository for GIF **preprocessing** strategies — denoising, cleanup, and other transforms applied *before* compression. Its purpose is to measure whether a preprocessing stage improves the quality/size tradeoff of a downstream compressor. gifprep is not a competitor to giflab; it is a consumer of giflab.

**Strategic source**: the [AI Preprocessing initiative in Linear](https://linear.app/animately/initiative/ai-preprocessing-177f09f5c33e) — the canonical cross-repo handshake. Its description (not a separate Linear document) carries the two-repo split, ownership matrix, library contract summary, sequencing, and collaboration model. Read it first if you are a Claude instance picking up this refactor. This repo-local document carries the tactical "what" and "how" — signatures, files to create, what not to touch. Never duplicate content between the two: strategic facts live in the initiative description, tactical facts live here.

**Short version of the split**:

| Concern | Owner |
|---|---|
| Compression engine wrappers, parameter grids, matrix benchmarks, ML dataset generation, feature extraction, SQLite storage | **giflab** (this repo — unchanged) |
| Quality metrics (SSIM, MS-SSIM, PSNR, LPIPS, etc.) | **giflab** (this repo — unchanged) |
| Preprocessing strategies (denoising, cleanup, future AI-based transforms) | **gifprep** |
| Test corpus with provenance metadata for preprocessing evaluation | **gifprep** |
| Benchmark harness that runs strategy × GIF × quality-level and produces a Pareto report | **gifprep** |

gifprep depends on giflab as a library. giflab does not depend on gifprep. The dependency is one-way.

## The ask

gifprep needs two functions from giflab, exposed as a stable, documented public API that gifprep can import directly:

### 1. `giflab.compress(input_path, output_path, engine, params) → CompressResult`

Thin wrapper over the existing engine machinery. Runs a single compression engine on an input GIF, writes the result to `output_path`, and returns metadata.

**Proposed signature** (refine during the giflab spec if needed):

```python
from pathlib import Path
from dataclasses import dataclass
from typing import Literal

def compress(
    input_path: Path,
    output_path: Path,
    engine: Literal["animately", "gifsicle", "gifski", "imagemagick", "ffmpeg"],
    params: dict,  # engine-specific, e.g. {"lossy": 60, "colors": 256}
) -> CompressResult:
    ...

@dataclass(frozen=True)
class CompressResult:
    output_path: Path
    output_bytes: int
    render_ms: int
    engine: str
    engine_version: str
    params: dict
```

### 2. `giflab.measure(reference_path, candidate_path, metrics) → MeasureResult`

Thin wrapper over `calculate_comprehensive_metrics`. Computes a configurable subset of quality metrics between two GIFs.

**Proposed signature**:

```python
def measure(
    reference_path: Path,
    candidate_path: Path,
    metrics: list[Literal["ssim", "ms_ssim", "psnr", "lpips", "gmsd", "fsim", "chist"]] = ["ssim"],
) -> MeasureResult:
    ...

@dataclass(frozen=True)
class MeasureResult:
    ssim: float | None
    ms_ssim: float | None
    psnr: float | None
    lpips: float | None
    gmsd: float | None
    fsim: float | None
    chist: float | None
    # Populated only for metrics requested; others remain None
```

Both importable at the top level: `from giflab import compress, measure, CompressResult, MeasureResult`.

## What exists today (per a 2026-04-11 architectural survey of this repo)

- `src/giflab/metrics.py::calculate_comprehensive_metrics(ref_gif, comp_gif)` — computes 13 quality metrics. The `measure()` wrapper needs to expose a requested subset and a cleaner dataclass return.
- `src/giflab/external_engines/common.py::run_command(cmd, engine, output_path)` — runs a compression command and returns metadata. The `compress()` wrapper needs to construct the command from a high-level `(engine, params)` pair.
- Engine wrappers (`AnimatelyLossyCompressor`, `GifsicleFrameReducer`, etc.) in `src/giflab/external_engines/*.py`. These are the implementation target of `compress()` — the wrapper dispatches to the right class by `engine` string.
- `src/giflab/__init__.py` already exposes a clean selection of internal types. Adding `compress`, `measure`, and their result types to that export list is the public-facing change.

## What needs to change

1. **New module** `src/giflab/public_api.py` containing the two functions above and their result dataclasses. This module does not reimplement anything — it dispatches to existing internals.
2. **Export from `src/giflab/__init__.py`**: `from giflab.public_api import compress, measure, CompressResult, MeasureResult`.
3. **Tests** proving the functions work end-to-end:
   - `compress()` with animately and gifsicle at a non-trivial lossy setting.
   - `measure()` requesting SSIM, then SSIM + MS-SSIM, on a known reference/candidate pair.
   - The returned dataclasses contain the right fields populated.
4. **Documentation** — a short public-API section in the README or a dedicated `docs/public-api.md`, with the signatures, the invariants (deterministic, does not mutate input, overwrites `output_path`), and a usage example. This is the contract future gifprep versions bind to.
5. **Version bump and tag** a giflab release so gifprep can pin it by version string.

## What MUST NOT change

- The existing CLI, dataset generation pipeline, matrix benchmark, SQLite schema, feature extraction, tool interfaces, or parameter grids. **None of that is touched.**
- `SCOPE.md`'s scope is not broadened. No preprocessing concepts are added to giflab. No pipeline chaining. No new compression engines. No new metrics. This is **purely an API exposure pass** over code that already exists.
- Tool interfaces (`tool_interfaces.py`, `dynamic_pipeline.py`) are not restructured. Their three slots (`frame_reduction`, `color_reduction`, `lossy_compression`) remain compression-only. Preprocessing lives in gifprep, always.

## How to proceed (for a Claude instance picking this up)

1. Read the Linear "gifprep ↔ giflab split" document for strategic context.
2. Open a feature branch in this repo, e.g. `NNN-public-api-refactor`.
3. Run `/speckit.specify` with this document as the input, producing `specs/NNN-public-api-refactor/spec.md`. The spec should be tight — this is a one- or two-day refactor, not a greenfield feature.
4. `/speckit.plan` → `/speckit.tasks` → `/speckit.implement`.
5. Release the tagged version so gifprep can pin it.
6. Notify the gifprep repo: update `specs/002-benchmark-harness/spec.md` assumptions to reference the specific pinned version.
7. Mark this document "implemented — see `docs/public-api.md`" and leave it as a historical pointer. The live contract lives in the public-API docs going forward.

## Coordination rules

- The **library contract** (function signatures + semantics) has **one source of truth at a time**. While this refactor is in progress, that source is this document. Once the refactor ships, it moves to the giflab public-API docs, and this document becomes a historical pointer. **Do not duplicate the contract.**
- gifprep's `specs/002-benchmark-harness/spec.md` assumes this refactor exists. gifprep implementation waits on a pinned giflab release; gifprep spec approval and planning can proceed in parallel.
- If during implementation the proposed signatures prove awkward, update this document **and** the Linear "gifprep ↔ giflab split" document **and** raise a flag on the gifprep spec so it can be updated *before* gifprep planning finalises. Do not silently diverge.
- This refactor should ship with a changelog entry pinning the exact public API surface, so gifprep can reference a specific version tag.
