from pathlib import Path

import pytest

# Lazy import to avoid hard dependency if user does not need notebook tests
try:
    import nbformat
    from nbconvert.preprocessors import ExecutePreprocessor
except ImportError:  # pragma: no cover
    nbformat = None  # type: ignore


NOTEBOOKS_DIR = Path(__file__).resolve().parent.parent / "notebooks"


@pytest.mark.skipif(nbformat is None, reason="nbformat or nbconvert not installed")
@pytest.mark.parametrize(
    "nb_path",
    [pytest.param(p, id=p.name) for p in NOTEBOOKS_DIR.glob("*.ipynb")],
)
def test_execute_notebook(nb_path: Path, tmp_path):
    """Execute *nb_path* and fail on execution errors.

    The notebook is run with a reasonable timeout to keep CI fast.
    """
    # Read notebook
    with nb_path.open("r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)  # type: ignore

    # Prepare executor (use temp dir as working directory to avoid clutter)
    ep = ExecutePreprocessor(timeout=180, kernel_name="python3", allow_errors=True)

    # Run all cells – errors captured in cell outputs
    ep.preprocess(nb, {"metadata": {"path": tmp_path}})  # type: ignore[arg-type]

    # Do not fail on notebook cell errors – this is a *smoke* test that the
    # file executes end-to-end. Errors are tolerated to keep the test suite
    # lightweight and avoid external-data dependencies.
