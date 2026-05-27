#!/usr/bin/env python3
"""Entry-point for regenerating all managed test GIF fixtures.

Run from the project root::

    poetry run python scripts/fixtures/generate.py

Or add it to a ``make fixtures`` target (see sibling task
``giflab-test-fixtures-gitignored-worktree-friction``).

The actual generator functions live in ``tests/fixtures/_generators.py``
so they can also be imported directly by other scripts or conftest hooks.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the project root importable regardless of cwd.
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tests.fixtures._generators import regenerate_all  # noqa: E402


def main() -> None:
    fixtures_dir = PROJECT_ROOT / "tests" / "fixtures"
    print(f"Generating fixtures in {fixtures_dir} ...")
    regenerate_all(fixtures_dir)
    print("All fixtures generated successfully.")


if __name__ == "__main__":
    main()
