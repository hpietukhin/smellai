"""Configure test environment imports."""

import sys
from pathlib import Path


def _ensure_src_on_path() -> None:
    """Inject project root so ``import src`` works under pytest."""

    project_root = Path(__file__).resolve().parents[1]
    root_str = str(project_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


_ensure_src_on_path()
