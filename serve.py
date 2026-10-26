"""Repository-root entry point for ``python3 -m serve ALIAS``."""

from pathlib import Path
import runpy
import sys


if __name__ == "__main__":
    source = Path(__file__).resolve().parent / "src"
    # Preserve the source scripts' sibling imports and script entry point.
    sys.path.insert(0, str(source))
    runpy.run_path(str(source / "serve.py"), run_name="__main__")
