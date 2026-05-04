import os
import sys
from pathlib import Path


def _preparse_root(argv: list[str]) -> None:
    """Promote ``--root`` to ``MERGEDOG_ROOT`` before mergedog imports.

    ``mergedog.paths`` resolves the root directory at import time, and
    every other mergedog module imports paths transitively, so we have
    to seed the env var before ``from mergedog.cli import main`` runs.
    """
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--root" and i + 1 < len(argv):
            os.environ["MERGEDOG_ROOT"] = str(Path(argv[i + 1]).expanduser().resolve())
            return
        if a.startswith("--root="):
            value = a.split("=", 1)[1]
            os.environ["MERGEDOG_ROOT"] = str(Path(value).expanduser().resolve())
            return
        i += 1


_preparse_root(sys.argv[1:])

from mergedog.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
