"""Early argv handling for import-time mergedog settings."""
from __future__ import annotations

import os
from pathlib import Path


def promote_early_env(argv: list[str]) -> None:
    """Promote flags that must be known before importing ``mergedog.paths``.

    Several modules resolve filesystem paths and project policy at import time.
    Entry points call this before importing those modules so explicit command
    line flags beat ambient environment variables.
    """
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--root" and i + 1 < len(argv):
            os.environ["MERGEDOG_ROOT"] = str(
                Path(argv[i + 1]).expanduser().resolve()
            )
            i += 2
            continue
        if a.startswith("--root="):
            value = a.split("=", 1)[1]
            os.environ["MERGEDOG_ROOT"] = str(Path(value).expanduser().resolve())
            i += 1
            continue
        if a == "--repo" and i + 1 < len(argv):
            os.environ["MERGEDOG_REPO_SLUG"] = argv[i + 1]
            i += 2
            continue
        if a.startswith("--repo="):
            os.environ["MERGEDOG_REPO_SLUG"] = a.split("=", 1)[1]
            i += 1
            continue
        i += 1
