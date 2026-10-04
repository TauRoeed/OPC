"""Which code produced a result: the git commit of this checkout and whether tracked files had changes."""
from __future__ import annotations

import subprocess
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def code_commit() -> dict:
    """``{"commit": full hash, "dirty": tracked files modified}``; both None outside a git checkout."""
    def git(*args):
        return subprocess.run(["git", *args], cwd=_ROOT, capture_output=True, text=True, check=True).stdout.strip()

    try:
        return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain", "--untracked-files=no"))}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
