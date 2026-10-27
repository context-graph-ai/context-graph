#!/usr/bin/env python3
"""Discard disposable compiler scratch before compiler cache post hooks; keep the trybuild compile cache."""
import os
from pathlib import Path
import shutil

root = Path(os.environ.get("CARGO_TARGET_DIR", "target"))
scratch = root / "tests"
if scratch.is_symlink():
    scratch.unlink()
elif scratch.is_dir():
    for entry in scratch.iterdir():
        if entry.name == "trybuild" and not entry.is_symlink():
            continue
        if entry.is_symlink() or entry.is_file():
            entry.unlink()
        else:
            shutil.rmtree(entry)
