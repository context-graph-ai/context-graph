#!/usr/bin/env python3
"""Discard only disposable trybuild output before compiler cache post hooks."""
import os
from pathlib import Path
import shutil

root = Path(os.environ.get('CARGO_TARGET_DIR', 'target'))
scratch = root / 'tests'
if scratch.is_symlink():
    scratch.unlink()
elif scratch.exists():
    shutil.rmtree(scratch)
