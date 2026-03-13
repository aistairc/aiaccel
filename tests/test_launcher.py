from __future__ import annotations

import importlib
import os
from pathlib import Path
import sys

import aiaccel.launcher as launcher


def test_launcher_does_not_inject_cwd(tmp_path: Path) -> None:
    original_cwd = Path.cwd()
    original_sys_path = list(sys.path)
    try:
        os.chdir(tmp_path)
        sys.path[:] = [entry for entry in sys.path if entry != str(tmp_path)]

        reloaded = importlib.reload(launcher)
        assert str(tmp_path) not in sys.path
        assert callable(reloaded.main)
    finally:
        os.chdir(original_cwd)
        sys.path[:] = original_sys_path
        importlib.reload(launcher)
