from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import aiaccel.launcher as launcher


def test_launcher_injects_cwd(tmp_path) -> None:
    original_cwd = Path.cwd()
    original_sys_path = list(sys.path)
    tmp_str = str(tmp_path)
    try:
        os.chdir(tmp_path)
        sys.path[:] = [entry for entry in sys.path if entry != tmp_str]

        reloaded = importlib.reload(launcher)
        assert tmp_str in sys.path
        assert callable(reloaded.main)
    finally:
        os.chdir(original_cwd)
        sys.path[:] = original_sys_path
        importlib.reload(launcher)
