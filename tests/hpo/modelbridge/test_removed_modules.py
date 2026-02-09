from __future__ import annotations

import importlib
from pathlib import Path

import pytest


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def test_ops_module_is_removed() -> None:
    assert not (_repo_root() / "aiaccel" / "hpo" / "modelbridge" / "ops.py").exists()
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("aiaccel.hpo.modelbridge.ops")


def test_makefile_module_is_removed() -> None:
    assert not (_repo_root() / "aiaccel" / "hpo" / "modelbridge" / "makefile.py").exists()
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("aiaccel.hpo.modelbridge.makefile")
