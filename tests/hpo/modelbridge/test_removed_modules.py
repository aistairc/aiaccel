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


@pytest.mark.parametrize(
    "module_name",
    [
        "aiaccel.hpo.modelbridge.layout",
        "aiaccel.hpo.modelbridge.contracts",
        "aiaccel.hpo.modelbridge.storage",
        "aiaccel.hpo.modelbridge.modeling",
        "aiaccel.hpo.modelbridge.role_target",
        "aiaccel.hpo.modelbridge.toolkit",
        "aiaccel.hpo.modelbridge.results",
        "aiaccel.hpo.modelbridge.step_registry",
    ],
)
def test_rev01_removed_internal_modules_are_not_importable(module_name: str) -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module_name)
