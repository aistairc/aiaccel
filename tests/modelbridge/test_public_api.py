# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from __future__ import annotations

import importlib

import pytest

import aiaccel.modelbridge.apps as modelbridge_apps
from aiaccel.modelbridge.apps import collect, evaluate, fit_model, prepare


def test_app_package_exports_modules() -> None:
    assert modelbridge_apps.__all__ == [
        "collect",
        "evaluate",
        "fit_model",
        "main",
        "prepare",
    ]
    assert modelbridge_apps.collect is collect
    assert modelbridge_apps.evaluate is evaluate
    assert modelbridge_apps.fit_model is fit_model
    assert modelbridge_apps.prepare is prepare


def test_tools_are_not_importable_from_the_legacy_root_package() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("aiaccel.modelbridge.prepare")


def test_legacy_hpo_modelbridge_package_is_not_importable() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("aiaccel.hpo.apps.modelbridge.prepare")
