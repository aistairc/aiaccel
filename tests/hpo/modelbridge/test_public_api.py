# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from __future__ import annotations

import importlib

import pytest

import aiaccel.modelbridge as modelbridge
from aiaccel.modelbridge import collect, evaluate, fit_model, prepare


def test_package_root_exports_modules() -> None:
    assert modelbridge.__all__ == [
        "collect",
        "evaluate",
        "fit_model",
        "main",
        "prepare",
    ]
    assert modelbridge.collect is collect
    assert modelbridge.evaluate is evaluate
    assert modelbridge.fit_model is fit_model
    assert modelbridge.prepare is prepare


def test_legacy_modelbridge_package_is_not_importable() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("aiaccel.hpo.apps.modelbridge.prepare")
