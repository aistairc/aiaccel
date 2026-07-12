# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from __future__ import annotations

import importlib

import pytest

import aiaccel.hpo.apps.modelbridge as modelbridge
from aiaccel.hpo.apps.modelbridge import collect, evaluate, fit_model, prepare


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


def test_modelbridge_tools_are_not_importable_from_hpo_root() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("aiaccel.hpo.modelbridge.prepare")
