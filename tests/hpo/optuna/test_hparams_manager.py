# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from typing import Any

import pytest

from aiaccel.hpo.optuna.hparams import Const, Float, Vector
from aiaccel.hpo.optuna.hparams_manager import HparamsManager


@pytest.mark.parametrize(
    ("value", "expected_type"),
    [
        (Const(value="fixed"), Const),
        ([0.0, 1.0], Float),
        ({"range": [0.0, 1.0], "dim": 2}, Vector),
        (3, Const),
        ({"a": 1}, Const),
        ({"range": [0, 1]}, Const),
        ({"dim": 2}, Const),
    ],
)
def test_init_selects_hparam_type(value: Any, expected_type: type[Any]) -> None:
    params = HparamsManager(param=value)

    assert isinstance(params.params["param"], expected_type)

    if callable(value):
        assert params.params["param"] is value
