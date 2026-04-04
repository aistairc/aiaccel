# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

import optuna

from aiaccel.hpo.optuna.hparams import (
    Categorical,
    Const,
    Float,
    Int,
)


def test_const() -> None:
    const = Const(value=0.5)
    assert const(trial=None, name="x1") == 0.5


def test_float() -> None:
    suggest_float = Float(low=0.0, high=1.0, step=None, log=False)
    trial = optuna.create_study().ask()

    assert isinstance(suggest_float(trial=trial, name="x2"), float)


def test_int() -> None:
    suggest_int = Int(low=0, high=10, step=1, log=False)
    trial = optuna.create_study().ask()

    assert isinstance(suggest_int(trial=trial, name="x3"), int)


def test_categorical() -> None:
    suggest_categorical = Categorical(choices=[0, 1, 2])
    trial = optuna.create_study().ask()

    assert suggest_categorical(trial=trial, name="x4") in [0, 1, 2]


def test_discrete_uniform() -> None:
    suggest_discrete_uniform = Float(low=0.0, high=1.0, step=0.1)
    trial = optuna.create_study().ask()

    assert isinstance(suggest_discrete_uniform(trial=trial, name="x5"), float)


def test_log_uniform() -> None:
    suggest_log_uniform = Float(low=0.1, high=1.0, log=True)
    trial = optuna.create_study().ask()

    assert isinstance(suggest_log_uniform(trial=trial, name="x6"), float)
