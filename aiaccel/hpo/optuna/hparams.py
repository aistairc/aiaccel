# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from typing import Any, Generic, TypeVar

from collections.abc import Sequence
from dataclasses import dataclass

from optuna.trial import Trial

T = TypeVar("T")


@dataclass
class Hparam(Generic[T]):
    def __call__(self, trial: Trial, name: str) -> T:
        raise NotImplementedError


@dataclass
class Const(Hparam[T]):
    value: T

    def __call__(self, trial: Trial | None, name: str | None) -> T:
        return self.value


class FormattedList(list[Any]):
    def __str__(self) -> str:
        return " ".join(map(str, self))


@dataclass
class Vector(Hparam[FormattedList], Generic[T]):
    dim: int
    hparam: Hparam[T]

    def __call__(self, trial: Trial, name: str) -> FormattedList:
        return FormattedList([self.hparam(trial, f"{name}[{index}]") for index in range(self.dim)])


@dataclass
class Float(Hparam[float]):
    low: float
    high: float
    step: float | None = None
    log: bool = False

    def __call__(self, trial: Trial, name: str) -> float:
        return trial.suggest_float(name=name, low=self.low, high=self.high, step=self.step, log=self.log)


@dataclass
class Int(Hparam[int]):
    low: int
    high: int
    step: int = 1
    log: bool = False

    def __call__(self, trial: Trial, name: str) -> int:
        return trial.suggest_int(name=name, low=self.low, high=self.high, step=self.step, log=self.log)


@dataclass
class Categorical(Hparam[None | bool | int | float | str]):
    choices: Sequence[None | bool | int | float | str]

    def __call__(self, trial: Trial, name: str) -> None | bool | int | float | str:
        return trial.suggest_categorical(name=name, choices=self.choices)
