from typing import Generic, TypeVar

from collections.abc import Sequence
from dataclasses import dataclass

from optuna.trial import Trial

T = TypeVar("T")


@dataclass
class Suggest(Generic[T]):
    name: str

    def __call__(self, _: Trial) -> T:
        raise NotImplementedError


@dataclass
class Const(Suggest[T]):
    value: T

    def __call__(self, _: Trial | None) -> T:
        return self.value


@dataclass
class SuggestFloat(Suggest[float]):
    name: str
    low: float
    high: float
    step: float | None = None
    log: bool = False

    def __call__(self, trial: Trial) -> float:
        return trial.suggest_float(self.name, self.low, self.high, step=self.step, log=self.log)


@dataclass
class SuggestInt(Suggest[int]):
    name: str
    low: int
    high: int
    step: int = 1
    log: bool = False

    def __call__(self, trial: Trial) -> int:
        return trial.suggest_int(name=self.name, low=self.low, high=self.high, step=self.step, log=self.log)


@dataclass
class SuggestCategorical(Suggest[None | bool | int | float | str]):
    name: str
    choices: Sequence[None | bool | int | float | str]

    def __call__(self, trial: Trial) -> None | bool | int | float | str:
        return trial.suggest_categorical(self.name, self.choices)


@dataclass
class SuggestDiscreteUniform(Suggest[float]):
    name: str
    low: float
    high: float
    q: float

    def __call__(self, trial: Trial) -> float:
        return trial.suggest_float(self.name, self.low, self.high, step=self.q)


@dataclass
class SuggestLogUniform(Suggest[float]):
    name: str
    low: float
    high: float

    def __call__(self, trial: Trial) -> float:
        return trial.suggest_loguniform(self.name, self.low, self.high)


@dataclass
class SuggestUniform(Suggest[float]):
    name: str
    low: float
    high: float

    def __call__(self, trial: Trial) -> float:
        return trial.suggest_float(self.name, self.low, self.high)
