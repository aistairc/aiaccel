from __future__ import annotations

import queue
import warnings
from collections.abc import Generator
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import optuna
from optuna.distributions import BaseDistribution
from optuna.study import Study
from optuna.trial import FrozenTrial, TrialState


@dataclass
class NelderMeadCoefficient:
    r: float = 1.0
    ic: float = -0.5
    oc: float = 0.5
    e: float = 2.0
    s: float = 0.5


class NelderMeadAlgorism:
    vertices: np.ndarray[Any, Any]
    values: np.ndarray[Any, Any]

    def __init__(
        self,
        search_space: dict[str, list[float]],
        coeff: NelderMeadCoefficient | None = None,
        seed: int | None = None,
    ) -> None:
        self._search_space = search_space
        self.coeff = coeff if coeff is not None else NelderMeadCoefficient()

        self.dimension = len(search_space)

        self.value_queue: queue.Queue[float] = queue.Queue()
        self._rng: np.random.RandomState = np.random.RandomState(seed)
        self.is_ready: bool = True

    def __iter__(self) -> Generator[np.ndarray[Any, Any], None, None]:
        # initialization
        lows, highs = zip(*self._search_space.values())
        self.vertices = self._rng.uniform(lows, highs, (self.dimension + 1, self.dimension))

        yield from iter(self.vertices[:-1])
        self.is_ready = False
        yield self.vertices[-1]
        self.values = np.array([self.value_queue.get() for _ in range(len(self.vertices))])

        # main loop
        shrink_requied = False
        while True:
            # sort self.vertices by their self.values
            order = np.argsort(self.values)
            self.vertices, self.values = self.vertices[order], self.values[order]

            # reflect
            yc = self.vertices[:-1].mean(axis=0)
            yield (yr := yc + self.coeff.r * (yc - self.vertices[-1]))

            fr = self.value_queue.get()

            if self.values[0] <= fr < self.values[-2]:
                self.vertices[-1], self.values[-1] = yr, fr
            elif fr < self.values[0]:  # expand
                yield (ye := yc + self.coeff.e * (yc - self.vertices[-1]))

                fe = self.value_queue.get()

                self.vertices[-1], self.values[-1] = (ye, fe) if fe < fr else (yr, fr)
            elif self.values[-2] <= fr < self.values[-1]:  # outside contract
                yield (yoc := yc + self.coeff.oc * (yc - self.vertices[-1]))
                foc = self.value_queue.get()

                if foc <= fr:
                    self.vertices[-1], self.values[-1] = yoc, foc
                else:
                    shrink_requied = True
            elif self.values[-1] <= fr:  # inside contract
                yield (yic := yc + self.coeff.ic * (yc - self.vertices[-1]))
                fic = self.value_queue.get()

                if fic < self.values[-1]:
                    self.vertices[-1], self.values[-1] = yic, fic
                else:
                    shrink_requied = True

            # shrink
            if shrink_requied:
                self.vertices = self.vertices[0] + self.coeff.s * (self.vertices - self.vertices[0])
                self.is_ready = True
                yield from iter(self.vertices[1:-1])
                self.is_ready = False
                yield self.vertices[-1]

                self.values[1:] = [self.value_queue.get() for _ in range(len(self.vertices) - 1)]

                shrink_requied = False


class NelderMeadSampler(optuna.samplers.BaseSampler):
    def __init__(
        self,
        search_space: dict[str, list[float]],
        seed: int | None = None,
        coeff: NelderMeadCoefficient | None = None,
    ) -> None:
        self.param_names = []  # パラメータの順序を記憶
        self._search_space = {}
        for param_name, param_distribution in sorted(search_space.items()):
            self.param_names.append(param_name)
            self._search_space[param_name] = list(param_distribution)

        self.nm = NelderMeadAlgorism(self._search_space, coeff, seed)
        self.nm_generator = iter(self.nm)
        self.num_running_trial = 0

        self.running_trial_id: list[int] = []
        self.stack: dict[int, float] = {}

    def is_within_range(self, coordinates: np.ndarray[Any, Any]) -> bool:
        return all(low < x < high for x, (low, high) in zip(coordinates, self._search_space.values()))

    def infer_relative_search_space(self, study: Study, trial: FrozenTrial) -> dict[str, BaseDistribution]:
        return {}

    def sample_relative(
        self,
        study: Study,
        trial: FrozenTrial,
        search_space: dict[str, BaseDistribution],
    ) -> dict[str, Any]:
        return {}

    def before_trial(self, study: Study, trial: FrozenTrial) -> None:
        # Raise RuntimeError if cannot output parameters. (include parallel execution)
        # TODO: support parallel execution
        if not self.nm.is_ready and self.num_running_trial > 0:
            raise RuntimeError("Cannot output parameters.")
        # Raise RuntimeError if use study.enqueue_trial()
        # TODO: support study.enqueue_trial()
        if "fixed_params" in trial.system_attrs:
            raise RuntimeError("NelderMeadSampler does not support enqueue_trial.")
        trial.set_user_attr("Coordinate", next(self.nm_generator))
        trial.set_user_attr("IsReady", self.nm.is_ready)
        if self.is_within_range(trial.user_attrs["Coordinate"]):
            self.num_running_trial += 1
            self.running_trial_id.append(trial._trial_id)
        else:
            self.nm.value_queue.put(np.inf)
            self.before_trial(study, trial)

    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        if trial.user_attrs["Coordinate"] is None:
            raise ValueError('trial.user_attrs["Coordinate"] is None')
        if param_name not in self._search_space:
            raise ValueError(f"The parameter name, {param_name}, is not found in the given search_space.")
        param_index = self.param_names.index(param_name)
        param_value = trial.user_attrs["Coordinate"][param_index]
        contains = param_distribution._contains(param_distribution.to_internal_repr(param_value))
        if not contains:
            warnings.warn(
                f"The value `{param_value}` is out of range of the parameter `{param_name}`. "
                f"The value will be used but the actual distribution is: `{param_distribution}`.", stacklevel=2)
        return param_value

    def after_trial(
        self,
        study: Study,
        trial: FrozenTrial,
        state: TrialState,
        values: Sequence[float] | None,
    ) -> None:
        if isinstance(values, list):
            self.num_running_trial -= 1
            self.stack[trial._trial_id] = values[0]
            if self.num_running_trial == 0:
                for trial_id in self.running_trial_id:
                    self.nm.value_queue.put(self.stack[trial_id])
                self.running_trial_id = []
                self.stack = {}
