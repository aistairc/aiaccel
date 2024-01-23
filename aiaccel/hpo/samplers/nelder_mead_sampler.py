from __future__ import annotations

import dataclasses
import itertools
import queue
from collections.abc import Generator
from typing import Any, Sequence

import numpy as np
import optuna
from optuna import distributions
from optuna.distributions import BaseDistribution
from optuna.study import Study
from optuna.trial import FrozenTrial, TrialState


@dataclasses.dataclass
class Coef:
    r: float = 1.0
    ic: float = -0.5
    oc: float = 0.5
    e: float = 2.0
    s: float = 0.5


class NelderMeadAlgorism:
    def __init__(self,
                 search_space: dict[str, list[float]],
                 coef: Coef,
                 seed: int | None = None,
                 num_iterations: int = 0) -> None:
        self._search_space = search_space
        self.dimension: int = len(search_space)
        # self._rng: LazyRandomState = LazyRandomState(seed)
        self.vertices: np.ndarray[float, float] = np.array([])
        self.values: np.ndarray[float, float] = np.array([])
        self.coef: Coef = coef
        self.num_iterations: int = num_iterations
        self.num_initial_create_trial: int = 0
        self.vertex_queue: queue.Queue[float] = queue.Queue()
        np.random.seed(seed)

    def initial(self) -> Generator[np.ndarray[float, float], None, None]:
        # initial_params = []
        initial_params = np.random.uniform(
                [param_distribution[0] for param_distribution in self._search_space.values()],
                [param_distribution[1] for param_distribution in self._search_space.values()],
                [self.dimension + 1, self.dimension])

        for initial_param in initial_params:
            self.num_initial_create_trial += 1
            yield np.array(initial_param)
        self.vertices, self.values = (
            np.array(initial_params),
            np.array([self.vertex_queue.get() for _ in range(self.dimension + 1)])
            )

    def shrink(self) -> Generator[np.ndarray[float, float], None, None]:
        for i in range(1, len(self.vertices)):
            yield (ysh := self.vertices[0] + self.coef.s * (self.vertices[i] - self.vertices[0]))
            self.vertices[i] = ysh
        for i in range(1, len(self.vertices)):
            self.values[i] = self.vertex_queue.get()

    def __iter__(self) -> Generator[np.ndarray[float, float], None, None]:
        # initial
        yield from self.initial()
        # nelder_mead
        shrink_requied = False
        for _ in range(self.num_iterations) if self.num_iterations > 0 else itertools.count():
            # sort vertices by their values
            order = np.argsort(self.values)
            self.vertices, self.values = self.vertices[order], self.values[order]
            # reflect
            yc = self.vertices[:-1].mean(axis=0)
            yield (yr := yc + self.coef.r * (yc - self.vertices[-1]))
            fr = self.vertex_queue.get()

            if self.values[0] <= fr < self.values[-2]:
                self.vertices[-1], self.values[-1] = yr, fr
            elif fr < self.values[0]:
                # expand
                yield (ye := yc + self.coef.e * (yc - self.vertices[-1]))
                fe = self.vertex_queue.get()

                self.vertices[-1], self.values[-1] = (ye, fe) if fe < fr else (yr, fr)
            elif self.values[-2] <= fr < self.values[-1]:
                # outside contract
                yield (yoc := yc + self.coef.oc * (yc - self.vertices[-1]))
                foc = self.vertex_queue.get()
                if foc <= fr:
                    self.vertices[-1], self.values[-1] = yoc, foc
                else:
                    shrink_requied = True
            elif self.values[-1] <= fr:
                # inside contract
                yield (yic := yc + self.coef.ic * (yc - self.vertices[-1]))
                fic = self.vertex_queue.get()
                if fic < self.values[-1]:
                    self.vertices[-1], self.values[-1] = yic, fic
                else:
                    shrink_requied = True
            if shrink_requied:
                # shrink
                yield from self.shrink()
                shrink_requied = False


class NelderMeadSampler(optuna.samplers.BaseSampler):
    def __init__(self,
                 search_space: dict[str, list[float]],
                 seed: int | None = None,
                 num_iterations: int = 0,
                 **coef: float
                 ) -> None:
        self.param_names = []  # パラメータの順序を記憶
        self._search_space = {}
        for param_name, param_distribution in sorted(search_space.items()):
            self.param_names.append(param_name)
            self._search_space[param_name] = list(param_distribution)

        self.nelder_mead: NelderMeadAlgorism = NelderMeadAlgorism(
            self._search_space, Coef(**coef), seed, num_iterations
            )
        self.generator = self.nelder_mead.__iter__()
        self.num_running_trial: int = 0

        self.stack: dict[int, float] = {}

    def is_within_range(self, coordinates: np.ndarray[float, float]) -> bool:
        return all(not (co < ss[0] or ss[1] < co) for ss, co in zip(self._search_space.values(), coordinates))

    def infer_relative_search_space(self, study: Study, trial: FrozenTrial) -> dict[str, BaseDistribution]:
        return {}

    def sample_relative(
        self, study: Study, trial: FrozenTrial, search_space: dict[str, BaseDistribution]
    ) -> dict[str, Any]:
        return {}

    def before_trial(self, study: Study, trial: FrozenTrial) -> None:
        if self.num_running_trial == 0:
            trial.user_attrs["Coordinate"] = next(self.generator)
            if self.is_within_range(trial.user_attrs["Coordinate"]):
                self.num_running_trial += 1
            else:
                self.nelder_mead.vertex_queue.put(np.inf)
                self.before_trial(study, trial)
        else:
            trial.user_attrs["Coordinate"] = None

    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: distributions.BaseDistribution,
    ) -> Any:
        if trial.user_attrs["Coordinate"] is None:
            raise ValueError('trial.user_attrs["Coordinate"] is None')
        param_index = self.param_names.index(param_name)
        param_value = trial.user_attrs["Coordinate"][param_index]

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
                for value in [item[1] for item in sorted(self.stack.items())]:
                    self.nelder_mead.vertex_queue.put(value)
                self.stack = {}
