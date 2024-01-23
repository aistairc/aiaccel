from __future__ import annotations

import dataclasses
import itertools
import queue
from collections.abc import Generator
from typing import Any, Sequence

import numpy as np
import optuna
from optuna import distributions
from optuna._transform import _SearchSpaceTransform
from optuna.distributions import BaseDistribution
from optuna.samplers._lazy_random_state import LazyRandomState
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
        self._rng: LazyRandomState = LazyRandomState(seed)
        self.vertices: np.ndarray[float, float] = np.array([])
        self.values: np.ndarray[float, float] = np.array([])
        self.coef: Coef = coef
        self.num_iterations: int = num_iterations
        self.num_initial_create_trial: int = 0
        self.vertex_queue: queue.Queue[float] = queue.Queue()

    def initial(self) -> Generator[np.ndarray[float, float], None, None]:
        initial_params = []
        for _ in range(self.dimension + 1):
            initial_param = []
            for param_name, param_distribution in self._search_space.items():
                search_space = {
                    param_name: optuna.distributions.FloatDistribution(param_distribution[0], param_distribution[1])
                    }
                trans = _SearchSpaceTransform(search_space)
                trans_params = self._rng.rng.uniform(trans.bounds[:, 0], trans.bounds[:, 1])
                initial_param.append(trans.untransform(trans_params)[param_name])
            self.num_initial_create_trial += 1
            initial_params.append(np.array(initial_param))
            yield np.array(initial_param), self.dimension + 1 - self.num_initial_create_trial
        self.vertices = np.array(initial_params)
        self.values = np.array([self.vertex_queue.get() for _ in range(self.dimension + 1)])

    def order_by(self) -> None:
        order = np.argsort(self.values)

        self.vertices = self.vertices[order]
        self.values = self.values[order]

    def shrink(self) -> Generator[np.ndarray[float, float], None, None]:
        for i in range(1, len(self.vertices)):
            yield (ysh := self.vertices[0] + self.coef.s * (self.vertices[i] - self.vertices[0])), 0
            self.vertices[i] = ysh
        for i in range(1, len(self.vertices)):
            self.values[i] = self.vertex_queue.get()

    def __iter__(self) -> Generator[np.ndarray[float, float], None, None]:
        yield from self.initial()
        for _ in range(self.num_iterations) if self.num_iterations > 0 else itertools.count():
            shrink_requied = False
            self.order_by()
            self.yc = self.vertices[:-1].mean(axis=0)
            yield (yr := self.yc + self.coef.r * (self.yc - self.vertices[-1])), 0
            fr = self.vertex_queue.get()

            if self.values[0] <= fr < self.values[-2]:
                self.vertices[-1] = yr
                self.values[-1] = fr
            elif fr < self.values[0]:
                yield (ye := self.yc + self.coef.e * (self.yc - self.vertices[-1])), 0
                fe = self.vertex_queue.get()
                if fe < fr:
                    self.vertices[-1] = ye
                    self.values[-1] = fe
                else:
                    self.vertices[-1] = yr
                    self.values[-1] = fr
            elif self.values[-2] <= fr < self.values[-1]:
                yield (yoc := self.yc + self.coef.oc * (self.yc - self.vertices[-1])), 0
                foc = self.vertex_queue.get()
                if foc <= fr:
                    self.vertices[-1] = yoc
                    self.values[-1] = foc
                else:
                    shrink_requied = True
            elif self.values[-1] <= fr:
                yield (yic := self.yc + self.coef.ic * (self.yc - self.vertices[-1])), 0
                fic = self.vertex_queue.get()
                if fic < self.values[-1]:
                    shrink_requied = False
                    self.vertices[-1] = yic
                    self.values[-1] = fic
                else:
                    shrink_requied = True
            if shrink_requied:
                yield from self.shrink()


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

        self.NelderMead: NelderMeadAlgorism = NelderMeadAlgorism(self._search_space, Coef(**coef), seed, num_iterations)
        self.generator = self.NelderMead.__iter__()
        self.parallel_limit: int = len(search_space) + 1
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
        if self.num_running_trial == 0 or self.parallel_limit > 0:
            trial.user_attrs["Coordinate"], self.parallel_limit = next(self.generator)
            if self.is_within_range(trial.user_attrs["Coordinate"]):
                trial.user_attrs["ParallelEnabled"] = self.parallel_limit > 0
                self.num_running_trial += 1
            else:
                self.NelderMead.vertex_queue.put(np.inf)
                self.before_trial(study, trial)
        else:
            trial.user_attrs["Coordinate"] = None
            trial.user_attrs["ParallelEnabled"] = False

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
                    self.NelderMead.vertex_queue.put(value)
                self.stack = {}
