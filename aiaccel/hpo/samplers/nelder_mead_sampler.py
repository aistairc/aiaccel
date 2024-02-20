from __future__ import annotations

import queue
import threading
import warnings
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
    vertices: np.ndarray
    values: np.ndarray

    def __init__(
        self,
        search_space: dict[str, list[float]],
        coeff: NelderMeadCoefficient | None = None,
        rng: np.random.RandomState | None = None,
    ) -> None:
        self._search_space = search_space
        self.coeff = coeff if coeff is not None else NelderMeadCoefficient()

        self.dimension = len(search_space)

        self.vertex_queue: queue.Queue[np.ndarray] = queue.Queue()
        self.value_queue: queue.Queue[float] = queue.Queue()
        self._rng = rng if rng is not None else np.random.RandomState()

        self.lock = threading.Lock()
        self.lock.acquire()
        self.num_running_trials: int = 0

    def put_vertices(self, vertices: list[np.ndarray]) -> None:
        self.num_running_trials = len(vertices)
        for vertex in vertices:
            self.vertex_queue.put(vertex)
        self.lock.release()

    def put_values(self, values: list[float]) -> None:
        self.num_running_trials -= len(values)
        for value in values:
            self.value_queue.put(value)
        if self.num_running_trials == 0:
            self.lock.acquire()

    def generator(self) -> None:
        # initialization
        lows, highs = zip(*self._search_space.values())
        self.vertices = self._rng.uniform(lows, highs, (self.dimension + 1, self.dimension))

        self.put_vertices(self.vertices)
        self.values = np.array([self.value_queue.get() for _ in range(len(self.vertices))])

        # main loop
        shrink_requied = False
        while True:
            # sort self.vertices by their self.values
            order = np.argsort(self.values)
            self.vertices, self.values = self.vertices[order], self.values[order]

            # reflect
            yc = self.vertices[:-1].mean(axis=0)
            self.put_vertices([yr := yc + self.coeff.r * (yc - self.vertices[-1])])

            fr = self.value_queue.get()

            if self.values[0] <= fr < self.values[-2]:
                self.vertices[-1], self.values[-1] = yr, fr
            elif fr < self.values[0]:  # expand
                self.put_vertices([ye := yc + self.coeff.e * (yc - self.vertices[-1])])

                fe = self.value_queue.get()

                self.vertices[-1], self.values[-1] = (ye, fe) if fe < fr else (yr, fr)
            elif self.values[-2] <= fr < self.values[-1]:  # outside contract
                self.put_vertices([yoc := yc + self.coeff.oc * (yc - self.vertices[-1])])

                foc = self.value_queue.get()

                if foc <= fr:
                    self.vertices[-1], self.values[-1] = yoc, foc
                else:
                    shrink_requied = True
            elif self.values[-1] <= fr:  # inside contract
                self.put_vertices([yic := yc + self.coeff.ic * (yc - self.vertices[-1])])

                fic = self.value_queue.get()

                if fic < self.values[-1]:
                    self.vertices[-1], self.values[-1] = yic, fic
                else:
                    shrink_requied = True

            # shrink
            if shrink_requied:
                self.vertices = self.vertices[0] + self.coeff.s * (self.vertices - self.vertices[0])
                self.put_vertices(self.vertices[1:])

                self.values[1:] = [self.value_queue.get() for _ in range(len(self.vertices) - 1)]

                shrink_requied = False


class NelderMeadSampler(optuna.samplers.BaseSampler):
    def __init__(
        self,
        search_space: dict[str, list[float]],
        seed: int | None = None,
        coeff: NelderMeadCoefficient | None = None,
    ) -> None:
        self._search_space = {name: list(dist) for name, dist in search_space.items()}  # Memorise parameter order.

        self.nm = NelderMeadAlgorism(self._search_space, coeff, np.random.RandomState(seed))
        self.nm_generator = threading.Thread(target=self.nm.generator, daemon=True)
        self.nm_generator.start()

        self.running_trial_id: list[int] = []
        self.stack: dict[int, float] = {}
        self.is_ready = True

    def is_within_range(self, coordinates: np.ndarray) -> bool:
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
        # TODO: support parallel execution
        # TODO: support study.enqueue_trial()
        # TODO: system_attrs is deprecated.
        if "fixed_params" in trial.system_attrs:
            raise RuntimeError("NelderMeadSampler does not support enqueue_trial.")

        trial.set_user_attr("Coordinate", self._get_cooridinate())
        self.running_trial_id.append(trial._trial_id)

    def _get_cooridinate(self) -> np.ndarray:
        self.nm.lock.acquire()
        try:
            cooridinate = self.nm.vertex_queue.get(block=False)
        except queue.Empty as e:
            raise e
        finally:
            self.nm.lock.release()

        if self.is_within_range(cooridinate):
            return cooridinate
        else:
            self.nm.put_values([np.inf])
            return self._get_cooridinate()

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

        param_index = list(self._search_space.keys()).index(param_name)
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
            self.stack[trial._trial_id] = values[0]

            if len(self.running_trial_id) == len(self.stack):
                self.nm.put_values([self.stack[trial_id] for trial_id in self.running_trial_id])
                self.running_trial_id = []
                self.stack = {}
                self.is_ready = True
