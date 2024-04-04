from __future__ import annotations

import queue
import threading
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


class UpdateByEnqueue(Exception):
    def __init__(self, additional_vertices: list[np.ndarray], additional_values: list[float]) -> None:
        self.additional_vertices = additional_vertices
        self.additional_values = additional_values


class NelderMeadAlgorism:
    vertices: np.ndarray
    values: np.ndarray

    def __init__(
        self,
        search_space: dict[str, tuple[float, float]],
        coeff: NelderMeadCoefficient | None = None,
        rng: np.random.RandomState | None = None,
        block: bool = True,
        timeout: int | None = None
    ) -> None:
        self._search_space = search_space
        self.coeff = coeff if coeff is not None else NelderMeadCoefficient()
        self._rng = rng if rng is not None else np.random.RandomState()
        self.block = block
        self.timeout = timeout

        self.value_queue: queue.Queue[float] = queue.Queue()
        self.generator = iter(self._generator())
        self.lock = threading.Lock()

        self.enqueue_vertex_queue: queue.Queue[np.ndarray] = queue.Queue()
        self.enqueue_value_queue: queue.Queue[float] = queue.Queue()
        self.num_enqueued = 0

    def get_vertex(self) -> np.ndarray:
        with self.lock:
            return next(self.generator)

    def put_enqueue_vertex_queue(self, enqueue_params: dict[str, float]) -> np.ndarray:
        vertex = np.array([
            enqueue_params[param_name] if param_name in enqueue_params  # enqueue
            else self._rng.uniform(*param_distrbution)  # random
            for param_name, param_distrbution in self._search_space.items()
        ])
        self.enqueue_vertex_queue.put(vertex)
        self.num_enqueued += 1
        return vertex

    def get_enqueue_vertex_queue(self, block: bool = False, timeout: int | None = None) -> np.ndarray:
        return self.enqueue_vertex_queue.get(block=block, timeout=timeout)

    def put_value_queue(self, value: float) -> None:
        self.value_queue.put(value)

    def get_value_queue(self, block: bool = False, timeout: int | None = None) -> float:
        return self.value_queue.get(block=block, timeout=timeout)

    def put_enqueue_value_queue(self, value: float) -> None:
        self.enqueue_value_queue.put(value)

    def get_enqueue_value_queue(self, block: bool = False, timeout: int | None = None) -> float:
        return self.enqueue_value_queue.get(block=block, timeout=timeout)

    def _waiting_for_float(self) -> Generator[None, None, tuple[float, list[float]]]:
        result, enqueue_values = yield from self._waiting_for_list(1)
        return result[0], enqueue_values

    def _waiting_for_list(self, num_waiting: int) -> Generator[None, None, tuple[list[float], list[float]]]:
        # values of nelder mead vertices
        values: list[float] = []
        while len(values) < num_waiting:
            try:
                values.append(self.get_value_queue(self.block, self.timeout))
            except queue.Empty:
                yield None

        # values of enqueue vertices
        enqueue_values: list[float] = []
        while len(enqueue_values) < self.num_enqueued:
            try:
                enqueue_values.append(self.get_enqueue_value_queue(self.block, self.timeout))
            except queue.Empty:
                yield None

        self.num_enqueued = 0

        return values, enqueue_values

    def _initialization(self) -> Generator[np.ndarray, None, None]:
        dimension = len(self._search_space)
        lows, highs = zip(*self._search_space.values())

        # random
        num_generate_random = 0
        vertices_of_random = []
        while num_generate_random + self.num_enqueued < dimension + 1:
            yield (yield_vertex := self._rng.uniform(lows, highs, dimension))
            vertices_of_random.append(yield_vertex)
            num_generate_random += 1

        values_of_random, enqueue_values = yield from self._waiting_for_list(num_generate_random)

        # enqueue
        enqueue_vertices = []
        while not self.enqueue_vertex_queue.empty():
            enqueue_vertices.append(self.get_enqueue_vertex_queue())

        self.vertices = np.array(vertices_of_random + enqueue_vertices)
        self.values = np.array(values_of_random + enqueue_values)

        if len(self.vertices) > dimension + 1:
            order = np.argsort(self.values)
            self.vertices, self.values = self.vertices[order][:dimension + 1], self.values[order][:dimension + 1]

    def _validate_better_param_in_enqueue(
            self,
            vertices_before_processing: list[np.ndarray],
            values_before_processing: list[float],
            enqueue_values: list[float]
            ) -> None:
        enqueue_vertices = []
        while not self.enqueue_vertex_queue.empty():
            enqueue_vertices.append(self.get_enqueue_vertex_queue())

        worst_value = self.values[-1] if len(self.values) > 0 else None

        if worst_value is not None and len([v for v in enqueue_values if v < worst_value]) > 0:
            raise UpdateByEnqueue(
                vertices_before_processing + enqueue_vertices,
                values_before_processing + enqueue_values
                )

    def _generator(self) -> Generator[np.ndarray, None, None]:
        # initialization
        yield from self._initialization()

        # main loop
        shrink_requied = False
        while True:
            try:
                # sort self.vertices by their self.values
                order = np.argsort(self.values)
                self.vertices, self.values = self.vertices[order], self.values[order]

                # reflect
                yc = self.vertices[:-1].mean(axis=0)
                yield (yr := yc + self.coeff.r * (yc - self.vertices[-1]))

                fr, enqueue_values = yield from self._waiting_for_float()
                self._validate_better_param_in_enqueue([yr], [fr], enqueue_values)

                if self.values[0] <= fr < self.values[-2]:
                    self.vertices[-1], self.values[-1] = yr, fr
                elif fr < self.values[0]:  # expand
                    yield (ye := yc + self.coeff.e * (yc - self.vertices[-1]))

                    fe, enqueue_values = yield from self._waiting_for_float()
                    self._validate_better_param_in_enqueue([yr, ye], [fr, fe], enqueue_values)

                    self.vertices[-1], self.values[-1] = (ye, fe) if fe < fr else (yr, fr)

                elif self.values[-2] <= fr < self.values[-1]:  # outside contract
                    yield (yoc := yc + self.coeff.oc * (yc - self.vertices[-1]))

                    foc, enqueue_values = yield from self._waiting_for_float()
                    self._validate_better_param_in_enqueue([yr, yoc], [fr, foc], enqueue_values)

                    if foc <= fr:
                        self.vertices[-1], self.values[-1] = yoc, foc
                    else:
                        shrink_requied = True
                elif self.values[-1] <= fr:  # inside contract
                    yield (yic := yc + self.coeff.ic * (yc - self.vertices[-1]))

                    fic, enqueue_values = yield from self._waiting_for_float()
                    self._validate_better_param_in_enqueue([yr, yic], [fr, fic], enqueue_values)

                    if fic < self.values[-1]:
                        self.vertices[-1], self.values[-1] = yic, fic
                    else:
                        shrink_requied = True

                # shrink
                if shrink_requied:
                    self.vertices = self.vertices[0] + self.coeff.s * (self.vertices - self.vertices[0])
                    yield from self.vertices[1:]

                    self.values[1:], enqueue_values = yield from self._waiting_for_list(len(self.vertices[1:]))
                    shrink_requied = False

                    self._validate_better_param_in_enqueue([], [], enqueue_values)

            except UpdateByEnqueue as e:
                # recontract_simplex
                dimension = len(self._search_space)
                new_vertices = np.array(list(self.vertices) + e.additional_vertices)
                new_values = np.array(list(self.values) + e.additional_values)

                order = np.argsort(new_values)
                self.vertices, self.values = new_vertices[order][:dimension + 1], new_values[order][:dimension + 1]


class NelderMeadSampler(optuna.samplers.BaseSampler):
    def __init__(
        self,
        search_space: dict[str, tuple[float, float]],
        seed: int | None = None,
        rng: np.random.RandomState | None = None,
        coeff: NelderMeadCoefficient | None = None,
        parallel_enabled: bool = False,
    ) -> None:
        self._search_space = search_space
        _rng = rng if rng is not None else np.random.RandomState(seed) if seed is not None else None

        self.nm = NelderMeadAlgorism(
            search_space=self._search_space,
            coeff=coeff,
            rng=_rng,
            block=parallel_enabled,
            timeout=None
            )

        self.running_trial_id: list[int] = []
        self.enqueue_running_trial_id: list[int] = []
        self.result_stack: dict[int, float] = {}

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
        # TODO: system_attrs is deprecated.
        if "fixed_params" in trial.system_attrs:
            # enqueue_trial
            params = self.nm.put_enqueue_vertex_queue(trial.system_attrs["fixed_params"])
            self.enqueue_running_trial_id.append(trial._trial_id)
        else:
            while True:
                params = self.nm.get_vertex()
                if params is None:
                    raise RuntimeError("No more parallel calls to ask() are possible.")

                if all(low < x < high for x, (low, high) in zip(params, self._search_space.values())):
                    break
                else:
                    self.nm.put_value_queue(np.inf)
            self.running_trial_id.append(trial._trial_id)

        trial.set_user_attr("params", params)

    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        if trial.user_attrs["params"] is None:
            raise ValueError('trial.user_attrs["params"] is None')
        if param_name not in self._search_space:
            raise ValueError(f"The parameter name, {param_name}, is not found in the given search_space.")

        param_index = list(self._search_space.keys()).index(param_name)
        param_value = trial.user_attrs["params"][param_index]

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
            self.result_stack[trial._trial_id] = values[0]

            if len(self.running_trial_id) + len(self.enqueue_running_trial_id) == len(self.result_stack):
                for trial_id in self.running_trial_id:
                    self.nm.put_value_queue(self.result_stack[trial_id])
                for trial_id in self.enqueue_running_trial_id:
                    self.nm.put_enqueue_value_queue(self.result_stack[trial_id])
                self.running_trial_id = []
                self.result_stack = {}
                self.enqueue_running_trial_id = []
