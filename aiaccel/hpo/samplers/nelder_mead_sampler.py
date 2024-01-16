from __future__ import annotations

import dataclasses
from enum import Enum
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


class NelderMeadState(Enum):
    Initial = 0
    Reflect = 1
    Expand = 2
    InsideContract = 3
    OutsideContract = 4
    Shrink = 5


class Simplex:
    def __init__(self, coef: Coef) -> None:
        self.vertices: np.ndarray[float, float] = np.array([])
        self.values: np.ndarray[float, float] = np.array([])
        self.coef: Coef = coef

    def add_vertices(self, vertex: np.ndarray[float, float], value: float | None = None) -> None:
        if len(self.vertices) == 0:
            self.vertices = np.array([vertex])
        else:
            self.vertices = np.append(self.vertices, [vertex], axis=0)
        self.values = np.append(self.values, value)

    def update_vertices(self, index: int, vertex: np.ndarray[float, float], value: float | None = None) -> None:
        self.vertices[index] = vertex
        self.values[index] = value

    def num_of_vertices(self) -> int:
        return len(self.vertices)

    def order_by(self) -> None:
        order = np.argsort(self.values)

        self.vertices = self.vertices[order]
        self.values = self.values[order]

    def calc_centroid(self) -> None:
        self.order_by()
        self.centroid = self.vertices[:-1].mean(axis=0)

    def reflect(self) -> np.ndarray[float, float]:
        xr = self.centroid + ((self.centroid - self.vertices[-1]) * self.coef.r)
        return xr

    def expand(self) -> np.ndarray[float, float]:
        xe = self.centroid + ((self.centroid - self.vertices[-1]) * self.coef.e)
        return xe

    def inside_contract(self) -> np.ndarray[float, float]:
        xic = self.centroid + ((self.centroid - self.vertices[-1]) * self.coef.ic)
        return xic

    def outside_contract(self) -> np.ndarray[float, float]:
        xoc = self.centroid + ((self.centroid - self.vertices[-1]) * self.coef.oc)
        return xoc

    def shrink(self) -> np.ndarray[float, float]:
        shrinked_vertices = np.array([self.vertices[0]])
        for i in range(1, len(self.vertices)):
            shrinked_vertex = self.vertices[0] + (self.vertices[i] - self.vertices[0]) * self.coef.s
            shrinked_vertices = np.append(shrinked_vertices, [shrinked_vertex], axis=0)

        self.vertices = self.vertices[:1]
        self.values = self.values[:1]

        return shrinked_vertices


class NelderMeadAlgorism:
    def __init__(self,
                 search_space: dict[str, list[float]],
                 **coef: float) -> None:
        self.dimension: int = len(search_space)

        self._search_space = {}
        for param_name, param_values in sorted(search_space.items()):
            self._search_space[param_name] = list(param_values)

        self.simplex: Simplex = Simplex(Coef(**coef))
        self.state: NelderMeadState = NelderMeadState.Initial
        self.xs: np.ndarray[float, float] = np.array([])

        self.r: np.ndarray[float, float] | None = None  # reflect
        self.e: np.ndarray[float, float] | None = None  # expand
        self.ic: np.ndarray[float, float] | None = None  # inside_contract
        self.oc: np.ndarray[float, float] | None = None  # outside_contract
        self.s: np.ndarray[float, float] | None = None  # shrink
        self.yr: float | None = None  # value of reflect vertex

    def after_initialize(self) -> None:
        self.state = NelderMeadState.Reflect

    def reflect(self) -> np.ndarray[float, float]:
        self.simplex.calc_centroid()
        self.r = self.simplex.reflect()
        return self.r

    def after_reflect(self, yr: float) -> None:
        self.yr = yr
        if self.simplex.values[0] <= yr < self.simplex.values[-2]:
            self.simplex.update_vertices(-1, self.r, yr)
            self.state = NelderMeadState.Reflect
        elif yr < self.simplex.values[0]:
            self.state = NelderMeadState.Expand
        elif self.simplex.values[-2] <= yr < self.simplex.values[-1]:
            self.state = NelderMeadState.OutsideContract
        elif self.simplex.values[-1] <= yr:
            self.state = NelderMeadState.InsideContract
        else:
            self.state = NelderMeadState.Reflect

    def expand(self) -> np.ndarray[float, float]:
        self.e = self.simplex.expand()
        return self.e

    def after_expand(self, ye: float) -> None:
        if self.yr is None:
            raise TypeError
        if ye < self.yr:
            self.simplex.update_vertices(-1, self.e, ye)
        else:
            self.simplex.update_vertices(-1, self.r, self.yr)
        self.state = NelderMeadState.Reflect

    def inside_contract(self) -> np.ndarray[float, float]:
        self.ic = self.simplex.inside_contract()
        return self.ic

    def after_inside_contract(self, yic: float) -> None:
        if yic < self.simplex.values[-1]:
            self.simplex.update_vertices(-1, self.ic, yic)
            self.state = NelderMeadState.Reflect
        else:
            self.state = NelderMeadState.Shrink

    def outside_contract(self) -> np.ndarray[float, float]:
        self.oc = self.simplex.outside_contract()
        return self.oc

    def after_outside_contract(self, yoc: float) -> None:
        if self.yr is None:
            raise TypeError
        if yoc <= self.yr:
            self.simplex.update_vertices(-1, self.oc, yoc)
            self.state = NelderMeadState.Reflect
        else:
            self.state = NelderMeadState.Shrink

    def shrink(self) -> np.ndarray[float, float]:
        self.s = self.simplex.shrink()
        return self.s

    def after_shrink(self) -> None:
        self.state = NelderMeadState.Reflect

    def is_within_range(self, coordinates: np.ndarray[float, float]) -> bool:
        return all(not (co < ss[0] or ss[1] < co) for ss, co in zip(self._search_space.values(), coordinates))

    def get_next_coordinates(self) -> np.ndarray[float, float] | None:
        if self.state == NelderMeadState.Initial:
            return None
        elif self.state == NelderMeadState.Shrink:
            if len(self.xs) == 0:
                self.xs = self.shrink()[1:]
            return self.xs[0]
        else:
            if self.state == NelderMeadState.Reflect:
                x = self.reflect()
            elif self.state == NelderMeadState.Expand:
                x = self.expand()
            elif self.state == NelderMeadState.InsideContract:
                x = self.inside_contract()
            elif self.state == NelderMeadState.OutsideContract:
                x = self.outside_contract()

            if not self.is_within_range(x):
                self.set_objective(x, np.inf)
                return self.get_next_coordinates()
            else:
                return x

    def set_objective(self, coordinates: np.ndarray[float, float], objective: float) -> None:
        if self.state == NelderMeadState.Initial:
            self.simplex.add_vertices(coordinates, objective)
            if self.simplex.num_of_vertices() == self.dimension + 1:
                self.after_initialize()
        elif self.state == NelderMeadState.Reflect:
            self.after_reflect(objective)
        elif self.state == NelderMeadState.Expand:
            self.after_expand(objective)
        elif self.state == NelderMeadState.InsideContract:
            self.after_inside_contract(objective)
        elif self.state == NelderMeadState.OutsideContract:
            self.after_outside_contract(objective)
        elif self.state == NelderMeadState.Shrink:
            self.simplex.add_vertices(coordinates, objective)
            self.xs = np.delete(self.xs, 0, axis=0)
            if len(self.xs) == 0:
                self.after_shrink()


class NelderMeadSampler(optuna.samplers.BaseSampler):
    def __init__(self,
                 search_space: dict[str, list[float]],
                 seed: int | None = None,
                 **coef: float
                 ) -> None:
        self._rng: LazyRandomState = LazyRandomState(seed)

        self.param_names = []  # パラメータの順序を記憶
        for param_name in sorted(search_space.keys()):
            self.param_names.append(param_name)

        self.NelderMead: NelderMeadAlgorism = NelderMeadAlgorism(search_space, **coef)
        self.x: np.ndarray[float, float] = np.array([])

    def infer_relative_search_space(self, study: Study, trial: FrozenTrial) -> dict[str, BaseDistribution]:
        return {}

    def sample_relative(
        self, study: Study, trial: FrozenTrial, search_space: dict[str, BaseDistribution]
    ) -> dict[str, Any]:
        return {}

    def before_trial(self, study: Study, trial: FrozenTrial) -> None:
        self.x = self.NelderMead.get_next_coordinates()

    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: distributions.BaseDistribution,
    ) -> Any:
        if self.NelderMead.state == NelderMeadState.Initial:
            # initial random search
            search_space = {param_name: param_distribution}
            trans = _SearchSpaceTransform(search_space)
            trans_params = self._rng.rng.uniform(trans.bounds[:, 0], trans.bounds[:, 1])

            return trans.untransform(trans_params)[param_name]
        else:
            # nelder-mead
            param_index = self.param_names.index(param_name)
            param_value = self.x[param_index]

            return param_value

    def after_trial(
        self,
        study: Study,
        trial: FrozenTrial,
        state: TrialState,
        values: Sequence[float] | None,
    ) -> None:
        coordinates = np.array([trial.params[name] for name in self.param_names])
        if isinstance(values, list):
            self.NelderMead.set_objective(coordinates, values[0])
