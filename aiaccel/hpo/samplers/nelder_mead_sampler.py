from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Union

import numpy as np
import optuna
from enum import Enum
from optuna import distributions
from optuna._transform import _SearchSpaceTransform
from optuna.distributions import BaseDistribution
from optuna.samplers._lazy_random_state import LazyRandomState
from optuna.study import Study
from optuna.trial import FrozenTrial, TrialState

coef: dict[str, float] = {"r": 1.0, "ic": -0.5, "oc": 0.5, "e": 2.0, "s": 0.5}


class Vertex:
    def __init__(self, xs: np.ndarray[Any, Any], value: Any = None) -> None:
        self.xs: np.ndarray[Any, Any] = xs
        self.value: Any = value

    @property
    def coordinates(self) -> np.ndarray[Any, Any]:
        return self.xs

    def set_value(self, value: Any) -> None:
        self.value = value

    def update(self, xs: np.ndarray[Any, Any], value: Any) -> None:
        self.xs = xs
        self.value = value

    def __add__(self, other: Vertex | Any) -> Vertex:  # Add +
        if isinstance(other, Vertex):
            new_vertex = Vertex(self.coordinates + other.coordinates)
            return new_vertex
        try:
            new_vertex = Vertex(self.coordinates + other)
            return new_vertex
        except TypeError as err:
            raise TypeError("Unsupported operand type for +") from err

    def __sub__(self, other: Vertex | Any) -> Vertex:  # Subtract -
        if isinstance(other, Vertex):
            new_vertex = Vertex(self.coordinates - other.coordinates)
            return new_vertex
        try:
            new_vertex = Vertex(self.xs - other)
            return new_vertex
        except TypeError as err:
            raise TypeError("Unsupported operand type for -") from err

    def __mul__(self, other: Any) -> Vertex:  # Multiply *
        new_vertex = Vertex(self.xs * other)
        return new_vertex

    def __eq__(self, other: Vertex | Any) -> bool:  # Equal ==
        if isinstance(other, Vertex):
            return self.value == other.value
        try:
            return self.value == other
        except TypeError as err:
            raise TypeError("Unsupported operand type for ==") from err

    def __ne__(self, other: Vertex | Any) -> bool:  # Not Equal !=
        if isinstance(other, Vertex):
            return self.value != other.value
        try:
            return self.value != other
        except TypeError as err:
            raise TypeError("Unsupported operand type for !=") from err

    def __lt__(self, other: Vertex | Any) -> bool:  # Less Than <
        if isinstance(other, Vertex):
            return self.value < other.value
        try:
            return self.value < other
        except TypeError as err:
            raise TypeError("Unsupported operand type for <") from err

    def __le__(self, other: Vertex | Any) -> bool:  # Less Than or Equal <=
        if isinstance(other, Vertex):
            return self.value <= other.value
        try:
            return self.value <= other
        except TypeError as err:
            raise TypeError("Unsupported operand type for <=") from err

    def __gt__(self, other: Vertex | Any) -> bool:  # Greater Than >
        if isinstance(other, Vertex):
            return self.value > other.value
        try:
            return self.value > other
        except TypeError as err:
            raise TypeError("Unsupported operand type for >") from err

    def __ge__(self, other: Vertex | Any) -> bool:  # Greater Than or Equal >=
        if isinstance(other, Vertex):
            return self.value >= other.value
        try:
            return self.value >= other
        except TypeError as err:
            raise TypeError("Unsupported operand type for >=") from err


class Simplex:
    def __init__(self) -> None:
        self.vertices: list[Vertex] = []
        self.coef: dict[str, float] = coef

    def add_vertices(self, v: Vertex) -> None:
        self.vertices.append(v)

    def num_of_vertices(self) -> int:
        return len(self.vertices)

    def get_simplex_coordinates(self) -> np.ndarray[Any, Any]:
        return np.array([v.xs for v in self.vertices])

    def order_by(self) -> None:
        order = np.argsort([v.value for v in self.vertices])
        self.vertices = [self.vertices[i] for i in order]

    def calc_centroid(self) -> None:
        self.order_by()
        xs = self.get_simplex_coordinates()
        self.centroid = Vertex(xs[:-1].mean(axis=0))

    def reflect(self) -> Vertex:
        xr = self.centroid + ((self.centroid - self.vertices[-1]) * self.coef["r"])
        return xr

    def expand(self) -> Vertex:
        xe = self.centroid + ((self.centroid - self.vertices[-1]) * self.coef["e"])
        return xe

    def inside_contract(self) -> Vertex:
        xic = self.centroid + ((self.centroid - self.vertices[-1]) * self.coef["ic"])
        return xic

    def outside_contract(self) -> Vertex:
        xoc = self.centroid + ((self.centroid - self.vertices[-1]) * self.coef["oc"])
        return xoc

    def shrink(self) -> list[Vertex]:
        for i in range(1, len(self.vertices)):
            self.vertices[i] = self.vertices[0] + (self.vertices[i] - self.vertices[0]) * self.coef["s"]
        return self.vertices


class Store:
    def __init__(self) -> None:
        self.r: Vertex | Any = None  # reflect
        self.e: Vertex | Any = None  # expand
        self.ic: Vertex | Any = None  # inside_contract
        self.oc: Vertex | Any = None  # outside_contract
        self.s: list[Vertex] | Any = None  # shrink


class NelderMeadState(Enum):
    Initial = 0
    Reflect = 1
    Expand = 2
    InsideContract = 3
    OutsideContract = 4
    Shrink = 5


class NelderMeadSampler(optuna.samplers.BaseSampler):
    def __init__(self, search_space: Mapping[str, Sequence[Union[float, float]]], seed: int | None = None) -> None:
        self._rng: LazyRandomState = LazyRandomState(seed)
        self.dimension: int = len(search_space)

        self._search_space = {}
        self.param_names = []  # パラメータの順序を記憶
        for param_name, param_values in sorted(search_space.items()):
            self._search_space[param_name] = list(param_values)
            self.param_names.append(param_name)

        self.simplex: Simplex = Simplex()
        self.state: NelderMeadState = NelderMeadState.Initial
        self.store: Store = Store()
        self.x: np.ndarray[Any, Any] = np.array([])
        self.xs: list[Vertex] = []

    def infer_relative_search_space(self, study: Study, trial: FrozenTrial) -> Dict[str, BaseDistribution]:
        return {}

    def sample_relative(
        self, study: Study, trial: FrozenTrial, search_space: Dict[str, BaseDistribution]
    ) -> Dict[str, Any]:
        return {}

    def after_initialize(self) -> None:
        self.state = NelderMeadState.Reflect

    def reflect(self) -> Vertex:
        self.simplex.calc_centroid()
        self.store.r = self.simplex.reflect()
        return self.store.r

    def after_reflect(self, yr: float) -> None:
        self.store.r.set_value(yr)
        if self.simplex.vertices[0] <= self.store.r < self.simplex.vertices[-2]:
            self.simplex.vertices[-1].update(self.store.r.coordinates, self.store.r.value)
            self.state = NelderMeadState.Reflect
        elif self.store.r < self.simplex.vertices[0]:
            self.state = NelderMeadState.Expand
        elif self.simplex.vertices[-2] <= self.store.r < self.simplex.vertices[-1]:
            self.state = NelderMeadState.OutsideContract
        elif self.simplex.vertices[-1] <= self.store.r:
            self.state = NelderMeadState.InsideContract
        else:
            self.state = NelderMeadState.Reflect

    def expand(self) -> Vertex:
        self.store.e = self.simplex.expand()
        return self.store.e

    def after_expand(self, ye: float) -> None:
        self.store.e.set_value(ye)
        if self.store.e < self.store.r:
            self.simplex.vertices[-1].update(self.store.e.coordinates, self.store.e.value)
        else:
            self.simplex.vertices[-1].update(self.store.r.coordinates, self.store.r.value)
        self.state = NelderMeadState.Reflect

    def inside_contract(self) -> Vertex:
        self.store.ic = self.simplex.inside_contract()
        return self.store.ic

    def after_inside_contract(self, yic: float) -> None:
        self.store.ic.set_value(yic)
        if self.store.ic < self.simplex.vertices[-1]:
            self.simplex.vertices[-1].update(self.store.ic.coordinates, self.store.ic.value)
            self.state = NelderMeadState.Reflect
        else:
            self.state = NelderMeadState.Shrink

    def outside_contract(self) -> Vertex:
        self.store.oc = self.simplex.outside_contract()
        return self.store.oc

    def after_outside_contract(self, yoc: float) -> None:
        self.store.oc.set_value(yoc)
        if self.store.oc <= self.store.r:
            self.simplex.vertices[-1].update(self.store.oc.coordinates, self.store.oc.value)
            self.state = NelderMeadState.Reflect
        else:
            self.state = NelderMeadState.Shrink

    def shrink(self) -> list[Vertex]:
        self.store.s = self.simplex.shrink()
        new_simplex = Simplex()
        new_simplex.add_vertices(self.store.s[0])
        self.simplex = new_simplex
        return self.store.s

    def after_shrink(self) -> None:
        self.state = NelderMeadState.Reflect

    def is_within_range(self, coordinates: np.ndarray[Any, Any]) -> bool:
        for ss, co in zip(self._search_space.values(), coordinates):
            if co < ss[0] or ss[1] < co:
                return False
        return True

    def get_next_coordinates(self) -> None:
        if self.state == NelderMeadState.Shrink:
            if len(self.xs) == 0:
                self.xs = [v.coordinates for v in self.shrink()[1:]]
            self.x = self.xs[0]
        else:
            if self.state == NelderMeadState.Initial:
                return
            elif self.state == NelderMeadState.Reflect:
                self.x = self.reflect().coordinates
            elif self.state == NelderMeadState.Expand:
                self.x = self.expand().coordinates
            elif self.state == NelderMeadState.InsideContract:
                self.x = self.inside_contract().coordinates
            elif self.state == NelderMeadState.OutsideContract:
                self.x = self.outside_contract().coordinates

            if not self.is_within_range(self.x):
                self.set_objective(self.x, np.inf)
                self.get_next_coordinates()

    def before_trial(self, study: Study, trial: FrozenTrial) -> None:
        self.get_next_coordinates()

    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: distributions.BaseDistribution,
    ) -> Any:
        if self.state == NelderMeadState.Initial:
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

    def set_objective(self, coordinates: np.ndarray[Any, Any], objective: float) -> None:
        if self.state == NelderMeadState.Initial:
            self.simplex.add_vertices(Vertex(coordinates, objective))
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
            self.simplex.add_vertices(Vertex(coordinates, objective))
            self.xs.pop(0)
            if len(self.xs) == 0:
                self.after_shrink()

    def after_trial(
        self,
        study: Study,
        trial: FrozenTrial,
        state: TrialState,
        values: Optional[Sequence[float]],
    ) -> None:
        coordinates = np.array([trial.params[name] for name in self.param_names])
        if isinstance(values, list):
            self.set_objective(coordinates, values[0])
