from __future__ import annotations

import string
from typing import Any

import numpy as np
from numpy.random import RandomState

from aiaccel.optimizer.value import Value

# constants for Nelder-Mead
# r: reflect
# e: expand
# ic: inside_contract
# oc: outside_contract
# s: shrink
coef: dict[str, float] = {"r": 1.0, "ic": -0.5, "oc": 0.5, "e": 2.0, "s": 0.5}

name_rng = RandomState(None)


class Vertex:
    def __init__(self, xs: np.ndarray[Any, Any], value: Any = None) -> None:
        self.xs: np.ndarray[Any, Any] = xs
        self.value: Any = value
        self.id: str = self.generate_random_name()

    def generate_random_name(self, length: int = 10) -> str:
        if length < 1:
            raise ValueError("Name length should be greater than 0.")
        rands = [name_rng.choice(list(string.ascii_letters + string.digits))[0] for _ in range(length)]
        return "".join(rands)

    @property
    def coordinates(self) -> np.ndarray[Any, Any]:
        return self.xs

    def set_value(self, value: Any) -> None:
        self.value = value

    def set_id(self, vertex_id: str) -> None:
        self.id = vertex_id

    def set_new_id(self) -> None:
        self.id = self.generate_random_name()

    def set_xs(self, xs: np.ndarray[Any, Any]) -> None:
        self.xs = xs

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
        except TypeError:
            raise TypeError("Unsupported operand type for +")

    def __sub__(self, other: Vertex | Any) -> Vertex:  # Subtract -
        if isinstance(other, Vertex):
            new_vertex = Vertex(self.coordinates - other.coordinates)
            return new_vertex
        try:
            new_vertex = Vertex(self.xs - other)
            return new_vertex
        except TypeError:
            raise TypeError("Unsupported operand type for -")

    def __mul__(self, other: Any) -> Vertex:  # Multiply *
        new_vertex = Vertex(self.xs * other)
        new_vertex.set_id(self.id)
        return new_vertex

    def __eq__(self, other: Vertex | Any) -> bool:  # Equal ==
        if isinstance(other, Vertex):
            return self.value == other.value
        try:
            return self.value == other
        except TypeError:
            raise TypeError("Unsupported operand type for ==")

    def __ne__(self, other: Vertex | Any) -> bool:  # Not Equal !=
        if isinstance(other, Vertex):
            return self.value != other.value
        try:
            return self.value != other
        except TypeError:
            raise TypeError("Unsupported operand type for !=")

    def __lt__(self, other: Vertex | Any) -> bool:  # Less Than <
        if isinstance(other, Vertex):
            return self.value < other.value
        try:
            return self.value < other
        except TypeError:
            raise TypeError("Unsupported operand type for <")

    def __le__(self, other: Vertex | Any) -> bool:  # Less Than or Equal <=
        if isinstance(other, Vertex):
            return self.value <= other.value
        try:
            return self.value <= other
        except TypeError:
            raise TypeError("Unsupported operand type for <=")

    def __gt__(self, other: Vertex | Any) -> bool:  # Greater Than >
        if isinstance(other, Vertex):
            return self.value > other.value
        try:
            return self.value > other
        except TypeError:
            raise TypeError("Unsupported operand type for >")

    def __ge__(self, other: Vertex | Any) -> bool:  # Greater Than or Equal >=
        if isinstance(other, Vertex):
            return self.value >= other.value
        try:
            return self.value >= other
        except TypeError:
            raise TypeError("Unsupported operand type for >=")


class Simplex:
    def __init__(self, simplex_coordinates: np.ndarray[Any, Any]) -> None:
        self.n_dim = simplex_coordinates.shape[1]
        self.vertices: list[Vertex] = []
        self.centroid: Vertex = Vertex(np.array([0.0] * self.n_dim))
        self.coef = coef
        for xs in simplex_coordinates:
            self.vertices.append(Vertex(xs))

    def get_simplex_coordinates(self) -> np.ndarray[Any, Any]:
        return np.array([v.xs for v in self.vertices])

    def set_value(self, vertex_id: str, value: Any) -> bool:
        for v in self.vertices:
            if v.id == vertex_id:
                v.set_value(value)
                return True
        return False

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


class NelderMead:
    def __init__(self, initial_parameters: np.ndarray[Any, Any]) -> None:
        self.simplex: Simplex = Simplex(initial_parameters)
        self.state = "initialize"
        self.store = Store()
        self.waits = {
            "initialize": self.simplex.n_dim + 1,
            "shrink": self.simplex.n_dim,
            "expand": 1,
            "inside_contract": 1,
            "outside_contract": 1,
            "reflect": 1,
        }
        self.n_waits = self.waits[self.state]

    def get_n_waits(self) -> int:
        return self.n_waits

    def get_n_dim(self) -> int:
        return self.simplex.n_dim

    def get_state(self) -> str:
        return self.state

    def change_state(self, state: str) -> None:
        self.state = state

    def set_value(self, vertex_id: str, value: float | int) -> None:
        self.simplex.set_value(vertex_id, value)

    def initialize(self) -> list[Vertex]:
        self.n_waits = self.waits["initialize"]
        return self.simplex.vertices

    def after_initialize(self, yis: list[Value]) -> None:
        for y in yis:
            self.simplex.set_value(y.id, y.value)
        self.change_state("reflect")

    def reflect(self) -> Vertex:
        self.n_waits = self.waits["reflect"]
        self.simplex.calc_centroid()
        self.store.r = self.simplex.reflect()
        return self.store.r

    def after_reflect(self, yr: Value) -> None:
        self.store.r.set_value(yr.value)
        if self.simplex.vertices[0] <= self.store.r < self.simplex.vertices[-2]:
            self.simplex.vertices[-1].update(self.store.r.coordinates, self.store.r.value)
            self.change_state("reflect")
        elif self.store.r < self.simplex.vertices[0]:
            self.change_state("expand")
        elif self.simplex.vertices[-2] <= self.store.r < self.simplex.vertices[-1]:
            self.change_state("outside_contract")
        elif self.simplex.vertices[-1] <= self.store.r:
            self.change_state("inside_contract")
        else:
            self.change_state("reflect")

    def expand(self) -> Vertex:
        self.n_waits = self.waits["expand"]
        self.store.e = self.simplex.expand()
        return self.store.e

    def after_expand(self, ye: Value) -> None:
        self.store.e.set_value(ye.value)
        if self.store.e < self.store.r:
            self.simplex.vertices[-1].update(self.store.e.coordinates, self.store.e.value)
        else:
            self.simplex.vertices[-1].update(self.store.r.coordinates, self.store.r.value)
        self.change_state("reflect")

    def inside_contract(self) -> Vertex:
        self.n_waits = self.waits["inside_contract"]
        self.store.ic = self.simplex.inside_contract()
        return self.store.ic

    def after_inside_contract(self, yic: Value) -> None:
        self.store.ic.set_value(yic.value)
        if self.store.ic < self.simplex.vertices[-1]:
            self.simplex.vertices[-1].update(self.store.ic.coordinates, self.store.ic.value)
            self.change_state("reflect")
        else:
            self.change_state("shrink")

    def outside_contract(self) -> Vertex:
        self.n_waits = self.waits["outside_contract"]
        self.store.oc = self.simplex.outside_contract()
        return self.store.oc

    def after_outside_contract(self, yoc: Value) -> None:
        self.store.oc.set_value(yoc.value)
        if self.store.oc <= self.store.r:
            self.simplex.vertices[-1].update(self.store.oc.coordinates, self.store.oc.value)
            self.change_state("reflect")
        else:
            self.change_state("shrink")

    def shrink(self) -> list[Vertex]:
        self.n_waits = self.waits["shrink"]
        self.store.s = self.simplex.shrink()
        return self.store.s

    def aftter_shrink(self, yss: list[Value]) -> None:
        for ys in yss:
            if not self.simplex.set_value(ys.id, ys.value):
                raise BaseException("Error: vertex is not found.")
        self.change_state("reflect")

    def search(self) -> list[Vertex]:
        if self.state == "initialize":
            xs = self.initialize()
            self.change_state("initialize_pending")
            return xs

        elif self.state == "initialize_pending":
            return []

        elif self.state == "reflect":
            x = self.reflect()
            self.change_state("reflect_pending")
            return [x]

        elif self.state == "reflect_pending":
            return []

        elif self.state == "expand":
            x = self.expand()
            self.change_state("expand_pending")
            return [x]

        elif self.state == "expand_pending":
            return []

        elif self.state == "inside_contract":
            x = self.inside_contract()
            self.change_state("inside_contract_pending")
            return [x]

        elif self.state == "inside_contract_pending":
            return []

        elif self.state == "outside_contract":
            x = self.outside_contract()
            self.change_state("outside_contract_pending")
            return [x]

        elif self.state == "outside_contract_pending":
            return []

        elif self.state == "shrink":
            xs = self.shrink()
            self.change_state("shrink_pending")
            return xs[1:]

        elif self.state == "shrink_pending":
            return []

        else:
            raise ValueError("Unknown state.")
