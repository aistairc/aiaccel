import queue
import threading
from collections.abc import Generator
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass
class NelderMeadCoefficient:
    r: float = 1.0
    ic: float = -0.5
    oc: float = 0.5
    e: float = 2.0
    s: float = 0.5


class NelderMeadEmpty(Exception):
    pass


@dataclass
class UnexpectedVerticesUpdate(Exception):
    updated_vertices: npt.NDArray[np.float64]
    updated_values: npt.NDArray[np.float64]


class NelderMeadAlgorism:
    vertices: npt.NDArray[np.float64]
    values: npt.NDArray[np.float64]

    def __init__(
        self,
        search_space: dict[str, tuple[float, float]],
        coeff: NelderMeadCoefficient | None = None,
        rng: np.random.RandomState | None = None,
        block: bool = True,
        timeout: int | None = None,
    ) -> None:
        self._search_space = search_space
        self.coeff = coeff if coeff is not None else NelderMeadCoefficient()

        self._rng = rng if rng is not None else np.random.RandomState()

        self.generator = iter(self._generator())
        self.lock = threading.Lock()

        self.results: queue.Queue[tuple[npt.NDArray[np.float64], float, bool]] = queue.Queue()

        self.block = block
        self.timeout = timeout

        self.simplex_size = len(self._search_space) + 1
        self.num_enqueued = 0

    def get_vertex(self) -> npt.NDArray[np.float64]:
        with self.lock:
            vertex = next(self.generator)

        if vertex is None:
            raise NelderMeadEmpty("Cannot generate new vertex now. Maybe get_vertex is called in parallel.")

        return vertex

    def enqueued(self) -> None:
        self.num_enqueued += 1

    def put_value(
        self,
        vertex: npt.NDArray[np.float64],
        value: float,
        enqueue: bool = False,
    ) -> None:
        self.results.put((vertex, value, enqueue))

    def _waiting_for_result(
        self,
    ) -> Generator[None, None, float]:
        values = yield from self._waiting_for_results(1)
        return values[0]

    def _waiting_for_results(
        self,
        num_waiting: int,
    ) -> Generator[None, None, list[float]]:
        # collect results
        vertices, values = list[npt.NDArray[np.float64]](), list[float]()
        enqueued_vertices, enqueued_values = list[npt.NDArray[np.float64]](), list[float]()
        while len(values) < num_waiting + self.num_enqueued:
            try:
                vertex, value, enqueue = self.results.get(block=self.block, timeout=self.timeout)
                if enqueue:
                    enqueued_vertices.append(vertex)
                    enqueued_values.append(value)
                    self.num_enqueued -= 1
                else:
                    vertices.append(vertex)
                    values.append(value)
            except queue.Empty:
                yield None

        # check if enqueued vertices change ordering
        if len(enqueued_values) > 0 and max(list(self.values)) > min(enqueued_values):
            new_vertices = np.array(list(self.vertices) + vertices + enqueued_vertices)
            new_values = np.array(list(self.values) + values + enqueued_values)

            raise UnexpectedVerticesUpdate(new_vertices, new_values)

        return values

    def _initialization(
        self,
    ) -> Generator[npt.NDArray[np.float64] | None, None, None]:
        lows, highs = zip(*self._search_space.values(), strict=False)
        vertices, values = list[npt.NDArray[np.float64]](), list[float]()
        num_random_vertices = 0

        while len(vertices) + num_random_vertices < self.simplex_size + self.num_enqueued:
            random_vertex = self._rng.uniform(lows, highs, len(self._search_space))
            num_random_vertices += 1
            yield random_vertex

        while len(values) < self.simplex_size + self.num_enqueued:
            try:
                vertex, value, _ = self.results.get(block=self.block)
                vertices.append(vertex)
                values.append(value)
            except queue.Empty:
                yield None

        self.vertices = np.array(vertices)
        self.values = np.array(values)
        self.num_enqueued = 0

    def _generator(self) -> Generator[npt.NDArray[np.float64] | None, None, None]:
        # initialization
        yield from self._initialization()

        # main loop
        shrink_requied = False
        while True:
            try:
                # sort self.vertices by their self.values
                order = np.argsort(self.values)[: self.simplex_size]
                self.vertices, self.values = self.vertices[order], self.values[order]

                # reflect
                yc = self.vertices[:-1].mean(axis=0)
                yield (yr := yc + self.coeff.r * (yc - self.vertices[-1]))

                fr = yield from self._waiting_for_result()

                if self.values[0] <= fr < self.values[-2]:
                    self.vertices[-1], self.values[-1] = yr, fr
                elif fr < self.values[0]:  # expand
                    yield (ye := yc + self.coeff.e * (yc - self.vertices[-1]))

                    fe = yield from self._waiting_for_result()

                    self.vertices[-1], self.values[-1] = (ye, fe) if fe < fr else (yr, fr)

                elif self.values[-2] <= fr < self.values[-1]:  # outside contract
                    yield (yoc := yc + self.coeff.oc * (yc - self.vertices[-1]))

                    foc = yield from self._waiting_for_result()

                    if foc <= fr:
                        self.vertices[-1], self.values[-1] = yoc, foc
                    else:
                        shrink_requied = True
                elif self.values[-1] <= fr:  # inside contract
                    yield (yic := yc + self.coeff.ic * (yc - self.vertices[-1]))

                    fic = yield from self._waiting_for_result()

                    if fic < self.values[-1]:
                        self.vertices[-1], self.values[-1] = yic, fic
                    else:
                        shrink_requied = True

                # shrink
                if shrink_requied:
                    self.vertices = self.vertices[0] + self.coeff.s * (self.vertices - self.vertices[0])
                    yield from self.vertices[1:]

                    self.values[1:] = yield from self._waiting_for_results(len(self.vertices[1:]))
                    shrink_requied = False

            except UnexpectedVerticesUpdate as e:
                self.vertices, self.values = e.updated_vertices, e.updated_values
