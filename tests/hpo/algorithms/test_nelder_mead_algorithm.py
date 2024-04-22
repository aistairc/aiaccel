from unittest.mock import patch

import numpy as np
import numpy.typing as npt
import pytest

from aiaccel.hpo.algorithms.nelder_mead_algorithm import (
    NelderMeadAlgorism,
    NelderMeadEmptyError,
    UnexpectedVerticesUpdateError,
)


@pytest.fixture
def search_space() -> dict[str, tuple[float, float]]:
    return {"x": (-5.0, 5.0), "y": (-5.0, 5.0)}


@pytest.fixture
def nm(search_space: dict[str, tuple[float, float]]) -> NelderMeadAlgorism:
    return NelderMeadAlgorism(search_space=search_space, block=False)


@pytest.fixture
def vertices() -> list[npt.NDArray[np.float64]]:
    return list(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))


@pytest.fixture
def values() -> list[float]:
    return [5.0, 3.0, 7.0]


class TestNelderMeadAlgorism:
    def test_waiting_for_result_empty(self, nm: NelderMeadAlgorism) -> None:
        nm.vertices, nm.values = [], []

        # queue is Empty
        result_value = next(nm._wait_for_result())
        assert result_value is None

    def test_waiting_for_results_empty(self, nm: NelderMeadAlgorism) -> None:
        nm.vertices, nm.values = [], []

        # queue is Empty
        result_value = next(nm._wait_for_results(2))
        assert result_value is None

    def test_waiting_for_result(self, nm: NelderMeadAlgorism) -> None:
        nm.vertices, nm.values = [], []

        # queue is not Empty
        expected_value = 1.0
        nm.put_value(np.zeros(2), expected_value)
        try:
            next(nm._wait_for_result())
            raise AssertionError()
        except StopIteration as e:
            result = e.value
        assert result == expected_value

    def test_waiting_for_results(self, nm: NelderMeadAlgorism) -> None:
        nm.vertices, nm.values = [], []

        # queue is not Empty
        expected_value1 = 1.0
        nm.put_value(np.zeros(2), expected_value1)
        expected_value2 = 2.0
        nm.put_value(np.zeros(2), expected_value2)
        try:
            next(nm._wait_for_results(2))
            raise AssertionError()
        except StopIteration as e:
            result_values = e.value[1]
        assert result_values == [expected_value1, expected_value2]

    def test_waiting_for_results_enqueue_update(self, nm: NelderMeadAlgorism) -> None:
        with patch(
            "aiaccel.hpo.samplers.nelder_mead_sampler.NelderMeadAlgorism._collect_enqueued_results"
        ) as mock_iter:
            mock_iter.side_effect = [([np.array([-1.0, 0.0])], [0.5])]

            nm.vertices, nm.values = [], []

            # queue is not Empty
            expected_value1 = 1.0
            nm.put_value(np.zeros(2), expected_value1)
            expected_value2 = 2.0
            nm.put_value(np.zeros(2), expected_value2)
            try:
                next(nm._wait_for_results(2))
                raise AssertionError()
            except UnexpectedVerticesUpdateError:
                assert True

    def test_collect_enqueued_results_empty(self, nm: NelderMeadAlgorism) -> None:
        # queue is Empty
        result_vertices, result_values = nm._collect_enqueued_results()
        assert result_vertices == []
        assert result_values == []

    def test_collect_enqueued_results(self, nm: NelderMeadAlgorism) -> None:
        # queue is not Empty
        expected_vertex1 = np.array([1.0, 2.0])
        expected_value1 = 1.0
        nm.put_value(expected_vertex1, expected_value1, enqueue=True)
        expected_vertex2 = np.array([3.0, 4.0])
        expected_value2 = 2.0
        nm.put_value(expected_vertex2, expected_value2, enqueue=True)

        result_vertices, result_values = nm._collect_enqueued_results()
        assert np.array_equal(result_vertices, [expected_vertex1, expected_vertex2])
        assert result_values == [expected_value1, expected_value2]

    def test_initialize(self, search_space: dict[str, tuple[float, float]], nm: NelderMeadAlgorism) -> None:
        for _ in range(len(search_space) + 1):
            random_xs = nm.get_vertex()
            for random_x, distribution in zip(random_xs, search_space.values(), strict=False):
                assert isinstance(random_x, float) and distribution[0] <= random_x <= distribution[1]
        try:
            nm.get_vertex()
            raise AssertionError()
        except NelderMeadEmptyError:
            assert True

    def test_initialize_enqueued1(self, search_space: dict[str, tuple[float, float]], nm: NelderMeadAlgorism) -> None:
        # enqueued
        enqueued_vertex = np.array([1.0, 2.0])
        enqueued_value = 1.0
        nm.put_value(enqueued_vertex, enqueued_value, enqueue=True)

        for _ in range(len(search_space)):
            random_xs = nm.get_vertex()
            for random_x, distribution in zip(random_xs, search_space.values(), strict=False):
                assert isinstance(random_x, float) and distribution[0] <= random_x <= distribution[1]

        assert np.array_equal(nm.vertices, [enqueued_vertex])
        assert nm.values == [enqueued_value]

        try:
            nm.get_vertex()
            raise AssertionError()
        except NelderMeadEmptyError:
            assert True

    def test_initialize_enqueued2(self, nm: NelderMeadAlgorism) -> None:
        # enqueued
        enqueud_vertices = [np.array([1.0, 2.0]), np.array([2.0, 3.0]), np.array([4.0, 5.0]), np.array([6.0, 7.0])]
        enqueud_values = [1.0, 4.0, 2.0, 3.0]
        expected_vertex = np.array([-1.0, 0.0])

        for enqueued_vertex, enqueued_value in zip(enqueud_vertices, enqueud_values, strict=False):
            nm.put_value(enqueued_vertex, enqueued_value, enqueue=True)

        x = nm.get_vertex()
        assert np.array_equal(x, expected_vertex)

    @pytest.mark.parametrize(
        "expected_results",  # (vertex, value, enqueue)
        [
            # reflect
            ([([-1.0, 0.0], None, False)]),
            # reflect -> reflect
            ([([-1.0, 0.0], 4.0, False), ([1.0, 2.0], None, False)]),
            # reflect -> expand -> fe < fr -> reflect
            ([([-1.0, 0.0], 2.0, False), ([-4.0, -3.0], 1.0, False), ([-2.0, -1.0], None, False)]),
            # reflect -> expand -> else (fe > fr) -> reflect
            ([([-1.0, 0.0], 2.0, False), ([-4.0, -3.0], 3.0, False), ([1.0, 2.0], None, False)]),
            # reflect -> outside_contract -> foc <= fr -> reflect
            ([([-1.0, 0.0], 6.0, False), ([0.5, 1.5], 5.5, False), ([3.5, 4.5], None, False)]),
            # reflect -> outside_contract -> shrink -> reflect
            (
                [
                    ([-1.0, 0.0], 6.0, False),
                    ([0.5, 1.5], 7.0, False),
                    ([2.0, 3.0], 1.0, False),
                    ([4.0, 5.0], 2.0, False),
                    ([3.0, 4.0], None, False),
                ]
            ),
            # reflect -> inside_contract -> fic < self.values[-1] -> reflect
            ([([-1.0, 0.0], 8.0, False), ([3.5, 4.5], 6.0, False), ([0.5, 1.5], None, False)]),
            # reflect -> inside_contract -> else (fic > self.values[-1]) -> shrink -> reflect
            (
                [
                    ([-1.0, 0.0], 8.0, False),
                    ([3.5, 4.5], 8.5, False),
                    ([2.0, 3.0], 1.0, False),
                    ([4.0, 5.0], 2.0, False),
                    ([3.0, 4.0], None, False),
                ]
            ),
            # reflect -> UnexpectedVerticesUpdateError -> reflect
            ([([-1.0, 0.0], 4.0, False), ([1.0, 3.0], 2.0, True), ([5.0, 7.0], None, False)]),
        ],
    )
    def test_compare_results(
        self,
        vertices: list[npt.NDArray[np.float64]],
        values: list[float],
        nm: NelderMeadAlgorism,
        expected_results: list[tuple[list[float], float, bool]],
    ) -> None:
        for vertex, value in zip(vertices, values, strict=False):
            nm.put_value(vertex, value, True)

        # main loop
        for expected_result in expected_results:
            expected_vertex, expected_value, enqueued = expected_result

            if not enqueued:
                x = nm.get_vertex()
                assert np.array_equal(x, expected_vertex)

            if expected_value is None:
                break

            nm.put_value(np.array(expected_vertex), expected_value, enqueued)
