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


def test_waiting_for_result_empty(nm: NelderMeadAlgorism) -> None:
    nm.vertices, nm.values = [], []

    # queue is Empty
    result_value = next(nm._wait_for_result())
    assert result_value is None


def test_waiting_for_results_empty(nm: NelderMeadAlgorism) -> None:
    nm.vertices, nm.values = [], []

    # queue is Empty
    result_value = next(nm._wait_for_results(2))
    assert result_value is None


def test_waiting_for_result(nm: NelderMeadAlgorism) -> None:
    nm.vertices, nm.values = [], []

    # queue is not Empty
    expected_value = 1.0
    nm.put_value(np.zeros(2), expected_value)
    with pytest.raises(StopIteration) as e:
        next(nm._wait_for_result())
    assert e.value.value == expected_value


def test_waiting_for_results(nm: NelderMeadAlgorism) -> None:
    nm.vertices, nm.values = [], []

    # queue is not Empty
    expected_value1 = 1.0
    nm.put_value(np.zeros(2), expected_value1)
    expected_value2 = 2.0
    nm.put_value(np.zeros(2), expected_value2)
    with pytest.raises(StopIteration) as e:
        next(nm._wait_for_results(2))
    result_values = e.value.value[1]
    assert result_values == [expected_value1, expected_value2]


def test_waiting_for_results_enqueue_update(nm: NelderMeadAlgorism) -> None:
    with patch("aiaccel.hpo.samplers.nelder_mead_sampler.NelderMeadAlgorism._collect_enqueued_results") as mock_iter:
        mock_iter.side_effect = [([np.array([-1.0, 0.0])], [0.5])]

        nm.vertices, nm.values = [], []

        # queue is not Empty
        expected_value1 = 1.0
        nm.put_value(np.zeros(2), expected_value1)
        expected_value2 = 2.0
        nm.put_value(np.zeros(2), expected_value2)
        with pytest.raises(UnexpectedVerticesUpdateError):
            next(nm._wait_for_results(2))


def test_collect_enqueued_results_empty(nm: NelderMeadAlgorism) -> None:
    # queue is Empty
    result_vertices, result_values = nm._collect_enqueued_results()
    assert result_vertices == []
    assert result_values == []


def test_collect_enqueued_results(nm: NelderMeadAlgorism) -> None:
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


def test_initialize(search_space: dict[str, tuple[float, float]], nm: NelderMeadAlgorism) -> None:
    for _ in range(len(search_space) + 1):
        random_xs = nm.get_vertex()
        for random_x, distribution in zip(random_xs, search_space.values(), strict=False):
            assert isinstance(random_x, float) and distribution[0] <= random_x <= distribution[1]

    with pytest.raises(NelderMeadEmptyError):
        nm.get_vertex()


def test_initialize_enqueued1(search_space: dict[str, tuple[float, float]], nm: NelderMeadAlgorism) -> None:
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

    with pytest.raises(NelderMeadEmptyError):
        nm.get_vertex()


def test_initialize_enqueued2(nm: NelderMeadAlgorism) -> None:
    # enqueued
    enqueud_vertices = [np.array([1.0, 2.0]), np.array([2.0, 3.0]), np.array([4.0, 5.0]), np.array([6.0, 7.0])]
    enqueud_values = [1.0, 4.0, 2.0, 3.0]
    expected_vertex = np.array([-1.0, 0.0])

    for enqueued_vertex, enqueued_value in zip(enqueud_vertices, enqueud_values, strict=False):
        nm.put_value(enqueued_vertex, enqueued_value, enqueue=True)

    x = nm.get_vertex()
    assert np.array_equal(x, expected_vertex)


@pytest.mark.parametrize(
    "expected_results",
    [
        pytest.param([{"vertex": [-1.0, 0.0], "value": None, "enqueue": False}], id="reflect"),
        pytest.param(
            [
                {"vertex": [-1.0, 0.0], "value": 4.0, "enqueue": False},
                {"vertex": [1.0, 2.0], "value": None, "enqueue": False},
            ],
            id="reflect -> reflect",
        ),
        pytest.param(
            [
                {"vertex": [-1.0, 0.0], "value": 2.0, "enqueue": False},
                {"vertex": [-4.0, -3.0], "value": 1.0, "enqueue": False},
                {"vertex": [-2.0, -1.0], "value": None, "enqueue": False},
            ],
            id="reflect -> expand -> fe < fr -> reflect",
        ),
        pytest.param(
            [
                {"vertex": [-1.0, 0.0], "value": 2.0, "enqueue": False},
                {"vertex": [-4.0, -3.0], "value": 3.0, "enqueue": False},
                {"vertex": [1.0, 2.0], "value": None, "enqueue": False},
            ],
            id="reflect -> expand -> else (fe > fr) -> reflect",
        ),
        pytest.param(
            [
                {"vertex": [-1.0, 0.0], "value": 6.0, "enqueue": False},
                {"vertex": [0.5, 1.5], "value": 5.5, "enqueue": False},
                {"vertex": [3.5, 4.5], "value": None, "enqueue": False},
            ],
            id="reflect -> outside_contract -> foc <= fr -> reflect",
        ),
        pytest.param(
            [
                {"vertex": [-1.0, 0.0], "value": 6.0, "enqueue": False},
                {"vertex": [0.5, 1.5], "value": 7.0, "enqueue": False},
                {"vertex": [2.0, 3.0], "value": 1.0, "enqueue": False},
                {"vertex": [4.0, 5.0], "value": 2.0, "enqueue": False},
                {"vertex": [3.0, 4.0], "value": None, "enqueue": False},
            ],
            id="reflect -> outside_contract -> shrink -> reflect",
        ),
        pytest.param(
            [
                {"vertex": [-1.0, 0.0], "value": 8.0, "enqueue": False},
                {"vertex": [3.5, 4.5], "value": 6.0, "enqueue": False},
                {"vertex": [0.5, 1.5], "value": None, "enqueue": False},
            ],
            id="reflect -> inside_contract -> fic < self.values[-1] -> reflect",
        ),
        pytest.param(
            [
                {"vertex": [-1.0, 0.0], "value": 8.0, "enqueue": False},
                {"vertex": [3.5, 4.5], "value": 8.5, "enqueue": False},
                {"vertex": [2.0, 3.0], "value": 1.0, "enqueue": False},
                {"vertex": [4.0, 5.0], "value": 2.0, "enqueue": False},
                {"vertex": [3.0, 4.0], "value": None, "enqueue": False},
            ],
            id="reflect -> inside_contract -> else (fic > self.values[-1]) -> shrink -> reflect",
        ),
        pytest.param(
            [
                {"vertex": [-1.0, 0.0], "value": 4.0, "enqueue": False},
                {"vertex": [1.0, 3.0], "value": 2.0, "enqueue": True},
                {"vertex": [5.0, 7.0], "value": None, "enqueue": False},
            ],
            id="reflect -> UnexpectedVerticesUpdateError -> reflect",
        ),
    ],
)
def test_compare_results(
    vertices: list[npt.NDArray[np.float64]],
    values: list[float],
    nm: NelderMeadAlgorism,
    expected_results: list[dict[str, list[float] | float | bool]],
) -> None:
    for vertex, value in zip(vertices, values, strict=False):
        nm.put_value(vertex, value, True)

    # main loop
    for expected_result in expected_results:
        expected_vertex, expected_value, enqueued = expected_result.values()

        if not enqueued:
            x = nm.get_vertex()
            assert np.array_equal(x, expected_vertex)

        if not isinstance(expected_value, float):
            break

        assert isinstance(enqueued, bool)
        nm.put_value(np.array(expected_vertex), expected_value, enqueued)
