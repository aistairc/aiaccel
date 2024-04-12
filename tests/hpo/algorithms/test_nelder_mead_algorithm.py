import unittest
from collections.abc import Generator
from typing import Any

import numpy as np
import numpy.typing as npt

from aiaccel.hpo.algorithms.nelder_mead_algorithm import NelderMeadAlgorism, NelderMeadEmpty


class TestNelderMeadAlgorism(unittest.TestCase):
    def setUp(self) -> None:
        self.search_space = {"x": (-5.0, 5.0), "y": (-5.0, 5.0)}
        self.nm = NelderMeadAlgorism(search_space=self.search_space, block=False)
        self.vertices = list(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
        self.values = [5.0, 3.0, 7.0]

    def test_waiting_for(self) -> Generator[Any, Any, Any]:
        # queue is Empty
        result = yield from self.nm._wait_for_result()
        self.assertIsNone(result)

        result = yield from self.nm._wait_for_results(2)
        self.assertIsNone(result)

        # queue is not Empty
        value = 1.0
        self.nm.put_value(np.zeros(2), value)
        result = yield from self.nm._wait_for_result()
        self.assertEqual(result, value)

        value1 = 1.0
        self.nm.put_value(np.zeros(2), value1)
        value2 = 2.0
        self.nm.put_value(np.zeros(2), value2)
        result = yield from self.nm._wait_for_results(2)
        self.assertEqual(result, [value1, value2])

    def test_initialize(self) -> None:
        for _ in range(len(self.search_space) + 1):
            xi = self.nm.get_vertex()
            for co, ss in zip(xi, self.search_space.values(), strict=False):
                self.assertIsInstance(co, float)
                self.assertGreaterEqual(co, ss[0])
                self.assertLessEqual(co, ss[1])

        with self.assertRaises(NelderMeadEmpty):
            xi = self.nm.get_vertex()

    def compare_results(self, vertices: list[npt.NDArray[np.float64]], values: list[float] | None = None) -> None:
        if values is None:
            values = []

        for vertex, value in zip(self.vertices, self.values, strict=False):
            self.nm.put_value(vertex, value, True)

        # main loop
        x = self.nm.get_vertex()
        self.assertTrue(np.array_equal(x, vertices[0]))
        for i in range(len(values)):
            self.nm.put_value(vertices[i], values[i])
            x = self.nm.get_vertex()

            self.assertTrue(np.array_equal(x, vertices[i+1]))

    def test_reflect(self) -> None:
        reflect_xs = np.array([-1.0, 0.0])
        self.compare_results([reflect_xs])

    def test_reflect_to_reflect(self) -> None:
        reflect_xs = np.array([-1.0, 0.0])
        reflect_value = 4.0
        reflect_xs2 = np.array([1.0, 2.0])

        # reflect -> self.values[0] <= fr < self.values[-2] -> reflect
        self.compare_results([reflect_xs, reflect_xs2], [reflect_value])

    def test_reflect_to_expand_less_than_r(self) -> None:
        reflect_xs = np.array([-1.0, 0.0])
        reflect_value = 2.0
        expand_xs = np.array([-4.0, -3.0])
        expand_value = 1.0
        reflect_xs2 = np.array([-2.0, -1.0])

        # reflect -> fr < self.values[0] -> expand -> fe < fr -> reflect
        self.compare_results([reflect_xs, expand_xs, reflect_xs2], [reflect_value, expand_value])

    def test_reflect_to_expand_more_than_r(self) -> None:
        reflect_xs = np.array([-1.0, 0.0])
        reflect_value = 2.0
        expand_xs = np.array([-4.0, -3.0])
        expand_value = 3.0
        reflect_xs2 = np.array([1.0, 2.0])

        # reflect -> fr < self.values[0] -> expand -> else (fe > fr) -> reflect
        self.compare_results([reflect_xs, expand_xs, reflect_xs2], [reflect_value, expand_value])

    def test_reflect_to_outside_contract(self) -> None:
        reflect_xs = np.array([-1.0, 0.0])
        reflect_value = 6.0
        outside_contract_xs = np.array([0.5, 1.5])
        outside_contract_value = 5.5
        reflect_xs2 = np.array([3.5, 4.5])

        # reflect -> self.values[-2] <= fr < self.values[-1] -> outside_contract -> foc <= fr -> reflect
        self.compare_results([reflect_xs, outside_contract_xs, reflect_xs2], [reflect_value, outside_contract_value])

    def test_reflect_to_outside_contract_shrink(self) -> None:
        reflect_xs = np.array([-1.0, 0.0])
        reflect_value = 6.0
        outside_contract_xs = np.array([0.5, 1.5])
        outside_contract_value = 7.0
        shrink_xs1 = np.array([2.0, 3.0])
        shrink_value1 = 1.0
        shrink_xs2 = np.array([4.0, 5.0])
        shrink_value2 = 2.0
        reflect_xs2 = np.array([3.0, 4.0])

        # reflect -> self.values[-2] <= fr < self.values[-1] -> outside_contract -> else (foc > fr) -> shrink -> reflect
        self.compare_results(
            [reflect_xs, outside_contract_xs, shrink_xs1, shrink_xs2, reflect_xs2],
            [reflect_value, outside_contract_value, shrink_value1, shrink_value2],
        )

    def test_reflect_to_inside_contract(self) -> None:
        reflect_xs = np.array([-1.0, 0.0])
        reflect_value = 8.0
        inside_contract_xs = np.array([3.5, 4.5])
        inside_contract_value = 6.0
        reflect_xs2 = np.array([0.5, 1.5])

        # reflect -> self.values[-1] <= fr -> inside_contract -> fic < self.values[-1] -> reflect
        self.compare_results([reflect_xs, inside_contract_xs, reflect_xs2], [reflect_value, inside_contract_value])

    def test_reflect_to_inside_contract_shrink(self) -> None:
        reflect_xs = np.array([-1.0, 0.0])
        reflect_value = 8.0
        inside_contract_xs = np.array([3.5, 4.5])
        inside_contract_value = 8.5
        shrink_xs1 = np.array([2.0, 3.0])
        shrink_value1 = 1.0
        shrink_xs2 = np.array([4.0, 5.0])
        shrink_value2 = 2.0
        reflect_xs2 = np.array([3.0, 4.0])

        # reflect -> self.values[-1] <= fr -> inside_contract -> else (fic > self.values[-1]) -> shrink -> reflect
        self.compare_results(
            [reflect_xs, inside_contract_xs, shrink_xs1, shrink_xs2, reflect_xs2],
            [reflect_value, inside_contract_value, shrink_value1, shrink_value2],
        )
