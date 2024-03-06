from __future__ import annotations

import unittest
import optuna
import datetime

import numpy as np
from unittest.mock import patch

from aiaccel.hpo.samplers.nelder_mead_sampler import NelderMeadSampler
from aiaccel.hpo.samplers.nelder_mead_sampler import NelderMeadAlgorism


class TestNelderMeadAlgorism(unittest.TestCase):
    def setUp(self):
        self.search_space = {"x": [-5, 5], "y": [-5, 5]}
        self.nm = NelderMeadAlgorism(search_space=self.search_space, block=False)
        self.vertices = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        self.values = np.array([5.0, 3.0, 7.0])

    def test_put_value(self):
        self.nm.put_value(0.0)
        self.assertFalse(self.nm.value_queue.empty())

    def test_waiting_for(self):
        # queue is Empty
        result = yield from self.nm._waiting_for_float()
        self.assertIsNone(result)

        # queue is not Empty
        value = 1.0
        self.nm.value_queue.put(value)
        result = yield from self.nm._waiting_for_float()
        self.assertEqual(result, value)

    def test_initialize(self):
        for _ in range(len(self.search_space) + 1):
            xi = self.nm.get_vertex()
            for co, ss in zip(xi, self.search_space.values()):
                self.assertIsInstance(co, float)
                self.assertGreaterEqual(co, ss[0])
                self.assertLessEqual(co, ss[1])
        xi = self.nm.get_vertex()
        self.assertIsNone(xi)

    def compare_results(self, vertices: list[np.ndarray], values: list[float] | None = None):
        values = values if values is not None else []
        # initialize
        for _ in range(len(self.search_space) + 1):
            self.nm.get_vertex()
        self.nm.vertices = self.vertices
        for value in self.values:
            self.nm.put_value(value)

        # main loop
        x = self.nm.get_vertex()
        self.assertTrue(np.array_equal(x, vertices[0]))
        for vertex, value in zip(vertices[1:], values):
            self.nm.value_queue.put(value)
            x = self.nm.get_vertex()

            self.assertTrue(np.array_equal(x, vertex))

    def test_reflect(self):
        reflect_xs = np.array([-1.0, 0.0])
        self.compare_results([reflect_xs])

    def test_reflect_to_reflect(self):
        reflect_xs = np.array([-1.0, 0.0])
        reflect_value = 4.0
        reflect_xs2 = np.array([1.0, 2.0])

        # reflect -> self.values[0] <= fr < self.values[-2] -> reflect
        self.compare_results([reflect_xs, reflect_xs2], [reflect_value])

    def test_reflect_to_expand_less_than_r(self):
        reflect_xs = np.array([-1.0, 0.0])
        reflect_value = 2.0
        expand_xs = np.array([-4.0, -3.0])
        expand_value = 1.0
        reflect_xs2 = np.array([-2.0, -1.0])

        # reflect -> fr < self.values[0] -> expand -> fe < fr -> reflect
        self.compare_results([reflect_xs, expand_xs, reflect_xs2], [reflect_value, expand_value])

    def test_reflect_to_expand_more_than_r(self):
        reflect_xs = np.array([-1.0, 0.0])
        reflect_value = 2.0
        expand_xs = np.array([-4.0, -3.0])
        expand_value = 3.0
        reflect_xs2 = np.array([1.0, 2.0])

        # reflect -> fr < self.values[0] -> expand -> else (fe > fr) -> reflect
        self.compare_results([reflect_xs, expand_xs, reflect_xs2], [reflect_value, expand_value])

    def test_reflect_to_outside_contract(self):
        reflect_xs = np.array([-1.0, 0.0])
        reflect_value = 6.0
        outside_contract_xs = np.array([0.5, 1.5])
        outside_contract_value = 5.5
        reflect_xs2 = np.array([3.5, 4.5])

        # reflect -> self.values[-2] <= fr < self.values[-1] -> outside_contract -> foc <= fr -> reflect
        self.compare_results([reflect_xs, outside_contract_xs, reflect_xs2], [reflect_value, outside_contract_value])

    def test_reflect_to_outside_contract_shrink(self):
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
            [reflect_value, outside_contract_value, shrink_value1, shrink_value2]
            )

    def test_reflect_to_inside_contract(self):
        reflect_xs = np.array([-1.0, 0.0])
        reflect_value = 8.0
        inside_contract_xs = np.array([3.5, 4.5])
        inside_contract_value = 6.0
        reflect_xs2 = np.array([0.5, 1.5])

        # reflect -> self.values[-1] <= fr -> inside_contract -> fic < self.values[-1] -> reflect
        self.compare_results([reflect_xs, inside_contract_xs, reflect_xs2], [reflect_value, inside_contract_value])

    def test_reflect_to_inside_contract_shrink(self):
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
            [reflect_value, inside_contract_value, shrink_value1, shrink_value2]
            )


class TestNelderMeadSampler(unittest.TestCase):
    def setUp(self):
        self.search_space = {"x": [-5, 5], "y": [-5, 5]}
        self.sampler = NelderMeadSampler(search_space=self.search_space, seed=42)

        self.study = optuna.create_study(sampler=self.sampler)
        self.state = optuna.trial.TrialState.COMPLETE
        self.param_distribution = optuna.distributions.FloatDistribution(-5, 5)
        self.trial_id = 0
        self.trial = optuna.trial.FrozenTrial(
            number=0,
            state=self.state,
            value=[0.0],
            datetime_start=datetime.datetime.now(),
            datetime_complete=datetime.datetime.now(),
            params={"x": 0.0, "y": 1.0},
            distributions=self.param_distribution,
            user_attrs={},
            system_attrs={},
            intermediate_values={},
            trial_id=self.trial_id,
        )

    def test_is_within_range(self):
        # True
        coordinates = np.array([3.0, 4.0])
        self.assertTrue(self.sampler.is_within_range(coordinates))

        # False
        coordinates = np.array([-6.0, 4.0])
        self.assertFalse(self.sampler.is_within_range(coordinates))
        coordinates = np.array([6.0, 4.0])
        self.assertFalse(self.sampler.is_within_range(coordinates))
        coordinates = np.array([3.0, -6.0])
        self.assertFalse(self.sampler.is_within_range(coordinates))
        coordinates = np.array([3.0, 6.0])
        self.assertFalse(self.sampler.is_within_range(coordinates))

    def test_infer_relative_search_space(self):
        self.assertEqual(self.sampler.infer_relative_search_space(self.study, self.trial), {})

    def test_sample_relative(self):
        self.assertEqual(self.sampler.sample_relative(self.study, self.trial, self.param_distribution), {})

    def test_before_trial(self):
        with patch("aiaccel.hpo.samplers.nelder_mead_sampler.NelderMeadSampler._get_coordinate") as mock_iter:
            mock_iter.side_effect = [np.array([-1.0, 0.0])]

            self.sampler.before_trial(self.study, self.trial)
            self.assertEqual(self.sampler.running_trial_id, [self.trial_id])

            self.assertTrue(np.array_equal(self.trial.user_attrs["Coordinate"], np.array([-1.0, 0.0])))

    def test_get_coordinate(self):
        with patch("aiaccel.hpo.samplers.nelder_mead_sampler.NelderMeadAlgorism.get_vertex") as mock_iter:
            mock_iter.side_effect = [np.array([-1.0, 0.0])]

            coordinate = self.sampler._get_coordinate()

            self.assertTrue(np.array_equal(coordinate, np.array([-1.0, 0.0])))

    def test_get_coordinate_out_of_range(self):
        with patch("aiaccel.hpo.samplers.nelder_mead_sampler.NelderMeadAlgorism.get_vertex") as mock_iter:
            mock_iter.side_effect = [np.array([-6.0, 0.0]), np.array([-2.0, 0.0])]

            coordinate = self.sampler._get_coordinate()

            self.assertTrue(np.array_equal(coordinate, np.array([-2.0, 0.0])))

    def test_sample_independent(self):
        xs = np.array([-1.0, 0.0])
        self.trial.set_user_attr("Coordinate", xs)

        value = self.sampler.sample_independent(self.study, self.trial, "x", self.param_distribution)
        self.assertEqual(value, xs[0])

        value = self.sampler.sample_independent(self.study, self.trial, "y", self.param_distribution)
        self.assertEqual(value, xs[1])

    def test_after_trial(self):
        put_value = 4.0
        self.trial.set_user_attr("Coordinate", np.array([-1.0, 0.0]))
        self.sampler.running_trial_id.append(self.trial._trial_id)

        self.sampler.after_trial(self.study, self.trial, self.state, [put_value])
        self.assertEqual(self.sampler.running_trial_id, [])

        value = self.sampler.nm.value_queue.get(block=False)
        self.assertEqual(value, put_value)
