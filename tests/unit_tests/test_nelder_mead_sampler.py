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
        self.nm = NelderMeadAlgorism(self.search_space)
        self.nm_generator = iter(self.nm)
        self.vertices = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        self.values = np.array([5.0, 3.0, 7.0])

    def test_initialize(self):
        for i in range(len(self.search_space) + 1):
            xi = next(self.nm_generator)
            if i < len(self.search_space):
                self.assertTrue(self.nm.is_ready)
            else:
                self.assertFalse(self.nm.is_ready)
            for co, ss in zip(xi, self.search_space.values()):
                self.assertIsInstance(co, float)
                self.assertGreaterEqual(co, ss[0])
                self.assertLessEqual(co, ss[1])

    def setup_initialize(self):
        for _ in range(len(self.search_space) + 1):
            next(self.nm_generator)
        self.nm.vertices = self.vertices
        for value in self.values:
            self.nm.value_queue.put(value)

    def test_reflect(self):
        self.setup_initialize()
        reflect_xs = np.array([-1.0, 0.0])
        xr = next(self.nm_generator)

        self.assertTrue(np.array_equal(xr, reflect_xs))

    def test_reflect_to_reflect(self):
        self.setup_initialize()
        reflect_xs = np.array([-1.0, 0.0])
        reflect_xs2 = np.array([1.0, 2.0])
        reflect_value = 4.0

        # self.values[0] <= fr < self.values[-2]
        next(self.nm_generator)  # reflect
        self.nm.value_queue.put(reflect_value)

        xr2 = next(self.nm_generator)
        self.assertTrue(np.array_equal(xr2, reflect_xs2))

        self.assertTrue(np.array_equal(self.nm.vertices[1], reflect_xs))
        self.assertTrue(np.array_equal(self.nm.values[1], reflect_value))

    def test_reflect_to_expand_less_than_r(self):
        self.setup_initialize()
        reflect_value = 2.0
        expand_xs = np.array([-4.0, -3.0])
        expand_value = 1.0

        # fr < self.values[0]
        next(self.nm_generator)  # reflect
        self.nm.value_queue.put(reflect_value)

        xe = next(self.nm_generator)  # expand
        self.assertTrue(np.array_equal(xe, expand_xs))

        # fe < fr
        self.nm.value_queue.put(expand_value)
        next(self.nm_generator)  # reflect
        self.assertTrue(np.array_equal(self.nm.vertices[0], expand_xs))
        self.assertTrue(np.array_equal(self.nm.values[0], expand_value))

    def test_reflect_to_expand_more_than_r(self):
        self.setup_initialize()
        reflect_xs = np.array([-1.0, 0.0])
        reflect_value = 2.0
        expand_value = 3.0

        # fr < self.values[0]
        next(self.nm_generator)  # reflect
        self.nm.value_queue.put(reflect_value)

        next(self.nm_generator)  # expand

        # else (fe > fr)
        self.nm.value_queue.put(expand_value)
        next(self.nm_generator)  # reflect
        self.assertTrue(np.array_equal(self.nm.vertices[0], reflect_xs))
        self.assertTrue(np.array_equal(self.nm.values[0], reflect_value))

    def test_reflect_to_outside_contract(self):
        self.setup_initialize()
        reflect_value = 6.0
        outside_contract_xs = np.array([0.5, 1.5])
        outside_contract_value = 5.5

        # self.values[-2] <= fr < self.values[-1]
        next(self.nm_generator)  # reflect
        self.nm.value_queue.put(reflect_value)

        xoc = next(self.nm_generator)  # outside_contract
        self.assertTrue(np.array_equal(xoc, outside_contract_xs))

        # foc <= fr
        self.nm.value_queue.put(outside_contract_value)
        next(self.nm_generator)  # reflect
        self.assertTrue(np.array_equal(self.nm.vertices[2], outside_contract_xs))
        self.assertTrue(np.array_equal(self.nm.values[2], outside_contract_value))

    def test_reflect_to_outside_contract_shrink(self):
        self.setup_initialize()
        reflect_value = 6.0
        outside_contract_value = 7.0
        shrink_xss = np.array([[2.0, 3.0], [4.0, 5.0]])
        shrink_values = [1.0, 2.0]
        reflect_xs2 = [3.0, 4.0]

        # self.values[-2] <= fr < self.values[-1]
        next(self.nm_generator)  # reflect
        self.nm.value_queue.put(reflect_value)

        next(self.nm_generator)  # outside_contract

        # else (foc > fr)
        self.nm.value_queue.put(outside_contract_value)
        for i, shrink_xs in enumerate(shrink_xss):
            xsh = next(self.nm_generator)  # shrink
            if i < len(shrink_xss) - 1:
                self.assertTrue(self.nm.is_ready)
            else:
                self.assertFalse(self.nm.is_ready)
            self.assertTrue(np.array_equal(shrink_xs, xsh))

        for shrink_value in shrink_values:
            self.nm.value_queue.put(shrink_value)

        xr2 = next(self.nm_generator)
        self.assertTrue(np.array_equal(xr2, reflect_xs2))

    def test_reflect_to_inside_contract(self):
        self.setup_initialize()
        reflect_value = 8.0
        inside_contract_xs = np.array([3.5, 4.5])
        inside_contract_value = 6.0

        # self.values[-1] <= fr
        next(self.nm_generator)  # reflect
        self.nm.value_queue.put(reflect_value)
        xic = next(self.nm_generator)  # inside_contract
        self.assertTrue(np.array_equal(xic, inside_contract_xs))

        # fic < self.values[-1]
        self.nm.value_queue.put(inside_contract_value)
        next(self.nm_generator)  # reflect
        self.assertTrue(np.array_equal(self.nm.vertices[2], inside_contract_xs))
        self.assertTrue(np.array_equal(self.nm.values[2], inside_contract_value))

    def test_reflect_to_inside_contract_shrink(self):
        self.setup_initialize()
        # else (self.simplex.vertices[-1] <= self.ic)
        reflect_value = 8.0
        inside_contract_value = 8.5
        shrink_xss = np.array([[2.0, 3.0], [4.0, 5.0]])

        # self.values[-1] <= fr
        next(self.nm_generator)  # reflect
        self.nm.value_queue.put(reflect_value)
        next(self.nm_generator)  # inside_contract

        # else (fic > self.values[-1])
        self.nm.value_queue.put(inside_contract_value)
        for i, shrink_xs in enumerate(shrink_xss):
            xsh = next(self.nm_generator)  # shrink
            if i < len(shrink_xss) - 1:
                self.assertTrue(self.nm.is_ready)
            else:
                self.assertFalse(self.nm.is_ready)
            self.assertTrue(np.array_equal(shrink_xs, xsh))


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
        with patch("aiaccel.hpo.samplers.nelder_mead_sampler.NelderMeadAlgorism.__iter__") as mock_iter:
            def side_effect():
                yield np.array([-1.0, 0.0])
            mock_iter.side_effect = side_effect
            self.sampler.nm_generator = iter(self.sampler.nm)

            self.sampler.before_trial(self.study, self.trial)
            self.assertEqual(self.sampler.num_running_trial, 1)
            self.assertEqual(self.sampler.running_trial_id, [self.trial_id])

            self.assertTrue(np.array_equal(self.trial.user_attrs["Coordinate"], np.array([-1.0, 0.0])))
            self.assertFalse(self.trial.user_attrs["IsReady"])

    def test_before_trial_out_of_range(self):
        with patch("aiaccel.hpo.samplers.nelder_mead_sampler.NelderMeadAlgorism.__iter__") as mock_iter:
            def side_effect():
                yield np.array([-6.0, 0.0])
                yield np.array([-2.0, 0.0])
            mock_iter.side_effect = side_effect
            self.sampler.nm_generator = iter(self.sampler.nm)

            self.sampler.before_trial(self.study, self.trial)
            self.assertEqual(self.sampler.num_running_trial, 1)
            self.assertEqual(self.sampler.running_trial_id, [self.trial_id])

            self.assertTrue(np.array_equal(self.trial.user_attrs["Coordinate"], np.array([-2.0, 0.0])))

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
        self.sampler.num_running_trial += 1
        self.sampler.running_trial_id.append(self.trial._trial_id)

        self.sampler.after_trial(self.study, self.trial, self.state, [put_value])
        self.assertEqual(self.sampler.num_running_trial, 0)
        self.assertEqual(self.sampler.running_trial_id, [])

        value = self.sampler.nm.value_queue.get(block=False)
        self.assertEqual(value, put_value)
