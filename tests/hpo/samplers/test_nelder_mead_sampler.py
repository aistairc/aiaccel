import csv
import datetime
import time
import unittest
from collections.abc import Callable
from multiprocessing import Pool
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import optuna

from aiaccel.hpo.samplers.nelder_mead_sampler import NelderMeadEmpty, NelderMeadSampler


class TestNelderMeadSampler(unittest.TestCase):
    def setUp(self) -> None:
        self.search_space = {"x": (-5.0, 5.0), "y": (-5.0, 5.0)}
        self.sampler = NelderMeadSampler(search_space=self.search_space, seed=42)

        self.study = optuna.create_study(sampler=self.sampler)
        self.state = optuna.trial.TrialState.COMPLETE
        self.param_distribution = optuna.distributions.FloatDistribution(-5, 5)
        self.trial_id = 0
        self.trial = optuna.trial.FrozenTrial(
            number=0,
            state=self.state,
            value=0.0,
            datetime_start=datetime.datetime.now(),
            datetime_complete=datetime.datetime.now(),
            params={"x": 0.0, "y": 1.0},
            distributions={"x": self.param_distribution, "y": self.param_distribution},
            user_attrs={},
            system_attrs={},
            intermediate_values={},
            trial_id=self.trial_id,
        )

    def test_infer_relative_search_space(self) -> None:
        self.assertEqual(self.sampler.infer_relative_search_space(self.study, self.trial), {})

    def test_sample_relative(self) -> None:
        self.assertEqual(
            self.sampler.sample_relative(
                self.study,
                self.trial,
                {"x": self.param_distribution, "y": self.param_distribution},
            ),
            {},
        )

    def test_before_trial(self) -> None:
        with patch("aiaccel.hpo.samplers.nelder_mead_sampler.NelderMeadAlgorism.get_vertex") as mock_iter:
            mock_iter.side_effect = [np.array([-1.0, 0.0])]

            self.sampler.before_trial(self.study, self.trial)
            self.assertEqual(self.sampler.running_trial_id, [self.trial_id])

            self.assertTrue(np.array_equal(self.trial.user_attrs["params"], np.array([-1.0, 0.0])))

    def test_before_trial_out_of_range(self) -> None:
        with patch("aiaccel.hpo.samplers.nelder_mead_sampler.NelderMeadAlgorism.get_vertex") as mock_iter:
            mock_iter.side_effect = [np.array([-6.0, 0.0]), np.array([0.0, 6.0]), np.array([-2.0, 0.0])]

            self.sampler.before_trial(self.study, self.trial)
            self.assertEqual(self.sampler.running_trial_id, [self.trial_id])

            self.assertTrue(np.array_equal(self.trial.user_attrs["params"], np.array([-2.0, 0.0])))

    def test_sample_independent(self) -> None:
        xs = np.array([-1.0, 0.0])
        self.trial.set_user_attr("params", xs)

        value = self.sampler.sample_independent(self.study, self.trial, "x", self.param_distribution)
        self.assertEqual(value, xs[0])

        value = self.sampler.sample_independent(self.study, self.trial, "y", self.param_distribution)
        self.assertEqual(value, xs[1])

    def test_after_trial(self) -> None:
        put_value = 4.0
        self.trial.set_user_attr("params", np.array([-1.0, 0.0]))
        self.sampler.running_trial_id.append(self.trial._trial_id)

        self.sampler.after_trial(self.study, self.trial, self.state, [put_value])
        self.assertEqual(self.sampler.running_trial_id, [])

        value = self.sampler.nm.value_queue.get(block=False)
        self.assertEqual(value, put_value)


def ackley(x: list[float]) -> float:
    X = x[0]
    Y = x[1]

    # Ackley function
    y = (
        -20 * np.exp(-0.2 * np.sqrt(0.5 * (X**2 + Y**2)))
        - np.exp(0.5 * (np.cos(2 * np.pi * X) + np.cos(2 * np.pi * Y)))
        + np.e
        + 20
    )

    return float(y)


def shpere(x: list[float]) -> float:
    time.sleep(0.001)
    return float(np.sum(np.asarray(x) ** 2))


class AbstractTestNelderMead:
    def common_setUp(
        self,
        search_space: dict[str, tuple[float, float]],
        objective: Callable[[list[float]], float],
        result_file_name: str,
        study: optuna.Study,
        n_jobs: int = 1,
    ) -> None:
        self.search_space = search_space
        self.objective = objective
        self.study = study
        self.n_jobs = n_jobs

        cwd = Path(__file__).resolve().parent
        self.results_csv_path = cwd.joinpath(result_file_name)

    def test_sampler(self) -> None:
        self.optimize()

        with open(self.results_csv_path) as f:
            results = list(csv.DictReader(f))

        self.validation(results)

    def optimize(self) -> None:
        self.study.optimize(self.func, n_trials=30, n_jobs=self.n_jobs)

    def validation(self, results: list[dict[str | Any, str | Any]]) -> None:
        raise NotImplementedError()

    def func(self, trial: optuna.trial.Trial) -> float:
        params = []
        for name, distribution in self.search_space.items():
            params.append(trial.suggest_float(name, *distribution))
        return self.objective(params)


class TestNelderMeadAckley(AbstractTestNelderMead, unittest.TestCase):
    def setUp(self) -> None:
        search_space = {"x": (0.0, 10.0), "y": (0.0, 10.0)}
        sampler = NelderMeadSampler(search_space=search_space, seed=42)

        self.common_setUp(
            search_space=search_space,
            objective=ackley,
            result_file_name="results_ackley.csv",
            study=optuna.create_study(sampler=sampler),
        )

    def validation(self, results: list[dict[str | Any, str | Any]]) -> None:
        for trial, result in zip(self.study.trials, results, strict=False):
            self.assertAlmostEqual(trial.params["x"], float(result["x"]))
            self.assertAlmostEqual(trial.params["y"], float(result["y"]))
            self.assertAlmostEqual(trial.values[0], float(result["objective"]))


class TestNelderMeadSphereParallel(AbstractTestNelderMead, unittest.TestCase):
    def setUp(self) -> None:
        search_space = {"x": (-30.0, 30.0), "y": (-30.0, 30.0), "z": (-30.0, 30.0)}
        sampler = NelderMeadSampler(search_space=search_space, seed=42, parallel_enabled=True)

        self.common_setUp(
            search_space=search_space,
            objective=shpere,
            result_file_name="results_shpere_parallel.csv",
            study=optuna.create_study(sampler=sampler),
            n_jobs=4,
        )

    def validation(self, results: list[dict[str | Any, str | Any]]) -> None:
        for trial in self.study.trials:
            almost_equal_trial_exists = False
            for result in results:
                try:
                    self.assertAlmostEqual(trial.params["x"], float(result["x"]))
                    self.assertAlmostEqual(trial.params["y"], float(result["y"]))
                    self.assertAlmostEqual(trial.params["z"], float(result["z"]))
                    self.assertAlmostEqual(trial.values[0], float(result["objective"]))
                    almost_equal_trial_exists = True
                    break
                except AssertionError:
                    continue
            self.assertTrue(almost_equal_trial_exists)


class TestNelderMeadSphereEnqueue(AbstractTestNelderMead, unittest.TestCase):
    def setUp(self) -> None:
        search_space = {"x": (-30.0, 30.0), "y": (-30.0, 30.0), "z": (-30.0, 30.0)}
        self._rng = np.random.RandomState(seed=42)
        sampler = NelderMeadSampler(search_space=search_space, rng=self._rng)

        self.common_setUp(
            search_space=search_space,
            objective=shpere,
            result_file_name="results_shpere_enqueue.csv",
            study=optuna.create_study(sampler=sampler),
        )

    def optimize(self) -> None:
        num_trial = 0
        num_parallel = 5
        p = Pool(num_parallel)
        Trials = []
        params = []

        while num_trial < 30 * num_parallel:
            try:
                # nelder mead
                trial = self.study.ask()
                X = trial.suggest_float("x", *self.search_space["x"])
                Y = trial.suggest_float("y", *self.search_space["y"])
                Z = trial.suggest_float("z", *self.search_space["z"])
                Trials.append(trial)
                params.append([X, Y, Z])
                num_trial += 1

                continue
            except NelderMeadEmpty:
                pass

            while len(Trials) < num_parallel:
                # random
                self.study.enqueue_trial(
                    {
                        "x": self._rng.uniform(*self.search_space["x"]),
                        "y": self._rng.uniform(*self.search_space["y"]),
                        "z": self._rng.uniform(*self.search_space["z"]),
                    }
                )
                trial = self.study.ask()
                X = trial.suggest_float("x", *self.search_space["x"])
                Y = trial.suggest_float("y", *self.search_space["y"])
                Z = trial.suggest_float("z", *self.search_space["z"])
                Trials.append(trial)
                params.append([X, Y, Z])
                num_trial += 1

            results = []

            try:
                results = p.map(shpere, params)
            except Exception as e:
                print(e)

            for trial, obj in zip(Trials, results, strict=False):
                print(f"trial {trial._trial_id} parameters {trial.params} value {obj}")
                self.study.tell(trial, obj)

            Trials = []
            params = []

    def validation(self, results: list[dict[str | Any, str | Any]]) -> None:
        trials = [trial for trial in self.study.trials if len(trial.params) > 0]
        for trial, result in zip(trials, results, strict=False):
            self.assertAlmostEqual(trial.params["x"], float(result["x"]))
            self.assertAlmostEqual(trial.params["y"], float(result["y"]))
            self.assertAlmostEqual(trial.values[0], float(result["objective"]))
