import unittest
import optuna
import csv
import numpy as np

from pathlib import Path
from aiaccel.hpo.samplers.nelder_mead_sampler import NelderMeadSampler

import time


def ackley(x):
    X = x[0]
    Y = x[1]

    # Ackley function
    y = -20*np.exp(-0.2*np.sqrt(0.5*(X**2+Y**2)))-np.exp(0.5 *
                                                            (np.cos(2*np.pi*X)+np.cos(2*np.pi*Y)))+np.e+20

    return float(y)


def shpere(x):
    time.sleep(0.001)
    return np.sum(np.array(x) ** 2)


class AbstractTestNelderMead:
    def setUp(self, search_space, objective, result_file_name, study, n_jobs=1):
        self.search_space = search_space
        self.objective = objective
        self.study = study
        self.n_jobs = n_jobs

        cwd = Path(__file__).resolve().parent
        self.results_csv_path = cwd.joinpath(result_file_name)

    def test_sampler(self):
        self.study.optimize(self.func, n_trials=30, n_jobs=self.n_jobs)

        with open(self.results_csv_path) as f:
            reader = csv.DictReader(f)
            results = [row for row in reader]

        self.validation(results)

    def validation(self):
        raise NotImplementedError()

    def func(self, trial):
        params = []
        for name, distribution in self.search_space.items():
            params.append(trial.suggest_float(name, *distribution))
        return self.objective(params)


class TestNelderMeadAckley(AbstractTestNelderMead, unittest.TestCase):
    def setUp(self):
        search_space = {"x": [0, 10], "y": [0, 10]}
        sampler = NelderMeadSampler(search_space=search_space, seed=42)

        super().setUp(
            search_space=search_space,
            objective=ackley,
            result_file_name='results_ackley.csv',
            study=optuna.create_study(sampler=sampler)
        )

    def validation(self, results):
        for trial, result in zip(self.study.trials, results):
            self.assertAlmostEqual(trial.params["x"], float(result["x"]))
            self.assertAlmostEqual(trial.params["y"], float(result["y"]))
            self.assertAlmostEqual(trial.values[0], float(result["objective"]))


class TestNelderMeadSphereParallel(AbstractTestNelderMead, unittest.TestCase):
    def setUp(self):
        search_space = {"x": [-30, 30], "y": [-30, 30], "z": [-30, 30]}
        sampler = NelderMeadSampler(search_space=search_space, seed=42, parallel_enabled=True)

        super().setUp(
            search_space=search_space,
            objective=shpere,
            result_file_name='results_shpere_parallel.csv',
            study=optuna.create_study(sampler=sampler),
            n_jobs=4
        )

    def validation(self, results):
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
