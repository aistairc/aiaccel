import csv
import time
import unittest
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import optuna

from aiaccel.hpo.samplers.nelder_mead_sampler import NelderMeadSampler


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
        self.optimize()

        with open(self.results_csv_path) as f:
            reader = csv.DictReader(f)
            results = [row for row in reader]

        self.validation(results)

    def optimize(self):
        self.study.optimize(self.func, n_trials=30, n_jobs=self.n_jobs)

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


class TestNelderMeadSphereEnqueue(AbstractTestNelderMead, unittest.TestCase):
    def setUp(self):
        search_space = {"x": [-30, 30], "y": [-30, 30], "z": [-30, 30]}
        self._rng = np.random.RandomState(seed=42)
        sampler = NelderMeadSampler(search_space=search_space, rng=self._rng)

        super().setUp(
            search_space=search_space,
            objective=shpere,
            result_file_name='results_shpere_enqueue.csv',
            study=optuna.create_study(sampler=sampler)
        )

    def optimize(self):
        num_trial = 0
        num_parallel = 5
        p = Pool(num_parallel)
        Trials = []
        params = []
        lows = []

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
            except RuntimeError:
                pass

            while len(Trials) < num_parallel:
                # random
                self.study.enqueue_trial({
                    "x": self._rng.uniform(*self.search_space["x"]),
                    "y": self._rng.uniform(*self.search_space["y"]),
                    "z": self._rng.uniform(*self.search_space["z"])
                    })
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

            for trial, obj in zip(Trials, results):
                print(f"trial {trial._trial_id} parameters {trial.params} value {obj}")
                lows.append(list(trial.params.values()) + [obj])
                self.study.tell(trial, obj)

            Trials = []
            params = []

        with open('./results.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerows(lows)

    def validation(self, results):
        trials = [trial for trial in self.study.trials if len(trial.params) > 0]
        for trial, result in zip(trials, results):
            self.assertAlmostEqual(trial.params["x"], float(result["x"]))
            self.assertAlmostEqual(trial.params["y"], float(result["y"]))
            self.assertAlmostEqual(trial.values[0], float(result["objective"]))
