import csv
import time
import unittest
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple, Union

import numpy as np
import optuna

from aiaccel.hpo.samplers.nelder_mead_sampler import NelderMeadSampler


def ackley(x: List[float]) -> float:
    X = x[0]
    Y = x[1]

    # Ackley function
    y = -20*np.exp(-0.2*np.sqrt(0.5*(X**2+Y**2)))-np.exp(0.5 *
                                                            (np.cos(2*np.pi*X)+np.cos(2*np.pi*Y)))+np.e+20

    return float(y)


def shpere(x: List[float]) -> float:
    time.sleep(0.001)
    return np.sum(np.array(x) ** 2)


class AbstractTestNelderMead:
    def common_setUp(self,
                     search_space: Dict[str, Tuple[float, float]],
                     objective: Callable[[List[float]], float],
                     result_file_name: str,
                     study: optuna.study,
                     n_jobs: int = 1) -> None:
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

    def validation(self, results: List[Dict[Union[str, Any], Union[str, Any]]]) -> None:
        raise NotImplementedError()

    def func(self, trial: optuna.trial.FrozenTrial) -> float:
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
            result_file_name='results_ackley.csv',
            study=optuna.create_study(sampler=sampler)
        )

    def validation(self, results: List[Dict[Union[str, Any], Union[str, Any]]]) -> None:
        for trial, result in zip(self.study.trials, results):
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
            result_file_name='results_shpere_parallel.csv',
            study=optuna.create_study(sampler=sampler),
            n_jobs=4
        )

    def validation(self, results: List[Dict[Union[str, Any], Union[str, Any]]]) -> None:
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
            result_file_name='results_shpere_enqueue.csv',
            study=optuna.create_study(sampler=sampler)
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
                self.study.tell(trial, obj)

            Trials = []
            params = []

    def validation(self, results: List[Dict[Union[str, Any], Union[str, Any]]]) -> None:
        trials = [trial for trial in self.study.trials if len(trial.params) > 0]
        for trial, result in zip(trials, results):
            self.assertAlmostEqual(trial.params["x"], float(result["x"]))
            self.assertAlmostEqual(trial.params["y"], float(result["y"]))
            self.assertAlmostEqual(trial.values[0], float(result["objective"]))
