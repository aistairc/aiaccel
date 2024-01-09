import unittest
import optuna
import csv
import numpy as np

from pathlib import Path
from aiaccel.hpo.samplers.nelder_mead_sampler import NelderMeadSampler


class TestNelderMead(unittest.TestCase):
    def setUp(self):
        search_space = {"x": [0, 10], "y": [0, 10]}
        self.sampler = NelderMeadSampler(search_space=search_space, seed=42)

        cwd = Path(__file__).resolve().parent
        self.results_csv_path = cwd.joinpath('results.csv')

    def test_sampler(self):
        study = optuna.create_study(sampler=self.sampler)
        study.optimize(self.objective, n_trials=30)

        with open(self.results_csv_path) as f:
            reader = csv.DictReader(f)
            results = [row for row in reader]

        for trial, result in zip(study.trials, results):
            self.assertAlmostEqual(trial.params["x"], float(result["x"]))
            self.assertAlmostEqual(trial.params["y"], float(result["y"]))
            self.assertAlmostEqual(trial.values[0], float(result["objective"]))

    def objective(self, trial):
        X = trial.suggest_float("x", 0, 10)
        Y = trial.suggest_float("y", 0, 10)

        # Ackley function
        y = -20*np.exp(-0.2*np.sqrt(0.5*(X**2+Y**2)))-np.exp(0.5 *
                                                             (np.cos(2*np.pi*X)+np.cos(2*np.pi*Y)))+np.e+20

        return float(y)
