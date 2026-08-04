#!/usr/bin/env python3
# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

"""Generate CSV fixtures for ``test_nelder_mead_sampler.py``.

Place this script in ``tests/hpo/optuna/samplers`` and run it from the aiaccel
repository root.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
import csv
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import optuna
from test_nelder_mead_sampler import ackley, ackley_sleep, sphere_sleep

from aiaccel.hpo.optuna.samplers.nelder_mead_sampler import (
    NelderMeadEmptyError,
    NelderMeadSampler,
)


class BaseGenerateNelderMead:
    """Base class with the same execution structure as the sampler tests."""

    def setup_method(self) -> None:
        raise NotImplementedError()

    def common_setup(
        self,
        search_space: dict[str, tuple[int | float, int | float]],
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
        self.results_csv_path = cwd / result_file_name

    def generate_results_csv(self) -> None:
        """Run optimization and write its trials to the corresponding CSV."""
        self.optimize()

        with self.results_csv_path.open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([*self.search_space, "objective"])

            for trial in self.study.trials:
                if not trial.params or trial.value is None:
                    continue
                writer.writerow(list(trial.params.values()) + [trial.value])

    def optimize(self) -> None:
        self.study.optimize(self.func, n_trials=30, n_jobs=self.n_jobs)

    def func(self, trial: optuna.trial.Trial) -> float:
        params = [trial.suggest_float(name, *distribution) for name, distribution in self.search_space.items()]
        return self.objective(params)


class GenerateNelderMeadAckley(BaseGenerateNelderMead):
    """Generate ``results_ackley.csv``."""

    def setup_method(self) -> None:
        search_space = {"x": (0.0, 10.0), "y": (0.0, 10.0)}
        sampler = NelderMeadSampler(search_space=search_space, seed=42)
        self.common_setup(
            search_space,
            ackley,
            "results_ackley.csv",
            optuna.create_study(sampler=sampler),
        )


class GenerateNelderMeadSphereParallel(BaseGenerateNelderMead):
    """Generate ``results_sphere_parallel.csv``."""

    def setup_method(self) -> None:
        search_space = {
            "x": (-30.0, 30.0),
            "y": (-30.0, 30.0),
            "z": (-30.0, 30.0),
        }
        sampler = NelderMeadSampler(search_space=search_space, seed=42, block=True)
        self.common_setup(
            search_space,
            sphere_sleep,
            "results_sphere_parallel.csv",
            optuna.create_study(sampler=sampler),
            n_jobs=4,
        )


class GenerateNelderMeadSphereEnqueue(BaseGenerateNelderMead):
    """Generate ``results_sphere_enqueue.csv``."""

    def setup_method(self) -> None:
        search_space = {
            "x": (-30.0, 30.0),
            "y": (-30.0, 30.0),
            "z": (-30.0, 30.0),
        }
        self._rng = np.random.RandomState(seed=42)
        sampler = NelderMeadSampler(search_space=search_space, rng=self._rng)
        self.common_setup(
            search_space,
            sphere_sleep,
            "results_sphere_enqueue.csv",
            optuna.create_study(sampler=sampler),
        )

    def optimize(self) -> None:
        num_parallel = 5
        with Pool(num_parallel) as pool:
            for _ in range(30):
                trials = []
                params = []
                for _ in range(num_parallel):
                    try:
                        trial = self.study.ask()
                    except NelderMeadEmptyError:
                        self.study.enqueue_trial(
                            {name: self._rng.uniform(*distribution) for name, distribution in self.search_space.items()}
                        )
                        trial = self.study.ask()

                    values = [
                        trial.suggest_float(name, *distribution) for name, distribution in self.search_space.items()
                    ]
                    trials.append(trial)
                    params.append(values)

                for trial, value in zip(
                    trials,
                    pool.imap(self.objective, params),
                    strict=False,
                ):
                    frozen_trial = self.study.tell(trial, value)
                    self.study._log_completed_trial([value], frozen_trial.number, frozen_trial.params)


class GenerateNelderMeadAckleySubSampler(BaseGenerateNelderMead):
    """Generate ``results_ackley_sub_sampler.csv``."""

    def setup_method(self) -> None:
        search_space = {"x": (0.0, 10.0), "y": (0.0, 10.0)}
        tpe_sampler = optuna.samplers.TPESampler(seed=43)
        sampler = NelderMeadSampler(
            search_space=search_space,
            seed=42,
            block=False,
            sub_sampler=tpe_sampler,
        )
        self.common_setup(
            search_space,
            ackley_sleep,
            "results_ackley_sub_sampler.csv",
            optuna.create_study(sampler=sampler),
        )

    def optimize(self) -> None:
        num_parallel = 5
        with Pool(num_parallel) as pool:
            for _ in range(30):
                trials = []
                params = []
                for _ in range(num_parallel):
                    trial = self.study.ask()
                    values = [
                        trial.suggest_float(name, *distribution) for name, distribution in self.search_space.items()
                    ]
                    trials.append(trial)
                    params.append(values)

                for trial, value in zip(
                    trials,
                    pool.imap(self.objective, params),
                    strict=False,
                ):
                    frozen_trial = self.study.tell(trial, value)
                    self.study._log_completed_trial([value], frozen_trial.number, frozen_trial.params)


class GenerateNelderMeadAckleyInteger(BaseGenerateNelderMead):
    """Generate ``results_ackley_int.csv``."""

    def setup_method(self) -> None:
        search_space = {"x": (-10, 10), "y": (-10.0, 10.0)}
        sampler = NelderMeadSampler(search_space=search_space, seed=42)
        self.common_setup(
            search_space,
            ackley,
            "results_ackley_int.csv",
            optuna.create_study(sampler=sampler),
        )

    def func(self, trial: optuna.trial.Trial) -> float:
        params: list[int | float] = [
            trial.suggest_int("x", *[int(space) for space in self.search_space["x"]]),
            trial.suggest_float("y", *self.search_space["y"]),
        ]
        return self.objective(params)


class GenerateNelderMeadAckleyStep(BaseGenerateNelderMead):
    """Generate ``results_ackley_step.csv``."""

    def setup_method(self) -> None:
        search_space = {"x": (-30, 30), "y": (-30.0, 30.0)}
        sampler = NelderMeadSampler(search_space=search_space, seed=42)
        self.common_setup(
            search_space,
            ackley,
            "results_ackley_step.csv",
            optuna.create_study(sampler=sampler),
        )

    def func(self, trial: optuna.trial.Trial) -> float:
        params: list[int | float] = [
            trial.suggest_int(
                "x",
                *[int(space) for space in self.search_space["x"]],
                step=2,
            ),
            trial.suggest_float("y", *self.search_space["y"], step=0.5),
        ]
        return self.objective(params)


class GenerateNelderMeadAckleyLogScale(BaseGenerateNelderMead):
    """Generate ``results_ackley_logscale.csv``."""

    def setup_method(self) -> None:
        search_space = {"x": (1.0e-5, 1.0e5), "y": (1.0e-5, 1.0e5)}
        sampler = NelderMeadSampler(search_space=search_space, seed=42)
        self.common_setup(
            search_space,
            ackley,
            "results_ackley_logscale.csv",
            optuna.create_study(sampler=sampler),
        )

    def func(self, trial: optuna.trial.Trial) -> float:
        params = [
            trial.suggest_float(name, *distribution, log=True) for name, distribution in self.search_space.items()
        ]
        return self.objective(params)


GENERATORS: dict[str, type[BaseGenerateNelderMead]] = {
    "ackley": GenerateNelderMeadAckley,
    "sphere_parallel": GenerateNelderMeadSphereParallel,
    "sphere_enqueue": GenerateNelderMeadSphereEnqueue,
    "ackley_sub_sampler": GenerateNelderMeadAckleySubSampler,
    "ackley_int": GenerateNelderMeadAckleyInteger,
    "ackley_step": GenerateNelderMeadAckleyStep,
    "ackley_logscale": GenerateNelderMeadAckleyLogScale,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        action="append",
        choices=GENERATORS,
        help="Generate only this case (repeatable).",
    )
    args = parser.parse_args()

    for name in args.case or GENERATORS:
        generator = GENERATORS[name]()
        generator.setup_method()
        generator.generate_results_csv()
        print(generator.results_csv_path)


if __name__ == "__main__":
    main()
