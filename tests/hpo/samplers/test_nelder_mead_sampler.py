import csv
import datetime
import math
import time
from collections.abc import Callable
from multiprocessing import Pool
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import numpy.typing as npt
import optuna
import pytest

from aiaccel.hpo.samplers.nelder_mead_sampler import NelderMeadEmptyError, NelderMeadSampler


@pytest.fixture
def search_space() -> dict[str, tuple[float, float]]:
    return {"x": (-5.0, 5.0), "y": (-5.0, 5.0)}


@pytest.fixture
def sampler(search_space: dict[str, tuple[float, float]]) -> NelderMeadSampler:
    return NelderMeadSampler(search_space=search_space, seed=42)

@pytest.fixture
def sampler_with_sub_sampler(search_space: dict[str, tuple[float, float]]) -> NelderMeadSampler:
    return NelderMeadSampler(search_space=search_space, seed=42, sub_sampler=optuna.samplers.RandomSampler())


@pytest.fixture
def study(sampler: NelderMeadSampler) -> optuna.study.Study:
    return optuna.create_study(sampler=sampler)

@pytest.fixture
def study_with_sub_sampler(sampler_with_sub_sampler: NelderMeadSampler) -> optuna.study.Study:
    return optuna.create_study(sampler=sampler_with_sub_sampler)


@pytest.fixture
def state() -> optuna.trial.TrialState:
    return optuna.trial.TrialState.COMPLETE


@pytest.fixture
def param_distribution() -> optuna.distributions.FloatDistribution:
    return optuna.distributions.FloatDistribution(-5, 5)


@pytest.fixture
def trial_id() -> int:
    return 0


@pytest.fixture
def trial(
    state: optuna.trial.TrialState,
    param_distribution: optuna.distributions.FloatDistribution,
    trial_id: int,
) -> optuna.trial.FrozenTrial:
    return optuna.trial.FrozenTrial(
        number=0,
        state=state,
        value=0.0,
        datetime_start=datetime.datetime.now(),
        datetime_complete=datetime.datetime.now(),
        params={"x": 0.0, "y": 1.0},
        distributions={"x": param_distribution, "y": param_distribution},
        user_attrs={},
        system_attrs={},
        intermediate_values={},
        trial_id=trial_id,
    )

@pytest.fixture
def fixed_params() -> dict[str, float]:
    return {"x": 2.0, "y": 3.0}


@pytest.fixture
def trial_with_fixed_params(
    state: optuna.trial.TrialState,
    param_distribution: optuna.distributions.FloatDistribution,
    trial_id: int,
    fixed_params: dict[str, tuple[float, float]]
) -> optuna.trial.FrozenTrial:
    return optuna.trial.FrozenTrial(
        number=0,
        state=state,
        value=0.0,
        datetime_start=datetime.datetime.now(),
        datetime_complete=datetime.datetime.now(),
        params={"x": 0.0, "y": 1.0},
        distributions={"x": param_distribution, "y": param_distribution},
        user_attrs={},
        system_attrs={"fixed_params": fixed_params},
        intermediate_values={},
        trial_id=trial_id,
    )


class TestNelderMeadSampler:
    def test_infer_relative_search_space(
        self,
        sampler: NelderMeadSampler,
        study: optuna.study.Study,
        trial: optuna.trial.FrozenTrial
    ) -> None:
        assert sampler.infer_relative_search_space(study, trial) == {}

    def test_sample_relative(
        self,
        sampler: NelderMeadSampler,
        study: optuna.study.Study,
        trial: optuna.trial.FrozenTrial,
        param_distribution: optuna.distributions.FloatDistribution,
    ) -> None:
        assert sampler.sample_relative(study, trial, {"x": param_distribution, "y": param_distribution}) == {}

    @pytest.mark.parametrize(
        "side_effect, expect_vertex",
        [
            ([np.array([-1.0, 0.0])], np.array([-1.0, 0.0])),
            (
                [np.array([-6.0, 0.0]), np.array([0.0, 6.0]), np.array([-2.0, 0.0])],
                np.array([-2.0, 0.0]),
            ),  # out_of_range
        ],
    )
    def test_before_trial(
        self,
        sampler: NelderMeadSampler,
        study: optuna.study.Study,
        trial: optuna.trial.FrozenTrial,
        side_effect: list[npt.NDArray[np.float64]],
        expect_vertex: npt.NDArray[np.float64],
    ) -> None:
        with patch("aiaccel.hpo.samplers.nelder_mead_sampler.NelderMeadAlgorism.get_vertex") as mock_iter:
            mock_iter.side_effect = side_effect

            sampler.before_trial(study, trial)

            assert sampler.running_trials == [trial]
            assert np.array_equal(trial.user_attrs["params"], expect_vertex)

    def test_before_trial_enqueued(
        self,
        sampler: NelderMeadSampler,
        study: optuna.study.Study,
        trial_with_fixed_params: optuna.trial.FrozenTrial
    ) -> None:
        sampler.before_trial(study, trial_with_fixed_params)

        assert sampler.running_trials == [trial_with_fixed_params]
        assert np.array_equal(
            trial_with_fixed_params.user_attrs["params"],
            list(trial_with_fixed_params.system_attrs["fixed_params"].values())
        )

    def test_before_trial_sub_sampler(
        self,
        sampler_with_sub_sampler: NelderMeadSampler,
        study_with_sub_sampler: optuna.study.Study,
        trial: optuna.trial.FrozenTrial,
        search_space: dict[str, tuple[float, float]],
    ) -> None:
        with patch("aiaccel.hpo.samplers.nelder_mead_sampler.NelderMeadAlgorism.get_vertex") as mock_iter:
            mock_iter.side_effect = NelderMeadEmptyError()

            sampler_with_sub_sampler.before_trial(study_with_sub_sampler, trial)

            assert sampler_with_sub_sampler.running_trials == [trial]
            for random_x, distribution in zip(trial.user_attrs["params"], search_space.values(), strict=False):
                assert isinstance(random_x, float) and distribution[0] <= random_x <= distribution[1]
            assert isinstance(trial.user_attrs["sub_trial"], optuna.trial.BaseTrial)

    def test_sample_independent(
        self,
        trial: optuna.trial.FrozenTrial,
        sampler: NelderMeadSampler,
        study: optuna.study.Study,
        param_distribution: optuna.distributions.FloatDistribution,
    ) -> None:
        xs = np.array([-1.0, 0.0])
        trial.set_user_attr("params", xs)

        value = sampler.sample_independent(study, trial, "x", param_distribution)
        assert value == xs[0]

        value = sampler.sample_independent(study, trial, "y", param_distribution)
        assert value == xs[1]

    def test_after_trial(
        self,
        trial: optuna.trial.FrozenTrial,
        sampler: NelderMeadSampler,
        study: optuna.study.Study,
        state: optuna.trial.TrialState,
    ) -> None:
        put_value = 4.0

        trial.set_user_attr("params", np.array([-1.0, 0.0]))

        sampler.after_trial(study, trial, state, [put_value])

        vertex, value, enqueue = sampler.nm.results.get(block=False)
        assert value == put_value

    def test_after_trial_sub_sampler(
        self,
        trial: optuna.trial.FrozenTrial,
        sampler_with_sub_sampler: NelderMeadSampler,
        study_with_sub_sampler: optuna.study.Study,
        state: optuna.trial.TrialState,
    ) -> None:
        with patch("optuna.study.Study.tell") as mock_iter:
            put_value = 4.0
            trial.set_user_attr("params", np.array([-1.0, 0.0]))
            trial.set_user_attr("sub_trial", study_with_sub_sampler.ask())

            sampler_with_sub_sampler.after_trial(study_with_sub_sampler, trial, state, [put_value])

            vertex, value, enqueue = sampler_with_sub_sampler.nm.results.get(block=False)
            assert value == put_value
            mock_iter.method.assert_not_called()


def ackley(x: list[float]) -> float:
    # Ackley function
    y = (
        -20 * np.exp(-0.2 * np.sqrt(0.5 * (x[0] ** 2 + x[1] ** 2)))
        - np.exp(0.5 * (np.cos(2 * np.pi * x[0]) + np.cos(2 * np.pi * x[1])))
        + np.e
        + 20
    )

    return float(y)


def ackley_sleep(x: list[float]) -> float:
    time.sleep(np.random.uniform(0.001, 0.01))
    # Ackley function
    y = (
        -20 * np.exp(-0.2 * np.sqrt(0.5 * (x[0] ** 2 + x[1] ** 2)))
        - np.exp(0.5 * (np.cos(2 * np.pi * x[0]) + np.cos(2 * np.pi * x[1])))
        + np.e
        + 20
    )

    return float(y)


def sphere_sleep(x: list[float]) -> float:
    time.sleep(np.random.uniform(0.001, 0.01))
    return float(np.sum(np.asarray(x) ** 2))


class BaseTestNelderMead:
    def common_setup(
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


class TestNelderMeadAckley(BaseTestNelderMead):
    def setup_method(self) -> None:
        search_space = {"x": (0.0, 10.0), "y": (0.0, 10.0)}
        sampler = NelderMeadSampler(search_space=search_space, seed=42)

        self.common_setup(
            search_space=search_space,
            objective=ackley,
            result_file_name="results_ackley.csv",
            study=optuna.create_study(sampler=sampler),
        )

    def validation(self, results: list[dict[str | Any, str | Any]]) -> None:
        for trial, result in zip(self.study.trials, results, strict=False):
            assert math.isclose(trial.params["x"], float(result["x"]), rel_tol=0.000001)
            assert math.isclose(trial.params["y"], float(result["y"]), rel_tol=0.000001)
            assert math.isclose(trial.values[0], float(result["objective"]), rel_tol=0.000001)


class TestNelderMeadAckleyParallel(BaseTestNelderMead):
    def setup_method(self) -> None:
        search_space = {"x": (0.0, 10.0), "y": (0.0, 10.0)}
        sampler = NelderMeadSampler(search_space=search_space, seed=42, block=True)

        self.common_setup(
            search_space=search_space,
            objective=ackley_sleep,
            result_file_name="results_ackley.csv",
            study=optuna.create_study(sampler=sampler),
            n_jobs=3,
        )

    def validation(self, results: list[dict[str | Any, str | Any]]) -> None:
        for trial in self.study.trials:
            almost_equal_trial_exists = False
            for result in results:
                try:
                    assert math.isclose(trial.params["x"], float(result["x"]), rel_tol=0.000001)
                    assert math.isclose(trial.params["y"], float(result["y"]), rel_tol=0.000001)
                    assert math.isclose(trial.values[0], float(result["objective"]), rel_tol=0.000001)
                    almost_equal_trial_exists = True
                    break
                except AssertionError:
                    continue
            assert almost_equal_trial_exists


class TestNelderMeadSphereParallel(BaseTestNelderMead):
    def setup_method(self) -> None:
        search_space = {"x": (-30.0, 30.0), "y": (-30.0, 30.0), "z": (-30.0, 30.0)}
        sampler = NelderMeadSampler(search_space=search_space, seed=42, block=True)

        self.common_setup(
            search_space=search_space,
            objective=sphere_sleep,
            result_file_name="results_shpere_parallel.csv",
            study=optuna.create_study(sampler=sampler),
            n_jobs=4,
        )

    def validation(self, results: list[dict[str | Any, str | Any]]) -> None:
        for trial in self.study.trials:
            almost_equal_trial_exists = False
            for result in results:
                try:
                    assert math.isclose(trial.params["x"], float(result["x"]), rel_tol=0.000001)
                    assert math.isclose(trial.params["y"], float(result["y"]), rel_tol=0.000001)
                    assert math.isclose(trial.params["z"], float(result["z"]), rel_tol=0.000001)
                    assert math.isclose(trial.values[0], float(result["objective"]), rel_tol=0.000001)
                    almost_equal_trial_exists = True
                    break
                except AssertionError:
                    continue
            assert almost_equal_trial_exists


class TestNelderMeadSphereEnqueue(BaseTestNelderMead):
    def setup_method(self) -> None:
        search_space = {"x": (-30.0, 30.0), "y": (-30.0, 30.0), "z": (-30.0, 30.0)}
        self._rng = np.random.RandomState(seed=42)
        sampler = NelderMeadSampler(search_space=search_space, rng=self._rng)

        self.common_setup(
            search_space=search_space,
            objective=sphere_sleep,
            result_file_name="results_shpere_enqueue.csv",
            study=optuna.create_study(sampler=sampler),
        )

    def optimize(self) -> None:
        num_parallel = 5
        with Pool(num_parallel) as p:
            for _ in range(30):
                trials = []
                params = []
                for _ in range(num_parallel):
                    try:  # nelder-mead
                        trial = self.study.ask()
                    except NelderMeadEmptyError:  # random sampling
                        self.study.enqueue_trial(
                            {
                                "x": self._rng.uniform(*self.search_space["x"]),
                                "y": self._rng.uniform(*self.search_space["y"]),
                                "z": self._rng.uniform(*self.search_space["z"]),
                            }
                        )
                        trial = self.study.ask()

                    x = trial.suggest_float("x", *self.search_space["x"])
                    y = trial.suggest_float("y", *self.search_space["y"])
                    z = trial.suggest_float("z", *self.search_space["z"])

                    trials.append(trial)
                    params.append([x, y, z])

                for trial, value in zip(trials, p.imap(sphere_sleep, params), strict=False):
                    frozen_trial = self.study.tell(trial, value)
                    self.study._log_completed_trial(frozen_trial)

    def validation(self, results: list[dict[str | Any, str | Any]]) -> None:
        trials = [trial for trial in self.study.trials if len(trial.params) > 0]
        for trial, result in zip(trials, results, strict=False):
            assert math.isclose(trial.params["x"], float(result["x"]), rel_tol=0.000001)
            assert math.isclose(trial.params["y"], float(result["y"]), rel_tol=0.000001)
            assert math.isclose(trial.params["z"], float(result["z"]), rel_tol=0.000001)
            assert math.isclose(trial.values[0], float(result["objective"]), rel_tol=0.000001)


class TestNelderMeadAckleySubSampler(BaseTestNelderMead):
    def setup_method(self) -> None:
        search_space = {"x": (0.0, 10.0), "y": (0.0, 10.0)}
        tpe_sampler = optuna.samplers.TPESampler(seed=43)
        sampler = NelderMeadSampler(search_space=search_space, seed=42, block=False, sub_sampler=tpe_sampler)

        self.common_setup(
            search_space=search_space,
            objective=ackley_sleep,
            result_file_name="results_ackley_sub_sampler.csv",
            study=optuna.create_study(sampler=sampler),
        )

    def optimize(self) -> None:
        num_parallel = 5
        with Pool(num_parallel) as p:
            for _ in range(30):
                trials = []
                params = []
                for _ in range(num_parallel):
                    trial = self.study.ask()

                    x = trial.suggest_float("x", *self.search_space["x"])
                    y = trial.suggest_float("y", *self.search_space["y"])

                    trials.append(trial)
                    params.append([x, y])

                for trial, value in zip(trials, p.imap(ackley_sleep, params), strict=False):
                    frozen_trial = self.study.tell(trial, value)
                    self.study._log_completed_trial(frozen_trial)

    def validation(self, results: list[dict[str | Any, str | Any]]) -> None:
        trials = [trial for trial in self.study.trials if len(trial.params) > 0]
        for trial, result in zip(trials, results, strict=False):
            assert math.isclose(trial.params["x"], float(result["x"]), rel_tol=0.000001)
            assert math.isclose(trial.params["y"], float(result["y"]), rel_tol=0.000001)
            assert math.isclose(trial.values[0], float(result["objective"]), rel_tol=0.000001)
