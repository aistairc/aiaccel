# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from __future__ import annotations

import numpy.typing as npt
from typing import Any

from collections.abc import Sequence
import math
import warnings

import numpy as np

import optuna
from optuna.distributions import BaseDistribution
from optuna.study import Study
from optuna.trial import FrozenTrial, TrialState

from aiaccel.hpo.algorithms import NelderMeadAlgorism, NelderMeadCoefficient, NelderMeadEmptyError

__all__ = ["NelderMeadSampler", "NelderMeadEmptyError"]


class NelderMeadSampler(optuna.samplers.BaseSampler):
    """Sampler using the NelderMead algorithm

    Only the initial point and shrink (number of parameters - 1) can be calculated in parallel.
    Others are basically series calculations.
    (Even if set by e.g. optuna.optimize(n_jobs=2),
    the calculation is performed in series except in initial and shrink.)
    If parallelisation is enabled, set block = True.

    When using optuna.enqueue_trial(),
    the enqueued parameters are calculated separately from the parameters determined by NelderMeadSampler
    and are taken into NelderMead if a good result is obtained.
    (Simplex is reconstituted).
    The enqueued parameters are calculated in parallel
    with the parameters determined by NelderMead if parallelisation is enabled.

    Example:

        An example of a single-objective optimization is as follows:

        .. testcode::

            import optuna
            from aiaccel.hpo.optuna.samplers.nelder_mead_sampler import NelderMeadSampler


            def objective(trial):
                x = trial.suggest_float("x", -10, 10)
                return x**2


            search_space = {"x": {"low": -10, "high": 10}}
            study = optuna.create_study(sampler=NelderMeadSampler(search_space=search_space, seed=42))
            study.optimize(objective, n_trials=10)

    Args:
        search_space: dict[str, tuple[float, float]]
            Parameter names and corresponding lower and upper limits.
            Must be set separately from suggest_uniform
            (as the parameters must be determined at the time of before_trial).
        seed: int | None = None
            Random seed used for initial point calculation.
        rng: np.random.RandomState | None = None
            RandomState used for initial point calculation.
            If specified with seed, rng takes precedence.
        coeff: NelderMeadCoefficient | None = None
            Parameters used in NelderMead Algorism.
        block: bool = False
            Indicates whether the queue used internally is blocked or not.
            If parallelisation by optuna.optimize is enabled, it must be set with block = True
        sub_sampler: optuna.samplers.BaseSampler | None = None
            Sampler to output parameters when NelderMead cannot output parameters.
            Mainly intended for use on free computation nodes in parallel.
            If the sub_sampler function is enabled, it must be set with block = False.

    Attributes:
        nm: NelderMeadAlgorism
            Instance of a class that manages the NelderMead algorithm.

    """

    def __init__(
        self,
        search_space: dict[str, tuple[int | float, int | float]],
        seed: int | None = None,
        rng: np.random.RandomState | None = None,
        coeff: NelderMeadCoefficient | None = None,
        block: bool = False,
        sub_sampler: optuna.samplers.BaseSampler | None = None,
    ) -> None:
        self._search_space = search_space
        _rng = rng if rng is not None else np.random.RandomState(seed) if seed is not None else None

        self.nm = NelderMeadAlgorism(
            dimensions=len(self._search_space),
            coeff=coeff,
            rng=_rng,
            block=block,
        )
        self.sub_sampler = sub_sampler
        self.num_trial = 1

    def infer_relative_search_space(self, study: Study, trial: FrozenTrial) -> dict[str, BaseDistribution]:
        return {}

    def sample_relative(
        self,
        study: Study,
        trial: FrozenTrial,
        search_space: dict[str, BaseDistribution],
    ) -> dict[str, Any]:
        return {}

    def _get_params(self, study: Study, trial: FrozenTrial) -> npt.NDArray[np.float64] | None:
        try:
            params = self.nm.get_vertex()
        except NelderMeadEmptyError as e:
            if self.sub_sampler is None:
                raise e
            else:
                self.sub_sampler.before_trial(study, trial)
                return None
        return params

    def _put_params(self, study: Study, trial: FrozenTrial, state: TrialState, values: Sequence[float] | None) -> None:
        if isinstance(values, list):
            system_attr = study._storage.get_trial_system_attrs(trial._trial_id)
            if "params" in system_attr and "fixed_params" not in system_attr:
                params = np.array(system_attr["params"])
            else:  # sub_sampler or enqueued
                it = zip(trial.params.values(), self._search_space.values(), strict=False)
                params = np.array([(value - low) / (high - low) for value, (low, high) in it])

            self.nm.put_value(
                params,
                values[0],
                enqueue="fixed_params" in system_attr or "sub_trial" in system_attr,
            )
            if "sub_trial" in system_attr and self.sub_sampler is not None:
                self.sub_sampler.after_trial(study, trial, state, values)

    def _resumption(self, study: Study) -> None:
        trials = study._storage.get_all_trials(study._study_id, deepcopy=False)

        for trial in trials:
            self.num_trial += 1
            if trial.values is None:
                continue

            # ask
            system_attr = study._storage.get_trial_system_attrs(trial._trial_id)
            if "fixed_params" not in system_attr:  # not enqueued trial
                self._get_params(study, trial)

            # tell
            self._put_params(study, trial, trial.state, trial.values)

    def before_trial(self, study: Study, trial: FrozenTrial) -> None:
        """Trial pre-processing.

        Preprocessing of the trial

        This determines the parameters for NelderMead.
        The determined parameters are stored in trial.user_attr["params"].
        If the NelderMead parameters cannot be output and sub_sampler is None, a NelderMeadEmptyError is raised.
        If sub_sampler is specified, sub_sampler.before_trial() is executed,
        and trial.user_attr["sub_trial"] is set to True.

        Args:
            study: Study
                Target study object.
            trial: FrozenTrial
                Target trial object.

        Returns:
            None

        """
        if self.num_trial < trial._trial_id:  # resumption
            self._resumption(study)
        self.num_trial += 1
        params: npt.NDArray[np.float64] | None

        if "fixed_params" in trial.system_attrs:  # enqueued trial
            fixed_params = trial.system_attrs["fixed_params"]
            if fixed_params.keys() != self._search_space.keys():
                raise RuntimeError("All parameters must be given when executing enqueue_trial.")
            study._storage.set_trial_system_attr(trial._trial_id, "fixed_params", trial.system_attrs["fixed_params"])

            params = np.array([fixed_params[name] for name in self._search_space])
        else:
            params = self._get_params(study, trial)
            if params is None:  # sub trial
                study._storage.set_trial_system_attr(trial._trial_id, "sub_trial", True)
                return

        study._storage.set_trial_system_attr(trial._trial_id, "params", list(params))

    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        """Sample a parameter

        For NelderMeadSampler, since the parameters are already determined in before_trial,
        return the corresponding parameters in trial.user_attr["params"].
        If trial.user_attr["sub_trial"] = True,
        execute sub_sampler.sample_independent() and return its parameters.

        Args:
            study: Study
                Target study object.
            trial: FrozenTrial
                Target trial object.
                Take a copy before modifying this object.
            param_name: str
                Name of the sampled parameter.
            param_distribution: BaseDistribution
                Distribution object that specifies a prior and/or scale of the sampling algorithm.

        Returns:
            Any
                A parameter value.

        """
        system_attr = study._storage.get_trial_system_attrs(trial._trial_id)
        if "sub_trial" in system_attr and self.sub_sampler is not None:
            param_value = self.sub_sampler.sample_independent(study, trial, param_name, param_distribution)
            if self._search_space[param_name][0] <= param_value <= self._search_space[param_name][1]:
                return param_value
            else:
                raise ValueError(
                    f"Sub_sampler {self.sub_sampler} outputs out-of-range parameters. {param_name} : {param_value}"
                )

        if param_name not in self._search_space:
            raise ValueError(f"The parameter name, {param_name}, is not found in the given search_space.")

        param_index = list(self._search_space.keys()).index(param_name)
        param_value = system_attr["params"][param_index]

        # reverse normalization
        assert hasattr(param_distribution, "high")
        assert hasattr(param_distribution, "low")
        assert hasattr(param_distribution, "log")
        if param_distribution.log:  # log scale
            high = math.log(param_distribution.high)
            low = math.log(param_distribution.low)
            param_value = math.exp((high - low) * param_value + low)
        else:
            param_value = (param_distribution.high - param_distribution.low) * param_value + param_distribution.low

        if isinstance(param_distribution, optuna.distributions.IntDistribution):
            param_value = int(param_value)
        if hasattr(param_distribution, "step") and param_distribution.step is not None:
            param_value -= (param_value - param_distribution.low) % param_distribution.step

        contains = param_distribution._contains(param_distribution.to_internal_repr(param_value))
        if not contains:
            warnings.warn(
                f"The value `{param_value}` is out of range of the parameter `{param_name}`. "
                f"The value will be used but the actual distribution is: `{param_distribution}`.",
                stacklevel=2,
            )
        return param_value

    def after_trial(
        self,
        study: Study,
        trial: FrozenTrial,
        state: TrialState,
        values: Sequence[float] | None,
    ) -> None:
        """Trial post-processing.

        Pass the parameter-result pairs to the NelderMead algorithm.
        If trial.user_attr["sub_trial"] = True, execute sub_sampler.after_trial().

        Args:
            study: Study
                Target study object.
            trial: FrozenTrial
                Target trial object.
                Take a copy before modifying this object.
            state: TrialState
                Resulting trial state.
            values: Sequence[float] | None
                Resulting trial values. Guaranteed to not be :obj:`None` if trial succeeded.

        Returns:
            None

        """
        if isinstance(values, Sequence) and len(values) != 1:  # trial was finished as multiobjective
            raise RuntimeError(
                "Multidimentional trial values are obtained. "
                "NelderMeadSampler supports only single objective optimization."
            )
        self._put_params(study, trial, state, values)
