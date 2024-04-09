from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
import optuna
from optuna.distributions import BaseDistribution
from optuna.study import Study
from optuna.trial import FrozenTrial, TrialState

from aiaccel.hpo.algorithms import NelderMeadAlgorism, NelderMeadCoefficient, NelderMeadEmpty

__all__ = ["NelderMeadSampler", "NelderMeadEmpty"]


class NelderMeadSampler(optuna.samplers.BaseSampler):
    def __init__(
        self,
        search_space: dict[str, tuple[float, float]],
        seed: int | None = None,
        rng: np.random.RandomState | None = None,
        coeff: NelderMeadCoefficient | None = None,
        parallel_enabled: bool = False,
    ) -> None:
        self._search_space = search_space
        _rng = rng if rng is not None else np.random.RandomState(seed) if seed is not None else None

        self.nm = NelderMeadAlgorism(
            search_space=self._search_space,
            coeff=coeff,
            rng=_rng,
            block=parallel_enabled,
            timeout=None,
        )

        self.running_trials: list[FrozenTrial] = []
        self.finished_trials: list[tuple[FrozenTrial, float]] = []

    def infer_relative_search_space(self, study: Study, trial: FrozenTrial) -> dict[str, BaseDistribution]:
        return {}

    def sample_relative(
        self,
        study: Study,
        trial: FrozenTrial,
        search_space: dict[str, BaseDistribution],
    ) -> dict[str, Any]:
        return {}

    def before_trial(self, study: Study, trial: FrozenTrial) -> None:
        if "fixed_params" in trial.system_attrs:  # enqueued trial
            self.nm.enqueued()
            fixed_params = trial.system_attrs["fixed_params"]
            if fixed_params.keys() == self._search_space.keys():
                params = np.array([fixed_params[name] for name in self._search_space])
            else:
                raise RuntimeError("All parameters must be given when executing enqueue_trial.")
        else:
            while True:
                params = self.nm.get_vertex()
                if all(low < x < high for x, (low, high) in zip(params, self._search_space.values(), strict=False)):
                    break

                self.nm.put_value(params, np.inf)

        trial.set_user_attr("params", params)

        self.running_trials.append(trial)

    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        if trial.user_attrs["params"] is None:
            raise ValueError('trial.user_attrs["params"] is None')
        if param_name not in self._search_space:
            raise ValueError(f"The parameter name, {param_name}, is not found in the given search_space.")

        param_index = list(self._search_space.keys()).index(param_name)
        param_value = trial.user_attrs["params"][param_index]

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
        if isinstance(values, Sequence) and len(values) != 1:  # trial was finished as multiobjective
            raise RuntimeError(
                "Multidimentional trial values are obtained. "
                "NelderMeadSampler supports only single objective optimization."
            )

        self.finished_trials.insert(0, (trial, values[0] if isinstance(values, Sequence) else np.inf))

        for tgt_idx, target_trial in enumerate(self.running_trials):
            for fin_idx, (trial, value) in enumerate(self.finished_trials):
                if trial._trial_id == target_trial._trial_id:
                    self.nm.put_value(
                        trial.user_attrs["params"],
                        value,
                        enqueue="fixed_params" in trial.system_attrs,
                    )

                    self.finished_trials.pop(fin_idx)

                    break
            else:
                self.running_trials = self.running_trials[tgt_idx:]
                break
        else:
            self.running_trials = []
