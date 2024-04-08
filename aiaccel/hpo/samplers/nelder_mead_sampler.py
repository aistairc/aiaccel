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
            search_space=self._search_space, coeff=coeff, rng=_rng, block=parallel_enabled, timeout=None
        )

        self.running_trial_id: list[int] = []
        self.enqueue_running_trial_id: list[int] = []
        self.result_stack: dict[int, float] = {}

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
        # TODO: system_attrs is deprecated.
        if "fixed_params" in trial.system_attrs:  # enqueue_trial
            params = self.nm.enqueue_vertex(trial.system_attrs["fixed_params"])
            self.enqueue_running_trial_id.append(trial._trial_id)
        else:  # nelder mead
            while True:
                params = self.nm.get_vertex()

                if all(low < x < high for x, (low, high) in zip(params, self._search_space.values(), strict=False)):
                    break
                else:
                    self.nm.put_value(np.inf)
            self.running_trial_id.append(trial._trial_id)

        trial.set_user_attr("params", params)

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
        if isinstance(values, list):
            self.result_stack[trial._trial_id] = values[0]

            if len(self.running_trial_id) + len(self.enqueue_running_trial_id) == len(self.result_stack):
                for trial_id in self.running_trial_id:
                    self.nm.put_value(self.result_stack[trial_id])
                for trial_id in self.enqueue_running_trial_id:
                    self.nm.put_enqueue_value_queue(self.result_stack[trial_id])
                self.running_trial_id = []
                self.result_stack = {}
                self.enqueue_running_trial_id = []
