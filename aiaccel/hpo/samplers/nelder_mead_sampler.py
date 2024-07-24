from __future__ import annotations

import copy
import math
import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
import optuna
from optuna.distributions import BaseDistribution
from optuna.study import Study
from optuna.trial import FrozenTrial, TrialState

from aiaccel.hpo.algorithms import NelderMeadAlgorism, NelderMeadCoefficient, NelderMeadEmptyError

__all__ = ["NelderMeadSampler", "NelderMeadEmptyError"]


class NelderMeadSampler(optuna.samplers.BaseSampler):
    def __init__(
        self,
        search_space: dict[str, tuple[int | float, int | float] | tuple[int | float, int | float, bool]],
        seed: int | None = None,
        rng: np.random.RandomState | None = None,
        coeff: NelderMeadCoefficient | None = None,
        block: bool = False,
        sub_sampler: optuna.samplers.BaseSampler | None = None,
    ) -> None:
        _search_space = copy.deepcopy(search_space)
        self._search_space: dict[str, tuple[int | float, int | float]] = {
            key: (math.log(value[0]), math.log(value[1])) if len(value) == 3 and value[2] else (value[0], value[1])
            for key, value in _search_space.items()
        }
        self._log = {key: value[2] if len(value) == 3 else False for key, value in _search_space.items()}
        _rng = rng if rng is not None else np.random.RandomState(seed) if seed is not None else None

        self.nm = NelderMeadAlgorism(
            search_space=self._search_space,
            coeff=coeff,
            rng=_rng,
            block=block,
        )
        self.sub_sampler = sub_sampler

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
            fixed_params = trial.system_attrs["fixed_params"]
            if fixed_params.keys() != self._search_space.keys():
                raise RuntimeError("All parameters must be given when executing enqueue_trial.")

            params = np.array([fixed_params[name] for name in self._search_space])
        else:
            try:
                while True:
                    params = self.nm.get_vertex()

                    if all(low < x < high for x, (low, high) in zip(params, self._search_space.values(), strict=False)):
                        break

                    self.nm.put_value(params, np.inf)
            except NelderMeadEmptyError as e:
                if self.sub_sampler is None:
                    raise e
                else:
                    self.sub_sampler.before_trial(study, trial)
                    trial.set_user_attr("sub_trial", True)
                    return

        trial.set_user_attr("params", params)

    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        if "sub_trial" in trial.user_attrs and self.sub_sampler is not None:
            param_value = self.sub_sampler.sample_independent(study, trial, param_name, param_distribution)
            value = math.log(param_value) if self._log[param_name] else param_value
            if self._search_space[param_name][0] <= value <= self._search_space[param_name][1]:
                return param_value
            else:
                raise ValueError(
                    f"Sub_sampler {self.sub_sampler} outputs out-of-range parameters. {param_name} : {param_value}"
                )

        if param_name not in self._search_space:
            raise ValueError(f"The parameter name, {param_name}, is not found in the given search_space.")

        param_index = list(self._search_space.keys()).index(param_name)
        param_value = trial.user_attrs["params"][param_index]

        if param_name in self._log and self._log[param_name]:
            param_value = math.exp(param_value)

        if isinstance(param_distribution, optuna.distributions.IntDistribution):
            param_value = int(param_value)

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
        if isinstance(values, list):
            params = [math.log(value) if self._log[key] else value for key, value in trial.params.items()]
            self.nm.put_value(
                np.array(params),
                values[0],
                enqueue="fixed_params" in trial.system_attrs or "sub_trial" in trial.user_attrs,
            )
            if "sub_trial" in trial.user_attrs and self.sub_sampler is not None:
                self.sub_sampler.after_trial(study, trial, state, values)
