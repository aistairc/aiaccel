# NelderMeadSampler

## class NelderMeadSampler(search_space, seed = None, rng=None, coeff=None, block=False, sub_sampler=None)

Sampler using the NelderMead algorithm.

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

## Example

An example of a single-objective optimization is as follows:

```python

import optuna
from aiaccel.hpo.optuna.samplers.nelder_mead_sampler import NelderMeadSampler


def objective(trial):
    x = trial.suggest_float("x", -10, 10)
    return x**2


search_space = {"x": {"low": -10, "high": 10}}
sampler = NelderMeadSampler(search_space=search_space, seed=42)
study.optimize(objective, n_trials=10)

```

## Parameters:

- search_space: dict[str, tuple[float, float]] - 
    Parameter names and corresponding lower and upper limits.
    Must be set separately from suggest_uniform
    (as the parameters must be determined at the time of before_trial).
- seed: int | None = None - 
    Random seed used for initial point calculation.
- rng: np.random.RandomState | None = None - 
    RandomState used for initial point calculation.
    If specified with seed, rng takes precedence.
- coeff: NelderMeadCoefficient | None = None - 
    Parameters used in NelderMead Algorism.
- block: bool = False - 
    Indicates whether the queue used internally is blocked or not.
    If parallelisation by optuna.optimize is enabled, it must be set with block = True
- sub_sampler: optuna.samplers.BaseSampler | None = None - 
    Sampler to output parameters when NelderMead cannot output parameters.
    Mainly intended for use on free computation nodes in parallel.
    If the sub_sampler function is enabled, it must be set with block = False.

## Methods

### before_trial(study, trial)

Trial pre-processing.

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


### sample_independent(study, trial, param_name, param_distribution)

Sample a parameter

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

### after_trial(study, trial, state, values)

Trial post-processing.

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