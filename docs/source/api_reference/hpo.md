# HPO

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

## API reference

```{toctree}
:maxdepth: 2
:caption: Code Reference

aiaccel.hpo.rst
```
