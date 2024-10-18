import numpy as np

import optuna

from aiaccel.hpo.optuna.samplers.nelder_mead_sampler import NelderMeadSampler, SearchSpace

search_space: dict[str, SearchSpace] = {
    "x": {"low": -10.0, "high": 10.0},
    "y": {"low": -10.0, "high": 10.0},
}


def sphere(trial: optuna.trial.Trial) -> float:
    params = []
    for name, distribution in search_space.items():
        params.append(trial.suggest_float(name, distribution["low"], distribution["high"]))

    return float(np.sum(np.asarray(params) ** 2))


if __name__ == "__main__":
    study = optuna.create_study(sampler=NelderMeadSampler(search_space=search_space, seed=42))
    study.optimize(func=sphere, n_trials=100)
