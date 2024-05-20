import numpy as np
import optuna

from aiaccel.hpo.samplers.nelder_mead_sampler import NelderMeadSampler

search_space={"x": (0.0, 10.0), "y": (0.0, 10.0)}

def sphere(trial: optuna.trial.Trial) -> float:
    params = []
    for name, distribution in search_space.items():
        params.append(trial.suggest_float(name, *distribution))

    return float(np.sum(np.asarray(params) ** 2))

if __name__ == "__main__":
    study = optuna.create_study(sampler=NelderMeadSampler(search_space=search_space, seed=42))
    study.optimize(func=sphere, n_trials=100)
