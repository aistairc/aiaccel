from multiprocessing import Pool

import numpy as np

import optuna

from aiaccel.hpo.optuna.samplers.nelder_mead_sampler import NelderMeadEmptyError, NelderMeadSampler, SearchSpace

search_space: dict[str, SearchSpace] = {
    "x": {"low": -10.0, "high": 10.0},
    "y": {"low": -10.0, "high": 10.0},
}


def sphere(params: list[float]) -> float:
    return float(np.sum(np.asarray(params) ** 2))


if __name__ == "__main__":
    study = optuna.create_study(sampler=NelderMeadSampler(search_space=search_space, seed=42))
    _rng = np.random.RandomState(seed=42)
    num_parallel = 5

    with Pool(num_parallel) as p:
        for _ in range(30):
            trials = []
            params = []
            for _ in range(num_parallel):
                try:  # nelder-mead
                    trial = study.ask()
                except NelderMeadEmptyError:  # random sampling
                    study.enqueue_trial(
                        {
                            "x": _rng.uniform(search_space["x"]["low"], search_space["x"]["high"]),
                            "y": _rng.uniform(search_space["y"]["low"], search_space["y"]["high"]),
                        }
                    )
                    trial = study.ask()

                x = trial.suggest_float("x", search_space["x"]["low"], search_space["x"]["high"])
                y = trial.suggest_float("y", search_space["y"]["low"], search_space["y"]["high"])

                trials.append(trial)
                params.append([x, y])

            for trial, value in zip(trials, p.imap(sphere, params), strict=False):
                frozen_trial = study.tell(trial, value)
                study._log_completed_trial(frozen_trial)
