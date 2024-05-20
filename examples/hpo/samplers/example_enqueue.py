from multiprocessing import Pool

import numpy as np
import optuna

from aiaccel.hpo.samplers.nelder_mead_sampler import NelderMeadEmptyError, NelderMeadSampler

search_space={"x": (0.0, 10.0), "y": (0.0, 10.0)}

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
                            "x": _rng.uniform(*search_space["x"]),
                            "y": _rng.uniform(*search_space["y"]),
                        }
                    )
                    trial = study.ask()

                x = trial.suggest_float("x", *search_space["x"])
                y = trial.suggest_float("y", *search_space["y"])

                trials.append(trial)
                params.append([x, y])

            for trial, value in zip(trials, p.imap(sphere, params), strict=False):
                frozen_trial = study.tell(trial, value)
                study._log_completed_trial(frozen_trial)
