import optuna

from aiaccel import JobDispatcher


def objective(hparams: dict) -> float:
    x = hparams["x"]
    return (x - 2) ** 2


if __name__ == "__main__":
    sampler = optuna.samplers.TPESampler(seed=42)
    # sampler = optuna.samplers.RandomSampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    n_trials = 50
    n_jobs = 1

    jobs = JobDispatcher(objective, n_trials, n_jobs=n_jobs)

    for n in range(n_trials):
        trial = study.ask()
        hparams = {"x": trial.suggest_float("x", 0, 10)}
        jobs.submit(hparams, tag=trial, job_name=f"hpo-{n:08}")
        for y, trial in jobs.collect_results():
            study.tell(trial, y)

    [print(result) for result in jobs.results]
