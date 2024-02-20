import optuna

from aiaccel import JobDispatcher


def objective(hparams: dict) -> float:
    x = hparams["x"]
    return (x - 2) ** 2


def param_to_args_fn(param: dict) -> str:
    """
    Example:
    param = {
        'x': 0.5,
        'y': 0.3,
        ...
    }
    return "x=0.5 y=0.3 ..."
    """
    return " ".join([f"{k}={v}" for k, v in param.items()])


if __name__ == "__main__":
    # sampler = optuna.samplers.TPESampler(seed=42)
    sampler = optuna.samplers.RandomSampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    n_trials = 50
    n_jobs = 4

    jobs = JobDispatcher(
        objective, n_trials, n_jobs=n_jobs, param_to_args_fn=param_to_args_fn
    )

    for n in range(n_trials):
        trial = study.ask()
        hparams = {"x": trial.suggest_float("x", 0, 10)}
        jobs.submit(hparams, tag=trial, job_name=f"hpo-{n:04}")

        # y = jobs.result()  # n_jobs = 1 の場合
        # study.tell(trial, y)

        for y, trial in jobs.collect_results():  # n_jobs > 1 の場合
            study.tell(trial, y)
