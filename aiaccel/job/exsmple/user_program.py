import optuna

from aiaccel import JobDispatcher


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
    return " ".join([f"--{k}={v}" for k, v in param.items()])


if __name__ == "__main__":
    sampler = optuna.samplers.TPESampler(seed=42)
    # sampler = optuna.samplers.RandomSampler(seed=42)

    study = optuna.create_study(direction="minimize", sampler=sampler)

    n_trials = 50
    n_jobs = 4

    jobs = JobDispatcher("job.sh", n_jobs=n_jobs, param_to_args_fn=param_to_args_fn)

    # ====================================
    # # n_jobs = 1 の場合の例
    # ====================================
    for n in range(n_trials):
        trial = study.ask()
        hparams = {
            "x1": trial.suggest_float("x", 0, 10),
            "x2": trial.suggest_float("x", 0, 10),
        }

        jobs.submit(hparams, job_name=f"hpo-{n:04}")  # ジョブプールが空かないと帰ってこない

        y = jobs.result()
        study.tell(trial, y)

    """
    # ====================================
    # n_jobs > 1 の場合の例
    # ====================================
    n = 0
    while True:
        if jobs.finished_job_count >= n_trials:
            break

        trial = study.ask()
        hparams = {
            "x1": trial.suggest_float("x", 0, 10),
            "x2": trial.suggest_float("x", 0, 10),
        }

        jobs.submit(hparams, tag=trial, job_name=f"hpo-{n:04}")  # ジョブプールが空かないと帰ってこない

        for y, trial in jobs.collect_results():
            study.tell(trial, y)
        n += 1
    """
