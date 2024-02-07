import optuna
from aiaccel import JobDispatcher


def objective(hparams: dict) -> float:
    x = hparams["x"]
    return (x - 2) ** 2


if __name__ == "__main__":
    search_space = {"x": [0, 10], "y": [0, 10]}
    sampler = optuna.samplers.RandomSampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    jobs = JobDispatcher(n_jobs=4)
    n_trial = 50

    while True:
        n = min(n_trial - jobs.finished_job_count, jobs.abvailable_worker_count)
        for _ in range(n):
            trial = study.ask()
            hparams = {"x": trial.suggest_float("x", 0, 10)}
            jobs.submit(objective, hparams, trial.number, trial)

        jobs.wait()

        for y, trial in jobs.collect_results():
            study.tell(trial, y)

        if jobs.finished_job_count >= n_trial:
            break

    [print(r) for r in jobs.all_result]
