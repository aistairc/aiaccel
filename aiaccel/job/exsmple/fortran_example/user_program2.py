import optuna

from aiaccel import AbciJobExecutor


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


sampler = optuna.samplers.TPESampler(seed=42)
# sampler = optuna.samplers.RandomSampler(seed=42)

study = optuna.create_study(direction="minimize", sampler=sampler)

n_trials = 50
n_jobs = 4

jobs = AbciJobExecutor("job2.sh", n_jobs=n_jobs)

# ====================================
# # n_jobs = 1 の場合の例
# ====================================
# for n in range(n_trials):
#     trial = study.ask()
#     args = [
#         f"{trial.suggest_float('x1', 0, 10):.4f}",
#         f"{trial.suggest_float('x2', 0, 10):.4f}",
#     ]

#     job = jobs.submit(args, job_name=f"hpo-{n:04}")  # ジョブプールが空かないと帰ってこない
#     y = job.result()
#     study.tell(trial, y)

# ====================================
# n_jobs > 1 の場合の例
# ====================================
n = 0
while True:
    if jobs.finished_job_count >= n_trials:
        break

    trial = study.ask()
    args = [
        f"{trial.suggest_float('x1', 0, 10):.4f}",
        f"{trial.suggest_float('x2', 0, 10):.4f}",
    ]

    jobs.submit(args, tag=trial, job_name=f"hpo-{n:04}")  # ジョブプールが空かないと帰ってこない

    for y, trial in jobs.get_results():
        study.tell(trial, y)
    n += 1

print(study.best_params)
