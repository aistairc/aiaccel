import optuna

from aiaccel.job import AbciJobExecutor

sampler = optuna.samplers.TPESampler(seed=42)
# sampler = optuna.samplers.RandomSampler(seed=42)

study = optuna.create_study(direction="minimize", sampler=sampler)

n_trials = 50

# ====================================
# n_jobs = 1 の場合の例
# ====================================
jobs = AbciJobExecutor("job.sh", n_jobs=1)

for n in range(n_trials):
    trial = study.ask()
    args = [
        "--x1",
        f"{trial.suggest_float('x1', 0, 10):.4f}",
        "--x2",
        f"{trial.suggest_float('x2', 0, 10):.4f}",
    ]

    job = jobs.submit(args, "", job_name=f"hpo-{n:04}")  # ジョブプールが空かないと帰ってこない
    y = job.get_result()
    study.tell(trial, float(y))  # y は str のため float に変換


# ====================================
# n_jobs > 1 の場合の例
# ====================================
jobs = AbciJobExecutor("job.sh", n_jobs=4)

n = 0
while True:
    if jobs.finished_job_count >= n_trials:
        break

    trial = study.ask()
    args = [
        "--x1",
        f"{trial.suggest_float('x1', 0, 10):.4f}",
        "--x2",
        f"{trial.suggest_float('x2', 0, 10):.4f}",
    ]

    jobs.submit(args, "", tag=trial, job_name=f"hpo-{n:04}")  # ジョブプールが空かないと帰ってこない

    for y, trial in jobs.get_results():
        study.tell(trial, float(y))  # y は str のため float に変換
    n += 1
