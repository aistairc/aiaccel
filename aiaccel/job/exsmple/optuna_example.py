import optuna


def objective(trial) -> float:
    x = trial.suggest_float("x", 0, 10)
    return (x - 2) ** 2


if __name__ == "__main__":
    n_trials = 50
    n_jobs = 4

    sampler = optuna.samplers.TPESampler(seed=42)
    # sampler = optuna.samplers.RandomSampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)
