import optuna

from aiaccel import JobDispatcher

template = """
#!/bin/bash

#$-l rt_C.small=1
#$-cwd

source /etc/profile.d/modules.sh
module load gcc/12.2.0
module load python/3.10/3.10.10
module load cuda/11.8
module load cudnn/8.6
"""


def objective(hparams: dict) -> float:
    x1 = hparams["x1"]
    x2 = hparams["x2"]
    print("Your job 41780660 (hpo-0002.sh) has been submitted")
    return (x1**2) - (4.0 * x1) + (x2**2) - x2 - (x1 * x2)


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
    sampler = optuna.samplers.TPESampler(seed=42)
    # sampler = optuna.samplers.RandomSampler(seed=42)

    study = optuna.create_study(direction="minimize", sampler=sampler)

    n_trials = 50
    n_jobs = 4

    jobs = JobDispatcher(
        objective,
        n_trials,
        n_jobs=n_jobs,
        param_to_args_fn=param_to_args_fn,
        template=template,
    )

    for n in range(n_trials):
        trial = study.ask()
        hparams = {
            "x1": trial.suggest_float("x", 0, 10),
            "x2": trial.suggest_float("x", 0, 10),
        }

        jobs.submit(hparams, tag=trial, job_name=f"hpo-{n:04}")  # ジョブプールが空かないと帰ってこない

        # y = jobs.result()  # n_jobs = 1 の場合
        # study.tell(trial, y)

        for y, trial in jobs.collect_results():  # n_jobs > 1 の場合
            study.tell(trial, y)
