import aiaccel
import optuna

from aiaccel import NelderMeadSampler


def objective(hparams: dict) -> float:
    x = hparams["x"]
    return (x - 2) ** 2


if __name__ == "__main__":
    search_space = {"x": [0, 10], "y": [0, 10]}
    sampler = NelderMeadSampler(search_space=search_space, seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    jobs = aiaccel.JobDispatcher()
    n_trial = 100

    # Run the optimization loop
    for _ in range(n_trial):
        trial = study.ask()
        hparams = {
            "x": trial.suggest_float("x", 0, 10),
        }
        y = jobs.submit(objective, hparams, trial._trial_id)
        study.tell(trial, y)

    # ===================================================
    # Run the optimization loop with parallel execution
    # ===================================================
    # while True:
    #     for _ in range(jobs.availavle_n_jobs):
    #         trial = jobs.ask()
    #         hparams = {
    #             'x': trial.suggest_float('x', 0, 10),
    #             'job_id': 0
    #         }
    #         trial, y = jobs.submit(objective, hparams, _tag_=trial)
    #
    #     jobs.wait()
    #
    #     for y, trial in jobs.collect_results():
    #         jobs.tell(trial, y)
