<<<<<<< HEAD
import pickle as pkl
from pathlib import Path
=======
from pathlib import Path
import pickle as pkl
>>>>>>> 0caccda (update v2 torch   (#383))

import optuna

from aiaccel.job import AbciJobExecutor


def main() -> None:
    n_trials = 50
    n_max_jobs = 4

    job_filename = Path("objective.sh")
    job_group = "gaa50000"

    result_filename_template = "{job.cwd}/{job.job_name}_result.pkl"

    jobs = AbciJobExecutor(job_filename, job_group, n_max_jobs=n_max_jobs)

    study = optuna.create_study(direction="minimize")

    finished_job_count = 0

    while finished_job_count <= n_trials:
        for _ in range(jobs.available_slots()):
            trial = study.ask()
            hparams = {
                "x1": trial.suggest_float("x1", 0, 10),
                "x2": trial.suggest_float("x2", 0, 10),
            }

            jobs.job_name = str(jobs.job_filename) + f"_{trial.number}"

            job = jobs.submit(
                args=[result_filename_template] + sum([[f"--{k}", f"{v:.5f}"] for k, v in hparams.items()], []),
                tag=trial,
            )

        for job in jobs.collect_finished():
            trial = job.tag

            with open(result_filename_template.format(job=job), "rb") as f:
                y = pkl.load(f)

            study.tell(trial, y)

            finished_job_count += 1


if __name__ == "__main__":
    main()
