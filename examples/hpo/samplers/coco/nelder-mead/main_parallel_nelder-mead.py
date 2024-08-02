from pathlib import Path

from aiaccel.job import AbciJobExecutor


def main() -> None:
    job_filename = Path("objective.sh")
    job_group = "xxx"

    jobs = AbciJobExecutor(job_filename, job_group)

    func_ids = list(range(1, 25))
    dims = [2, 3, 5, 10, 20, 40]
    instances = list(range(1, 16))
    optuna_seeds = list(range(1, 16))

    for func_id in func_ids:
        for dim in dims:
            for instance, optuna_seed in zip(instances, optuna_seeds, strict=False):
                jobs.submit(
                    args=[
                        "-l",
                        "h_rt=0:05:00",
                        "--func_id",
                        f"{func_id}",
                        "--dim",
                        f"{dim}",
                        "--instance",
                        f"{instance}",
                        "--optuna_seed",
                        f"{optuna_seed}",
                    ],
                )


if __name__ == "__main__":
    main()
