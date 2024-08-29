from itertools import product
from pathlib import Path

from aiaccel.job.abci_job import AbciJob


def main() -> None:
    job_filename = Path("objective.sh")
    job_group = "xxx"

    sampler_names = ["nelder-mead", "nelder-mead-subTPE", "TPE"]
    func_ids = list(range(1, 25))
    dims = [2, 3, 5, 10, 20, 40]
    execute_times = ["0:01:00", "0:02:00", "0:03:00", "0:10:00", "0:30:00", "3:00:00"]
    instances = list(range(1, 16))
    optuna_seeds = list(range(1, 16))

    combinations = product(
        sampler_names, func_ids, zip(dims, execute_times, strict=False), zip(instances, optuna_seeds, strict=False)
    )

    for sampler_name, func_id, (dim, execute_time), (instance, optuna_seed) in combinations:
        execute_time = "0:05:00" if sampler_name == "nelder-mead" else execute_time
        print(sampler_name, (func_id, execute_time), dim, (instance, optuna_seed))
        job = AbciJob(
            job_filename,
            job_group,
            qsub_args=[
                "-l",
                f"h_rt={execute_time}",
            ],
            args=[
                "--func_id",
                f"{func_id}",
                "--dim",
                f"{dim}",
                "--instance",
                f"{instance}",
                "--optuna_seed",
                f"{optuna_seed}",
                "--sampler_name",
                f"{sampler_name}",
            ],
        )
        job.submit()


if __name__ == "__main__":
    main()
