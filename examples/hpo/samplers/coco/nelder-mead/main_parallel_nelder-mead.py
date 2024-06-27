from pathlib import Path

from aiaccel.job import AbciJobExecutor


def main() -> None:
    job_filename = Path("objective.sh")
    job_group = "gaa50073"

    jobs = AbciJobExecutor(job_filename, job_group)

    func_ids = list(range(1, 25))
    dims = [2, 3, 5, 10, 20, 40]

    for func_id in func_ids:
        for dim in dims:
            jobs.submit(
                args=["--func_id", f"{func_id}", "--dim", f"{dim}"],
            )


if __name__ == "__main__":
    main()
