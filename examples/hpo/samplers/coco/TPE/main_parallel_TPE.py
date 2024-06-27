from pathlib import Path

from aiaccel.job import AbciJobExecutor


def main() -> None:
    dims = [2, 3, 5, 10, 20, 40]

    for dim in dims:
        job_filename = Path(f"objective_dim{dim}.sh")
        job_group = "gaa50073"

        jobs = AbciJobExecutor(job_filename, job_group)

        func_ids = list(range(1, 25))

        for func_id in func_ids:
            jobs.submit(
                args=["--func_id", f"{func_id}", "--dim", f"{dim}"],
            )
            print(f"submit --func_id {func_id} --dim {dim}")


if __name__ == "__main__":
    main()
