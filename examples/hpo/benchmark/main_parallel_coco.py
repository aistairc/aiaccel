# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from concurrent.futures import ThreadPoolExecutor
from itertools import product
import subprocess


def main() -> None:
    sampler_names = ["nelder-mead", "nelder-mead-subTPE", "TPE"]
    func_ids = list(range(1, 25))
    dims = [2, 3, 5, 10, 20, 40]
    execute_times = ["0:01:00", "0:02:00", "0:03:00", "0:10:00", "0:30:00", "3:00:00"]
    instances = list(range(1, 16))
    optuna_seeds = list(range(1, 16))

    combinations = product(
        sampler_names, func_ids, zip(dims, execute_times, strict=False), zip(instances, optuna_seeds, strict=False)
    )

    with ThreadPoolExecutor() as pool:
        for sampler_name, func_id, (dim, execute_time), (instance, optuna_seed) in combinations:
            execute_time = "0:05:00" if sampler_name == "nelder-mead" else execute_time
            print(sampler_name, (func_id, execute_time), dim, (instance, optuna_seed))

            aiaccel_job_command = f"""\
aiaccel-job pbs --config job_config.yaml cpu --walltime {execute_time} log/job_{func_id}_{dim}_{instance}.log \
-- python3.13 experiment_coco.py --func_id {func_id} --dim {dim} \
--instance {instance} --optuna_seed {optuna_seed} --sampler_name {sampler_name}
"""

            pool.submit(subprocess.run, aiaccel_job_command, shell=True)


if __name__ == "__main__":
    main()
