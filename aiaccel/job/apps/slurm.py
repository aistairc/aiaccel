#! /usr/bin/env python

# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from aiaccel.job.apps import SchedulerJobApp


class SlurmJobApp(SchedulerJobApp):
    array_task_id_variable = "SLURM_ARRAY_TASK_ID"
    array_job_log_suffix = ".%a.log"
    array_job_status_suffix = ".${SLURM_ARRAY_TASK_ID}.out"
    submit_command_key = "sbatch"
    submit_args_key = "sbatch_args"

    def __init__(self) -> None:
        super().__init__("slurm.yaml")

    def build_job_script(self) -> str:
        return f"""\
#! /bin/bash
#SBATCH -o {self.job_log_filename}
#SBATCH -t {self.args.walltime}

set -eE -o pipefail
trap 'echo $? > {self.job_status_filename}' ERR EXIT  # at error and exit


{self.config.script_prologue}

{self.job}
"""


def main() -> None:
    SlurmJobApp().run()


if __name__ == "__main__":
    main()
