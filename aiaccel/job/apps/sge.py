#! /usr/bin/env python


# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from aiaccel.job.apps import SchedulerJobApp


class SgeJobApp(SchedulerJobApp):
    config_name = "sge.yaml"
    array_task_id_variable = "SGE_TASK_ID"
    array_job_log_suffix = ".$TASK_ID.log"
    array_job_status_suffix = ".${SGE_TASK_ID}.out"
    submit_command_key = "qsub"
    submit_args_key = "qsub_args"

    def build_job_script(self) -> str:
        return f"""\
#! /bin/bash

#$-j y
#$-cwd
#$-o {self.job_log_filename}

set -eE -o pipefail
trap 'echo $? > {self.job_status_filename}' ERR EXIT  # at error and exit
trap 'echo 143 > {self.job_status_filename}' TERM  # at termination (by job scheduler)

if [ -n "$PBS_O_WORKDIR" ] && [ "$PBS_ENVIRONMENT" != "PBS_INTERACTIVE" ]; then
    cd $PBS_O_WORKDIR
fi


{self.config.script_prologue}

{self.job}
"""


def main() -> None:
    SgeJobApp().run()


if __name__ == "__main__":
    main()
