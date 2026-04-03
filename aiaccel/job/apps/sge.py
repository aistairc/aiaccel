#! /usr/bin/env python


# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from aiaccel.job.apps import SchedulerJobApp


class SgeJobApp(SchedulerJobApp):
    def __init__(self) -> None:
        super().__init__("sge.yaml")

    def build_submit_command(self) -> tuple[str, str]:
        return (
            self.config.qsub.format(args=self.args),
            self.config[self.mode].qsub_args.format(args=self.args),
        )

    def prepare_array_job_context(self) -> None:
        self.job = f"""\
for LOCAL_PROC_INDEX in {{1..{self.args.n_procs}}}; do
    TASK_INDEX=$(( SGE_TASK_ID + {self.args.n_tasks_per_proc} * (LOCAL_PROC_INDEX - 1) ))

    if [[ $TASK_INDEX -gt {self.args.n_tasks} ]]; then
        break
    fi

    TASK_INDEX=$TASK_INDEX \\
    TASK_STEPSIZE={self.args.n_tasks_per_proc} \\
        {self.job} > \\
        {self.args.log_filename.with_suffix("")}.${{SGE_TASK_ID}}-${{LOCAL_PROC_INDEX}}.log 2>&1 &

    pids[$LOCAL_PROC_INDEX]=$!
done

for i in "${{!pids[@]}}"; do
    wait ${{pids[$i]}}
done
"""
        self.job_log_filename = self.args.log_filename.with_suffix(".$TASK_ID.log").resolve()
        self.job_status_filename = self.args.log_filename.with_suffix(".${SGE_TASK_ID}.out").resolve()
        self.status_filename_list = [
            self.args.log_filename.with_suffix(f".{array_idx + 1}.out").resolve()
            for array_idx in range(
                0,
                self.args.n_tasks,
                self.args.n_tasks_per_proc * self.args.n_procs,
            )
        ]

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
