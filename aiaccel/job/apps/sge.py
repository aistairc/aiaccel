#! /usr/bin/env python


# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

import re
import shlex

from aiaccel.job.apps import SchedulerJobApp


def _append_notify_option_if_needed(qsub: str, qsub_args: str) -> str:
    notify_option_pattern = re.compile(r"(?<!\S)-notify(?=\s|$)")
    has_notify_in_qsub = notify_option_pattern.search(qsub) is not None
    has_notify_in_qsub_args = notify_option_pattern.search(qsub_args) is not None
    if has_notify_in_qsub or has_notify_in_qsub_args:
        return qsub_args

    return f"-notify {qsub_args}" if qsub_args else "-notify"


class SgeJobApp(SchedulerJobApp):
    """Job application for submitting scripts to an SGE scheduler."""

    def __init__(self) -> None:
        super().__init__("sge.yaml")

    def build_submit_command(self) -> tuple[str, str]:
        """Build the SGE submission command."""
        job_name = (
            str(self.config.get("job_name", self.args.log_filename.with_suffix(""))).replace("/", "_").replace(" ", "_")
        )
        if job_name[:1].isdigit():
            job_name = f"_{job_name}"

        qsub = self.config.qsub.format(args=self.args, job_name=job_name)
        qsub_args = self.config[self.mode].qsub_args.format(args=self.args)
        qsub_args = _append_notify_option_if_needed(qsub, qsub_args)

        return (qsub, qsub_args)

    def prepare_array_job_context(self) -> None:
        """Prepare SGE-specific context for array jobs."""
        log_filename_prefix = shlex.quote(str(self.args.log_filename.with_suffix("").absolute()))
        self.job = f"""\
for LOCAL_PROC_INDEX in {{1..{self.args.n_procs}}}; do
    TASK_INDEX=$(( SGE_TASK_ID + {self.args.n_tasks_per_proc} * (LOCAL_PROC_INDEX - 1) ))

    if [[ $TASK_INDEX -gt {self.args.n_tasks} ]]; then
        break
    fi

    TASK_INDEX=$TASK_INDEX \\
    TASK_STEPSIZE={self.args.n_tasks_per_proc} \\
        {self.job} > \\
        {log_filename_prefix}.${{SGE_TASK_ID}}-${{LOCAL_PROC_INDEX}}.log 2>&1 &

    pids[$LOCAL_PROC_INDEX]=$!
done

for i in "${{!pids[@]}}"; do
    wait ${{pids[$i]}}
done
"""
        self.job_log_filename = shlex.quote(str(self.args.log_filename.with_suffix(".$TASK_ID.log").absolute()))
        self.job_status_filename = self.args.log_filename.with_suffix(".${SGE_TASK_ID}.out").absolute()
        self.status_filename_list = [
            self.args.log_filename.with_suffix(f".{array_idx + 1}.out").absolute()
            for array_idx in range(
                0,
                self.args.n_tasks,
                self.args.n_tasks_per_proc * self.args.n_procs,
            )
        ]

    def build_job_script(self) -> str:
        """Build the SGE job script."""
        return f"""\
#! /bin/bash

#$-S /bin/bash
#$-j y
#$-cwd
#$-o {self.job_log_filename}

set -eE -o pipefail
trap 'echo $? > "{self.job_status_filename}"' ERR EXIT  # at error and exit
trap 'echo 143 > "{self.job_status_filename}"' TERM  # at termination (by job scheduler)
trap 'echo 140 > "{self.job_status_filename}"' USR2


{self.config.script_prologue}

{self.job}
"""


def main() -> None:
    """Run the SGE job application entry point."""
    SgeJobApp().run()


if __name__ == "__main__":
    main()
