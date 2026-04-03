#! /usr/bin/env python3


# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

import logging
from math import ceil
from pathlib import Path
import subprocess

from aiaccel.job.apps import JobApp

logger = logging.getLogger(__name__)


class LocalJobApp(JobApp):
    def __init__(self) -> None:
        super().__init__("local.yaml")

        for key in ["walltime", "n_nodes", "n_tasks_per_proc"]:
            if getattr(self.args, key, None) is not None:
                logger.warning(
                    f"Argument '{key}' is defined for compatibility and will not be used in aiaccel-job local."
                )

    def build_submit_command(self) -> tuple[str, str]:
        return ("bash", "")

    def prepare_array_job_context(self) -> None:
        n_tasks_per_proc = ceil(self.args.n_tasks / self.args.n_procs)
        self.job = f"""\
for LOCAL_PROC_INDEX in {{1..{self.args.n_procs}}}; do
    TASK_INDEX=$(( 1 + {n_tasks_per_proc} * (LOCAL_PROC_INDEX - 1) ))

    if [[ $TASK_INDEX -gt {self.args.n_tasks} ]]; then
        break
    fi

    TASK_INDEX=$TASK_INDEX \\
    TASK_STEPSIZE={n_tasks_per_proc} \\
        {self.job} 2>&1 | tee {self.args.log_filename.with_suffix("")}.${{LOCAL_PROC_INDEX}}.log &

    pids[$LOCAL_PROC_INDEX]=$!
done

for i in "${{!pids[@]}}"; do
    wait ${{pids[$i]}}
done
"""
        self.job_log_filename = self.args.log_filename.resolve()
        self.job_status_filename = self.args.log_filename.with_suffix(".out").resolve()
        self.status_filename_list = [self.job_status_filename]

    def build_job_script(self) -> str:
        return f"""\
#! /bin/bash

set -eE -o pipefail
trap 'echo $? > {self.job_status_filename}' ERR EXIT  # at error and exit
trap 'echo 143 > {self.job_status_filename}' TERM  # at termination (by job scheduler)
trap 'kill 0' INT
exec > >(tee -a {self.job_log_filename}) 2>&1


{self.config.script_prologue}

{self.job}
"""

    def run(self) -> None:
        self.build_job()
        self.prepare_job_context()
        job_script = self.build_job_script()

        self.args.log_filename.parent.mkdir(exist_ok=True, parents=True)

        job_filename: Path = self.args.log_filename.with_suffix(".sh")
        with open(job_filename, "w") as f:
            f.write(job_script)

        subprocess.run(f"bash {job_filename}", shell=True, check=True)


def main() -> None:
    LocalJobApp().run()


if __name__ == "__main__":
    main()
