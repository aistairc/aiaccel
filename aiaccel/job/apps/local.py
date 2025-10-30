#! /usr/bin/env python3


import logging
from math import ceil
from pathlib import Path
import shlex
import subprocess

from aiaccel.job.apps import prepare_argument_parser

logger = logging.getLogger(__name__)


def main() -> None:
    # Load configuration (from the default YAML string)
    config, parser, sub_parsers = prepare_argument_parser("local.yaml")

    args = parser.parse_args()
    mode = args.mode + "-array" if getattr(args, "n_tasks", None) is not None else args.mode

    for key in ["walltime", "n_nodes", "n_tasks_per_procs"]:
        if getattr(args, key, None) is not None:
            logger.warning(f"Argument '{key}' is defined for compatibility and will not be used in aiaccel-job local.")

    # Prepare the job script and arguments
    job = config[mode].job.format(command=shlex.join(args.command), args=args)

    if mode in ["cpu-array", "gpu-array"]:
        n_tasks_per_proc = ceil(args.n_tasks / args.n_procs)
        job = f"""\
for LOCAL_PROC_INDEX in {{1..{args.n_procs}}}; do
    TASK_INDEX=$(( 1 + {n_tasks_per_proc} * (LOCAL_PROC_INDEX - 1) ))

    if [[ $TASK_INDEX -gt {args.n_tasks} ]]; then
        break
    fi

    TASK_INDEX=$TASK_INDEX \\
    TASK_STEPSIZE={n_tasks_per_proc} \\
        {job} 2>&1 | tee {args.log_filename.with_suffix("")}.${{LOCAL_PROC_INDEX}}.log &

    pids[$LOCAL_PROC_INDEX]=$!
done

for i in "${{!pids[@]}}"; do
    wait ${{pids[$i]}}
done
"""
    else:
        job = f"{job} 2>&1 | tee {args.log_filename}"

    job_script = f"""\
#! /bin/bash

set -eE -o pipefail
trap 'exit $?' ERR EXIT  # at error and exit
trap 'echo 143' TERM  # at termination (by job scheduler)
trap 'kill 0' INT EXIT


{config.script_prologue}

{job}
"""

    # Create the job script file, remove old status files, and run the job
    args.log_filename.parent.mkdir(exist_ok=True, parents=True)

    job_filename: Path = args.log_filename.with_suffix(".sh")
    with open(job_filename, "w") as f:
        f.write(job_script)

    subprocess.run(f"bash {job_filename}", shell=True, check=True)


if __name__ == "__main__":
    main()
