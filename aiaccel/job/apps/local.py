#! /usr/bin/env python3


from functools import partial
import logging
from multiprocessing import Pool
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

    for key in ["walltime", "n_nodes"]:
        if getattr(args, key, None) is not None:
            logger.warning(f"Argument '{key}' is defined for compatibility and will not be used in aiaccel-job local.")

    # Prepare the job script and arguments
    job = config[mode].job.format(command=shlex.join(args.command), args=args)

    if mode in ["cpu-array", "gpu-array"]:
        job = f"TASK_STEPSIZE={args.n_tasks_per_proc} {job}"
        log_filename = f"{args.log_filename.with_suffix('')}.${{TASK_INDEX}}.log"
    else:
        log_filename = args.log_filename

    job_script = f"""\
#! /bin/bash

set -eE -o pipefail
trap 'exit $?' ERR EXIT  # at error and exit
trap 'echo 143' TERM  # at termination (by job scheduler)


{config.script_prologue}

{job} 2>&1 | tee {log_filename}
"""

    # Create the job script file, remove old status files, and run the job
    args.log_filename.parent.mkdir(exist_ok=True, parents=True)

    job_filename: Path = args.log_filename.with_suffix(".sh")
    with open(job_filename, "w") as f:
        f.write(job_script)

    if mode in ["cpu-array", "gpu-array"]:
        worker = partial(subprocess.run, shell=True, check=True)
        with Pool(processes=args.n_procs) as pool:
            cmd_list = []
            for ii in range(0, args.n_tasks, args.n_tasks_per_proc):
                cmd_list.append(f"TASK_INDEX={ii + 1} bash {job_filename}")

            for _ in pool.imap_unordered(worker, cmd_list):
                pass
    else:
        subprocess.run(f"bash {job_filename}", shell=True, check=True)


if __name__ == "__main__":
    main()
