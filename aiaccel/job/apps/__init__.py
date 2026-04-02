# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from typing import cast

from abc import ABC, abstractmethod
from argparse import ArgumentParser, Namespace, _SubParsersAction
from importlib import resources
import os
from pathlib import Path
import shlex
import subprocess
import time

from omegaconf import DictConfig

from aiaccel.config import prepare_config, print_config, setup_omegaconf

setup_omegaconf()


def _prepare_argument_parser(
    default_config_name: str,
) -> tuple[DictConfig, ArgumentParser, _SubParsersAction]:  # type: ignore
    """Prepare a parser and load the job application configuration.

    Args:
        default_config_name (str): Default configuration filename used when no explicit
            config path is provided.

    Returns:
        tuple[DictConfig, ArgumentParser, _SubParsersAction]: Loaded config, parser, and
        sub-parser container.
    """
    parser = ArgumentParser(add_help=False)
    parser.add_argument("--print_config", action="store_true")
    parser.add_argument("--config", type=Path, default=None)
    args, _ = parser.parse_known_args()

    args.config = Path(
        args.config
        or os.environ.get("AIACCEL_JOB_CONFIG")
        or (Path(str(resources.files(__package__) / "config")) / default_config_name)
    )  # type: ignore

    config = cast(DictConfig, prepare_config(args.config))

    if args.print_config:
        print_config(config)

    parser = ArgumentParser()
    parser.add_argument("--print_config", action="store_true")
    parser.add_argument("--config", type=Path)
    sub_parsers = parser.add_subparsers(dest="mode", required=True)

    parent_parser = ArgumentParser(add_help=False)
    parent_parser.add_argument("--walltime", type=str, default=config.walltime)
    parent_parser.add_argument("log_filename", type=Path)
    parent_parser.add_argument("command", nargs="+")

    sub_parser = sub_parsers.add_parser("cpu", parents=[parent_parser])
    sub_parser.add_argument("--n_tasks", type=int)
    sub_parser.add_argument("--n_tasks_per_proc", type=int, default=config["cpu-array"].n_tasks_per_proc)
    sub_parser.add_argument("--n_procs", type=int, default=config["cpu-array"].n_procs)

    sub_parser = sub_parsers.add_parser("gpu", parents=[parent_parser])
    sub_parser.add_argument("--n_tasks", type=int)
    sub_parser.add_argument("--n_tasks_per_proc", type=int, default=config["gpu-array"].n_tasks_per_proc)
    sub_parser.add_argument("--n_procs", type=int, default=config["gpu-array"].n_procs)

    sub_parser = sub_parsers.add_parser("mpi", parents=[parent_parser])
    sub_parser.add_argument("--n_procs", type=int, required=True)
    sub_parser.add_argument("--n_nodes", type=int, default=config["mpi"].n_nodes)

    sub_parser = sub_parsers.add_parser("train", parents=[parent_parser])
    sub_parser.add_argument("--n_gpus", type=int)

    return config, parser, sub_parsers


def _is_skip_job_submission(job_filename: Path, job_script: str, status_filename_list: list[Path]) -> bool:
    has_same_job_script = job_filename.exists() and job_filename.read_text() == job_script
    has_success_status_files = all(
        status_filename.exists() and status_filename.read_text().strip() == "0"
        for status_filename in status_filename_list
    )
    return has_same_job_script and has_success_status_files


class JobApp(ABC):
    """Base class for job application entry points.

    On initialization, this class loads the configuration, parses command line
    arguments, and determines the execution mode including array variants.
    """

    config_name: str

    def __init__(self) -> None:
        self.config, parser, _ = _prepare_argument_parser(self.config_name)
        self.args = parser.parse_args()
        self.mode = self.args.mode + "-array" if getattr(self.args, "n_tasks", None) is not None else self.args.mode

    def build_job(self) -> None:
        """Build the base job command from the current config and arguments."""
        self.job = cast(str, self.config[self.mode].job.format(command=shlex.join(self.args.command), args=self.args))

    @abstractmethod
    def build_job_script(self) -> str:
        """Build the job script to execute."""
        pass


class SchedulerJobApp(JobApp):
    """Base class for scheduler-backed job applications.

    This class extends :class:`JobApp` with scheduler-specific handling such as
    array job expansion, status file management, and job submission.
    """

    array_task_id_variable: str
    array_job_log_suffix: str
    array_job_status_suffix: str
    submit_command_key: str
    submit_args_key: str

    def prepare_job_context(self) -> None:
        """Prepare scheduler-specific job and status file context."""
        if self.mode in ["cpu-array", "gpu-array"]:
            self.job = f"""\
for LOCAL_PROC_INDEX in {{1..{self.args.n_procs}}}; do
    TASK_INDEX=$(( {self.array_task_id_variable} + {self.args.n_tasks_per_proc} * (LOCAL_PROC_INDEX - 1) ))

    if [[ $TASK_INDEX -gt {self.args.n_tasks} ]]; then
        break
    fi

    TASK_INDEX=$TASK_INDEX \\
    TASK_STEPSIZE={self.args.n_tasks_per_proc} \\
        {self.job} > \\
        {self.args.log_filename.with_suffix("")}.${{{self.array_task_id_variable}}}-${{LOCAL_PROC_INDEX}}.log 2>&1 &

    pids[$LOCAL_PROC_INDEX]=$!
done

for i in "${{!pids[@]}}"; do
    wait ${{pids[$i]}}
done
"""
            self.job_log_filename = self.args.log_filename.with_suffix(self.array_job_log_suffix).resolve()
            self.job_status_filename = self.args.log_filename.with_suffix(self.array_job_status_suffix).resolve()
            self.status_filename_list = [
                self.args.log_filename.with_suffix(f".{array_idx + 1}.out").resolve()
                for array_idx in range(
                    0,
                    cast(int, self.args.n_tasks),
                    cast(int, self.args.n_tasks_per_proc) * cast(int, self.args.n_procs),
                )
            ]
        else:
            self.job_log_filename = self.args.log_filename.resolve()
            self.job_status_filename = self.args.log_filename.with_suffix(".out").resolve()
            self.status_filename_list = [self.job_status_filename]

    def submit_job_and_wait(self, job_script: str) -> None:
        """Submit the job script and wait for completion via status files.

        Args:
            job_script (str): Job script content to write and submit.
        """
        submit_command = cast(str, self.config[self.submit_command_key].format(args=self.args))
        submit_args = cast(str, self.config[self.mode][self.submit_args_key].format(args=self.args))
        log_filename = self.args.log_filename
        job_filename = log_filename.with_suffix(".sh")

        if _is_skip_job_submission(job_filename, job_script, self.status_filename_list):
            print(
                "A successfully completed .out file exists"
                f"({[str(status_filename) for status_filename in self.status_filename_list]}), "
                "so the job will not be submitted."
            )
            for status_filename in self.status_filename_list:
                status_filename.unlink(missing_ok=True)
            return

        log_filename.parent.mkdir(exist_ok=True, parents=True)

        with open(job_filename, "w") as f:
            f.write(job_script)

        for status_filename in self.status_filename_list:
            status_filename.unlink(missing_ok=True)

        subprocess.run(f"{submit_command} {submit_args} {job_filename}", shell=True, check=True)

        for status_filename in self.status_filename_list:
            while not status_filename.exists():
                time.sleep(1.0)

                if self.config.get("use_scandir", False):  # Reflesh the file system if needed
                    os.scandir(status_filename.parent)

            status = int(status_filename.read_text())
            if status != 0:
                raise RuntimeError(f"Job failed with {status} exit code.")
            status_filename.unlink()

    def run(self) -> None:
        """Execute the standard scheduler job workflow."""
        self.build_job()
        self.prepare_job_context()
        job_script = self.build_job_script()
        self.submit_job_and_wait(job_script)
