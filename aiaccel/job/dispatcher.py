from __future__ import annotations

import copy
import sys
import time
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Callable
import uuid
from aiaccel.job.job_creator import JobCreator
from aiaccel.job.functions import param_to_args_key_value
from aiaccel.job.eval import param_str_eval


__DEFAULT_WORK_DIR__ = "./work"
__DEFAULT_TOUT_SEC__ = -1  # no timeout
__DEFAULT_RETRY_NUM__ = 0  # no retry
__DEFAULT_N_JOBS__ = 1  # no parallel execution
__DEFAULT_PYTHON_CMD__ = "python"

hp_args = {}

parser = ArgumentParser()
parser.add_argument(
    "-e", action="store_true", required=False, help="Execute the objective function."
)
parser.add_argument(
    "--params",
    nargs="+",
    required=False,
    help="Hyperparameter values to override the default values.",
)  # --params key1=value key2=value ...
args = parser.parse_known_args()[0]
if args.params:
    for option in args.params:
        key, value = option.split("=")
        hp_args[key] = param_str_eval(value)


class JobDispatcher:
    def __init__(
        self,
        func: Callable | str,
        # n_trials: int,
        platform: str = "",
        group: str = "",
        template: str = "",
        template_file: Path | str | None = None,
        n_jobs: int = __DEFAULT_N_JOBS__,
        param_to_args_fn: Callable = param_to_args_key_value,
        retry_num: int = __DEFAULT_RETRY_NUM__,
        timeout_seconds: int = __DEFAULT_TOUT_SEC__,
        work_dir: str = __DEFAULT_WORK_DIR__,
        python_execute_cmd: str = __DEFAULT_PYTHON_CMD__,
    ):
        self.func = func
        if args.e:
            _run_job(self.func, hp_args)
            sys.exit(0)
        # ====
        if isinstance(func, str):
            self.execute_cmd = func
            self.func = None
        else:
            self.execute_cmd = None

        # self.n_trials = n_trials
        self.platform = platform.lower()
        self.group = group
        self._n_jobs = n_jobs
        self.param_to_args_fn = param_to_args_fn
        self.template = template
        if isinstance(template_file, str):
            self.template_file = Path(template_file).resolve()
        elif isinstance(template_file, Path):
            self.template_file = template_file.resolve()
        else:
            self.template_file = template_file
        self.retry_num = retry_num  # not used yet
        self.timeout_seconds = timeout_seconds  # not used yet
        self.work_dir = Path(work_dir).resolve()
        self.python_execute_cmd = python_execute_cmd

        self.futures = []
        self._all_future = []
        self._submit_job_count = 0
        self._all_results = []

        self.script_name = sys.argv[0]

        if not self.work_dir.exists():
            self.work_dir.mkdir(parents=True)

        self.executor = ProcessPoolExecutor(max_workers=n_jobs)

    def submit(
        self,
        hparams: dict,
        tag: Any = None,
        job_name: int | None = None,
    ) -> None:
        """Submit a job to the job dispatcher."""
        self._submit_job_count += 1
        job_name = job_name if job_name is not None else _get_job_name()
        hparams_str = self.param_to_args_fn(hparams)
        future = self.executor.submit(
            _create_and_run,
            self.script_name,
            job_name,
            self.execute_cmd,
            self.python_execute_cmd,
            self.platform,
            self.group,
            self.template,
            self.template_file,
            self.timeout_seconds,
            self.work_dir,
            hparams,
            hparams_str,
        )
        self.futures.append((future, job_name, hparams, tag))
        self._all_future.append(future)

        # Wait for at least one available worker
        while True:
            if self.available_worker_count > 0:
                break
            # if self.all_done():
            #     break
            time.sleep(0.01)

    def _update_working_feature_list(self):
        self.futures = [f for f in self.futures if not f[0].done()]

    ########################################
    # collect result
    ########################################

    def collect_results(self) -> list[tuple[float, Any]]:
        """Collect the results of the finished jobs.

        return:
            List of tuples containing the objective value and the corresponding trial object.
        """
        fdone = [f for f in self.futures if f[0].done()]
        if len(fdone) == 0:
            return []

        collected_results = []
        for future in [f for f, _, _, _ in fdone]:
            result = future.result()  # wait for the completion of the job
            _, job_name, hparams, tag = next(
                (f, tid, hps, t) for f, tid, hps, t in self.futures if f == future
            )
            collected_results.append((result, tag))
            result = {"job_name": job_name, "value": result, "hparams": hparams}
            self._all_results.append(result)
            print(result)

        self._update_working_feature_list()
        return collected_results

    def result(self) -> Any:
        """Get the result of the job dispatcher."""
        future = self.futures.pop(0)  # get the first finished job
        y = future[0].result()  # wait for the completion of the job
        result = {"job_name": future[1], "value": y, "hparams": future[2]}
        self._all_results.append(result)
        print(result)
        self._update_working_feature_list()
        return y

    @property
    def results(self) -> list[dict]:
        _all_results = copy.deepcopy(self._all_results)
        _all_results.sort(key=lambda x: x["job_name"])
        return _all_results

    ########################################
    # status
    ########################################

    # @property
    # def available_worker_count(self) -> int:
    #     _working_feature_count = len([f for f in self.futures if not f[0].done()])
    #     return min(
    #         self.n_trials - self.submit_job_count, self._n_jobs - _working_feature_count
    #     )

    @property
    def available_worker_count(self) -> int:
        _working_feature_count = len([f for f in self.futures if not f[0].done()])
        return self._n_jobs - _working_feature_count

    @property
    def finished_job_count(self) -> int:
        """Get the number of finished jobs."""
        if len(self._all_future) == 0:
            return 0
        return len([f for f in self._all_future if f.done()])

    @property
    def submit_job_count(self) -> int:
        """Get the number of submitted jobs."""
        return self._submit_job_count

    # def all_done(self) -> bool:
    #     if len(self._all_future) >= self.n_trials:
    #         return all([f.done() for f in self._all_future])
    #     else:
    #         return False

    ...


def _get_job_name() -> int:
    return str(uuid.uuid4())


def _run_job(objective: Callable, hparams: dict):
    # Called by the job script file (***.sh), not invoked by the job dispatcher.
    # Retrieve hyperparameter values from command-line arguments and update the parameter object.
    # Execute the objective function and print the result.
    y = _run_objective(objective, hparams)
    sys.stdout.write(f"{str(y)}\n")
    sys.stdout.flush()


def _create_and_run(
    script_name: str,
    job_name: int,
    execute_cmd: str | None,
    python_execute_cmd,
    platform: str,
    group: str,
    template: str,
    template_file: Path | None,
    timeout_seconds: int,
    work_dir: Path,
    hparams: dict,
    hparams_str: str,
):
    job = JobCreator(
        script_name,
        job_name,
        execute_cmd,
        python_execute_cmd,
        platform,
        group,
        template,
        template_file,
        timeout_seconds,
        work_dir,
    )
    job.create()  # create job file (***.sh)
    job.run(hparams_str)
    y = job.collect_result()
    job.create_result_json(_create_result(job_name, hparams, float(y)))
    return param_str_eval(y)


def _run_objective(objective: Callable, params: dict) -> Any:
    """Run the objective function."""
    return objective(params)


def _create_result(job_name: int, hparams: dict, y: float) -> dict:
    """Create a result dictionary."""
    result = {"job_name": job_name, "velue": y}
    result.update(hparams)
    return result
