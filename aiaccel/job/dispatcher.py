from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Job(ABC):
    """Abstract class for job.
    It has the following features:
        - Create a batch file to run the job.
        - Run the job with the given hyperparameters.
        - Collect the result of the job.
    """

    def __init__(self, trial_id: int, max_retry: int = 3):
        self.trial = trial_id
        self.max_retry = max_retry
        self.retry_count = 0
        self._state = "waiting"  # waiting, running, completed, failed

    @abstractmethod
    def create_batch_file(self) -> str:
        """Create a batch file to run the job."""
        raise NotImplementedError

    @abstractmethod
    def run(self, trial) -> None:
        """Run the job with the given hyperparameters.
        subprocess.run(["qsub", self.job_file], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        """
        raise NotImplementedError

    @abstractmethod
    def collect_result(self) -> Any:
        """Collect the result of the job.

        return:
            The result of the job (objective value).
        """
        raise NotImplementedError

    @abstractmethod
    def cancel(self) -> None:
        """Cancel the job."""
        raise NotImplementedError

    @abstractmethod
    def retry(self) -> None:
        """Retry the job."""
        raise NotImplementedError

    @property
    def state(self) -> str:
        return self._state

    def change_state(self, state: str) -> None:
        self._state = state


class JobDispatcher(ABC):
    """Abstract class for job dispatcher.

    It has the following features:
        - Schedule jobs.
        - Submit jobs.
        - Cancel jobs.
        - Optimize the objective function.
    """

    def __init__(self, study, **kwargs):
        self.study = study
        self.wait: list[Job] = []  # waiting jobs
        ...

    # ========================
    # Job Management
    # ========================

    def get_running_jobs(self) -> list[Job]:
        """Get the list of running jobs."""
        raise NotImplementedError

    def get_running_job_count(self) -> int:
        """Get the number of running jobs."""
        raise NotImplementedError

    def get_waiting_jobs(self) -> list[Job]:
        """Get the list of waiting jobs."""
        raise NotImplementedError

    def get_waiting_job_count(self) -> int:
        """Get the number of waiting jobs."""
        raise NotImplementedError

    def get_completed_jobs(self) -> list[Job]:
        """Get the list of completed jobs."""
        raise NotImplementedError

    def get_completed_job_count(self) -> int:
        """Get the number of completed jobs."""
        raise NotImplementedError

    def get_abvailable_worker_count(self) -> int:
        """Get the number of available workers."""
        raise NotImplementedError

    def schedule(self, job: Job) -> None:
        """Schedule jobs.

        Schedule jobs in the waiting list.
        self.wait.append(job)
        """
        raise NotImplementedError

    # ========================
    # Ask and Tell
    # ========================

    def ask(self) -> Any:
        raise NotImplementedError

    def tell(self, trial, objective_value) -> None:
        raise NotImplementedError

    # ========================
    # Optimize
    # ========================

    def optimize(self, n_trials) -> None:
        """Optimize the objective function.

        Example:
            for _ in range(n_trials):
                trial = self.ask()
                objective_value = self.run(trial)
                self.tell(trial, objective_value)
        """
        raise NotImplementedError

    def _optimize_parallel(self, n_trials, n_jobs) -> None:
        raise NotImplementedError

    # ========================
    # Run objective function
    # ========================

    def run(self, trial) -> None:
        """Run the job with the given hyperparameters.

        Example:
            job = Job(trial)
            job.run(trial)
            objective_value = job.collect_result()
            return float(objective_value)
        """
        raise NotImplementedError


"""
Example:

    from aiaccel import NelderMeadSampler, Parameter
    import numpy as np

    def objective(hp: Parameter) -> float:
        x1 = hp.values["x1"]
        x2 = hp.values["x2"]
        y = -20*np.exp(-0.2*np.sqrt(0.5*(x1**2+x2**2)))-np.exp(0.5 *
            (np.cos(2*np.pi*x1)+np.cos(2*np.pi*x2)))+np.e+20
        return float(y)


    if __name__ == "__main__":
        param = Parameter([
            {"name": "x1", "type": "float", "bounds": [0, 10], "step": 1, "log": False},
            {"name": "x2", "type": "float", "bounds": [0, 10], "step": 1, "log": False},
        ])

        study = aiaccel.create_study(direction='minimize', sampler=NelderMeadSampler(search_space=param.bounds, seed=42))
        jobs = aiaccel.JobDispatcher(study, func=objective, parameter=param, ...)
        n_trial = 100

        for _ in range(n_trial):
            trial = jobs.ask()
            y = jobs.run(trial)
            jobs.tell(trial, y)

        # or
        # jobs.optimize(n_trial)
"""
