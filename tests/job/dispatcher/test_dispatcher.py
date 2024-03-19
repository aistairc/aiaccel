from __future__ import annotations

import unittest
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch


class TestAbciJob(unittest.TestCase):
    def test___init__(self) -> None:
        from aiaccel.job.dispatcher import AbciJob
        job_file_path = "job.sh"
        n_jobs = 1
        work_dir = Path("./test_work").resolve()
        job = AbciJob(job_file_path, n_jobs, work_dir=str(work_dir))
        self.assertEqual(job.__class__.__name__, "AbciJob")
        if work_dir.exists():
            shutil.rmtree(work_dir)

    def test_is_finished(self) -> None:
        from aiaccel.job.dispatcher import AbciJob
        job_file_path = "job.sh"
        n_jobs = 1
        work_dir = Path("./test_work").resolve()
        job = AbciJob(job_file_path, n_jobs, work_dir=str(work_dir))
        self.assertEqual(job.is_finished(), False)
        if work_dir.exists():
            shutil.rmtree(work_dir)


class TestAbciJobExecutor(unittest.TestCase):
    def test___init__(self) -> None:
        from aiaccel.job.dispatcher import AbciJobExecutor
        job_file_path = "job.sh"
        n_jobs = 1
        work_dir = Path("./test_work").resolve()
        dispatcher = AbciJobExecutor(job_file_path, n_jobs, work_dir=str(work_dir))
        self.assertEqual(dispatcher.__class__.__name__, "AbciJobExecutor")
        if work_dir.exists():
            shutil.rmtree(work_dir)

    def test__create_work_dir(self) -> None:
        from aiaccel.job.dispatcher import AbciJobExecutor
        job_file_path = "job.sh"
        n_jobs = 1
        work_dir = Path("./test_work").resolve()
        dispatcher = AbciJobExecutor(job_file_path, n_jobs, work_dir=str(work_dir))
        assert dispatcher._create_work_dir() is None
        if work_dir.exists():
            shutil.rmtree(work_dir)

    # @patch("aiaccel.job.dispatcher.subprocess.run")
    # def test_submit(self, mock_run) -> None:
    #     from aiaccel.job.dispatcher import AbciJobExecutor
    #     job_file_path = "job.sh"
    #     n_jobs = 1
    #     work_dir = Path("./test_work").resolve()

    #     mock_stdout = MagicMock()
    #     mock_stdout.configure_mock(
    #         **{
    #             "stdout": "0.12",
    #             "stderr": "",
    #             "returncode": 0,
    #         }
    #     )

    #     dispatcher = AbciJobExecutor(job_file_path, n_jobs, work_dir=str(work_dir))
    #     args = ["--x1", "0.0", "--x2", "0.0"]

    #     with patch("aiaccel.job.dispatcher.subprocess.run", return_value=mock_stdout):
    #         assert dispatcher.submit(args, job_name="test") is None

    def test_sibmit(self) -> None:
        pass

    def test_qsub(self) -> None:
        pass

    def test__execute_qsub(self) -> None:
        pass

    def test_get_results(self) -> None:
        pass

    def test_get_result(self) -> None:
        pass

    def test_update_state_batch(self) -> None:
        pass

    def test_available_worker_count(self) -> None:
        pass

    def test_finished_job_count(self) -> None:
        pass

    def test_is_finished(self) -> None:
        pass

    def test_submit_job_count(self) -> None:
        pass



def test_create_run_command() -> None:
    pass


def test_create_qsub_command() -> None:
    pass


def test_collect_result() -> None:
    pass


def test_get_job_number() -> None:
    pass


def test_get_job_numbers() -> None:
    pass


def test_get_job_status() -> None:
    pass
