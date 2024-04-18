from __future__ import annotations

import shutil
import unittest
from pathlib import Path
from unittest.mock import patch
from typing import Any

import pytest

from aiaccel.job.dispatcher import AbciJob, AbciJobExecutor, JobStatus


class Output:
    stdout = 'Your job 42183931 ("hpo-0006") has been submitted'
    stderr = ""


# queue_info: 2 jobs
# job_info  : None

qstat_xml_dict = {
    "job_info": {
        "queue_info": {
            "job_list": [
                {
                    "@state": "running",
                    "JB_job_number": "42183931",
                    "JAT_prio": "0.25586",
                    "JB_name": "hpo-0006",
                    "JB_owner": "owner",
                    "state": "r",
                    "JAT_start_time": "2024-03-26T09:49:16.592",
                    "queue_name": "gpu@g0007",
                    "jclass_name": None,
                    "slots": "10",
                },
                {
                    "@state": "running",
                    "JB_job_number": "42183935",
                    "JAT_prio": "0.25586",
                    "JB_name": "hpo-0007",
                    "JB_owner": "owner",
                    "state": "r",
                    "JAT_start_time": "2024-03-26T09:49:36.770",
                    "queue_name": "gpu@g0007",
                    "jclass_name": None,
                    "slots": "10",
                },
            ]
        },
        "job_info": None,
    }
}


class TestAbciJob(unittest.TestCase):

    @pytest.fixture
    def tmpdir_fixture(self, tmpdir):  # type: ignore
        tmpdir.mkdir("test_work")
        self.work_dir = tmpdir / "test_work"
        return tmpdir

    def test___init__(self) -> None:
        job_file_path = Path("job.sh")
        work_dir = Path("./test_work").resolve()
        job = AbciJob(
            job_name="test",
            args=[],
            tag=None,
            job_file_path=job_file_path,
            stdout_file_path=Path("stdout"),
            stderr_file_path=Path("stderr"),
        )
        self.assertEqual(job.__class__.__name__, "AbciJob")
        if work_dir.exists():
            shutil.rmtree(work_dir)

    def test_run(self) -> None:
        with patch("subprocess.run", return_value=Output()) as mock_run:
            job_file_path = Path("job.sh")
            job = AbciJob(
                job_name="test",
                args=[],
                tag=None,
                job_file_path=job_file_path,
                stdout_file_path=Path("stdout"),
                stderr_file_path=Path("stderr"),
            )
            with patch("aiaccel.job.dispatcher.qstat_xml", return_value=qstat_xml_dict):
                job.run("qsub -g test -N test -o stdout -e stderr job.sh")

    def test_get_job_status(self) -> None:
        test_job_nummber = 42183931
        expected_state = "r"
        job_file_path = Path("job.sh")
        job = AbciJob(
            job_name="test",
            args=[],
            tag=None,
            job_file_path=job_file_path,
            stdout_file_path=Path("stdout"),
            stderr_file_path=Path("stderr"),
        )
        with patch("aiaccel.job.dispatcher.qstat_xml", return_value=qstat_xml_dict):
            assert job.get_job_status(test_job_nummber) == expected_state

    def test_set_state(self) -> None:
        job_file_path = Path("job.sh")
        job = AbciJob(
            job_name="test",
            args=[],
            tag=None,
            job_file_path=job_file_path,
            stdout_file_path=Path("stdout"),
            stderr_file_path=Path("stderr"),
        )

        job.set_state(JobStatus.RUNNING)
        assert job.status == JobStatus.RUNNING

    @pytest.mark.usefixtures("tmpdir_fixture")
    def test_collect_result(self) -> None:
        stdout_file_path = self.work_dir / "test.o"
        stderr_file_path = self.work_dir / "test.e"
        job = AbciJob(
            job_name="test",
            args=[],
            tag=None,
            job_file_path=Path("job.sh"),
            stdout_file_path=stdout_file_path,
            stderr_file_path=stderr_file_path,
        )
        with open(stdout_file_path, "w") as f:
            f.write("0.12")
        assert job.collect_result() == "0.12"

    def test_get_result(self) -> None:
        job_file_path = Path("job.sh")
        job = AbciJob(
            job_name="test",
            args=[],
            tag=None,
            job_file_path=job_file_path,
            stdout_file_path=Path("stdout"),
            stderr_file_path=Path("stderr"),
        )
        with patch(
            "aiaccel.job.dispatcher.AbciJob.collect_result", return_value="0.12"
        ):
            assert job.get_result() == "0.12"


class TestAbciJobExecutor(unittest.TestCase):
    job_file_path = "job.sh"
    n_jobs = 4
    work_dir = None

    @pytest.fixture
    def tmpdir_fixture(self, tmpdir):  # type: ignore
        tmpdir.mkdir("test_work")
        self.work_dir = tmpdir / "test_work"
        return tmpdir

    @pytest.mark.usefixtures("tmpdir_fixture")  # type: ignore
    def test___init__(self) -> None:
        dispatcher = AbciJobExecutor(
            self.job_file_path, self.n_jobs, work_dir=str(self.work_dir)
        )
        self.assertEqual(dispatcher.__class__.__name__, "AbciJobExecutor")

    @pytest.mark.usefixtures("tmpdir_fixture")  # type: ignore
    def test_submit(self) -> None:
        args = ["--x1", "0.0", "--x2", "0.0"]
        job_name = "test"
        group = "test_group"
        tag = None
        dispatcher = AbciJobExecutor(
            self.job_file_path, self.n_jobs, work_dir=str(self.work_dir)
        )
        job = AbciJob(
            job_name=job_name,
            args=args,
            tag=tag,
            job_file_path=Path("job.sh"),
            stdout_file_path=Path("stdout"),
            stderr_file_path=Path("stderr"),
        )
        with patch("aiaccel.job.dispatcher.AbciJobExecutor._execute", return_value=job):
            assert dispatcher.submit(args, group, job_name, tag) is job

    @pytest.mark.usefixtures("tmpdir_fixture")  # type: ignore
    def test_execute(self) -> None:
        pass

    @pytest.mark.usefixtures("tmpdir_fixture")  # type: ignore
    def test_get_results(self) -> None:
        JB_name = "hpo-0006"
        stdout_file_path = self.work_dir / f"{JB_name}.o"
        stderr_file_path = self.work_dir / f"{JB_name}.e"

        dispatcher = AbciJobExecutor(
            self.job_file_path, self.n_jobs, work_dir=self.work_dir
        )
        job = AbciJob(
            job_name=JB_name,
            args=[],
            tag=None,
            job_file_path=Path("job.sh"),
            stdout_file_path=stdout_file_path,
            stderr_file_path=stderr_file_path,
        )

        job.job_number = 42183931
        job.status = JobStatus.FINISHED
        dispatcher.working_job_list.append(job)

        with open(stdout_file_path, "w") as f:
            f.write("0.12")

        qstat_xml_dict = {"job_info": {"queue_info": None, "job_info": None}}
        with patch("aiaccel.job.dispatcher.qstat_xml", return_value=qstat_xml_dict):
            with patch(
                "aiaccel.job.dispatcher.AbciJob.collect_result", return_value="0.12"
            ):
                for y, trial in dispatcher.get_results():
                    assert y == "0.12"

    @pytest.mark.usefixtures("tmpdir_fixture")  # type: ignore
    def test_get_result(self) -> None:
        JB_name = "hpo-0006"
        stdout_file_path = self.work_dir / f"{JB_name}.o"
        stderr_file_path = self.work_dir / f"{JB_name}.e"

        dispatcher = AbciJobExecutor(
            self.job_file_path, self.n_jobs, work_dir=self.work_dir
        )
        job = AbciJob(
            job_name=JB_name,
            args=[],
            tag=None,
            job_file_path=Path("job.sh"),
            stdout_file_path=stdout_file_path,
            stderr_file_path=stderr_file_path,
        )

        job.job_number = 42183931
        job.status = JobStatus.FINISHED
        dispatcher.working_job_list.append(job)

        with open(stdout_file_path, "w") as f:
            f.write("0.12")

        qstat_xml_dict = {"job_info": {"queue_info": None, "job_info": None}}
        with patch("aiaccel.job.dispatcher.qstat_xml", return_value=qstat_xml_dict):
            with patch(
                "aiaccel.job.dispatcher.AbciJob.collect_result", return_value="0.12"
            ):
                y = dispatcher.get_result()
                assert y == "0.12"

    @pytest.mark.usefixtures("tmpdir_fixture")  # type: ignore
    def test_update_state_batch(self) -> None:
        JB_names = ["hpo-0006", "hpo-0007"]
        job_numbers = [42183931, 42183935]
        dispatcher = AbciJobExecutor(
            self.job_file_path, self.n_jobs, work_dir=self.work_dir
        )

        for JB_name, job_number in zip(JB_names, job_numbers):
            stdout_file_path = self.work_dir / f"{JB_name}.o"
            stderr_file_path = self.work_dir / f"{JB_name}.e"

            job = AbciJob(
                job_name=JB_name,
                args=[],
                tag=None,
                job_file_path=Path("job.sh"),
                stdout_file_path=stdout_file_path,
                stderr_file_path=stderr_file_path,
            )
            job.job_number = job_number
            job.set_state(JobStatus.RUNNING)
            dispatcher.working_job_list.append(job)

        qstat_xml_dict = {"job_info": {"queue_info": None, "job_info": None}}

        with patch("aiaccel.job.dispatcher.qstat_xml", return_value=qstat_xml_dict):
            dispatcher.update_state_batch()
            for job in dispatcher.working_job_list:
                assert job.status == JobStatus.FINISHED

    @pytest.mark.usefixtures("tmpdir_fixture")  # type: ignore
    def test_get_available_worker_count(self) -> None:
        dispatcher = AbciJobExecutor(
            self.job_file_path, self.n_jobs, work_dir=str(self.work_dir)
        )
        for i in range(10):
            dispatcher.working_job_list.append(
                AbciJob(
                    job_name=f"test_job_{i}",
                    args=[],
                    tag=None,
                    job_file_path=Path("job.sh"),
                    stdout_file_path=Path("stdout"),
                    stderr_file_path=Path("stderr"),
                )
            )
        for i in range(0, 7):
            dispatcher.working_job_list[i].status = JobStatus.FINISHED
        for i in range(7, 10):
            dispatcher.working_job_list[i].status = JobStatus.RUNNING
        assert dispatcher.get_available_worker_count() == 1
