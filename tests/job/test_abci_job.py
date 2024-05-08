import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from aiaccel.job import AbciJob, JobStatus


class SubprocessReturn:
    stdout = ""
    stderr = ""


def qstat_xml(txt_data_path: str = "tests/job/qstat_dat.txt") -> SubprocessReturn:
    p = SubprocessReturn()
    with open(txt_data_path) as f:
        p.stdout = f.read()
    return p


class TestAbciJob(unittest.TestCase):
    def setUp(self) -> None:
        self.job_filename = "job.sh"
        self.job_group = "group"
        self.job_name = "job"
        self.cwd = tempfile.mkdtemp()
        self.stdout_filename = Path(self.cwd) / f"{self.job_name}.o"
        self.stderr_filename = Path(self.cwd) / f"{self.job_name}.e"
        self.qsub_args = ["-l"]
        self.args = ["arg1", "arg2"]
        self.tag = None

        self.job = AbciJob(
            self.job_filename,
            self.job_group,
            job_name=self.job_name,
            cwd=self.cwd,
            stdout_filename=self.stdout_filename,
            stderr_filename=self.stderr_filename,
            qsub_args=self.qsub_args,
            args=self.args,
            tag=self.tag
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.cwd)

    def test_init(self) -> None:
        self.assertEqual(self.job.job_filename, Path(self.job_filename))
        self.assertEqual(self.job.job_group, self.job_group)
        self.assertEqual(self.job.job_name, self.job_name)
        self.assertEqual(self.job.cwd, Path(self.cwd))
        self.assertEqual(self.job.stdout_filename, Path(self.stdout_filename))
        self.assertEqual(self.job.stderr_filename, Path(self.stderr_filename))
        self.assertEqual(self.job.tag, self.tag)
        self.assertEqual(self.job.status, JobStatus.UNSUBMITTED)
        self.assertIsNone(self.job.job_number)
        self.assertEqual(
            self.job.cmd,
            [
                "qsub", "-g", self.job_group, "-o", str(self.stdout_filename), "-e", str(self.stderr_filename),
                "-N", self.job_name, "-l", str(self.job_filename), "arg1", "arg2"
            ]
        )

    def test_submit(self) -> None:
        self.job.status = JobStatus.UNSUBMITTED
        self.job_number = 123456

        with patch("subprocess.run") as mock_run:
            mock_run.return_value.stdout = "Your job 123456"
            result = self.job.submit()

        self.assertEqual(result, self.job)
        self.assertEqual(result.job_number, self.job_number)
        self.assertEqual(result.status, JobStatus.WAITING)
        mock_run.assert_called_once_with(self.job.cmd, capture_output=True, text=True, check=True)

    def test_submit_already_submitted(self) -> None:
        self.job.status = JobStatus.WAITING

        with self.assertRaises(RuntimeError):
            self.job.submit()

    def test_submit_qsub_result_cannot_be_parsed(self) -> None:
        self.job.status = JobStatus.UNSUBMITTED

        with patch("subprocess.run") as mock_run:
            mock_run.return_value.stdout = "Invalid qsub result"

            with self.assertRaises(RuntimeError):
                self.job.submit()

    def test_update_status(self) -> None:
        self.job.status = JobStatus.WAITING
        self.job.job_number = 42340793

        with patch("subprocess.run", return_value=qstat_xml("tests/job/qstat_dat_1r.txt")):
            result = self.job.update_status()

        self.assertEqual(result, JobStatus.RUNNING)

    def test_wait(self) -> None:
        self.job.status = JobStatus.WAITING

        with patch.object(self.job, "update_status") as mock_update_status:
            mock_update_status.side_effect = [JobStatus.RUNNING, JobStatus.RUNNING, JobStatus.FINISHED]
            result = self.job.wait(sleep_time=0.1)

        self.assertEqual(result, self.job)
        self.assertEqual(mock_update_status.call_count, 3)
        mock_update_status.assert_called_with()

    def test_update_status_batch(self) -> None:
        """
        - Job status:
            - WAITING: 4
            - RUNNING: 0

        ↓↓↓↓↓

        - Job status:
            - WAITING: 0
            - RUNNING: 4
        """

        job1 = AbciJob(self.job_filename, self.job_group, job_name=self.job_name)
        job1.status = JobStatus.WAITING
        job1.job_number = 42340793

        job2 = AbciJob(self.job_filename, self.job_group, job_name=self.job_name)
        job2.status = JobStatus.WAITING
        job2.job_number = 42340795

        job3 = AbciJob(self.job_filename, self.job_group, job_name=self.job_name)
        job3.status = JobStatus.WAITING
        job3.job_number = 42340797

        job4 = AbciJob(self.job_filename, self.job_group, job_name=self.job_name)
        job4.status = JobStatus.WAITING
        job4.job_number = 42340799

        job_list = [job1, job2, job3, job4]

        with patch("subprocess.run", return_value=qstat_xml("tests/job/qstat_dat.txt")):
            AbciJob.update_status_batch(job_list)

        self.assertEqual(job1.status, JobStatus.RUNNING)
        self.assertEqual(job2.status, JobStatus.RUNNING)
        self.assertEqual(job3.status, JobStatus.RUNNING)
        self.assertEqual(job4.status, JobStatus.RUNNING)


if __name__ == "__main__":
    unittest.main()
