import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from aiaccel.job import AbciJob, AbciJobExecutor, JobStatus


class SubprocessReturn:
    stdout = ""
    stderr = ""


def qstat_xml(txt_data_path: str = "tests/job/qstat_dat.txt") -> SubprocessReturn:
    p = SubprocessReturn()
    with open(txt_data_path) as f:
        p.stdout = f.read()
    return p


class TestAbciJobExecutor(unittest.TestCase):
    def setUp(self) -> None:
        self.job_filename = "main.sh"
        self.job_group = "group"
        self.job_name = "job"
        self.work_dir = tempfile.mkdtemp()
        self.n_max_jobs = 100

        self.executor = AbciJobExecutor(
            self.job_filename,
            self.job_group,
            job_name=self.job_name,
            work_dir=self.work_dir,
            n_max_jobs=self.n_max_jobs
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.work_dir)

    def test_available_slots_full(self) -> None:
        """
        - Job status:
            - RUNNING: 4
        - available: 0
        """

        job1 = AbciJob(self.job_filename, self.job_group, job_name=self.job_name)
        job1.status = JobStatus.RUNNING
        job1.job_number = 42340793

        job2 = AbciJob(self.job_filename, self.job_group, job_name=self.job_name)
        job2.status = JobStatus.RUNNING
        job2.job_number = 42340795

        job3 = AbciJob(self.job_filename, self.job_group, job_name=self.job_name)
        job3.status = JobStatus.RUNNING
        job3.job_number = 42340797

        job4 = AbciJob(self.job_filename, self.job_group, job_name=self.job_name)
        job4.status = JobStatus.RUNNING
        job4.job_number = 42340799

        self.executor.job_list = [job1, job2, job3, job4]

        with patch("subprocess.run", return_value=qstat_xml("tests/job/qstat_dat.txt")):
            result = self.executor.available_slots()

        num_available_slots = (
            self.n_max_jobs
            - len([j for j in self.executor.job_list if j.status == JobStatus.RUNNING or JobStatus.WAITING])
        )
        self.assertEqual(result, num_available_slots)

    def test_available_slots_pending(self) -> None:
        """
        - Job status:
            - RUNNING: 2
            - WAITING: 1
        - available: 1
        """

        job1 = AbciJob(self.job_filename, self.job_group, job_name=self.job_name)
        job1.status = JobStatus.RUNNING
        job1.job_number = 42340793

        job2 = AbciJob(self.job_filename, self.job_group, job_name=self.job_name)
        job2.status = JobStatus.RUNNING
        job2.job_number = 42340795

        job3 = AbciJob(self.job_filename, self.job_group, job_name=self.job_name)
        job3.status = JobStatus.WAITING
        job3.job_number = 42340797

        self.executor.job_list = [job1, job2, job3]

        with patch("subprocess.run", return_value=qstat_xml("tests/job/qstat_dat 2r_1qw.txt")):
            result = self.executor.available_slots()

        num_available_slots = (
            self.n_max_jobs
            - len([j for j in self.executor.job_list if j.status == JobStatus.RUNNING or JobStatus.WAITING])
        )
        self.assertEqual(result, num_available_slots)

    def test_collect_finished(self) -> None:
        job1 = MagicMock(spec=AbciJob)
        job1.status = JobStatus.FINISHED

        job2 = MagicMock(spec=AbciJob)
        job2.status = JobStatus.RUNNING

        job3 = MagicMock(spec=AbciJob)
        job3.status = JobStatus.FINISHED

        self.executor.job_list = [job1, job2, job3]

        result = self.executor.collect_finished()

        self.assertEqual(result, [job1, job3])
        self.assertNotIn(job1, self.executor.job_list)
        self.assertNotIn(job3, self.executor.job_list)


if __name__ == "__main__":
    unittest.main()
