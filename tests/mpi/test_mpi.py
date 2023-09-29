import omegaconf
import unittest
import tempfile
import shutil
import os

from unittest.mock import MagicMock, patch
from aiaccel.util.mpi import Mpi, MpiOutputHandler
from subprocess import PIPE, STDOUT, Popen, run

from pathlib import Path


def get_root():
    return Path(os.environ['GITHUB_WORKSPACE'])/'mpi_work'


def get_rank_log():
    return get_root()/'work/mpi/rank_log'


def test_logf():
    s = (get_root()/'logf').read_text()
    assert 'Scheduler INFO     1/1, finished, ready: 0, running: 0' in s
    assert 'value : 40.076' in s

def test_rank_log_0_csv():
    s = (get_rank_log()/'0.csv').read_text()
    assert ',"prepare: rank=0 tag=0",' in s
    assert ',"submit start: recv: tag=1 trial_id=0 list=[1,' in s


def test_rank_log_1_csv():
    s = (get_rank_log()/'1.csv').read_text()
    assert ',"_func_sub(): tag=1 command=[' in s
    assert ',"_func_sub(): debug: line=40.076' in s


class TestMpi(unittest.TestCase):
    def test_make_hostfile(self):
        config = omegaconf.OmegaConf.create({
            "resource": {
                "mpi_bat_root_dir": "/path/to/root",
                "mpi_bat_config_dir": "config",
                "mpi_hostfile": "hostfile",
                "mpi_bat_rt_num": 1,
                "mpi_npernode": 2,
                "mpi_gpu_mode": True
            }
        })
        os.environ["SGE_JOB_HOSTLIST"] = "hoge"
        with patch("pathlib.Path.read_text", return_value="g0073"):
            logger = MagicMock()
            with tempfile.TemporaryDirectory() as tmpdir:
                config.resource.mpi_bat_root_dir = tmpdir
                Path(f"{tmpdir}/config").mkdir(parents=True, exist_ok=True)
                Mpi._make_hostfile(config, logger)
                expected_output = "g0073 slots=3\n"
                with open(f"{tmpdir}/config/hostfile", "r") as f:
                    actual_output = f.read()
                assert actual_output == expected_output

    @patch("aiaccel.util.mpi.run")
    def test_run_bat_file(self, mock_run):
        config = omegaconf.OmegaConf.create({
            "resource": {
                "mpi_bat_file": "mpi.bat",
            },
            "ABCI": {"group": "test_group"}
        })
        logger = MagicMock()
        mock_proc = MagicMock()
        mock_proc.stdout.decode.return_value = "output"
        mock_run.return_value = mock_proc
        Mpi._run_bat_file(config, logger)
        mock_run.assert_called_once_with(["qsub", "-g", "test_group", "mpi.bat"], stdout=PIPE, stderr=STDOUT)
        logger.info.assert_called_once_with("output < qsub -g test_group mpi.bat")


    def test_make_bat_file(self):
        config = omegaconf.OmegaConf.create({
            "resource": {
                "mpi_bat_root_dir": "/path/to/root",
                "mpi_bat_venv_dir": "venv",
                "mpi_bat_aiaccel_dir": "aiaccel",
                "mpi_bat_config_dir": "config",
                "mpi_bat_file": "mpi.bat",
                "mpi_hostfile": "hostfile",
                "mpi_bat_rt_type": "F",
                "mpi_bat_rt_num": 4,
                "mpi_bat_h_rt": "01:00:00",
                "num_workers": 8
            }
        })
        logger = MagicMock()
        with tempfile.TemporaryDirectory() as tmpdir:
            config.resource.mpi_bat_root_dir = tmpdir
            os.makedirs(os.path.join(tmpdir, "config")) # create the directory
            Mpi._make_bat_file(config, logger)
            expected_output = """#!/bin/bash

#$ -l rt_F=4
#$ -l h_rt=01:00:00
#$ -j y
#$ -cwd

source /etc/profile.d/modules.sh
module load python/3.11
module load hpcx-mt/2.12
source {}/venv/bin/activate
export PYTHONPATH={}/aiaccel/:$PYTHONPATH

cd {}/config

python -m aiaccel.cli.start --config config.yaml --make_hostfile

mpiexec -n 9 -hostfile {}/config/hostfile python -m mpi4py.futures -m aiaccel.cli.start --config config.yaml --clean --from_mpi_bat

deactivate
    """.format(tmpdir, tmpdir, tmpdir, tmpdir)
            with open(os.path.join(tmpdir, "config", "mpi.bat"), "r") as f:
                actual_output = f.read()
            self.assertEqual(actual_output.strip(), expected_output.strip())


    def test_get_returncode(self):
        handler = MpiOutputHandler(None, False, "", 0, "", 0)
        self.assertIsNone(handler.get_returncode())
        handler._returncode = 0
        self.assertEqual(handler.get_returncode(), 0)
