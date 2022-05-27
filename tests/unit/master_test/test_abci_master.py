import pathlib
from aiaccel.master.abci_master import AbciMaster
from aiaccel.scheduler.abci_scheduler import AbciScheduler
from aiaccel.util.filesystem import get_dict_files
from tests.base_test import BaseTest
import pytest
import subprocess
import time
import sys
from unittest.mock import patch


def callback_qstat():
    time.sleep(2)
    raise subprocess.TimeoutExpired


def callback_return():
    return


class TestAbciMaster(BaseTest):

    @pytest.fixture(autouse=True)
    def setup_master(self, clean_work_dir):
        config_path = str(self.config_json).split("/")
        config_path.pop(-1)
        config_path.append("config_abci.json")
        config_path = str(pathlib.Path("/".join(config_path)))

        commandline_args = [
            "start.py",
            "--config",
            config_path
        ]

        with patch.object(sys, 'argv', commandline_args):
            from aiaccel import start
            self.master = start.Master()

        # self.master = AbciMaster(self.config_json)
        yield
        self.master = None

    def test_pre_process(self, cd_work, work_dir):
        self.master.pre_process()
        alive_files = get_dict_files(work_dir.joinpath('alive'), '*.yml')

        for f in alive_files:
            f.unlink()

        # self.master.scheduler_proc.wait()
        # self.master.optimizer_proc.wait()
        assert type(self.master.runner_files) is list

    def test_get_stats(self, cd_work, data_dir, fake_process):
        xml_path = data_dir.joinpath('qstat.xml')
        fake_process.register_subprocess(
            ['qstat', '-xml'], stdout=[]
        )
        self.master.get_stats()
        with open(xml_path, 'r') as f:
            xml_string = f.read()

        fake_process.register_subprocess(
            ['qstat', '-xml'], stdout=[xml_string]
        )
        self.master.get_stats()
        assert type(self.master.stats) is list

        # TODO: Confirm why cannot cover subprocess.TimeoutExpired in
        #  get_stats()
        fake_process.register_subprocess(
            ['qstat', '-xml'], callback=callback_qstat
        )
        # with pytest.raises(subprocess.TimeoutExpired):
        #    master.get_stats()
        self.master.get_stats()

    def test_inner_loop_post_process(self, cd_work, fake_process):
        fake_process.register_subprocess(
            ['qstat', '-xml'], callback=callback_return
        )
        assert self.master.inner_loop_post_process()

    def test_loop_post_process(self):
        assert self.master.loop_post_process() is None
