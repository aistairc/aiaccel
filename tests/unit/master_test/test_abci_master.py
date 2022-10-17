import pathlib
import subprocess
import sys
import time
from unittest.mock import patch

import pytest
from aiaccel.argument import Arguments
from aiaccel.master.abci_master import AbciMaster
from aiaccel.master.create import create_master
from aiaccel.scheduler.abci_scheduler import AbciScheduler
from aiaccel.util.filesystem import get_dict_files

from tests.base_test import BaseTest


def callback_qstat():
    time.sleep(2)
    raise subprocess.TimeoutExpired


def callback_return():
    return


class TestAbciMaster(BaseTest):

    def get_confit_path(self):
        config_path = str(self.config_json).split("/")
        config_path.pop(-1)
        config_path.append("config_abci.json")
        config_path = str(pathlib.Path("/".join(config_path)))
        return config_path

    @pytest.fixture(autouse=True)
    def setup_master(self, clean_work_dir):
        config_path = str(self.config_json).split("/")
        config_path.pop(-1)
        config_path.append("config_abci.json")
        config_path = str(pathlib.Path("/".join(config_path)))

        self.workspace.clean()
        self.workspace.create()

        commandline_args = [
            "start.py",
            "--config",
            config_path
        ]

        with patch.object(sys, 'argv', commandline_args):
            options = Arguments()
            self.master = AbciMaster(options)

        yield
        self.master = None

    def test_pre_process(
        self,
        cd_work,
        work_dir,
        database_remove
    ):
        database_remove()
        commandline_args = [
            "start.py",
            "--config",
            self.get_confit_path()
        ]
        with patch.object(sys, 'argv', commandline_args):
            options = Arguments()
            master = AbciMaster(options)
        master.pre_process()
        assert type(master.runner_files) is list

    def test_get_stats(
        self,
        cd_work,
        data_dir,
        fake_process,
        database_remove
    ):
        database_remove()
        commandline_args = [
            "start.py",
            "--config",
            self.get_confit_path()
        ]
        with patch.object(sys, 'argv', commandline_args):
            options = Arguments()
            master = AbciMaster(options)

        xml_path = data_dir.joinpath('qstat.xml')
        fake_process.register_subprocess(
            ['qstat', '-xml'], stdout=[]
        )
        master.get_stats()
        with open(xml_path, 'r') as f:
            xml_string = f.read()

        fake_process.register_subprocess(
            ['qstat', '-xml'], stdout=[xml_string]
        )
        master.get_stats()
        assert type(master.stats) is list

        # TODO: Confirm why cannot cover subprocess.TimeoutExpired in
        #  get_stats()
        fake_process.register_subprocess(
            ['qstat', '-xml'], callback=callback_qstat
        )
        # with pytest.raises(subprocess.TimeoutExpired):
        #    master.get_stats()
        master.get_stats()

    def test_inner_loop_post_process(
        self,
        cd_work,
        fake_process,
        database_remove
    ):
        database_remove()
        commandline_args = [
            "start.py",
            "--config",
            self.get_confit_path()
        ]
        with patch.object(sys, 'argv', commandline_args):
            options = Arguments()
            master = AbciMaster(options)

        fake_process.register_subprocess(
            ['qstat', '-xml'], callback=callback_return
        )
        assert master.inner_loop_post_process()

    def test_loop_post_process(self, database_remove):
        database_remove()
        commandline_args = [
            "start.py",
            "--config",
            self.get_confit_path()
        ]
        with patch.object(sys, 'argv', commandline_args):
            options = Arguments()
            master = AbciMaster(options)
        assert master.loop_post_process() is None
