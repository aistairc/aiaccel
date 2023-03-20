import pathlib
import subprocess
import sys
import time
from unittest.mock import patch

import pytest

from aiaccel.master.abci_master import AbciMaster
from tests.arguments import parse_arguments
from tests.base_test import BaseTest


def callback_qstat(fake_process):
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
    def setup_master(self, create_tmp_config):
        self.config_path = self.test_data_dir.joinpath("config_abci.json")
        self.config_path = create_tmp_config(self.config_path)

        self.workspace.clean()
        self.workspace.create()

        commandline_args = ["start.py", "--config", str(self.config_path)]

        with patch.object(sys, "argv", commandline_args):
            # from aiaccel import start
            # self.master = start.Master()
            options = parse_arguments()
            # self.master = create_master(options['config'])(options)
            self.master = AbciMaster(options)

        yield
        self.master = None

    def test_pre_process(self, cd_work, database_remove):
        database_remove()
        commandline_args = ["start.py", "--config", str(self.config_path)]
        with patch.object(sys, "argv", commandline_args):
            options = parse_arguments()
            master = AbciMaster(options)
        master.pre_process()
        assert type(master.runner_files) is list

    def test_get_stats(self, cd_work, data_dir, fake_process, database_remove):
        database_remove()
        commandline_args = ["start.py", "--config", str(self.config_path)]
        with patch.object(sys, "argv", commandline_args):
            options = parse_arguments()
            master = AbciMaster(options)

        xml_path = data_dir.joinpath("qstat.xml")
        fake_process.register_subprocess(["qstat", "-xml"], stdout=[])
        master.get_stats()
        with open(xml_path, "r") as f:
            xml_string = f.read()

        fake_process.register_subprocess(["qstat", "-xml"], stdout=[xml_string])
        master.get_stats()
        assert type(master.stats) is list

        # TODO: Confirm why cannot cover subprocess.TimeoutExpired in
        #  get_stats()
        fake_process.register_subprocess(["qstat", "-xml"], callback=callback_qstat)
        fake_process.keep_last_process(True)
        # with pytest.raises(subprocess.TimeoutExpired):
        #    master.get_stats()
        master.get_stats()
