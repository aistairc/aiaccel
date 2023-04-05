import pathlib
import subprocess
import sys
import time
from unittest.mock import patch

import pytest
from aiaccel.master import AbciMaster
from tests.arguments import parse_arguments
from tests.base_test import BaseTest
from aiaccel.config import load_config


def callback_qstat(fake_process):
    time.sleep(2)
    raise subprocess.TimeoutExpired


def callback_return():
    return


class TestAbciMaster(BaseTest):

    @pytest.fixture(autouse=True)
    def setup_master(self, clean_work_dir):
        self.workspace.clean()
        self.workspace.create()

        self.master = AbciMaster(self.load_config_for_test(self.configs["config_abci_json"]))

        yield
        self.master = None

    def test_pre_process(
        self,
        cd_work,
        database_remove
    ):
        database_remove()

        master = AbciMaster(self.load_config_for_test(self.configs["config_abci_json"]))
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

        master = AbciMaster(self.load_config_for_test(self.configs["config_abci_json"]))

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
        fake_process.keep_last_process(True)
        # with pytest.raises(subprocess.TimeoutExpired):
        #    master.get_stats()
        master.get_stats()
