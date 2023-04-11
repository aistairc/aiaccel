import os
from pathlib import Path
from unittest.mock import patch

import pytest

from aiaccel.abci import create_qsub_command
from aiaccel.common import dict_output, dict_runner


def test_create_qsub_command(load_test_config):
    config = load_test_config()
    optimizer_file = os.path.join(os.path.join("", dict_runner), "")
    qsub_command = create_qsub_command(config, optimizer_file)
    assert type(qsub_command) is list

    path = Path(config.generic.workspace).resolve()
    command = ["qsub", "-g", f"{config.ABCI.group}", "-j", "y", "-o", f"{path / dict_output}", str(optimizer_file)]

    tmp_config = config
    tmp_config.ABCI.job_execution_options = None
    assert create_qsub_command(tmp_config, optimizer_file) == command

    tmp_config.ABCI.job_execution_options = ""
    assert create_qsub_command(tmp_config, optimizer_file) == command

    # tmp_config.ABCI.job_execution_options = []
    # assert create_qsub_command(tmp_config, optimizer_file) == command

    tmp_config.ABCI.job_execution_options = "aaa bbb"
    command_tmp = command.copy()
    for cmd in "aaa bbb".split(" "):
        command_tmp.insert(-1, cmd)
    assert create_qsub_command(tmp_config, optimizer_file) == command_tmp

    # tmp_config.ABCI.job_execution_options = ['aaa bbb']
    # command_tmp = command.copy()
    # for option in ['aaa bbb']:
    #     for cmd in option.split(' '):
    #         command_tmp.insert(-1, cmd)
    # assert create_qsub_command(tmp_config, optimizer_file) == command_tmp

    # tmp_config.ABCI.job_execution_options = 1
    # with pytest.raises(ValueError):
    #     create_qsub_command(tmp_config, optimizer_file)
