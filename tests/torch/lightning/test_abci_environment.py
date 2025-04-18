import os
from unittest import mock

import pytest

from aiaccel.torch.lightning.abci_environment import ABCIEnvironment


@mock.patch.dict(
    os.environ,
    {
        "OMPI_COMM_WORLD_SIZE": "8",
        "OMPI_COMM_WORLD_RANK": "6",
        "OMPI_COMM_WORLD_LOCAL_RANK": "2",
        "OMPI_COMM_WORLD_LOCAL_SIZE": "4",
        "MAIN_ADDR": "192.168.0.1",
        "MAIN_PORT": "3000",
    },
)
def test_abci_environment() -> None:
    env = ABCIEnvironment()

    assert env.creates_processes_externally

    assert env.main_address == "192.168.0.1"
    assert env.main_port == 3000

    assert env.detect()
    assert env.world_size() == 8
    assert env.global_rank() == 6
    assert env.node_rank() == 1
    assert env.local_rank() == 2

    env.validate_settings(4, 2)

    with pytest.raises(ValueError, match=r"^`num_devices` should match.*"):
        env.validate_settings(3, 2)

    with pytest.raises(ValueError, match=r"^`num_devices \* num_nodes` should match.*"):
        env.validate_settings(4, 1)
