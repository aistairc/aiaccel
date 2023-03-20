from aiaccel.master.abci_master import AbciMaster
from aiaccel.master.create import create_master
from aiaccel.master.local_master import LocalMaster


def test_create():
    config_abci = "tests/test_data/config_abci.json"
    assert create_master(config_abci) == AbciMaster

    config_local = "tests/test_data/config.json"
    assert create_master(config_local) == LocalMaster

    config_local = "tests/test_data/config_invalid_resource.json"
    assert create_master(config_local) is None
