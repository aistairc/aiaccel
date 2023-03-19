
from aiaccel.master import AbciMaster
from aiaccel.master import LocalMaster
from aiaccel.master import create_master


def test_create():
    config_abci = "tests/test_data/config_abci.json"
    assert create_master(config_abci) == AbciMaster

    config_local = "tests/test_data/config.json"
    assert create_master(config_local) == LocalMaster

    config_local = "tests/test_data/config_invalid_resource.json"
    assert create_master(config_local) is None
