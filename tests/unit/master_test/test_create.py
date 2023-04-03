import pytest

from aiaccel.master import AbciMaster
from aiaccel.master import LocalMaster
from aiaccel.master import PylocalMaster
from aiaccel.master import create_master


def test_create():
    config_abci = "tests/test_data/config_abci.json"
    assert create_master(config_abci) == AbciMaster

    config_local = "tests/test_data/config.json"
    assert create_master(config_local) == LocalMaster

    config_python_local = "tests/test_data/config_python_local.json"
    assert create_master(config_python_local) == PylocalMaster

    with pytest.raises(ValueError):
        config_invalid = "tests/test_data/config_invalid_resource.json"
        _ = create_master(config_invalid)
