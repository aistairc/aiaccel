import pytest

from aiaccel.master import AbciMaster, LocalMaster, PylocalMaster, create_master


def test_create():
    assert create_master("abci") == AbciMaster
    assert create_master("local") == LocalMaster
    assert create_master("python_local") == PylocalMaster
    with pytest.raises(ValueError):
        create_master("invalid")
