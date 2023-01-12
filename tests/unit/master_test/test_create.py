
from aiaccel.master.abci_master import AbciMaster
from aiaccel.master.local_master import LocalMaster
from aiaccel.master.create import create_master


def test_create():
    assert create_master('abci') == AbciMaster
    assert create_master('local') == LocalMaster
    assert create_master('invalid') is None
