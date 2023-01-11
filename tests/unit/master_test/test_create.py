
from aiaccel.master.abci_master import AbciMaster
from aiaccel.master.local_master import LocalMaster
from aiaccel.master.create import create_master

<<<<<<< HEAD
=======

def test_create():
    config_abci = "tests/test_data/config_abci.json"
    assert create_master(config_abci) == AbciMaster

    config_local = "tests/test_data/config.json"
    assert create_master(config_local) == LocalMaster
>>>>>>> 392d1634b3b761e737cfcbca38507b668d7ab129

def test_create():
    assert create_master('abci') == AbciMaster
    assert create_master('local') == LocalMaster
    assert create_master('invalid') is None
