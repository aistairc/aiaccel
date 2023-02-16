from aiaccel.scheduler.abci_scheduler import AbciScheduler
from aiaccel.scheduler.local_scheduler import LocalScheduler
from aiaccel.scheduler.pylocal_scheduler import PylocalScheduler
from aiaccel.scheduler.create import create_scheduler

def test_create():
    config_abci = "tests/test_data/config_abci.json"
    assert create_scheduler(config_abci) == AbciScheduler

    config_local = "tests/test_data/config.json"
    assert create_scheduler(config_local) == LocalScheduler

    config_python_local = "tests/test_data/config_python_local.json"
    assert create_scheduler(config_python_local) == PylocalScheduler

    config_invalid = "tests/test_data/config_invalid_resource.json"
    assert create_scheduler(config_invalid) is None
