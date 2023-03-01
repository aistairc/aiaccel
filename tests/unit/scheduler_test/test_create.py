from aiaccel.scheduler.abci_scheduler import AbciScheduler
from aiaccel.scheduler.create import create_scheduler
from aiaccel.scheduler.local_scheduler import LocalScheduler
from aiaccel.scheduler.pylocal_scheduler import PylocalScheduler


def test_create():
    assert create_scheduler('abci') == AbciScheduler
    assert create_scheduler('local') == LocalScheduler
    assert create_scheduler('python_local') == PylocalScheduler
    assert create_scheduler('invalid') is None
