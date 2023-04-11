import pytest

from aiaccel.scheduler import AbciScheduler
from aiaccel.scheduler import LocalScheduler
from aiaccel.scheduler import PylocalScheduler
from aiaccel.scheduler import create_scheduler


def test_create():
    assert create_scheduler('abci') == AbciScheduler
    assert create_scheduler('local') == LocalScheduler
    assert create_scheduler('python_local') == PylocalScheduler
    with pytest.raises(ValueError):
        assert create_scheduler('invalid')
