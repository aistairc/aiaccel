import pytest

from aiaccel.scheduler import AbciScheduler, LocalScheduler, PylocalScheduler, create_scheduler


def test_create():
    assert create_scheduler("abci") == AbciScheduler
    assert create_scheduler("local") == LocalScheduler
    assert create_scheduler("python_local") == PylocalScheduler
    with pytest.raises(ValueError):
        assert create_scheduler("invalid")
