import pytest

from aiaccel.util.retry import retry


@retry(_MAX_NUM=5, _DELAY=0.1)
def dummy_normal():
    return None


@retry(_MAX_NUM=0, _DELAY=0.1)
def dummy_normal_1():
    return None


@retry(_MAX_NUM=5, _DELAY=0.1)
def dummy_error():
    raise ValueError


def test_retry():
    assert dummy_normal() is None
    assert dummy_normal_1() is None
    with pytest.raises(ValueError):
        dummy_error()
