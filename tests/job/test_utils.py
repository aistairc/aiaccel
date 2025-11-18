import pytest

from aiaccel.job.utils import slice_tasks


def test_slice_tasks_no_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TASK_INDEX", raising=False)
    monkeypatch.delenv("TASK_STEPSIZE", raising=False)
    sequence = [1, 2, 3]
    assert slice_tasks(sequence) == sequence


def test_slice_tasks_with_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TASK_INDEX", "2")
    monkeypatch.setenv("TASK_STEPSIZE", "2")
    assert slice_tasks([10, 20, 30, 40]) == [20, 30]


def test_slice_tasks_invalid_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TASK_INDEX", "0")
    monkeypatch.setenv("TASK_STEPSIZE", "1")
    with pytest.raises(ValueError):
        slice_tasks([1, 2])

    monkeypatch.setenv("TASK_INDEX", "2")
    monkeypatch.setenv("TASK_STEPSIZE", "0")
    with pytest.raises(ValueError):
        slice_tasks([1, 2])


def test_slice_tasks_missing_stepsize(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TASK_INDEX", "1")
    monkeypatch.delenv("TASK_STEPSIZE", raising=False)
    with pytest.raises(ValueError):
        slice_tasks([1, 2])
