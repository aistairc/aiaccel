from pathlib import Path
import numpy as np
from aiaccel.storage.variable import Serializer


def test_serialize():
    file_name = Path('/tmp/storage.db')
    if file_name.exists():
        file_name.unlink()
    var = Serializer(file_name=file_name)
    var.register(process_name='optimizer', labels=['hoge', 'foo', 'bar'])

    # if labels[i] not in self.d.keys()
    var.register(process_name='optimizer', labels=['hoge', 'foo', 'bar'])

    var.d['hoge'].set(trial_id=1, value=1.1)
    var.d['hoge'].set(trial_id=2, value=2.2)
    var.d['hoge'].set(trial_id=3, value=3.3)

    assert var.d['hoge'].get(trial_id=1) == 1.1
    assert var.d['hoge'].get(trial_id=2) == 2.2
    assert var.d['hoge'].get(trial_id=3) == 3.3

    var.d['hoge'].set(trial_id=3, value=4.4)
    assert var.d['hoge'].get(trial_id=3) == 4.4

    var.register(process_name='optimizer', labels=['random_state'])
    rs = np.random.get_state()
    var.d['hoge'].set(trial_id=1, value=rs)
    var.d['hoge'].set(trial_id=1, value=rs)


    d = var.d['hoge'].get(trial_id=1)
    for i in range(len(d)):
        if type(d[i]) is np.ndarray:
            assert all(d[i] == rs[i])
        else:
            assert d[i] == rs[i]


def test_get():
    file_name = Path('/tmp/storage.db')
    if file_name.exists():
        file_name.unlink()
    var = Serializer(file_name=file_name)
    var.register(process_name='optimizer', labels=['hoge', 'foo', 'bar'])

    var.d['hoge'].set(trial_id=1, value=1.1)
    assert var.d['hoge'].get(trial_id=1) == 1.1
    assert var.d['hoge'].get(trial_id=2) is None

def test_all_delete():
    file_name = Path('/tmp/storage.db')
    if file_name.exists():
        file_name.unlink()
    var = Serializer(file_name=file_name)
    var.register(process_name='optimizer', labels=['hoge', 'foo', 'bar'])
    assert var.d['hoge'].all_delete() is None

def test_all_delete():
    file_name = Path('/tmp/storage.db')
    if file_name.exists():
        file_name.unlink()
    var = Serializer(file_name=file_name)
    var.register(process_name='optimizer', labels=['hoge', 'foo', 'bar'])
    assert var.d['hoge'].all_delete() is None

def test_delete_any_trial_variable():
    file_name = Path('/tmp/storage.db')
    if file_name.exists():
        file_name.unlink()
    var = Serializer(file_name=file_name)
    var.register(process_name='optimizer', labels=['hoge', 'foo', 'bar'])

    var.d['hoge'].set(trial_id=1, value=1.1)
    var.d['hoge'].set(trial_id=2, value=2.2)
    var.d['hoge'].set(trial_id=3, value=3.3)

    assert var.d['hoge'].delete_any_trial_variable(
        trial_id=1,
        process_name='optimizer',
        label='hoge'
    ) is None

    assert var.d['hoge'].get(trial_id=1) is None
    assert var.d['hoge'].get(trial_id=2) == 2.2
    assert var.d['hoge'].get(trial_id=3) == 3.3

    assert var.d['hoge'].delete_any_trial_variable(
        trial_id=2,
        process_name='optimizer',
        label='hoge'
    ) is None

    assert var.d['hoge'].get(trial_id=1) is None
    assert var.d['hoge'].get(trial_id=2) is None
    assert var.d['hoge'].get(trial_id=3) == 3.3

    assert var.d['hoge'].delete_any_trial_variable(
        trial_id=3,
        process_name='optimizer',
        label='hoge'
    ) is None

    assert var.d['hoge'].get(trial_id=1) is None
    assert var.d['hoge'].get(trial_id=2) is None
    assert var.d['hoge'].get(trial_id=3) is None

def test_delete():
    file_name = Path('/tmp/storage.db')
    if file_name.exists():
        file_name.unlink()
    var = Serializer(file_name=file_name)
    var.register(process_name='optimizer', labels=['hoge', 'foo', 'bar'])

    var.d['hoge'].set(trial_id=1, value=1.1)
    var.d['hoge'].set(trial_id=2, value=2.2)
    var.d['hoge'].set(trial_id=3, value=3.3)

    assert var.d['hoge'].delete(trial_id=1) is None
    assert var.d['hoge'].get(trial_id=1) is None
    assert var.d['hoge'].get(trial_id=2) == 2.2
    assert var.d['hoge'].get(trial_id=3) == 3.3

    assert var.d['hoge'].delete(trial_id=2) is None
    assert var.d['hoge'].get(trial_id=1) is None
    assert var.d['hoge'].get(trial_id=2) is None
    assert var.d['hoge'].get(trial_id=3) == 3.3

    assert var.d['hoge'].delete(trial_id=3) is None
    assert var.d['hoge'].get(trial_id=1) is None
    assert var.d['hoge'].get(trial_id=2) is None
    assert var.d['hoge'].get(trial_id=3) is None


def test_delete_any_trial_variable():
    file_name = Path('/tmp/storage.db')
    if file_name.exists():
        file_name.unlink()
    var = Serializer(file_name=file_name)
    var.register(process_name='optimizer', labels=['hoge', 'foo', 'bar'])

    var.d['hoge'].set(trial_id=1, value=1.1)
    var.d['foo'].set(trial_id=1, value=2.2)
    var.d['bar'].set(trial_id=1, value=3.3)

    var.d['hoge'].set(trial_id=2, value=4.1)
    var.d['foo'].set(trial_id=2, value=5.2)
    var.d['bar'].set(trial_id=2, value=6.3)

    assert var.d['hoge'].get(trial_id=1) == 1.1
    assert var.d['foo'].get(trial_id=1) == 2.2
    assert var.d['bar'].get(trial_id=1) == 3.3
    assert var.d['hoge'].get(trial_id=2) == 4.1
    assert var.d['foo'].get(trial_id=2) == 5.2
    assert var.d['bar'].get(trial_id=2) == 6.3

    assert var.delete_any_trial_variable(trial_id=1) is None
    assert var.d['hoge'].get(trial_id=1) is None
    assert var.d['foo'].get(trial_id=1) is None
    assert var.d['bar'].get(trial_id=1) is None
    assert var.d['hoge'].get(trial_id=2) == 4.1
    assert var.d['foo'].get(trial_id=2) == 5.2
    assert var.d['bar'].get(trial_id=2) == 6.3

    assert var.delete_any_trial_variable(trial_id=2) is None
    assert var.d['hoge'].get(trial_id=1) is None
    assert var.d['foo'].get(trial_id=1) is None
    assert var.d['bar'].get(trial_id=1) is None
    assert var.d['hoge'].get(trial_id=2) is None
    assert var.d['foo'].get(trial_id=2) is None
    assert var.d['bar'].get(trial_id=2) is None


