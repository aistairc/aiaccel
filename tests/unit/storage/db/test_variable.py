from pathlib import Path
import numpy as np
from base import db_path, t_base, ws, init

from aiaccel.storage.variable import Serializer
from aiaccel.storage.storage import Storage
from undecorated import undecorated
from sqlalchemy.exc import SQLAlchemyError
import pytest

@t_base()
def test_serialize():
    storage = Storage(ws.path)
    storage.variable.register(process_name='optimizer', labels=['hoge', 'foo', 'bar'])

    # if labels[i] not in self.d.keys()
    storage.variable.register(process_name='optimizer', labels=['hoge', 'foo', 'bar'])

    storage.variable.d['hoge'].set(trial_id=1, value=1.1)
    storage.variable.d['hoge'].set(trial_id=2, value=2.2)
    storage.variable.d['hoge'].set(trial_id=3, value=3.3)

    assert storage.variable.d['hoge'].get(trial_id=1) == 1.1
    assert storage.variable.d['hoge'].get(trial_id=2) == 2.2
    assert storage.variable.d['hoge'].get(trial_id=3) == 3.3

    storage.variable.d['hoge'].set(trial_id=3, value=4.4)
    assert storage.variable.d['hoge'].get(trial_id=3) == 4.4

    storage.variable.register(process_name='optimizer', labels=['random_state'])
    rs = np.random.get_state()
    storage.variable.d['hoge'].set(trial_id=1, value=rs)
    storage.variable.d['hoge'].set(trial_id=1, value=rs)


    d = storage.variable.d['hoge'].get(trial_id=1)
    for i in range(len(d)):
        if type(d[i]) is np.ndarray:
            assert all(d[i] == rs[i])
        else:
            assert d[i] == rs[i]

@t_base()
def test_set_exception():
    storage = Storage(ws.path)
    storage.variable.register(process_name='optimizer', labels=['hoge', 'foo', 'bar'])

    init()
    with pytest.raises(SQLAlchemyError):
        set_any_trial_variable = undecorated(storage.variable.d['hoge'].set_any_trial_variable)
        set_any_trial_variable(
            storage.variable.d['hoge'],
            trial_id=1,
            process_name='optimizer',
            label='hoge',
            value=1.1,
            update_allow=True
        )

@t_base()
def test_get():
    storage = Storage(ws.path)
    storage.variable.register(process_name='optimizer', labels=['hoge', 'foo', 'bar'])

    storage.variable.d['hoge'].set(trial_id=1, value=1.1)
    assert storage.variable.d['hoge'].get(trial_id=1) == 1.1
    assert storage.variable.d['hoge'].get(trial_id=2) is None

@t_base()
def test_all_delete():
    storage = Storage(ws.path)
    storage.variable.register(process_name='optimizer', labels=['hoge', 'foo', 'bar'])
    assert storage.variable.d['hoge'].all_delete() is None

@t_base()
def test_all_delete_exception():
    storage = Storage(ws.path)
    storage.variable.register(process_name='optimizer', labels=['hoge', 'foo', 'bar'])

    init()
    with pytest.raises(SQLAlchemyError):
        all_delete = undecorated(storage.variable.d['hoge'].all_delete)
        all_delete(storage.variable.d['hoge'])

@t_base()
def test_delete_any_trial_variable():
    storage = Storage(ws.path)
    storage.variable.register(process_name='optimizer', labels=['hoge', 'foo', 'bar'])

    storage.variable.d['hoge'].set(trial_id=1, value=1.1)
    storage.variable.d['hoge'].set(trial_id=2, value=2.2)
    storage.variable.d['hoge'].set(trial_id=3, value=3.3)

    assert storage.variable.d['hoge'].delete_any_trial_variable(
        trial_id=1,
        process_name='optimizer',
        label='hoge'
    ) is None

    assert storage.variable.d['hoge'].get(trial_id=1) is None
    assert storage.variable.d['hoge'].get(trial_id=2) == 2.2
    assert storage.variable.d['hoge'].get(trial_id=3) == 3.3

    assert storage.variable.d['hoge'].delete_any_trial_variable(
        trial_id=2,
        process_name='optimizer',
        label='hoge'
    ) is None

    assert storage.variable.d['hoge'].get(trial_id=1) is None
    assert storage.variable.d['hoge'].get(trial_id=2) is None
    assert storage.variable.d['hoge'].get(trial_id=3) == 3.3

    assert storage.variable.d['hoge'].delete_any_trial_variable(
        trial_id=3,
        process_name='optimizer',
        label='hoge'
    ) is None

    assert storage.variable.d['hoge'].get(trial_id=1) is None
    assert storage.variable.d['hoge'].get(trial_id=2) is None
    assert storage.variable.d['hoge'].get(trial_id=3) is None


@t_base()
def test_delete_any_trial_variable_exception():
    storage = Storage(ws.path)
    storage.variable.register(process_name='optimizer', labels=['hoge', 'foo', 'bar'])

    storage.variable.d['hoge'].set(trial_id=1, value=1.1)

    init()
    with pytest.raises(SQLAlchemyError):
        delete_any_trial_variable = undecorated(storage.variable.d['hoge'].delete_any_trial_variable)
        delete_any_trial_variable(
            storage.variable.d['hoge'],
            trial_id=1,
            process_name='optimizer',
            label='hoge'
        )


@t_base()
def test_delete():
    storage = Storage(ws.path)
    storage.variable.register(process_name='optimizer', labels=['hoge', 'foo', 'bar'])

    storage.variable.d['hoge'].set(trial_id=1, value=1.1)
    storage.variable.d['hoge'].set(trial_id=2, value=2.2)
    storage.variable.d['hoge'].set(trial_id=3, value=3.3)

    assert storage.variable.d['hoge'].delete(trial_id=1) is None
    assert storage.variable.d['hoge'].get(trial_id=1) is None
    assert storage.variable.d['hoge'].get(trial_id=2) == 2.2
    assert storage.variable.d['hoge'].get(trial_id=3) == 3.3

    assert storage.variable.d['hoge'].delete(trial_id=2) is None
    assert storage.variable.d['hoge'].get(trial_id=1) is None
    assert storage.variable.d['hoge'].get(trial_id=2) is None
    assert storage.variable.d['hoge'].get(trial_id=3) == 3.3

    assert storage.variable.d['hoge'].delete(trial_id=3) is None
    assert storage.variable.d['hoge'].get(trial_id=1) is None
    assert storage.variable.d['hoge'].get(trial_id=2) is None
    assert storage.variable.d['hoge'].get(trial_id=3) is None


@t_base()
def test_delete_any_trial_variable():
    storage = Storage(ws.path)
    storage.variable.register(process_name='optimizer', labels=['hoge', 'foo', 'bar'])

    storage.variable.d['hoge'].set(trial_id=1, value=1.1)
    storage.variable.d['foo'].set(trial_id=1, value=2.2)
    storage.variable.d['bar'].set(trial_id=1, value=3.3)

    storage.variable.d['hoge'].set(trial_id=2, value=4.1)
    storage.variable.d['foo'].set(trial_id=2, value=5.2)
    storage.variable.d['bar'].set(trial_id=2, value=6.3)

    assert storage.variable.d['hoge'].get(trial_id=1) == 1.1
    assert storage.variable.d['foo'].get(trial_id=1) == 2.2
    assert storage.variable.d['bar'].get(trial_id=1) == 3.3
    assert storage.variable.d['hoge'].get(trial_id=2) == 4.1
    assert storage.variable.d['foo'].get(trial_id=2) == 5.2
    assert storage.variable.d['bar'].get(trial_id=2) == 6.3

    assert storage.variable.delete_any_trial_variable(trial_id=1) is None
    assert storage.variable.d['hoge'].get(trial_id=1) is None
    assert storage.variable.d['foo'].get(trial_id=1) is None
    assert storage.variable.d['bar'].get(trial_id=1) is None
    assert storage.variable.d['hoge'].get(trial_id=2) == 4.1
    assert storage.variable.d['foo'].get(trial_id=2) == 5.2
    assert storage.variable.d['bar'].get(trial_id=2) == 6.3

    assert storage.variable.delete_any_trial_variable(trial_id=2) is None
    assert storage.variable.d['hoge'].get(trial_id=1) is None
    assert storage.variable.d['foo'].get(trial_id=1) is None
    assert storage.variable.d['bar'].get(trial_id=1) is None
    assert storage.variable.d['hoge'].get(trial_id=2) is None
    assert storage.variable.d['foo'].get(trial_id=2) is None
    assert storage.variable.d['bar'].get(trial_id=2) is None
