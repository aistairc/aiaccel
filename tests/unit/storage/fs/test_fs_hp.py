from aiaccel.storage.storage import Storage
from tests.unit.storage.fs.base import t_base
from tests.unit.storage.fs.base import db_path
from tests.unit.storage.fs.base import ws
from tests.unit.storage.fs.base import config_path


# set_any_trial_param
@t_base()
def test_set_any_trial_param():
    storage = Storage(ws.path, fsmode=True, config_path=config_path)

    trial_id = 0
    param_name = "x1"
    param_value = 0.01
    param_type = "float"

    assert storage.hp.set_any_trial_param(
        trial_id=trial_id,
        param_name=param_name,
        param_value=param_value,
        param_type=param_type
    ) is None

# set_any_trial_param
@t_base()
def test_set_any_trial_params():
    storage = Storage(ws.path, fsmode=True, config_path=config_path)

    trial_id = 1
    params = [
        {
            "parameter_name": "x1",
            "value": 0.01,
            "type": "float"
        },
        {
            "parameter_name": "x2",
            "value": 0.02,
            "type": "float"
        },
        {
            "parameter_name": "x3",
            "value": 0.03,
            "type": "float"
        },
        {
            "parameter_name": "x4",
            "value": 0.04,
            "type": "float"
        },
        {
            "parameter_name": "x5",
            "value": 0.05,
            "type": "float"
        }
    ]
    assert storage.hp.set_any_trial_params(
        trial_id=trial_id,
        params=params
    ) is None


# get_any_trial_params
@t_base()
def test_get_any_trial_params():
    storage = Storage(ws.path, fsmode=True, config_path=config_path)

    trial_id = 2
    param_name = "x1"
    param_value = 0.01
    param_type = "float"

    storage.hp.set_any_trial_param(
        trial_id=trial_id,
        param_name=param_name,
        param_value=param_value,
        param_type=param_type
    )

    d = storage.hp.get_any_trial_params(trial_id)
    assert d[0].trial_id == trial_id
    assert d[0].param_name == param_name
    assert d[0].param_value == param_value
    assert d[0].param_type == param_type

    if param_type.lower() == "float":
        value = float(d[0].param_value)
    elif param_type.lower() == "int":
        value = int(d[0].param_value)
    elif param_type.lower() == "categorical":
        value == str(d[0].param_value)
    else:
        assert False


# all_delete
@t_base()
def test_all_delete():
    storage = Storage(ws.path, fsmode=True, config_path=config_path)

    trial_id = 3
    param_name = "x1"
    param_value = 0.01
    param_type = "float"

    storage.hp.set_any_trial_param(
        trial_id=trial_id,
        param_name=param_name,
        param_value=param_value,
        param_type=param_type
    )

    assert storage.hp.all_delete() is None
    assert storage.hp.get_any_trial_params(trial_id) is None

# delete_any_trial_params
@t_base()
def test_delete_any_trial_params():
    storage = Storage(ws.path)

    trial_id = 0
    params = [
        {
            "parameter_name": "x1",
            "value": 0.01,
            "type": "float"
        },
        {
            "parameter_name": "x2",
            "value": 0.02,
            "type": "float"
        }
    ]
    storage.hp.set_any_trial_params(
        trial_id=trial_id,
        params=params
    )

    trial_id = 1
    params = [
        {
            "parameter_name": "x1",
            "value": 0.01,
            "type": "float"
        },
        {
            "parameter_name": "x2",
            "value": 0.02,
            "type": "float"
        }
    ]
    storage.hp.set_any_trial_params(
        trial_id=trial_id,
        params=params
    )

    trial_id = 2
    params = [
        {
            "parameter_name": "x1",
            "value": 0.01,
            "type": "float"
        },
        {
            "parameter_name": "x2",
            "value": 0.02,
            "type": "float"
        }
    ]
    storage.hp.set_any_trial_params(
        trial_id=trial_id,
        params=params
    )

    assert storage.hp.get_any_trial_params(trial_id=0) is not None
    assert storage.hp.get_any_trial_params(trial_id=1) is not None
    assert storage.hp.get_any_trial_params(trial_id=2) is not None

    assert storage.hp.delete_any_trial_params(trial_id=0) is None
    assert storage.hp.get_any_trial_params(trial_id=0) is None
    assert storage.hp.get_any_trial_params(trial_id=1) is not None
    assert storage.hp.get_any_trial_params(trial_id=2) is not None

    assert storage.hp.delete_any_trial_params(trial_id=1) is None
    assert storage.hp.get_any_trial_params(trial_id=0) is None
    assert storage.hp.get_any_trial_params(trial_id=1) is None
    assert storage.hp.get_any_trial_params(trial_id=2) is not None

    assert storage.hp.delete_any_trial_params(trial_id=2) is None
    assert storage.hp.get_any_trial_params(trial_id=0) is None
    assert storage.hp.get_any_trial_params(trial_id=1) is None
    assert storage.hp.get_any_trial_params(trial_id=2) is None