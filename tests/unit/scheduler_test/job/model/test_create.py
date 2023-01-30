from aiaccel.scheduler.job.model.abci_model import AbciModel
from aiaccel.scheduler.job.model.local_model import LocalModel
from aiaccel.scheduler.job.model.create import create_model

def test_create():
    config_abci = "tests/test_data/config_abci.json"
    assert create_model(config_abci) == AbciModel

    config_local = "tests/test_data/config.json"
    assert create_model(config_local) == LocalModel

    config_invalid = "tests/test_data/config_invalid_resource.json"
    assert create_model(config_invalid) is None
