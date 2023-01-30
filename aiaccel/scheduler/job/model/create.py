from typing import Any

from aiaccel.config import Config
from aiaccel import resource_type_abci
from aiaccel import resource_type_local
from aiaccel.scheduler.job.model.abci_model import AbciModel
from aiaccel.scheduler.job.model.local_model import LocalModel


def create_model(config_path: str) -> Any:
    config = Config(config_path)
    resource = config.resource_type.get()

    if resource.lower() == resource_type_abci:
        return AbciModel
    elif resource.lower() == resource_type_local:
        return LocalModel
    else:
        return None
