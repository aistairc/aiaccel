from typing import Any

from aiaccel.config import Config
from aiaccel.scheduler.abci_scheduler import AbciScheduler
from aiaccel.scheduler.local_scheduler import LocalScheduler


def create_scheduler(config_path: str) -> Any:
    config = Config(config_path)
    resource = config.resource_type.get()

    if resource.lower() == "local" or resource.lower() == "python_local":
        return LocalScheduler

    elif resource.lower() == "abci":
        return AbciScheduler

    else:
        return None
