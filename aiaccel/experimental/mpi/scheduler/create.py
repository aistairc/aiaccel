from typing import Any

from aiaccel.scheduler.abci_scheduler import AbciScheduler
from aiaccel.scheduler.local_scheduler import LocalScheduler
from aiaccel.scheduler.pylocal_scheduler import PylocalScheduler

from aiaccel.experimental.mpi import resource_type_mpi
from aiaccel.experimental.mpi.config import MpiConfig
from aiaccel.experimental.mpi.scheduler.mpi_scheduler import MpiScheduler


def create_scheduler(config_path: str) -> Any:
    config = MpiConfig(config_path)
    resource = config.resource_type.get()

    if resource.lower() == "local":
        return LocalScheduler

    elif resource.lower() == "python_local":
        return PylocalScheduler

    elif resource.lower() == "abci":
        return AbciScheduler

    elif resource.lower() == resource_type_mpi:
        return MpiScheduler

    else:
        return None
