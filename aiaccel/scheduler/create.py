from typing import Any

from aiaccel.scheduler.abci_scheduler import AbciScheduler
from aiaccel.scheduler.local_scheduler import LocalScheduler
from aiaccel.scheduler.pylocal_scheduler import PylocalScheduler


def create_scheduler(resource_type: str) -> Any:

    if resource_type.lower() == "local":
        return LocalScheduler

    elif resource_type.lower() == "python_local":
        return PylocalScheduler

    elif resource_type.lower() == "abci":
        return AbciScheduler

    else:
        return None
