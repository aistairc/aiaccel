from __future__ import annotations

from aiaccel import (resource_type_abci, resource_type_local,
                     resource_type_python_local)
from aiaccel.config import Config
from aiaccel.scheduler.abci_scheduler import AbciScheduler
from aiaccel.scheduler.local_scheduler import LocalScheduler
from aiaccel.scheduler.pylocal_scheduler import PylocalScheduler


def create_scheduler(config_path: str) -> type | None:
    """Returns scheduler type.

    Args:
        config_path (str): Path to configuration file.

    Returns:
        type | None: `LocalScheduler`, `PylocalScheduler`, or `AbciScheduler`
            if resource type is 'local', 'python_local', or 'abci',
            respectively. Other cases, None.
    """
    config = Config(config_path)
    resource = config.resource_type.get()

    if resource.lower() == resource_type_local:
        return LocalScheduler

    elif resource.lower() == resource_type_python_local:
        return PylocalScheduler

    elif resource.lower() == resource_type_abci:
        return AbciScheduler

    else:
        return None
