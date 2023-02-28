from __future__ import annotations

from aiaccel.config import Config
from aiaccel.scheduler.abci_scheduler import AbciScheduler
from aiaccel.scheduler.local_scheduler import LocalScheduler
from aiaccel.scheduler.pylocal_scheduler import PylocalScheduler


def create_scheduler(config_path: str) -> type:
    """Returns scheduler type.

    Args:
        config_path (str): Path to configuration file.

    Returns:
        type: `LocalScheduler`, `PylocalScheduler`, or `AbciScheduler`
            if resource type is 'local', 'python_local', or 'abci',
            respectively.
    """
    config = Config(config_path)
    resource = config.resource_type.get()

    if resource.lower() == "local":
        return LocalScheduler

    elif resource.lower() == "python_local":
        return PylocalScheduler

    elif resource.lower() == "abci":
        return AbciScheduler

    else:
        raise ValueError("Resource type is 'local', 'python_local', or 'abci'")
