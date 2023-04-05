from __future__ import annotations

from aiaccel.scheduler.abci_scheduler import AbciScheduler
from aiaccel.scheduler.local_scheduler import LocalScheduler
from aiaccel.scheduler.pylocal_scheduler import PylocalScheduler


def create_scheduler(resource_type: str) -> type | None:
    """Returns scheduler type.

    Args:
        config_path (str): Path to configuration file.

    Returns:
        type | None: `LocalScheduler` , `PylocalScheduler` , or `AbciScheduler`
        if resource type is 'local', 'python_local', or 'abci', respectively.
        Other cases, None.
    """

    if resource_type.lower() == "local":
        return LocalScheduler

    elif resource_type.lower() == "python_local":
        return PylocalScheduler

    elif resource_type.lower() == "abci":
        return AbciScheduler

    else:
        return None
