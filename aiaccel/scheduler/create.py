from __future__ import annotations

from aiaccel.config import Config
from aiaccel.scheduler import AbciScheduler
from aiaccel.scheduler import LocalScheduler
from aiaccel.scheduler import PylocalScheduler


def create_scheduler(config_path: str) -> type | None:
    """Returns scheduler type.

    Args:
        config_path (str): Path to configuration file.

    Returns:
        type | None: `LocalScheduler` , `PylocalScheduler` , or `AbciScheduler`
        if resource type is 'local', 'python_local', or 'abci', respectively.
        Other cases, None.
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
        return None
