from __future__ import annotations

from typing import Any

from aiaccel.config import Config
from aiaccel.master.abci_master import AbciMaster
from aiaccel.master.local_master import LocalMaster
from aiaccel.master.pylocal_master import PylocalMaster


def create_master(config_path: str) -> Any:
    """Returns master type.

    Args:
        config_path (str): Path to configuration file.

    Returns:
        type | None: `LocalMaster`, `PylocalMaster`, or `AbciMaster`
            if resource type is 'local', 'python_local', or 'abci',
            respectively. Other cases, None.
    """
    config = Config(config_path)
    resource = config.resource_type.get()

    if resource.lower() == "local":
        return LocalMaster

    elif resource.lower() == "python_local":
        return PylocalMaster

    elif resource.lower() == "abci":
        return AbciMaster

    else:
        return None
