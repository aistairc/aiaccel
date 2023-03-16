from __future__ import annotations

from aiaccel import (resource_type_abci, resource_type_local,
                     resource_type_python_local)
from aiaccel.config import Config
from aiaccel.master.abci_master import AbciMaster
from aiaccel.master.local_master import LocalMaster
from aiaccel.master.pylocal_master import PylocalMaster


def create_master(config_path: str) -> type | None:
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

    if resource.lower() == resource_type_local:
        return LocalMaster

    elif resource.lower() == resource_type_python_local:
        return PylocalMaster

    elif resource.lower() == resource_type_abci:
        return AbciMaster

    else:
        return None
