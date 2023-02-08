from __future__ import annotations

from aiaccel.master.abci_master import AbciMaster
from aiaccel.master.local_master import LocalMaster
from aiaccel.master.pylocal_master import PylocalMaster


def create_master(resource_type: str) -> type | None:
    """ Returns master type.

    Args:
        config_path (str): Path to configuration file.

    Returns:
        type | None: `LocalMaster`, `PylocalMaster`, or `AbciMaster`
            if resource type is 'local', 'python_local', or 'abci',
            respectively. Other cases, None.
    """

    if resource_type.lower() == "local":
        return LocalMaster

    elif resource_type.lower() == "python_local":
        return PylocalMaster

    elif resource_type.lower() == "abci":
        return AbciMaster

    else:
        return None
