from typing import Any

from aiaccel.master.abci_master import AbciMaster
from aiaccel.master.local_master import LocalMaster
from aiaccel.master.pylocal_master import PylocalMaster


def create_master(resource_type: str) -> Any:
    """ Create a master class
        by selecting localmaster or abcimaster.
    """

    if resource_type.lower() == "local":
        return LocalMaster

    elif resource_type.lower() == "python_local":
        return PylocalMaster

    elif resource_type.lower() == "abci":
        return AbciMaster

    else:
        return None
