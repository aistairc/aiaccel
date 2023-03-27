from typing import Any

from aiaccel.master.abci_master import AbciMaster
from aiaccel.master.local_master import LocalMaster
from aiaccel.master.pylocal_master import PylocalMaster

from aiaccel.experimental.mpi.common import resource_type_mpi
from aiaccel.experimental.mpi.config import MpiConfig
from aiaccel.experimental.mpi.master.mpi_master import MpiMaster


def create_master(config_path: str) -> Any:
    """ Create a master class
        by selecting localmaster or abcimaster.
    """
    config = MpiConfig(config_path)
    resource = config.resource_type.get()

    if resource.lower() == "local":
        return LocalMaster

    elif resource.lower() == "python_local":
        return PylocalMaster

    elif resource.lower() == "abci":
        return AbciMaster

    elif resource.lower() == resource_type_mpi:
        return MpiMaster

    else:
        return None
