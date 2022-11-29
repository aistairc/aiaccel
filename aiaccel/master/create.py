from typing import Any

from aiaccel.config import Config
from aiaccel.master.abci_master import AbciMaster
from aiaccel.master.local_master import LocalMaster


def create_master(config_path: str) -> Any:
    """ Create a master class
        by selecting localmaster or abcimaster.
    """
    config = Config(config_path)
    resource = config.resource_type.get()

    if resource.lower() == "local" or resource.lower() == "python_local":
        return LocalMaster

    elif resource.lower() == "abci":
        return AbciMaster
    else:
        return None
