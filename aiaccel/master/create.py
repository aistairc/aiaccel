from typing import Any
from aiaccel.master.abci import AbciMaster
from aiaccel.master.local import LocalMaster
from aiaccel.config import Config


def create_master(config_path: str) -> Any:
    """ Create a master class
        by selecting localmaster or abcimaster.
    """
    config = Config(config_path)
    resource = config.resource_type.get()

    if resource.lower() == "local":
        return LocalMaster

    elif resource.lower() == "abci":
        return AbciMaster
    else:
        return None
