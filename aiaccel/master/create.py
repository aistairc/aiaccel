from typing import Any
from aiaccel.master.abci import AbciMaster
from aiaccel.master.local import LocalMaster
from aiaccel.config import Config


def create_master(options: dict) -> Any:
    """ Create a master class
        by selecting localmaster or abcimaster.
    """
    config_path = options['config']
    config = Config(config_path)
    resource = config.resource_type.get()

    if resource.lower() == "local":
        return LocalMaster(options)

    elif resource.lower() == "abci":
        return AbciMaster(options)

    else:
        return None
