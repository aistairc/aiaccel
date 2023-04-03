from __future__ import annotations

from typing import Type, Union

from aiaccel.common import (resource_type_abci, resource_type_local,
                            resource_type_python_local)
from aiaccel.config import Config
from aiaccel.master import AbciMaster, LocalMaster, PylocalMaster

# TODO: Replace typing.Type with builtins.type when aiaccel supports python>=3.9.
MasterType = Type[Union[AbciMaster, LocalMaster, PylocalMaster]]


def create_master(config_path: str) -> MasterType:
    """Returns master type.

    Args:
        config_path (str): Path to configuration file.

    Raises:
        ValueError: Causes when specified resource type is invalid.

    Returns:
        MasterType: `LocalMaster`, `PylocalMaster`, or `AbciMaster`
            if resource type is 'local', 'python_local', or 'abci',
            respectively.
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
        raise ValueError(
            f"Invalid resource type \"{resource}\". "
            "The resource type should be one of \"local\", \"python_local\", and \"abci\"."
        )
