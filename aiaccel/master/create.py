from __future__ import annotations

from aiaccel.master.abci_master import AbciMaster
from aiaccel.master.local_master import LocalMaster
from aiaccel.master.pylocal_master import PylocalMaster
from typing import Type
from typing import Union


# TODO: Replace typing.Type with builtins.type when aiaccel supports python>=3.9.
MasterType = Type[Union[AbciMaster, LocalMaster, PylocalMaster]]


def create_master(resource_type: str) -> type:
    """ Returns master type.

    Args:
        config_path (str): Path to configuration file.

    Raises:
        ValueError: Causes when specified resource type is invalid.

    Returns:
        MasterType: `LocalMaster`, `PylocalMaster`, or `AbciMaster`
            if resource type is 'local', 'python_local', or 'abci',
            respectively.
    """

    if resource_type.lower() == "local":
        return LocalMaster

    elif resource_type.lower() == "python_local":
        return PylocalMaster

    elif resource_type.lower() == "abci":
        return AbciMaster
    else:
        raise ValueError(
            f"Invalid resource type \"{resource_type}\". "
            "The resource type should be one of \"local\", \"python_local\", and \"abci\"."
        )
