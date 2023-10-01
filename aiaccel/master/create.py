from __future__ import annotations

from typing import Type, Union

from aiaccel.common import resource_type_abci, resource_type_local, resource_type_mpi, resource_type_python_local
from aiaccel.master.abci_master import AbciMaster
from aiaccel.master.local_master import LocalMaster
from aiaccel.master.mpi_master import MpiMaster
from aiaccel.master.pylocal_master import PylocalMaster

# TODO: Replace typing.Type with builtins.type when aiaccel supports python>=3.9.
MasterType = Type[Union[AbciMaster, LocalMaster, PylocalMaster, MpiMaster]]


def create_master(resource_type: str) -> type:
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

    if resource_type.lower() == resource_type_local:
        return LocalMaster
    elif resource_type.lower() == resource_type_python_local:
        return PylocalMaster
    elif resource_type.lower() == resource_type_abci:
        return AbciMaster
    elif resource_type.lower() == resource_type_mpi:
        return MpiMaster
    else:
        raise ValueError(
            f'Invalid resource type "{resource_type}".  \
            The resource type should be one of "{resource_type_local}", \
            "{resource_type_python_local}", and "{resource_type_abci}".'
        )
