from __future__ import annotations

from typing import Type, Union

from aiaccel.common import resource_type_abci, resource_type_local, resource_type_mpi, resource_type_python_local
from aiaccel.manager.abci_manager import AbciManager
from aiaccel.manager.local_manager import LocalManager
from aiaccel.manager.mpi_manager import MpiManager
from aiaccel.manager.pylocal_manager import PylocalManager

# TODO: Replace typing.Type with builtins.type when aiaccel supports python>=3.9.
ManagerType = Type[Union[AbciManager, LocalManager, PylocalManager, MpiManager]]


def create_manager(resource_type: str) -> type:
    """Returns manager type.

    Args:
        config_path (str): Path to configuration file.

    Raises:
        ValueError: Causes when specified resource type is invalid.

    Returns:
        type | None: `LocalManager` , `PylocalManager` , or `AbciManager`
        if resource type is 'local', 'python_local', or 'abci', respectively.
    """

    if resource_type.lower() == resource_type_local:
        return LocalManager
    elif resource_type.lower() == resource_type_python_local:
        return PylocalManager
    elif resource_type.lower() == resource_type_abci:
        return AbciManager
    elif resource_type.lower() == resource_type_mpi:
        return MpiManager
    else:
        raise ValueError(
            f'Invalid resource type "{resource_type}".  \
            The resource type should be one of "{resource_type_local}", \
            "{resource_type_python_local}", and "{resource_type_abci}".'
        )
