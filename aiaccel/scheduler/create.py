from __future__ import annotations

from typing import Type, Union

from aiaccel.common import resource_type_abci, resource_type_local, resource_type_mpi, resource_type_python_local
from aiaccel.scheduler.abci_scheduler import AbciScheduler
from aiaccel.scheduler.local_scheduler import LocalScheduler
from aiaccel.scheduler.mpi_scheduler import MpiScheduler
from aiaccel.scheduler.pylocal_scheduler import PylocalScheduler

# TODO: Replace typing.Type with builtins.type when aiaccel supports python>=3.9.
SchedulerType = Type[Union[AbciScheduler, LocalScheduler, PylocalScheduler, MpiScheduler]]


def create_scheduler(resource_type: str) -> type:
    """Returns scheduler type.

    Args:
        config_path (str): Path to configuration file.

    Raises:
        ValueError: Causes when specified resource type is invalid.

    Returns:
        type | None: `LocalScheduler` , `PylocalScheduler` , or `AbciScheduler`
        if resource type is 'local', 'python_local', or 'abci', respectively.
    """

    if resource_type.lower() == resource_type_local:
        return LocalScheduler
    elif resource_type.lower() == resource_type_python_local:
        return PylocalScheduler
    elif resource_type.lower() == resource_type_abci:
        return AbciScheduler
    elif resource_type.lower() == resource_type_mpi:
        return MpiScheduler
    else:
        raise ValueError(
            f'Invalid resource type "{resource_type}".  \
            The resource type should be one of "{resource_type_local}", \
            "{resource_type_python_local}", and "{resource_type_abci}".'
        )
