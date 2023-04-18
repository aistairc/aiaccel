from __future__ import annotations

from typing import Type, Union

from aiaccel.scheduler.abci_scheduler import AbciScheduler
from aiaccel.scheduler.local_scheduler import LocalScheduler
from aiaccel.scheduler.pylocal_scheduler import PylocalScheduler

# TODO: Replace typing.Type with builtins.type when aiaccel supports python>=3.9.
SchedulerType = Type[Union[AbciScheduler, LocalScheduler, PylocalScheduler]]


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

    if resource_type.lower() == "local":
        return LocalScheduler

    elif resource_type.lower() == "python_local":
        return PylocalScheduler

    elif resource_type.lower() == "abci":
        return AbciScheduler
    else:
        raise ValueError(
            f'Invalid resource type "{resource_type}". '
            'The resource type should be one of "local", "python_local", and "abci".'
        )
