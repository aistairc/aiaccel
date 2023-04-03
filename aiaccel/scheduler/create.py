from __future__ import annotations

from typing import Type, Union

from aiaccel.common import (resource_type_abci, resource_type_local,
                            resource_type_python_local)
from aiaccel.config import Config
from aiaccel.scheduler import AbciScheduler, LocalScheduler, PylocalScheduler

# TODO: Replace typing.Type with builtins.type when aiaccel supports python>=3.9.
SchedulerType = Type[Union[AbciScheduler, LocalScheduler, PylocalScheduler]]


def create_scheduler(config_path: str) -> SchedulerType:
    """Returns scheduler type.

    Args:
        config_path (str): Path to configuration file.

    Raises:
        ValueError: Causes when specified resource type is invalid.

    Returns:
        type | None: `LocalScheduler` , `PylocalScheduler` , or `AbciScheduler`
        if resource type is 'local', 'python_local', or 'abci', respectively.
    """
    config = Config(config_path)
    resource = config.resource_type.get()

    if resource.lower() == resource_type_local:
        return LocalScheduler

    elif resource.lower() == resource_type_python_local:
        return PylocalScheduler

    elif resource.lower() == resource_type_abci:
        return AbciScheduler
    else:
        raise ValueError(
            f"Invalid resource type \"{resource}\". "
            "The resource type should be one of \"local\", \"python_local\", and \"abci\"."
        )
