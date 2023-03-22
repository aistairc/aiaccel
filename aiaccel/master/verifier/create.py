from __future__ import annotations

from typing import Union
from typing import Type

from aiaccel.master.verifier import SingleObjectiveVerifier, MultiObjectiveVerifier
from aiaccel.config import Config, is_multi_objective


# TODO: Replace typing.Type with builtins.type when aiaccel supports python>=3.9.
VerifierClass = Type[Union[SingleObjectiveVerifier, MultiObjectiveVerifier]]


def create_verifier(config_path: str) -> VerifierClass:
    """Creates a verifier.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        VerifierClass: _description_
    """

    config = Config(config_path)

    if is_multi_objective(config):
        return MultiObjectiveVerifier
    else:
        return SingleObjectiveVerifier
