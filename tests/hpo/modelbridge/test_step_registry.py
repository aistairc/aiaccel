from __future__ import annotations

import importlib

import pytest

from aiaccel.hpo.apps.modelbridge import STEP_COMMANDS, _resolve_step_handler


def test_step_commands_are_stage_aligned() -> None:
    assert STEP_COMMANDS == (
        "prepare-train",
        "prepare-eval",
        "hpo-train",
        "hpo-eval",
        "collect-train",
        "collect-eval",
        "fit-regression",
        "evaluate-model",
        "publish-summary",
    )


@pytest.mark.parametrize("command", STEP_COMMANDS)
def test_each_step_command_has_handler(command: str) -> None:
    handler = _resolve_step_handler(command)
    assert callable(handler)


def test_pipeline_module_is_removed() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("aiaccel.hpo.modelbridge.pipeline")
