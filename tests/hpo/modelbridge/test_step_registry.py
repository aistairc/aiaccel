from __future__ import annotations

import importlib

import pytest

import aiaccel.hpo.modelbridge.pipeline as pipeline_module
from aiaccel.hpo.modelbridge.pipeline import (
    PIPELINE_PROFILES,
    STEP_ACTIONS,
    STEP_DEFINITIONS,
    STEP_SPECS,
    normalize_steps,
    steps_for_profile,
)


def test_steps_for_profile_order_is_deterministic() -> None:
    assert steps_for_profile("prepare") == ["prepare_train", "prepare_eval"]
    assert steps_for_profile("analyze") == [
        "collect_train",
        "collect_eval",
        "fit_regression",
        "evaluate_model",
        "publish_summary",
    ]


def test_steps_for_full_profile_contains_all_steps() -> None:
    assert steps_for_profile("full") == [
        "prepare_train",
        "prepare_eval",
        "collect_train",
        "collect_eval",
        "fit_regression",
        "evaluate_model",
        "publish_summary",
    ]


def test_normalize_steps_rejects_unknown_step() -> None:
    with pytest.raises(ValueError, match="Unknown step"):
        normalize_steps(["unknown_step"])


def test_step_specs_are_derived_from_definitions() -> None:
    expected = tuple(
        (step_name, cli_command, profiles)
        for step_name, cli_command, profiles, _action in STEP_DEFINITIONS
    )
    assert expected == STEP_SPECS


def test_pipeline_profile_choices_are_canonical() -> None:
    assert PIPELINE_PROFILES == ("prepare", "analyze", "full")


def test_pipeline_step_actions_keys_are_canonical() -> None:
    assert tuple(STEP_ACTIONS.keys()) == tuple(steps_for_profile("full"))


def test_step_registry_module_removed() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("aiaccel.hpo.modelbridge.step_registry")


def test_pipeline_module_has_no_legacy_registry_symbols() -> None:
    assert not hasattr(pipeline_module, "STEP_BY_NAME")
    assert not hasattr(pipeline_module, "STEP_BY_CLI_COMMAND")
    assert not hasattr(pipeline_module, "valid_steps")
    assert not hasattr(pipeline_module, "STEP_NAME_BY_CLI_COMMAND")
