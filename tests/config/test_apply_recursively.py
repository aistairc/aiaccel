# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from omegaconf import DictConfig, ListConfig
from omegaconf import OmegaConf as oc  # noqa: N813

from aiaccel.config.config import apply_recursively


def test_apply_recursively_visits_nested_containers_and_preserves_interpolations() -> None:
    visited_types: list[type] = []

    @apply_recursively
    def mark_processed(config: DictConfig | ListConfig) -> DictConfig | ListConfig:
        visited_types.append(type(config))
        if isinstance(config, DictConfig):
            config["_processed"] = True
        return config

    config = oc.create(
        {
            "branch": {"value": 1},
            "sequence": [{"value": 2}],
            "branch_alias": "${branch}",
            "sequence_alias": "${sequence}",
        }
    )

    original = oc.to_container(config, resolve=False)

    result = mark_processed(config)

    assert oc.to_container(config, resolve=False) == original

    assert isinstance(result, DictConfig)
    assert result["branch"]["_processed"] is True
    assert result["sequence"][0]["_processed"] is True
    assert ListConfig in visited_types

    assert oc.is_interpolation(result, "branch_alias")
    assert oc.is_interpolation(result, "sequence_alias")

    raw_result = oc.to_container(result, resolve=False)
    assert isinstance(raw_result, dict)
    assert raw_result["branch_alias"] == "${branch}"
    assert raw_result["sequence_alias"] == "${sequence}"


def test_apply_recursively_preserves_missing_values() -> None:
    @apply_recursively
    def identity(
        config: DictConfig | ListConfig,
    ) -> DictConfig | ListConfig:
        return config

    config = oc.create(
        {
            "required": "???",
            "nested": {
                "required": "???",
            },
            "sequence": [
                "???",
                {
                    "required": "???",
                },
            ],
        }
    )

    result = identity(config)

    assert oc.to_container(result, resolve=False) == {
        "required": "???",
        "nested": {
            "required": "???",
        },
        "sequence": [
            "???",
            {
                "required": "???",
            },
        ],
    }
