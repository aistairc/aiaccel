# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from omegaconf import DictConfig, ListConfig
from omegaconf import OmegaConf as oc  # noqa: N813
from omegaconf.errors import ConfigTypeError

import pytest

from aiaccel.config.config import _merge_config, remove_replace


@pytest.mark.parametrize(
    ("base", "override"),
    [
        (oc.create({"a": 1}), oc.create({"b": 2})),
        (oc.create({"x": {"a": 1}}), oc.create({"x": {"b": 2}})),
        (oc.create({"x": [1, 2]}), oc.create({"x": [3, 4]})),
        (oc.create([1, 2]), oc.create([3, 4])),
        (oc.create({"x": 1}), oc.create({"x": {"a": 1}})),
        (oc.create({"x": {"a": 1}}), oc.create({"x": 1})),
        (oc.create({"x": None}), oc.create({"x": {"a": 1}})),
        (oc.create({"x": {"a": 1}}), oc.create({"x": None})),
        (
            oc.create({"source": {"a": 1}, "alias": {"b": 2}}),
            oc.create({"alias": "${source}"}),
        ),
    ],
)
def test_merge_config_matches_omegaconf_without_replace(
    base: DictConfig | ListConfig,
    override: DictConfig | ListConfig,
) -> None:
    assert _merge_config(base, override) == oc.merge(base, override)


@pytest.mark.parametrize(
    ("base", "override"),
    [
        (oc.create({"a": 1}), oc.create([1, 2])),
        (oc.create([1, 2]), oc.create({"a": 1})),
        (oc.create({"x": [1, 2]}), oc.create({"x": {"a": 1}})),
        (oc.create({"x": {"a": 1}}), oc.create({"x": [1, 2]})),
    ],
)
def test_merge_config_matches_omegaconf_error_without_replace(
    base: DictConfig | ListConfig,
    override: DictConfig | ListConfig,
) -> None:
    with pytest.raises(ConfigTypeError):
        oc.merge(base, override)

    with pytest.raises(ConfigTypeError):
        _merge_config(base, override)


@pytest.mark.parametrize(
    ("base", "override", "expected"),
    [
        (
            oc.create({"x": [1, 2]}),
            oc.create({"x": {"_replace_": True, "a": 1}}),
            oc.create({"x": {"a": 1}}),
        ),
        (
            oc.create({"x": 1}),
            oc.create({"x": {"_replace_": True, "a": 1}}),
            oc.create({"x": {"a": 1}}),
        ),
    ],
)
def test_merge_config_replaces_different_types(
    base: DictConfig | ListConfig,
    override: DictConfig | ListConfig,
    expected: DictConfig | ListConfig,
) -> None:
    config = _merge_config(base, override)

    assert remove_replace(config) == expected


@pytest.mark.parametrize(
    ("base", "override"),
    [
        (oc.create({"x": 1}), oc.create({"x": "???"})),
        (oc.create({"x": "???"}), oc.create({"x": {"a": 1}})),
        (oc.create({}), oc.create({"x": "???"})),
        (
            oc.create({"source": {"a": 1}, "alias": "${source}"}),
            oc.create({"alias": {"b": 2}}),
        ),
    ],
)
def test_merge_config_matches_omegaconf_with_special_nodes(
    base: DictConfig,
    override: DictConfig,
) -> None:
    expected = oc.merge(base, override)
    actual = _merge_config(base, override)

    assert oc.to_container(actual, resolve=False) == oc.to_container(expected, resolve=False)


def test_merge_config_preserves_missing_value_semantics_with_replace() -> None:
    base = oc.create({"x": 1, "model": {"name": "base", "optimizer": "sgd"}})
    override = oc.create({"x": "???", "model": {"_replace_": True, "name": "override"}})

    actual = remove_replace(_merge_config(base, override))

    assert actual == {"x": 1, "model": {"name": "override"}}


def test_merge_config_preserves_base_interpolation_with_replace() -> None:
    base = oc.create(
        {
            "source": {"a": 1},
            "alias": "${source}",
            "model": {"name": "base", "optimizer": "sgd"},
        }
    )
    override = oc.create({"alias": {"b": 2}, "model": {"_replace_": True, "name": "override", "x": "???"}})
    expected = {
        "source": {"a": 1},
        "alias": {"a": 1, "b": 2},
        "model": {"name": "override", "x": "???"},
    }

    actual = remove_replace(_merge_config(base, override))

    assert oc.to_container(actual, resolve=False) == expected
