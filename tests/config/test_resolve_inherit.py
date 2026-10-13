# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from typing import Any

from omegaconf import OmegaConf as oc  # noqa: N813

import pytest

from aiaccel.config.config import resolve_inherit


def test_resolve_inherit() -> None:
    loaded_config = oc.create(
        {
            "A": [{"_inherit_": ["${B}", "${C}"], "AA": "aa"}, {"AAA": "aaa"}],
            "B": {"AA": "dummy", "BB": "bb"},
            "C": {"CC": "cc"},
            "D": {"_inherit_": "${E}"},
            "E": {"EE": "ee"},
        }
    )
    resolved_config = resolve_inherit(loaded_config)
    expected_config = {
        "A": [{"CC": "cc", "AA": "aa", "BB": "bb"}, {"AAA": "aaa"}],
        "B": {"AA": "dummy", "BB": "bb"},
        "C": {"CC": "cc"},
        "D": {"EE": "ee"},
        "E": {"EE": "ee"},
    }

    assert resolved_config == expected_config


def test_resolve_inherit_multiple_conflicting_sources() -> None:
    config = oc.create(
        {
            "base1": {"value": "base1", "base1_only": "base1"},
            "base2": {"value": "base2", "base2_only": "base2"},
            "child": {"_inherit_": ["${base1}", "${base2}"]},
        }
    )

    resolved_config = resolve_inherit(config)

    expected_config = {
        "base1": {"value": "base1", "base1_only": "base1"},
        "base2": {"value": "base2", "base2_only": "base2"},
        "child": {
            "value": "base1",
            "base1_only": "base1",
            "base2_only": "base2",
        },
    }

    assert resolved_config == expected_config


@pytest.mark.parametrize(
    ("config", "expected_config"),
    [
        (
            {
                "A": {"a": 1},
                "B": {"_inherit_": "${A}", "b": 2},
                "C": {"_inherit_": "${B}", "c": 3},
            },
            {
                "A": {"a": 1},
                "B": {"a": 1, "b": 2},
                "C": {"a": 1, "b": 2, "c": 3},
            },
        ),
        (
            {
                "C": {"_inherit_": "${B}", "c": 3},
                "B": {"_inherit_": "${A}", "b": 2},
                "A": {"a": 1},
            },
            {
                "C": {"a": 1, "b": 2, "c": 3},
                "B": {"a": 1, "b": 2},
                "A": {"a": 1},
            },
        ),
    ],
)
def test_resolve_inherit_chained(
    config: dict[str, Any],
    expected_config: dict[str, Any],
) -> None:
    resolved_config = resolve_inherit(oc.create(config))

    assert resolved_config == expected_config


def test_resolve_inherit_preserves_unrelated_interpolation() -> None:
    config = oc.create(
        {
            "source": {"value": "source"},
            "alias": "${source}",
        }
    )

    resolved_config = resolve_inherit(config)
    unresolved_config = oc.to_container(resolved_config, resolve=False)

    assert isinstance(unresolved_config, dict)
    assert unresolved_config["alias"] == "${source}"
    assert resolved_config.alias == {"value": "source"}
