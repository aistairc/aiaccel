# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

import copy
from pathlib import Path

from omegaconf import DictConfig, ListConfig
from omegaconf import OmegaConf as oc  # noqa: N813

import pytest

from aiaccel.config.config import pathlib2str_config


@pytest.mark.parametrize(
    ("src_conf", "expected_conf"),
    [
        (
            oc.create({"foo": {"bar": Path("test/path")}}),
            {"foo": {"bar": "test/path"}},
        ),
        (
            oc.create([Path("foo"), {"bar": Path("bar")}]),
            ["foo", {"bar": "bar"}],
        ),
    ],
)
def test_pathlib2str_config(
    src_conf: DictConfig | ListConfig,
    expected_conf: dict[str, object] | list[object],
) -> None:
    original_conf = copy.deepcopy(src_conf)

    dst_conf = pathlib2str_config(src_conf)

    assert dst_conf == expected_conf
    assert src_conf == original_conf
