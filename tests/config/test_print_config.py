# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

import io
from pathlib import Path

from omegaconf import OmegaConf as oc  # noqa: N813

import pytest

from aiaccel.config.config import print_config


def test_print_config(capfd: pytest.CaptureFixture[str]) -> None:
    conf = oc.create({"foo": {"bar": [1, 2, 3]}})
    print_config(conf)

    stdout, _ = capfd.readouterr()

    with open(Path(__file__).parent / "test_config_assets" / "print_config.txt") as f:
        stdout_target = f.read()

    assert stdout == stdout_target


def test_print_config_kwargs() -> None:
    buffer = io.StringIO()
    conf = oc.create({"foo": 1})
    print_config(conf, line_length=10, file=buffer)

    output = buffer.getvalue()
    assert "=" * 10 in output
