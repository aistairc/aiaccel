# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from pathlib import Path

from omegaconf import DictConfig
from omegaconf import OmegaConf as oc  # noqa: N813

import pytest

from aiaccel.config.config import load_config, pathlib2str_config, print_config, resolve_inherit


def test_load_config() -> None:
    _, loaded_config = load_config(Path(__file__).parent / "test_conf.yaml")
    assert isinstance(loaded_config, DictConfig)
    del loaded_config["config_path"]
    del loaded_config["working_directory"]
    expected_config = {
        "A": [{"CC": "cc", "AA": "aa", "BB": "bb"}, {"AAA": "aaa"}],
        "B": {"AA": "dummy", "BB": "bb"},
        "C": {"CC": "cc"},
        "D": {"EE": "ee"},
        "E": {"EE": "ee"},
        "Eval": 1.5,
    }

    assert loaded_config == expected_config


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


def test_resolve_path() -> None:
    _, loaded_config = load_config(Path(__file__).parent / "test_resolve_path.yaml")

    assert isinstance(loaded_config, DictConfig)


def test_print_config(capfd: pytest.CaptureFixture[str]) -> None:
    conf = oc.create({"foo": {"bar": [1, 2, 3]}})
    print_config(conf)

    stdout, _ = capfd.readouterr()

    # with open(Path(__file__).parent / "test_config_assets" / "print_config.txt", "w") as f:
    #     f.write(stdout)  # noqa: ERA001

    with open(Path(__file__).parent / "test_config_assets" / "print_config.txt") as f:
        stdout_target = f.read()

    assert stdout == stdout_target


def test_pathlib2str_config() -> None:
    src_conf = oc.create({"foo": {"bar": Path("test/path")}})
    dst_conf = pathlib2str_config(src_conf)

    assert isinstance(dst_conf.foo.bar, str)
