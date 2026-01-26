# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

import io
from pathlib import Path

from omegaconf import DictConfig
from omegaconf import OmegaConf as oc  # noqa: N813

import pytest

from aiaccel.config.config import pathlib2str_config, prepare_config, print_config, resolve_inherit


def test_load_config() -> None:
    config = prepare_config(Path(__file__).parent / "test_conf.yaml")
    assert isinstance(config, DictConfig)
    del config["config_path"]
    del config["working_directory"]
    expected_config = {
        "A": [{"CC": "cc", "AA": "aa", "BB": "bb"}, {"AAA": "aaa"}],
        "B": {"AA": "dummy", "BB": "bb"},
        "C": {"CC": "cc"},
        "D": {"EE": "ee"},
        "E": {"EE": "ee"},
        "Eval": 1.5,
        "Replace": {"AA": "aa"},
    }

    assert config == expected_config


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
    config = prepare_config(Path(__file__).parent / "test_resolve_path.yaml")

    assert isinstance(config, DictConfig)


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
    assert isinstance(src_conf.foo.bar, Path)


def test_load_config_print_option(capfd: pytest.CaptureFixture[str]) -> None:
    prepare_config(
        Path(__file__).parent / "test_conf.yaml",
        print_config=True,
        print_config_kwargs={"line_length": 40},
    )

    stdout, _ = capfd.readouterr()
    assert "=" * 40 in stdout


def test_print_config_kwargs() -> None:
    buffer = io.StringIO()
    conf = oc.create({"foo": 1})
    print_config(conf, line_length=10, file=buffer)

    output = buffer.getvalue()
    assert "=" * 10 in output


def test_load_config_save_option(tmp_path: Path) -> None:
    save_dir = tmp_path / "saved"
    config = prepare_config(
        Path(__file__).parent / "test_conf.yaml",
        working_directory=tmp_path,
        save_config=True,
        save_directory=save_dir,
        save_filename="custom.yaml",
    )

    save_path = save_dir / "custom.yaml"

    assert save_path.exists()
    assert isinstance(config.working_directory, str)

    reloaded_config = oc.load(save_path)
    assert "config_path" in reloaded_config
