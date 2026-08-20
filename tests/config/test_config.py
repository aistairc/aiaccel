# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

import io
from pathlib import Path

from omegaconf import DictConfig
from omegaconf import OmegaConf as oc  # noqa: N813

import pytest

from aiaccel.config.config import load_config, pathlib2str_config, prepare_config, print_config, resolve_inherit


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


def test_load_config_with_multiple_bases(tmp_path: Path) -> None:
    """Multiple base configs are merged from left to right."""
    base1_path = tmp_path / "base1.yaml"
    base1_path.write_text(
        """
model:
  name: resnet
  epochs: 50
  optimizer:
    name: adam
    lr: 0.001

base1_only: value1
""".lstrip()
    )

    base2_path = tmp_path / "base2.yaml"
    base2_path.write_text(
        """
model:
  epochs: 100
  optimizer:
    lr: 0.01

base2_only: value2
""".lstrip()
    )

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
_base_:
  - base1.yaml
  - base2.yaml

model:
  optimizer:
    name: sgd
""".lstrip()
    )

    config = load_config(config_path)

    expected_config = {
        "model": {
            "name": "resnet",
            "epochs": 100,
            "optimizer": {
                "name": "sgd",
                "lr": 0.01,
            },
        },
        "base1_only": "value1",
        "base2_only": "value2",
    }

    assert config == expected_config


def test_replace_config_with_multiple_bases(tmp_path: Path) -> None:
    base1_path = tmp_path / "base1.yaml"
    base1_path.write_text(
        """
x:
  a: 1
""".lstrip()
    )

    base2_path = tmp_path / "base2.yaml"
    base2_path.write_text(
        """
x:
  b: 2
""".lstrip()
    )

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
_base_:
  - base1.yaml
  - base2.yaml

x:
  _replace_: true
  c: 3
""".lstrip()
    )

    config = load_config(config_path)

    assert config == {
        "x": {
            "c": 3,
        }
    }


def test_replace_config_nested(tmp_path: Path) -> None:
    """Nested _replace_ replaces only the specified mapping."""
    base_path = tmp_path / "base.yaml"
    base_path.write_text(
        """
outer:
  keep: base
  inner:
    arg3: zzz
""".lstrip()
    )

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
_base_: base.yaml

outer:
  inner:
    _replace_: true
    arg1: xxx
    arg2: yyy
""".lstrip()
    )

    config = load_config(config_path)

    expected_config = {
        "outer": {
            "keep": "base",
            "inner": {
                "arg1": "xxx",
                "arg2": "yyy",
            },
        }
    }

    assert config == expected_config


def test_replace_config_with_recursive_base(tmp_path: Path) -> None:
    """_replace_ works when _base_ is recursively defined."""
    grand_base_path = tmp_path / "grand_base.yaml"
    grand_base_path.write_text(
        """
hogehoge:
  arg3: zzz
""".lstrip()
    )

    base_path = tmp_path / "base.yaml"
    base_path.write_text(
        """
_base_: grand_base.yaml

hogehoge:
  arg4: aaa
""".lstrip()
    )

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
_base_: base.yaml

hogehoge:
  _replace_: true
  arg1: xxx
  arg2: yyy
""".lstrip()
    )

    config = load_config(config_path)

    expected_config = {
        "hogehoge": {
            "arg1": "xxx",
            "arg2": "yyy",
        }
    }

    assert config == expected_config


def test_replace_config_with_interpolation(tmp_path: Path) -> None:
    """DictConfig interpolation is not treated as a concrete child mapping."""
    base_path = tmp_path / "base.yaml"
    base_path.write_text(
        """
source:
  arg3: zzz

alias:
  base_only: value
""".lstrip()
    )

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
_base_: base.yaml

source:
  _replace_: true
  arg1: xxx

alias: ${source}
""".lstrip()
    )

    config = load_config(config_path)

    # source itself should be replaced.
    assert config.source == {
        "arg1": "xxx",
    }

    # alias remains an interpolation internally.
    unresolved = oc.to_container(config, resolve=False)
    assert isinstance(unresolved, dict)
    assert unresolved["alias"] == "${source}"

    # When resolved, alias refers to the replaced source.
    assert config.alias == {
        "arg1": "xxx",
    }


def test_replace_config_at_root(tmp_path: Path) -> None:
    """Top-level _replace_ replaces the entire base configuration."""
    base_path = tmp_path / "base.yaml"
    base_path.write_text(
        """
arg3: zzz

nested:
  value: base
""".lstrip()
    )

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
_base_: base.yaml
_replace_: true

arg1: xxx
arg2: yyy
""".lstrip()
    )

    config = load_config(config_path)

    expected_config = {
        "arg1": "xxx",
        "arg2": "yyy",
    }

    assert config == expected_config
