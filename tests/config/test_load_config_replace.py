# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from pathlib import Path

from omegaconf import OmegaConf as oc  # noqa: N813

from aiaccel.config.config import load_config


def test_load_config_with_parent_config(tmp_path: Path) -> None:
    base_path = tmp_path / "base.yaml"
    base_path.write_text(
        """
model:
  name: resnet
  optimizer:
    name: sgd
    lr: 0.1
""".lstrip()
    )

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
_base_: base.yaml

model:
  optimizer:
    name: adam
""".lstrip()
    )

    parent_config = {"model": {"optimizer": {"lr": 0.01}}}

    config = load_config(config_path, parent_config=parent_config)

    assert config == {
        "model": {
            "name": "resnet",
            "optimizer": {
                "name": "adam",
                "lr": 0.01,
            },
        }
    }


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


def test_replace_in_later_base_discards_earlier_base(tmp_path: Path) -> None:
    base1_path = tmp_path / "base1.yaml"
    base1_path.write_text(
        """
x:
  _replace_: true
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
""".lstrip()
    )

    config = load_config(config_path)

    assert config == {
        "x": {
            "a": 1,
        }
    }


def test_replace_in_leaf_base_is_consumed(tmp_path: Path) -> None:
    base_path = tmp_path / "base.yaml"
    base_path.write_text(
        """
x:
  _replace_: true
  a: 1
""".lstrip()
    )

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
_base_: base.yaml

x:
  b: 2
""".lstrip()
    )

    config = load_config(config_path)

    assert config == {
        "x": {
            "a": 1,
            "b": 2,
        }
    }


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


def test_replace_in_intermediate_base(tmp_path: Path) -> None:
    grand_base_path = tmp_path / "grand_base.yaml"
    grand_base_path.write_text(
        """
x:
  a: 1
""".lstrip()
    )

    base_path = tmp_path / "base.yaml"
    base_path.write_text(
        """
_base_: grand_base.yaml

x:
  _replace_: true
  b: 2
""".lstrip()
    )

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
_base_: base.yaml

x:
  c: 3
""".lstrip()
    )

    config = load_config(config_path)

    assert config == {
        "x": {
            "b": 2,
            "c": 3,
        }
    }


def test_replace_in_lower_priority_base(tmp_path: Path) -> None:
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
  _replace_: true
  b: 2
""".lstrip()
    )

    base3_path = tmp_path / "base3.yaml"
    base3_path.write_text(
        """
x:
  c: 3
""".lstrip()
    )

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
_base_:
  - base1.yaml
  - base2.yaml
  - base3.yaml
""".lstrip()
    )

    config = load_config(config_path)

    assert config == {"x": {"a": 1, "b": 2}}


def test_replace_config_false(tmp_path: Path) -> None:
    base_path = tmp_path / "base.yaml"
    base_path.write_text(
        """
x:
  a: 1
""".lstrip()
    )

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
_base_: base.yaml

x:
  _replace_: false
  b: 2
""".lstrip()
    )

    config = load_config(config_path)

    assert config == {"x": {"a": 1, "b": 2}}


def test_replace_config_with_parent_config(tmp_path: Path) -> None:
    base_path = tmp_path / "base.yaml"
    base_path.write_text(
        """
model:
  name: resnet
  optimizer:
    name: sgd
""".lstrip()
    )

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
_base_: base.yaml

model:
  optimizer:
    name: adam
""".lstrip()
    )

    parent_config = {"model": {"_replace_": True, "name": "custom"}}

    config = load_config(config_path, parent_config=parent_config)

    assert config == {"model": {"name": "custom"}}
