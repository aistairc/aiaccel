# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from pathlib import Path

from aiaccel.config.config import load_config


def test_load_config_with_base(tmp_path: Path) -> None:
    base_path = tmp_path / "base.yaml"
    base_path.write_text(
        """
model:
  name: base
  epochs: 10
""".lstrip()
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
_base_: base.yaml

model:
  epochs: 100
""".lstrip()
    )

    config = load_config(config_path)

    expected_config = {
        "model": {
            "name": "base",
            "epochs": 100,
        },
    }

    assert config == expected_config


def test_load_config_with_multiple_bases(tmp_path: Path) -> None:
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
            "epochs": 50,
            "optimizer": {
                "name": "sgd",
                "lr": 0.001,
            },
        },
        "base1_only": "value1",
        "base2_only": "value2",
    }

    assert config == expected_config


def test_load_config_with_nested_base(tmp_path: Path) -> None:
    grand_base_path = tmp_path / "grand_base.yaml"
    grand_base_path.write_text(
        """
model:
  name: grand
  epochs: 10
  grand_only: grand
""".lstrip()
    )
    base_path = tmp_path / "base.yaml"
    base_path.write_text(
        """
_base_: grand_base.yaml

model:
  name: base
  base_only: base
""".lstrip()
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
_base_: base.yaml

model:
  epochs: 100
""".lstrip()
    )

    config = load_config(config_path)

    expected_config = {
        "model": {
            "name": "base",
            "epochs": 100,
            "grand_only": "grand",
            "base_only": "base",
        },
    }

    assert config == expected_config


def test_load_config_with_multiple_nested_bases(tmp_path: Path) -> None:
    grand_base1_path = tmp_path / "grand_base1.yaml"
    grand_base1_path.write_text(
        """
model:
  value: grand_base1
  grand_base1_only: grand_base1
""".lstrip()
    )
    base1_path = tmp_path / "base1.yaml"
    base1_path.write_text(
        """
_base_: grand_base1.yaml

model:
  value: base1
  base1_only: base1
""".lstrip()
    )
    grand_base2_path = tmp_path / "grand_base2.yaml"
    grand_base2_path.write_text(
        """
model:
  value: grand_base2
  grand_base2_only: grand_base2
""".lstrip()
    )
    base2_path = tmp_path / "base2.yaml"
    base2_path.write_text(
        """
_base_: grand_base2.yaml

model:
  value: base2
  base2_only: base2
""".lstrip()
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
_base_:
  - base1.yaml
  - base2.yaml

model:
  config_only: config
""".lstrip()
    )

    config = load_config(config_path)

    expected_config = {
        "model": {
            "value": "base1",
            "grand_base1_only": "grand_base1",
            "base1_only": "base1",
            "grand_base2_only": "grand_base2",
            "base2_only": "base2",
            "config_only": "config",
        },
    }

    assert config == expected_config


def test_load_config_resolves_relative_base_path(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs"
    base_dir = config_dir / "bases"
    base_dir.mkdir(parents=True)

    base_path = base_dir / "base.yaml"
    base_path.write_text(
        """
model:
  value: base
""".lstrip()
    )
    config_path = config_dir / "config.yaml"
    config_path.write_text(
        """
_base_: bases/base.yaml

model:
  name: config
""".lstrip()
    )

    config = load_config(config_path)

    expected_config = {
        "model": {
            "value": "base",
            "name": "config",
        },
    }

    assert config == expected_config
