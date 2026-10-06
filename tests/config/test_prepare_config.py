# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from pathlib import Path

from omegaconf import DictConfig
from omegaconf import OmegaConf as oc  # noqa: N813

import pytest

from aiaccel.config.config import prepare_config


def test_prepare_config(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("foo: bar\n")

    config = prepare_config(config_path)

    assert isinstance(config, DictConfig)
    assert config.foo == "bar"
    assert config.config_path == str(config_path)
    assert config.working_directory == str(tmp_path)


def test_prepare_config_merge_precedence(tmp_path: Path) -> None:
    """overwrite > parent > user config > base config."""
    base_path = tmp_path / "base.yaml"
    base_path.write_text(
        """
value: base
base_only: base
""".lstrip()
    )

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
_base_: base.yaml

value: config
config_only: config
""".lstrip()
    )

    config = prepare_config(
        config_path,
        overwrite_config={
            "value": "overwrite",
            "overwrite_only": "overwrite",
        },
        load_config_kwargs={
            "parent_config": {
                "value": "parent",
                "parent_only": "parent",
            },
        },
    )

    expected_config = {
        "value": "overwrite",
        "base_only": "base",
        "config_only": "config",
        "parent_only": "parent",
        "overwrite_only": "overwrite",
        "config_path": str(config_path),
        "working_directory": str(tmp_path),
    }

    assert config == expected_config


def test_prepare_config_print_option(tmp_path: Path, capfd: pytest.CaptureFixture[str]) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("foo: 1\n")

    prepare_config(
        config_path,
        print_config=True,
        print_config_kwargs={"line_length": 40},
    )

    stdout, _ = capfd.readouterr()
    assert "=" * 40 in stdout


def test_prepare_config_save_option(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("foo: 1\n")

    save_dir = tmp_path / "saved"
    config = prepare_config(
        config_path,
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


def test_prepare_config_eval_resolver(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
value: ${eval:"(21 + 9) / (4 + (8 % 3) ** 4)"}
""".lstrip()
    )

    config = prepare_config(config_path)

    assert config.value == 1.5


def test_prepare_config_resolve_pkg_path(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("_base_: ${resolve_pkg_path:aiaccel.hpo.apps.config}/default.yaml\n")

    config = prepare_config(config_path)

    assert isinstance(config, DictConfig)
