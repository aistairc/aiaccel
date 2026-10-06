# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

import io
from pathlib import Path

from omegaconf import DictConfig
from omegaconf import OmegaConf as oc  # noqa: N813

import pytest

from aiaccel.config.config import load_config, pathlib2str_config, prepare_config, print_config, resolve_inherit


def test_prepare_config(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("foo: bar\n")

    config = prepare_config(config_path)

    assert isinstance(config, DictConfig)
    assert config.foo == "bar"
    assert config.config_path == str(config_path)
    assert config.working_directory == str(tmp_path)


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


def test_eval_resolver(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
value: ${eval:"(21 + 9) / (4 + (8 % 3) ** 4)"}
""".lstrip()
    )

    config = prepare_config(config_path)

    assert config.value == 1.5


def test_resolve_path(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("_base_: ${resolve_pkg_path:aiaccel.hpo.apps.config}/default.yaml\n")

    config = prepare_config(config_path)

    assert isinstance(config, DictConfig)


def test_load_config_print_option(tmp_path: Path, capfd: pytest.CaptureFixture[str]) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("foo: 1\n")

    prepare_config(
        config_path,
        print_config=True,
        print_config_kwargs={"line_length": 40},
    )

    stdout, _ = capfd.readouterr()
    assert "=" * 40 in stdout


def test_load_config_save_option(tmp_path: Path) -> None:
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


def test_load_config_with_nested_base(tmp_path: Path) -> None:
    """Nested base configs are merged recursively."""
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
    """Multiple nested base configs follow the current merge precedence."""
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


def test_load_config_resolves_relative_base_from_parent_file(tmp_path: Path) -> None:
    """Relative _base_ paths are resolved from the file that contains them."""
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


def test_resolve_inherit_with_multiple_conflicting_sources() -> None:
    """Earlier _inherit_ entries take precedence over later entries."""
    config = oc.create(
        {
            "base1": {
                "value": "base1",
                "base1_only": "base1",
            },
            "base2": {
                "value": "base2",
                "base2_only": "base2",
            },
            "child": {
                "_inherit_": ["${base1}", "${base2}"],
            },
        }
    )

    resolved_config = resolve_inherit(config)

    expected_config = {
        "base1": {
            "value": "base1",
            "base1_only": "base1",
        },
        "base2": {
            "value": "base2",
            "base2_only": "base2",
        },
        "child": {
            "value": "base1",
            "base1_only": "base1",
            "base2_only": "base2",
        },
    }

    assert resolved_config == expected_config


def test_resolve_inherit_preserves_unrelated_interpolation() -> None:
    """resolve_inherit does not resolve unrelated interpolations."""
    config = oc.create(
        {
            "source": {
                "value": "source",
            },
            "alias": "${source}",
        }
    )

    resolved_config = resolve_inherit(config)

    unresolved_config = oc.to_container(resolved_config, resolve=False)

    assert isinstance(unresolved_config, dict)
    assert unresolved_config["alias"] == "${source}"
    assert resolved_config.alias == {
        "value": "source",
    }


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


def test_print_config_kwargs() -> None:
    buffer = io.StringIO()
    conf = oc.create({"foo": 1})
    print_config(conf, line_length=10, file=buffer)

    output = buffer.getvalue()
    assert "=" * 10 in output


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
