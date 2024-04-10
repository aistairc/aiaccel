from pathlib import Path

import pytest
from omegaconf import OmegaConf as oc

from aiaccel.utils.config import pathlib2str_config, print_config


def test_print_config(capfd: pytest.CaptureFixture[str]) -> None:
    conf = oc.create({"foo": {"bar": Path("test/path")}})
    print_config(conf)

    stdout, _ = capfd.readouterr()

    with open(Path(__file__).parent / "test_config_assets" / "print_config.txt") as f:
        stdout_target = f.read()

    assert stdout == stdout_target


def test_pathlib2str_config() -> None:
    src_conf = oc.create({"foo": {"bar": Path("test/path")}})
    dst_conf = pathlib2str_config(src_conf)

    assert isinstance(dst_conf.foo.bar, str)
