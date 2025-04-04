from pathlib import Path

from omegaconf import OmegaConf as oc  # noqa: N813

import pytest

from aiaccel.utils.config import load_config, pathlib2str_config, print_config, resolve_inherit


def test_load_config() -> None:
    loaded_config = load_config(Path(__file__).parent / "test_conf.yaml")
    expected_config = {
        "A": [{"_inherit_": ["${B}", "${C}"], "AA": "aa"}, {"AAA": "aaa"}],
        "D": {"_inherit_": "E"},
        "E": {"EE": "ee"},
        "B": {"AA": "dummy", "BB": "bb"},
        "C": {"CC": "cc"},
    }

    assert loaded_config == expected_config


def test_resolve_inherit() -> None:
    loaded_config = oc.create(
        {
            "A": [{"_inherit_": ["${B}", "${C}"], "AA": "aa"}, {"AAA": "aaa"}],
            "D": {"_inherit_": "E"},
            "E": {"EE": "ee"},
            "B": {"AA": "dummy", "BB": "bb"},
            "C": {"CC": "cc"},
        }
    )
    resolved_config = resolve_inherit(loaded_config)
    expected_config = {
        "A": [{"CC": "cc", "AA": "aa", "BB": "bb"}, {"AAA": "aaa"}],
        "D": {},
        "E": {"EE": "ee"},
        "B": {"AA": "dummy", "BB": "bb"},
        "C": {"CC": "cc"},
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
