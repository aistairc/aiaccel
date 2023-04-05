from aiaccel.config import (is_multi_objective, load_config)
import pytest


def test_load_config(config_json, config_yaml):
    json_config = load_config(config_json)
    yaml_config = load_config(config_yaml)
    assert json_config.generic.workspace == '/tmp/work'
    assert yaml_config.generic.workspace == './hoge'


def test_config_not_exists():
    with pytest.raises(ValueError):
        load_config("?")


def test_is_multi_objective(config_json):
    config = load_config(config_json)
    assert is_multi_objective(config) is False
