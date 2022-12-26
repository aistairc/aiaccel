from aiaccel.config import load_config

def test_load_config(config_json, config_yaml):
    json_config = load_config(config_json)
    yaml_config = load_config(config_yaml)
    assert json_config.generic.workspace == '/tmp/work'
    assert yaml_config.generic.workspace == './hoge'

