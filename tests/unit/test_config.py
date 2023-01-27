# from pathlib import Path
from aiaccel.config import load_config
from tests.base_test import BaseTest
import omegaconf


class TestConfig(BaseTest):

    def test_load_config(self):
        config_path = self.test_data_dir.joinpath('config.json')
        config = load_config(str(config_path))
        assert type(config) is omegaconf.dictconfig.DictConfig

        # typo config (typo trial_number)
        config_typo_path = self.test_data_dir.joinpath('config_typo.json')
        try:
            load_config(str(config_typo_path))
            assert False
        except omegaconf.errors.ValidationError:
            pass

        # user option
        config_user_option_path = self.test_data_dir.joinpath('config_user_option.json')
        config = load_config(str(config_user_option_path))
        assert type(config) is omegaconf.dictconfig.DictConfig
        assert type(config.user_option.option_str) is str
        assert type(config.user_option.option_int) is int
        assert type(config.user_option.option_bool) is bool
        assert type(config.user_option.option_float) is float
        assert type(config.user_option.option_list) is omegaconf.listconfig.ListConfig
