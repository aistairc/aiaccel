import omegaconf

from aiaccel.config import load_config
from tests.base_test import BaseTest


class TestConfig(BaseTest):
    def test_load_config(self):
        # default
        config_path = self.test_data_dir.joinpath("config.json")
        config = load_config(str(config_path))
        assert type(config) is omegaconf.dictconfig.DictConfig
        assert type(config.generic.workspace) is str
        assert type(config.resource.num_node) is int
        assert type(config.optimize.parameters) is omegaconf.listconfig.ListConfig
        assert type(config.optimize.parameters[0].lower) is float

        # typo config (typo trial_number)
        config_typo_path = self.test_data_dir.joinpath("config_typo.json")
        try:
            load_config(str(config_typo_path))
            assert False
        except omegaconf.errors.ValidationError:
            assert True

        # customize config (user option)
        config_user_option_path = self.test_data_dir.joinpath("config_user_option.json")
        config = load_config(str(config_user_option_path))
        # completely new item
        assert type(config) is omegaconf.dictconfig.DictConfig
        assert type(config.user_option.option_str) is str
        assert type(config.user_option.option_int) is int
        assert type(config.user_option.option_bool) is bool
        assert type(config.user_option.option_float) is float
        assert type(config.user_option.option_list) is omegaconf.listconfig.ListConfig
        # additions to existing entries
        assert type(config.generic.user_option_generic) is str
        assert type(config.resource.user_option_resource) is str
        assert type(config.ABCI.user_option_ABCI) is str
        assert type(config.optimize.user_option_optimize) is str
        assert type(config.job_setting.user_option_job_setting) is str
        assert type(config.logger.file.user_option_file) is str
        assert type(config.logger.log_level.user_option_log_level) is str
        assert type(config.logger.stream_level.user_option_stream_level) is str
        assert type(config.logger.user_option_logger) is str
