from aiaccel.command_line_options import CommandLineOptions
from aiaccel.optimizer import RandomOptimizer

from tests.base_test import BaseTest


class TestRandomOptimizer(BaseTest):

    def test_generate_parameter(self, create_tmp_config):
        self.config_random_path = create_tmp_config(self.config_random_path)
        options = CommandLineOptions(
            config=str(self.config_random_path),
            resume=None,
            clean=False,
            process_name="optimizer"
        )
        optimizer = RandomOptimizer(options)
        optimizer.pre_process()
        assert len(optimizer.generate_parameter()) > 0
