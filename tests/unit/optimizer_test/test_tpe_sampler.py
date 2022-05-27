from aiaccel.optimizer.tpe.sampler import TPESamplerWrapper
from tests.base_test import BaseTest


class TestTPESamplerWrapper(BaseTest):

    def test_get_startup_trials(self):
        tpe_sampler_wrapper = TPESamplerWrapper()
        assert tpe_sampler_wrapper.get_startup_trials() == 10
