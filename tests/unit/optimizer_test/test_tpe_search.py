import pytest
from aiaccel.config import Config
from aiaccel.parameter import load_parameter
from aiaccel.optimizer.tpe.search import TpeSearchOptimizer
from aiaccel.optimizer.tpe.search import create_distributions
from tests.base_test import BaseTest


class TestTpeSearchOptimizer(BaseTest):

    @pytest.fixture(autouse=True)
    def setup_optimizer(self, clean_work_dir, data_dir):
        options = {
            'config': data_dir / 'config_tpe.json',
            'resume': None,
            'clean': False,
            'fs': False,
            'process_name': 'optimizer'
        }
        self.optimizer = TpeSearchOptimizer(options)
        self.optimizer.storage.alive.init_alive()
        yield
        self.optimizer = None

    def test_pre_process(self):
        assert self.optimizer.pre_process() is None

    def test_post_process(self):
        self.optimizer.pre_process()
        assert self.optimizer.post_process() is None

    def test_check_result(self, setup_hp_finished, setup_result, work_dir):
        self.optimizer.pre_process()
        assert self.optimizer.check_result() is None

    def test_is_startup_trials(self):
        self.optimizer.pre_process()
        assert self.optimizer.is_startup_trials()

    def test_generate_parameter(self):
        self.optimizer.pre_process()
        assert self.optimizer.generate_parameter() is None

    def test_create_study(self):
        assert self.optimizer.create_study() is None

    def test_study_pickle_path(self):
        assert self.optimizer.study_pickle_path.name == 'study.pkl'

    def test_serialize(self):
        self.optimizer.storage.trial.set_any_trial_state(trial_id=0, state="ready")
        self.optimizer.trial_id.increment()
        assert self.optimizer._serialize() is None

    def test_deserialize(self):
        self.optimizer.pre_process()
        self.optimizer.serialize_datas = {
            'num_of_generated_parameter': None,
            'loop_count': 0
        }
        self.optimizer.storage.trial.set_any_trial_state(trial_id=0, state="finished")
        self.optimizer.trial_id.increment()
        self.optimizer._serialize()
        assert self.optimizer._deserialize(trial_id=0) is None


def test_create_distributions(data_dir):
    config = Config(data_dir / 'config_tpe.json')
    params = load_parameter(config.hyperparameters.get())
    dist = create_distributions(params)
    assert type(dist) is dict
