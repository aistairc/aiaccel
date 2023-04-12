from __future__ import annotations

from collections.abc import Callable
from collections.abc import Generator
from pathlib import Path

import pytest

from aiaccel.optimizer import AbstractOptimizer
from aiaccel.optimizer import BudgetSpecifiedGridOptimizer
from tests.base_test import BaseTest


class TestBudgetSpecifiedGridOptimizer(BaseTest):
    @pytest.fixture(autouse=True)
    def setup_optimizer(
        self,
        data_dir: Path,
        create_tmp_config: Callable[[Path], Path]
    ) -> Generator[None, None, None]:

        self.data_dir = data_dir
        self.optimizer = BudgetSpecifiedGridOptimizer(self.load_config_for_test(self.configs['config_tpe.json']))
        yield
        self.optimizer = None

    def test_init(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        with monkeypatch.context() as m:
            m.setattr(AbstractOptimizer, "__init__", lambda *_: None)

            self.optimizer.config.optimize.trial_number = 0
            self.optimizer.config.optimize.grid_accept_small_trial_number = False
            with pytest.raises(ValueError):
                self.optimizer.__init__(self.optimizer.config)
            self.optimizer.config.optimize.grid_accept_small_trial_number = True
            assert self.optimizer.__init__(self.optimizer.config) is None

            self.optimizer.config.optimize.trial_number = 9999
            self.optimizer.config.optimize.grid_accept_small_trial_number = False
            assert self.optimizer.__init__(self.optimizer.config) is None
            self.optimizer.config.optimize.grid_accept_small_trial_number = True
            assert self.optimizer.__init__(self.optimizer.config) is None

    def test_generate_parameter(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.optimizer.pre_process()

        with monkeypatch.context() as m:
            m.setattr(self.optimizer, 'all_parameters_generated', lambda: True)
            assert self.optimizer.generate_parameter() is None

        with monkeypatch.context() as m:
            m.setattr(self.optimizer._grid_point_generator, "all_grid_points_generated", lambda: True)
            assert self.optimizer.generate_parameter() is None

        self.optimizer.all_parameters_generated = False
        assert len(self.optimizer.generate_parameter()) > 0

    def test_generate_initial_parameter(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.optimizer.pre_process()

        with monkeypatch.context() as m:
            hyperparameters = []
            for hyperparameter in self.optimizer.params.get_parameter_list():
                hyperparameter.initial = None
                hyperparameters.append(hyperparameter)
            m.setattr(self.optimizer.params, 'get_parameter_list', lambda: hyperparameters)
            assert len(self.optimizer.generate_initial_parameter()) > 0

        with monkeypatch.context() as m:
            hyperparameters = []
            for hyperparameter in self.optimizer.params.get_parameter_list():
                hyperparameter.initial = hyperparameter.lower
                hyperparameters.append(hyperparameter)
            m.setattr(self.optimizer.params, 'get_parameter_list', lambda: hyperparameters)
            assert len(self.optimizer.generate_initial_parameter()) > 0

        with monkeypatch.context() as m:
            m.setattr(self.optimizer, 'generate_parameter', lambda: None)
            with pytest.raises(ValueError):
                _ = self.optimizer.generate_initial_parameter()
