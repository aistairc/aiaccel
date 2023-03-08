from __future__ import annotations

from collections.abc import Callable
from collections.abc import Generator
from pathlib import Path

import pytest

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
        self.config_path = create_tmp_config(
            self.data_dir / 'config_budget-specified-grid.json'
        )
        self.options = {
            'config': self.config_path,
            'resume': None,
            'clean': False,
            'process_name': 'optimizer'
        }
        self.optimizer = BudgetSpecifiedGridOptimizer(self.options)
        yield
        self.optimizer = None

    def test_init(
        self,
        data_dir: Path,
        create_tmp_config: Callable[[Path], Path],
    ) -> None:

        options = {
            'config': self.config_path,
            'resume': None,
            'clean': False,
            'process_name': 'optimizer',
            'accept_small_trial_number': True
        }
        assert isinstance(
            BudgetSpecifiedGridOptimizer(options),
            BudgetSpecifiedGridOptimizer
        )

        config_path = create_tmp_config(
            data_dir / 'config_budget-specified-grid_too_small_trials.json'
        )
        options = {
            'config': config_path,
            'resume': None,
            'clean': False,
            'process_name': 'optimizer'
        }
        with pytest.raises(ValueError):
            _ = BudgetSpecifiedGridOptimizer(options)

        options = {
            'config': config_path,
            'resume': None,
            'clean': False,
            'process_name': 'optimizer',
            'accept_small_trial_number': True
        }
        assert isinstance(
            BudgetSpecifiedGridOptimizer(options),
            BudgetSpecifiedGridOptimizer
        )

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
