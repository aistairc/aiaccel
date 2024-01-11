

import os
import time

from aiaccel.manager import Job, LocalManager
from aiaccel.optimizer import create_optimizer

from tests.base_test import BaseTest
from unittest.mock import patch
import numpy as np
from aiaccel.config import is_multi_objective

class TestLocalManager(BaseTest):

    def test_init(self, config_json):
        config = self.load_config_for_test(self.configs['config.json'])
        optimizer = create_optimizer(config.optimize.search_algorithm)(config)
        assert type(LocalManager(config, optimizer)) is LocalManager

    def test_pre_process(
        self,
        setup_hp_running,
        setup_result,
        database_remove
    ):
        database_remove()
        config = self.load_config_for_test(self.configs['config.json'])
        optimizer = create_optimizer(config.optimize.search_algorithm)(config)
        manager = LocalManager(config, optimizer)
        setup_hp_running(2)
        setup_result(1)

        manager.pre_process()

        manager = LocalManager(config, optimizer)
        with patch.object(manager.storage.trial, 'get_running', return_value=[]):
            assert manager.pre_process() is None

        with patch.object(manager.storage.trial, 'get_running', return_value=[0, 1, 2]):
            assert manager.pre_process() is None

    def test_post_process(self, database_remove):
        database_remove()
        class dummy_job:
            def __init__(self):
                pass

            def stop(self):
                pass

            def join(self):
                pass

        jobs = []
        for i in range(10):
            jobs.append({'thread': dummy_job()})

        config = self.load_config_for_test(self.configs['config.json'])
        optimizer = create_optimizer(config.optimize.search_algorithm)(config)
        manager = LocalManager(config, optimizer)
        assert manager.post_process() is None

        with patch.object(manager, 'jobs', jobs):
            assert manager.post_process() is None

        assert manager.post_process() is None

    def test_inner_loop_main_process(
        self,
        clean_work_dir,
        config_json,
        setup_hp_ready,
        database_remove
    ):
        database_remove()
        config = self.load_config_for_test(self.configs['config.json'])

        if is_multi_objective(config):
            user_main_file = self.test_data_dir.joinpath('original_main_mo.py')
        else:
            user_main_file = None
        with self.create_main(from_file_path=user_main_file):
            optimizer = create_optimizer(config.optimize.search_algorithm)(config)
            manager = LocalManager(config, optimizer)
            # manager.pre_process()
            # setup_hp_ready(1)
            assert manager.inner_loop_main_process() is True

    def test_serialize(
        self,
        clean_work_dir,
        config_json,
        database_remove
    ):
        database_remove()
        config = self.load_config_for_test(self.configs['config.json'])
        optimizer = create_optimizer(config.optimize.search_algorithm)(config)
        manager = LocalManager(config, optimizer)
        manager._rng = np.random.RandomState(0)
        manager.storage.trial.set_any_trial_state(trial_id=0, state="finished")
        assert manager.serialize(trial_id=0) is None

    def test_deserialize(
        self,
        clean_work_dir,
        config_json,
        database_remove
    ):
        database_remove()
        config = self.load_config_for_test(self.configs['config.json'])
        optimizer = create_optimizer(config.optimize.search_algorithm)(config)
        manager = LocalManager(config, optimizer)
        manager.storage.trial.set_any_trial_state(trial_id=0, state="finished")
        manager._rng = np.random.RandomState(0)
        manager.serialize(trial_id=0)
        assert manager.deserialize(trial_id=0) is None

    def test_is_error_free(self, config_json, database_remove):
        database_remove()
        config = self.load_config_for_test(self.configs['config.json'])
        optimizer = create_optimizer(config.optimize.search_algorithm)(config)
        manager = LocalManager(config, optimizer)
        assert manager.is_error_free() is True

        jobstates = [
            {'trial_id': 0, 'jobstate': 'failure'}
        ]

        with patch.object(manager, 'job_status', {1: 'failure'}):
            with patch.object(manager.storage.jobstate, 'get_all_trial_jobstate', return_value=jobstates):
                assert manager.is_error_free() is True

        with patch.object(manager, 'job_status', {0: 'failure'}):
            with patch.object(manager.storage.jobstate, 'get_all_trial_jobstate', return_value=jobstates):
                assert manager.is_error_free() is False

    def test_resume(self, config_json):
        config = self.load_config_for_test(self.configs['config.json'])
        optimizer = create_optimizer(config.optimize.search_algorithm)(config)
        manager = LocalManager(config, optimizer)
        manager.pre_process()
        manager.serialize(0)
        manager.serialize(1)

        manager.config.resume = 0
        assert manager.resume() is None

        manager.config.resume = None
        assert manager.resume() is None
