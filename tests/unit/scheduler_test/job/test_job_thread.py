import asyncio
import datetime
import json
import sys
import time
from unittest.mock import patch
from aiaccel.config import load_config

with patch('aiaccel.util.retry.retry', lambda function: function):
    import aiaccel
    import pytest
    from aiaccel.config import load_config

    from aiaccel.scheduler.create import create_scheduler
    from aiaccel.scheduler.job.job import (JOB_STATES, JOB_TRANSITIONS,
                                                CustomMachine, Job, create_model)
    from aiaccel.scheduler.job.model.abci_model import AbciModel
    from aiaccel.scheduler.job.model.local_model import LocalModel
    from aiaccel.scheduler.local_scheduler import LocalScheduler
    from aiaccel.util.time_tools import get_time_now_object
    from tests.arguments import parse_arguments
    from tests.base_test import BaseTest



async def async_start_job(job):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, job.start())


async def async_stop_job_after_sleep(job, sleep_time):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, time.sleep, sleep_time)
    job.get_machine().set_state('Sucess')
    job.join()


class TestModel(BaseTest):

    @pytest.fixture(autouse=True)
    def setup_job(
        self,
        clean_work_dir,
        config_json,
        load_test_config,
        setup_hp_ready,
        work_dir
    ):

        self.workspace.clean()
        self.workspace.create()

        config = self.configs['config.json']
        scheduler = create_scheduler(config.resource.type)(config)

        setup_hp_ready(1)
        trial_id = 0
        self.job = Job(
            config,
            scheduler,
            trial_id
        )
        self.model = create_model(config.resource.type)
        yield
        self.job = None
        self.model = None

    @pytest.fixture
    def setup_abci_job(
        self,
        config_json,
        work_dir,
        database_remove
    ):

        with open(config_json) as f:
            json_object = json.load(f)

        json_object['resource']['type'] = 'ABCI'
        
        config = load_config(config_json)
        config.resource.type = 'ABCI'

        scheduler = create_scheduler(config.resource.type)(config)

        # scheduler = LocalScheduler(config_json)
        trial_id = 1
        self.abci_job = Job(
            config,
            scheduler,
            trial_id
        )
        yield
        self.abci_job = None

    def test_after_confirmed(self, database_remove):
        assert self.model.after_confirmed(self.job) is None

    def test_before_failed(self, database_remove):
        assert self.model.before_failed(self.job) is None

    def test_conditions_confirmed(self, work_dir, database_remove):
        self.job.to_file = work_dir.joinpath(aiaccel.dict_hp_ready, '001.hp')
        self.job.next_state = 'ready'
        assert self.model.conditions_confirmed(self.job)

    # def test_before_file_move(self, work_dir):
    #     self.job.from_file = work_dir.joinpath(aiaccel.dict_hp_ready, '001.hp')
    #     self.job.to_file = work_dir.joinpath(aiaccel.dict_hp_ready, '002.hp')
    #     assert self.model.before_file_move(self.job) is None

    def test_after_runner(self, database_remove):
        assert self.model.after_runner(self.job) is None

    def test_before_runner_create(
        self,
        cd_work,
        setup_abci_job,
        work_dir,
        database_remove
    ):
        assert self.model.before_runner_create(self.job) is None

        self.abci_job.to_file = work_dir.joinpath(aiaccel.dict_runner, 'run_001.sh')
        assert self.model.before_runner_create(self.abci_job) is None

    def test_conditions_runner_confirmed(
        self,
        setup_abci_job,
        work_dir,
        database_remove
    ):
        assert self.model.conditions_runner_confirmed(self.job)

        self.abci_job.to_file = work_dir.joinpath(aiaccel.dict_hp_ready, '001.hp')
        assert self.model.conditions_runner_confirmed(self.abci_job)

    def test_after_running(self, database_remove):
        assert self.model.after_running(self.job) is None

    def test_after_job(self, database_remove):
        assert self.model.after_job(self.job) is None

    # def test_before_job_submitted(self, fake_process, setup_abci_job):
        # """
        #     Configファイルパスが環境によって変わるため，このUnitテストは不可．
        #     実機試験等,他のテストで補う
        # """
        # fake_process.register_subprocess(
        #     [
        #         'python', 'wrapper.py',
        #         '-i', '001',
        #         '-c',
        #         '/home/member/opt/working/opt/tests/test_data/config.json',
        #         '-x1=0.9932890709584586',
        #         '-x10=3.599465287952899',
        #         '-x2=-3.791100401941936',
        #         '-x3=-1.6730481463987088',
        #         '-x4=2.2148440758326835',
        #         '-x5=2.111917696952796',
        #         '-x6=4.364405867994597',
        #         '-x7=-0.7789300003858477',
        #         '-x8=3.30035693274327',
        #         '-x9=1.7030556641407104'
        #     ],
        #     stdout=[]
        # )
        # assert self.model.before_job_submitted(self.job) is None
        #
        # fake_process.register_subprocess(
        #     [
        #         'qsub', '-g', 'gaa*****',
        #         '-j', 'y',
        #         '-o', '/tmp/work/abci_output',
        #         '/tmp/work/runner/run_001.sh'
        #     ],
        #     stdout=[], stderr=[]
        # )
        # assert self.model.before_job_submitted(self.abci_job) is None

    def test_conditions_job_confirmed(self, database_remove):
        assert not self.model.conditions_job_confirmed(self.job)

        # self.job.scheduler.stats.append({'name': '001'})
        # self.job.scheduler.stats.append({'name': 0})
        self.job.scheduler.stats.append(
            {'name': '2 python user.py --trial_id 0 --config config.yaml --x1=1.0 --x2=1.0',}
        )
        assert self.model.conditions_job_confirmed(self.job)

    def test_after_result(self, database_remove):
        assert self.model.after_result(self.job) is None

    def test_after_wait_result(self, database_remove):
        assert self.model.after_wait_result(self.job) is None

    def test_conditions_result(self, database_remove):
        assert not self.model.conditions_result(self.job)

    def test_after_finished(self, database_remove):
        assert self.model.after_finished(self.job) is None

    """
    def test_before_finished(
        self,
        setup_hp_running,
        setup_result,
        work_dir,
        database_remove
    ):
        # setup_hp_running(0)
        # setup_result(0)
        # print(self.job.trial_id_str)
        # print(self.storage.result.get_result_trial_id_list())
        print(self.job.storage.result.get_all_result())
        for i in range(10):
            self.job.storage.result.set_any_trial_objective(trial_id=i, objective=i*1.0)
            for j in range(10):
                self.job.storage.hp.set_any_trial_param(
                    trial_id=i,
                    param_name=f'x{j+1}',
                    param_value=0.0,
                    param_type='float'
                )
        assert self.model.before_finished(self.job) is None

        # self.job.storage.trial.all_delete()
        # self.job.storage.hp.all_delete()

        # setup_hp_running(1)
        # setup_result(1)

        for i in range(10):
            self.job.storage.trial.set_any_trial_state(trial_id=i, state='finished')
            for j in range(10):
                self.job.storage.hp.set_any_trial_param(
                    trial_id=i,
                    param_name=f'x{j+1}',
                    param_value=0.0,
                    param_type='float'
                )
        print(self.job.trial_id)
        print([d.objective for d in self.job.storage.result.get_all_result()])
        print(self.job.storage.get_best_trial_dict('minimize'))

        self.job.next_state = 'finished'
        self.job.from_file = work_dir.joinpath(aiaccel.dict_hp_running, '001.hp')
        self.job.to_file = work_dir.joinpath(aiaccel.dict_hp_finished, '001.hp')
        assert self.model.before_finished(self.job) is None
    """

    def test_before_finished(
        self,
        setup_hp_running,
        setup_result,
        work_dir,
        database_remove
    ):
        # setup_hp_running(0)
        # setup_result(0)
        # print(self.job.trial_id_str)
        # print(self.storage.result.get_result_trial_id_list())
        print(self.job.storage.result.get_all_result())
        for i in range(10):
            self.job.storage.result.set_any_trial_objective(trial_id=i, objective=i*1.0)
            self.job.storage.hp.set_any_trial_params(
                trial_id=i,
                params=[
                    {'parameter_name': f'x{j+1}', 'value': 0.0, 'type': 'float'}
                    for j in range(10)
                ]
            )
            """
            for j in range(10):
                self.job.storage.hp.set_any_trial_param(
                    trial_id=i,
                    param_name=f'x{j+1}',
                    param_value=0.0,
                    param_type='float'
                )
            """
        assert self.model.before_finished(self.job) is None

        # self.job.storage.trial.all_delete()
        # self.job.storage.hp.all_delete()

        # setup_hp_running(1)
        # setup_result(1)

        for i in range(10):
            self.job.storage.trial.set_any_trial_state(trial_id=i, state='finished')
            self.job.storage.hp.set_any_trial_params(
                trial_id=i,
                params=[
                    {'parameter_name': f'x{j+1}', 'value': 0.0, 'type': 'float'}
                    for j in range(10)
                ]
            )
            """
            for j in range(10):
                self.job.storage.hp.set_any_trial_param(
                    trial_id=i,
                    param_name=f'x{j+1}',
                    param_value=0.0,
                    param_type='float'
                )
            """
        print(self.job.trial_id)
        print([d.objective for d in self.job.storage.result.get_all_result()])
        print(self.job.storage.get_best_trial_dict('minimize'))

        self.job.next_state = 'finished'
        self.job.from_file = work_dir.joinpath(aiaccel.dict_hp_running, '001.hp')
        self.job.to_file = work_dir.joinpath(aiaccel.dict_hp_finished, '001.hp')
        assert self.model.before_finished(self.job) is None

    def test_after_expire(self, database_remove):
        assert self.model.after_expire(self.job) is None

    def test_after_kill(self, database_remove):
        assert self.model.after_kill(self.job) is None

    def test_before_kill_submitted(self, fake_process, database_remove):
        assert self.model.before_kill_submitted(self.job) is None

        fake_pid = 99999999
        # self.job.scheduler.stats.append({'name': '001', 'job-ID': fake_pid})
        self.job.scheduler.stats.append(
            {
                'name': '2 python user.py --trial_id 1 --config config.yaml --x1=1.0 --x2=1.0',
                'job-ID': fake_pid
            }
        )
        fake_process.register_subprocess(
            [
                '/bin/kill', f'{fake_pid}'
            ],
            stdout=[]
        )
        assert self.model.before_kill_submitted(self.job) is None

    def test_conditions_kill_confirmed(self, database_remove):
        assert self.model.conditions_kill_confirmed(self.job)

        # self.job.scheduler.stats.append({'name': '001'})
        # self.job.scheduler.stats.append({'name': 0})
        self.job.scheduler.stats.append(
            {'name': '2 python user.py --trial_id 0 --config config.yaml --x1=1.0 --x2=1.0'}
        )
        assert not self.model.conditions_kill_confirmed(self.job)

    def test_after_check_result(self, database_remove):
        assert self.model.after_check_result(self.job) is None

    def test_after_cancel(
        self,
        setup_hp_running,
        work_dir,
        database_remove
    ):
        assert self.model.after_cancel(self.job) is None
        setup_hp_running(1)
        assert self.model.after_cancel(self.job) is None

class TestJob(BaseTest):

    @pytest.fixture(autouse=True)
    def setup_job(
        self,
        clean_work_dir,
        config_json,
        load_test_config,
        setup_hp_ready,
        work_dir
    ):
        self.workspace.clean()
        self.workspace.create()

        config = self.configs['config.json']
        scheduler = create_scheduler(config.resource.type)(config)

        setup_hp_ready(1)
        trial_id = 1
        self.job = Job(
            config,
            scheduler,
            trial_id
        )
        yield
        self.job = None

    def test_init(
        self,
        clean_work_dir,
        config_json,
        load_test_config,
        setup_hp_ready,
        work_dir,
        database_remove
    ):

        options = {
            'config': config_json,
            'resume': None,
            'clean': False,
            'fs': False,
            'process_name': 'scheduler'
        }

        config = self.configs['config.json']
        scheduler = LocalScheduler(config)
        # config = load_test_config()
        setup_hp_ready(1)
        trial_id = 1
        job = Job(
            config,
            scheduler,
            trial_id
        )
        assert type(job) is Job

    # def test_get_initial_timeout(self):
    #    assert type(self.job.get_initial_timeout()) is datetime.datetime

    def test_get_machine(self, database_remove):
        assert type(self.job.get_machine()) is CustomMachine

    def test_get_model(self, database_remove):
        assert type(self.job.get_model()) is LocalModel or AbciModel

    def test_get_state(self, database_remove):
        assert self.job.get_state().name == 'Init'

    def test_get_state_name(self, database_remove):
        assert self.job.get_state_name() == 'Init'

    def test_schedule(self, database_remove):
        self.job.get_machine().set_state('Scheduling')
        assert self.job.schedule() is None

    def test_run_2(self, database_remove):
        self.job.scheduler.pre_process()
        self.job.main()
        self.job.threshold_timeout = get_time_now_object()
        self.job.threshold_timeout =\
            get_time_now_object() + datetime.timedelta(10)
        self.job.get_machine().set_state('Init')
        self.job.count_retry = 100
        self.job.threshold_retry = 10
        self.job.get_machine().set_state('Success')
        self.job.main()
