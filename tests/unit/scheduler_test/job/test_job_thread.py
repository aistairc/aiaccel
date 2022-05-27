from aiaccel.config import ConfileWrapper
from aiaccel.scheduler.local_scheduler import LocalScheduler
from aiaccel.scheduler.job.job_thread import CustomMachine, Job, JOB_STATES,\
    JOB_TRANSITIONS, Model
from aiaccel.util.time_tools import get_time_now_object
from tests.base_test import BaseTest
import aiaccel
import asyncio
import datetime
import json
import pytest
import time
from pathlib import Path
import sys
from unittest.mock import patch


async def async_start_job(job):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, job.start())


async def async_stop_job_after_sleep(job, sleep_time):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, time.sleep, sleep_time)
    job.get_machine().set_state('Sucess')
    job.join()


def test_custom_machine():
    machine = CustomMachine(
        model=Model(),
        states=JOB_STATES,
        transitions=JOB_TRANSITIONS,
        initial=JOB_STATES[0]['name'],
        auto_transitions=False,
        ordered_transitions=False
    )
    assert type(machine) is CustomMachine


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

        commandline_args = [
            "start.py",
            "--config",
            format(config_json)
        ]

        with patch.object(sys, 'argv', commandline_args):
            from aiaccel import start
            scheduler = start.Scheduler()
        # scheduler = LocalScheduler(config_json)
        # config = load_test_config()
        setup_hp_ready(1)
        self.job = Job(
            self.config,
            Path(config_json).resolve(),
            scheduler,
            work_dir.joinpath(aiaccel.dict_hp_ready, '001.hp')
        )
        self.model = Model()
        yield
        self.job = None
        self.model = None

    @pytest.fixture
    def setup_abci_job(self, config_json, work_dir):
        with open(config_json) as f:
            json_object = json.load(f)

        json_object['resource']['type'] = 'ABCI'
        json_object_config = ConfileWrapper(json_object, 'json_object')
        
        commandline_args = [
            "start.py",
            "--config",
            format(config_json)
        ]

        with patch.object(sys, 'argv', commandline_args):
            from aiaccel import start
            scheduler = start.Scheduler()
        # scheduler = LocalScheduler(config_json)
        self.abci_job = Job(
            # json_object_config,
            self.config,
            Path(config_json).resolve(),
            scheduler,
            work_dir.joinpath(aiaccel.dict_hp_ready, '001.hp')
        )
        yield
        self.abci_job = None

    def test_after_confirmed(self):
        assert self.model.after_confirmed(self.job) is None

    def test_before_failed(self):
        assert self.model.before_failed(self.job) is None

    def test_conditions_file_confirmed(self, work_dir):
        self.job.to_file = work_dir.joinpath(aiaccel.dict_hp_ready, '001.hp')
        assert self.model.conditions_file_confirmed(self.job)

    def test_before_file_move(self, work_dir):
        self.job.from_file = work_dir.joinpath(aiaccel.dict_hp_ready, '001.hp')
        self.job.to_file = work_dir.joinpath(aiaccel.dict_hp_ready, '002.hp')
        assert self.model.before_file_move(self.job) is None

    def test_after_runner(self):
        assert self.model.after_runner(self.job) is None

    def test_before_runner_create(self, cd_work, setup_abci_job, work_dir):
        assert self.model.before_runner_create(self.job) is None

        self.abci_job.to_file = work_dir.joinpath(aiaccel.dict_runner, 'run_001.sh')
        assert self.model.before_runner_create(self.abci_job) is None

    def test_conditions_runner_confirmed(self, setup_abci_job, work_dir):
        assert self.model.conditions_runner_confirmed(self.job)

        self.abci_job.to_file = work_dir.joinpath(aiaccel.dict_hp_ready, '001.hp')
        assert self.model.conditions_runner_confirmed(self.abci_job)

    def test_after_running(self):
        assert self.model.after_running(self.job) is None

    def test_after_job(self):
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

    def test_conditions_job_confirmed(self):
        assert not self.model.conditions_job_confirmed(self.job)

        self.job.scheduler.stats.append({'name': '001'})
        assert self.model.conditions_job_confirmed(self.job)

    def test_after_result(self):
        assert self.model.after_result(self.job) is None

    def test_after_wait_result(self):
        assert self.model.after_wait_result(self.job) is None

    def test_conditions_result(self):
        assert not self.model.conditions_result(self.job)

    def test_after_finished(self):
        assert self.model.after_finished(self.job) is None

    def test_before_finished_move(self, setup_hp_running, setup_result,
                                  work_dir):
        assert self.model.before_finished_move(self.job) is None

        setup_hp_running(1)
        setup_result(1)
        self.job.from_file = work_dir.joinpath(aiaccel.dict_hp_running, '001.hp')
        self.job.to_file = work_dir.joinpath(aiaccel.dict_hp_finished, '001.hp')
        assert self.model.before_finished_move(self.job) is None

    def test_after_expire(self):
        assert self.model.after_expire(self.job) is None

    def test_after_kill(self):
        assert self.model.after_kill(self.job) is None

    def test_before_kill_submitted(self, fake_process):
        assert self.model.before_kill_submitted(self.job) is None

        fake_pid = 99999999
        self.job.scheduler.stats.append({'name': '001', 'job-ID': fake_pid})
        fake_process.register_subprocess(
            [
                '/bin/kill', '{}'.format(fake_pid)
            ],
            stdout=[]
        )
        assert self.model.before_kill_submitted(self.job) is None

    def test_conditions_kill_confirmed(self):
        assert self.model.conditions_kill_confirmed(self.job)

        self.job.scheduler.stats.append({'name': '001'})
        assert not self.model.conditions_kill_confirmed(self.job)

    def test_after_check_result(self):
        assert self.model.after_check_result(self.job) is None

    def test_after_cancel(self, setup_hp_running, work_dir):
        assert self.model.after_cancel(self.job) is None

        work_dir.joinpath(aiaccel.dict_hp_ready, '001.hp').unlink()
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

        commandline_args = [
            "start.py",
            "--config",
            format(config_json)
        ]
        with patch.object(sys, 'argv', commandline_args):
            from aiaccel import start
            scheduler = start.Scheduler()
        # scheduler = LocalScheduler(config_json)
        # config = load_test_config()
        setup_hp_ready(1)
        self.job = Job(
            self.config,
            Path(config_json).resolve(),
            scheduler,
            work_dir.joinpath(aiaccel.dict_hp_ready, '001.hp')
        )
        yield
        self.job = None

    def test_init(
        self,
        clean_work_dir,
        config_json,
        load_test_config,
        setup_hp_ready,
        work_dir
    ):
        options = {
            'config': config_json,
            'resume': None,
            'clean': False,
            'nosave': False,
            'dbg': False,
            'graph': False,
            'process_name': 'scheduler'
        }
        scheduler = LocalScheduler(options)
        # config = load_test_config()
        setup_hp_ready(1)
        job = Job(
            self.config,
            Path(config_json).resolve(),
            scheduler,
            work_dir.joinpath(aiaccel.dict_hp_ready, '001.hp')
        )
        assert type(job) is Job

    #def test_get_initial_timeout(self):
    #    assert type(self.job.get_initial_timeout()) is datetime.datetime

    def test_get_machine(self):
        assert type(self.job.get_machine()) is CustomMachine

    def test_get_model(self):
        assert type(self.job.get_model()) is Model

    def test_get_state(self):
        assert self.job.get_state().name == 'Init'

    def test_get_state_name(self):
        assert self.job.get_state_name() == 'Init'

    def test_is_local(self):
        assert self.job.is_local()

    def test_schedule(self):
        self.job.get_machine().set_state('Scheduling')
        assert self.job.schedule() is None

    def test_run_1(self):
        self.job.start()
        self.job.join()

    def test_run_2(self):
        self.job.scheduler.pre_process()
        self.job.start()
        time.sleep(1)
        self.job.threshold_timeout = get_time_now_object()
        time.sleep(1)
        self.job.threshold_timeout =\
            get_time_now_object() + datetime.timedelta(10)
        self.job.get_machine().set_state('Init')
        time.sleep(1)
        self.job.count_retry = 100
        self.job.threshold_retry = 10
        time.sleep(1)
        self.job.get_machine().set_state('Success')
        self.job.join()
