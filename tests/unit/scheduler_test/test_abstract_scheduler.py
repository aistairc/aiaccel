from aiaccel.scheduler.abstract_scheduler import AbstractScheduler,\
    parse_hashname
from tests.base_test import BaseTest
import aiaccel
import asyncio
import os
import time


async def async_function(func):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, func)


async def make_directory(sleep_time, d):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, time.sleep, sleep_time)
    os.mkdir(d)


async def stop_jobs(sleep_time, scheduler):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, time.sleep, sleep_time)
    for job in scheduler.jobs:
        machine = job['thread'].get_machine()
        machine.set_state('Success')


'''
def test_create_stat():
    assert create_stat(1, 'name') == {
        'job-ID': 1, 'prior': None, 'name': 'name', 'user': None, 'state': 'r',
        'submit/start at': None, 'queue': None, 'jclass': '', 'slots': None,
        'ja-task-ID': None
    }
'''


def test_parse_hashname():
    s = "python wrapper.py --index 001"
    name = parse_hashname(s)
    assert name in '001'


class TestAbstractScheduler(BaseTest):
    # TODO: Need to fix?
    #  logging modules are generated each method.

    def test_init(self, config_json):
        options = {
            'config': config_json,
            'resume': None,
            'clean': False,
            'nosave': False,
            'dbg': False,
            'graph': False,
            'process_name': 'scheduler'
        }
        assert type(AbstractScheduler(options)) is AbstractScheduler

    def test_check_finished_hp(
        self,
        clean_work_dir,
        config_json,
        setup_hp_running,
        setup_result,
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
        scheduler = AbstractScheduler(options)
        scheduler.print_dict_state()
        setup_hp_running(1)
        setup_result(1)
        assert scheduler.check_finished_hp() is None

    def test_get_stats(self, config_json, work_dir):
        options = {
            'config': config_json,
            'resume': None,
            'clean': False,
            'nosave': False,
            'dbg': False,
            'graph': False,
            'process_name': 'scheduler'
        }
        scheduler = AbstractScheduler(options)
        scheduler.print_dict_state()
        assert scheduler.get_stats() is None

    def test_start_job_thread(
        self,
        clean_work_dir,
        config_json,
        setup_hp_ready, work_dir
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
        scheduler = AbstractScheduler(options)
        scheduler.print_dict_state()
        setup_hp_ready(1)
        scheduler.start_job_thread(
            work_dir.joinpath(aiaccel.dict_hp_ready).joinpath('001.hp')
        )
        assert scheduler.start_job_thread(
            work_dir.joinpath(aiaccel.dict_hp_ready).joinpath('001.hp')
        ) is None

        for job in scheduler.jobs:
            machine = job['thread'].get_machine()
            machine.set_state('Success')
            job['thread'].join()

    def test_update_resource(self, config_json, work_dir):
        options = {
            'config': config_json,
            'resume': None,
            'clean': False,
            'nosave': False,
            'dbg': False,
            'graph': False,
            'process_name': 'scheduler'
        }
        scheduler = AbstractScheduler(options)
        scheduler.print_dict_state()
        assert scheduler.update_resource() is None

    def test_pre_process(
        self,
        clean_work_dir,
        config_json,
        setup_hp_running,
        setup_result,
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
        scheduler = AbstractScheduler(options)
        scheduler.print_dict_state()
        setup_hp_running(2)
        setup_result(1)

        scheduler.pre_process()

        for job in scheduler.jobs:
            machine = job['thread'].get_machine()
            machine.set_state('Success')
            job['thread'].join()

    def test_post_process(self, config_json):
        options = {
            'config': config_json,
            'resume': None,
            'clean': False,
            'nosave': False,
            'dbg': False,
            'graph': False,
            'process_name': 'scheduler'
        }
        scheduler = AbstractScheduler(options)
        assert scheduler.post_process() is None

    def test_loop_pre_process(self, config_json, work_dir):
        options = {
            'config': config_json,
            'resume': None,
            'clean': False,
            'nosave': False,
            'dbg': False,
            'graph': False,
            'process_name': 'scheduler'
        }
        scheduler = AbstractScheduler(options)
        work_dir.joinpath(aiaccel.dict_runner).rmdir()
        loop = asyncio.get_event_loop()
        gather = asyncio.gather(
            async_function(scheduler.loop_pre_process),
            make_directory(1, work_dir.joinpath(aiaccel.dict_runner))
        )
        loop.run_until_complete(gather)

    def test_loop_post_process(self, config_json):
        options = {
            'config': config_json,
            'resume': None,
            'clean': False,
            'nosave': False,
            'dbg': False,
            'graph': False,
            'process_name': 'scheduler'
        }
        scheduler = AbstractScheduler(options)
        assert scheduler.loop_post_process() is None

    def test_inner_loop_pre_process(
        self,
        clean_work_dir,
        config_json,
        setup_hp_finished,
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
        scheduler = AbstractScheduler(options)
        assert not scheduler.inner_loop_pre_process()

        scheduler.pre_process()
        assert scheduler.inner_loop_pre_process()

        setup_hp_ready(1)
        setup_hp_finished(10)
        scheduler.start_job_thread(
            work_dir.joinpath(aiaccel.dict_hp_ready, '001.hp')
        )
        loop = asyncio.get_event_loop()
        gather = asyncio.gather(
            async_function(scheduler.inner_loop_pre_process),
            stop_jobs(1, scheduler)
        )
        loop.run_until_complete(gather)

    def test_inner_loop_main_process(
        self,
        clean_work_dir,
        config_json,
        setup_hp_ready
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
        scheduler = AbstractScheduler(options)
        scheduler.pre_process()
        setup_hp_ready(1)

        assert scheduler.inner_loop_main_process()

        for job in scheduler.jobs:
            machine = job['thread'].get_machine()
            machine.set_state('Scheduling')

        assert scheduler.inner_loop_main_process()

        for job in scheduler.jobs:
            machine = job['thread'].get_machine()
            machine.set_state('Success')
            job['thread'].join()

    def test_inner_loop_post_process(self, config_json):
        options = {
            'config': config_json,
            'resume': None,
            'clean': False,
            'nosave': False,
            'dbg': False,
            'graph': False,
            'process_name': 'scheduler'
        }
        scheduler = AbstractScheduler(options)
        assert scheduler.inner_loop_post_process()

    def test_serialize(self, clean_work_dir, config_json):
        options = {
            'config': config_json,
            'resume': None,
            'clean': False,
            'nosave': False,
            'dbg': False,
            'graph': False,
            'process_name': 'scheduler'
        }
        scheduler = AbstractScheduler(options)
        serialized_dict = scheduler._serialize()
        assert 'loop_count' in serialized_dict

    def test_deserialize(self, clean_work_dir, config_json):
        options = {
            'config': config_json,
            'resume': None,
            'clean': False,
            'nosave': False,
            'dbg': False,
            'graph': False,
            'process_name': 'scheduler'
        }
        scheduler = AbstractScheduler(options)
        serialized_dict = scheduler._serialize()
        assert scheduler._deserialize(serialized_dict) is None
