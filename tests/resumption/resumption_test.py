from tests.integration.integration_test import IntegrationTest
from tests.integration.integration_test import start_master
from tests.integration.integration_test import wait_finish_wrapper
from unittest.mock import patch
import aiaccel
import aiaccel.util.filesystem as fs
import asyncio
import pathlib
import sys
import time


class ResumptionTest(IntegrationTest):
    search_algorithm = None

    def test_run(self, cd_work, data_dir, work_dir):
        config_file = data_dir.joinpath(
            'config_{}.json'.format(self.search_algorithm)
        )
        commandline_args = [
            "start.py",
            "--config",
            format(config_file)
        ]
        run_master(commandline_args, work_dir)
        final_result_at_one_time = get_final_result(work_dir)
        print('at one time', final_result_at_one_time)
        wait_alive(work_dir)
        base_clean_work_dir(data_dir, work_dir)

        config_file = data_dir.joinpath(
            'config_{}_resumption.json'.format(self.search_algorithm)
        )
        commandline_args = [
            "start.py",
            "--config",
            format(config_file)
        ]
        run_master(commandline_args, work_dir)
        print('resumed steps finished')
        wait_alive(work_dir)

        config_file = data_dir.joinpath(
            'config_{}.json'.format(self.search_algorithm)
        )
        dict_resume = sorted(
            [f for f in list(work_dir.joinpath(aiaccel.dict_state).iterdir()) if pathlib.Path.is_dir(f)]
        )[-1].name
        commandline_args = [
            "start.py",
            "--config",
            format(config_file),
            "--resume",
            dict_resume
        ]
        run_master(commandline_args, work_dir)
        final_result_resumption = get_final_result(work_dir)
        print('resumption steps finished', final_result_resumption)
        assert final_result_at_one_time == final_result_resumption


def get_final_result(work_dir):
    final_result = work_dir.joinpath(aiaccel.dict_result, aiaccel.file_final_result)
    final_result_yaml = fs.load_yaml(final_result)
    return final_result_yaml['result']


def run_master(commandline_args, work_dir):
    with patch.object(sys, 'argv', commandline_args):
        from aiaccel import start
        master = start.Master()
        loop = asyncio.get_event_loop()
        gather = asyncio.gather(
            start_master(master),
            wait_finish_wrapper(1, work_dir.joinpath(aiaccel.dict_alive), master)
        )
        loop.run_until_complete(gather)


def wait_alive(work_dir):
    alive_files = [
        work_dir.joinpath(aiaccel.dict_alive, aiaccel.alive_master),
        work_dir.joinpath(aiaccel.dict_alive, aiaccel.alive_optimizer),
        work_dir.joinpath(aiaccel.dict_alive, aiaccel.alive_scheduler)
    ]
    while True:
        alive = False
        for alive_file in alive_files:
            if fs.check_alive_file(alive_file):
                alive = True
                break
        if not alive:
            break
        time.sleep(0.1)


def base_clean_work_dir(data_dir, work_dir):
    from tests.conftest import WORK_FILES, WORK_SUB_DIRECTORIES
    import shutil

    while not work_dir.is_dir():
        if not work_dir.exists():
            work_dir.mkdir()

        if not work_dir.is_dir():
            work_dir.unlink()

    valid_dir = [work_dir.joinpath(wsd) for wsd in WORK_SUB_DIRECTORIES]
    valid_dir += [work_dir]
    work_files = [work_dir.joinpath(wf) for wf in WORK_FILES]

    fs.clean_directory(
        work_dir, exclude_file=work_files
    )

    for p in work_dir.glob('**/*'):
        # TODO: this part can be replaced using PurePath.is_relative_to()
        #  from version 3.9
        if p.is_dir() and \
                p not in [work_dir.joinpath(wsd) for wsd in WORK_SUB_DIRECTORIES] and \
                any([str(work_dir.joinpath(wsd)) in str(p) for wsd in WORK_SUB_DIRECTORIES]):
            shutil.rmtree(p)

    for d in valid_dir:
        while d.exists() and not d.is_dir():
            d.unlink()
            d.mkdir()

    for wsd in WORK_SUB_DIRECTORIES:
        if not work_dir.joinpath(wsd).exists():
            work_dir.joinpath(wsd).mkdir()

    for wf in WORK_FILES:
        if not work_dir.joinpath(wf).exists():
            shutil.copyfile(data_dir.joinpath(wf), work_dir.joinpath(wf))

    # re-create files
    for wf in WORK_FILES:
        shutil.copyfile(data_dir.joinpath(wf), work_dir.joinpath(wf))
