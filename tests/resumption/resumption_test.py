from tests.integration.integration_test import IntegrationTest
from tests.integration.integration_test import start_master
from tests.integration.integration_test import wait_finish_wrapper
from unittest.mock import patch
import aiaccel
import aiaccel.util.filesystem as fs
import asyncio
import sys
import time

from aiaccel.master.local import LocalMaster
from aiaccel.argument import Arguments
from aiaccel.storage.storage import Storage

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

        if self.workspace.path.exists():
            self.workspace.clean()
        self.workspace.create()

        self.create_main()

        with patch.object(sys, 'argv', commandline_args):
            options = Arguments()
            master = LocalMaster(options)
            master.storage.alive.init_alive()
            run_master(master)
            final_result_at_one_time = get_final_result(work_dir, master)
            print('at one time', final_result_at_one_time)
            base_clean_work_dir(data_dir, work_dir)

        config_file = data_dir / f'config_{self.search_algorithm}_resumption.json'

        commandline_args = [
            "start.py",
            "--config",
            format(config_file)
        ]

        if self.workspace.path.exists():
            self.workspace.clean()
        self.workspace.create()

        self.create_main()

        time.sleep(5)

        with patch.object(sys, 'argv', commandline_args):
            options = Arguments()
            master = LocalMaster(options)
            master.storage.alive.init_alive()
            run_master(master)
            print('resumed steps finished')

        time.sleep(5)

        config_file = data_dir.joinpath(
            'config_{}.json'.format(self.search_algorithm)
        )
        commandline_args = [
            "start.py",
            "--config",
            format(config_file),
            "--resume",
            # "5"
            "4"
        ]
        with patch.object(sys, 'argv', commandline_args):
            options = Arguments()
            master = LocalMaster(options)
            master.storage.alive.init_alive()
            run_master(master)
            final_result_resumption = get_final_result(work_dir, master)
            print('resumption steps finished', final_result_resumption)
            assert final_result_at_one_time == final_result_resumption


def get_final_result(work_dir, master):
    # final_result = work_dir.joinpath(aiaccel.dict_result, aiaccel.file_final_result)
    # final_result_yaml = fs.load_yaml(final_result)
    # return final_result_yaml['result']
    data = master.storage.result.get_all_result()
    return [d.objective for d in data][-1]


def run_master(master):
    loop = asyncio.get_event_loop()
    gather = asyncio.gather(
        start_master(master),
        wait_finish_wrapper(1, master.storage, master)
    )
    loop.run_until_complete(gather)


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
