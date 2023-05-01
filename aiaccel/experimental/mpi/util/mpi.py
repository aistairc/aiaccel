from __future__ import annotations

from pathlib import Path
from threading import Thread
from subprocess import Popen, PIPE, STDOUT, run
from os import environ, path as os_path, linesep
from logging import Logger

from mpi4py.futures import MPIPoolExecutor
from mpi4py.MPI import Get_processor_name, COMM_WORLD
from fasteners import InterProcessLock

from aiaccel.common import resource_type_abci
from aiaccel.experimental.mpi.common import dict_experimental
from aiaccel.experimental.mpi.common import dict_mpi
from aiaccel.experimental.mpi.common import file_mpi_lock
from aiaccel.experimental.mpi.common import dict_rank_log
from aiaccel.experimental.mpi.common import file_mpi_lock_timeout
from aiaccel.experimental.mpi.util.error import MpiError
from aiaccel.experimental.mpi.util.mpi_log import MpiLog
from aiaccel.experimental.mpi.config import MpiConfig

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from aiaccel.storage import Storage
    from aiaccel.scheduler import AbstractScheduler


class Mpi:
    func_end_id = 'MpiFuncEnd'
    executor: MPIPoolExecutor | None = None
    lock: InterProcessLock | None = None
    rank_log_path = None
    error_file_path = None
    log: MpiLog | None = None
    gpu_max = 0
    gpu_list: dict[str, list[list[int]]] = {}
    tag = 0
    trial_id_list: list[[int, int]] = [] # [[trial_id, tag]]

    @classmethod
    def prepare(cls, workspace_path: Path) -> None:
        mpi_path = workspace_path / dict_experimental / dict_mpi
        cls.error_file_path = mpi_path / 'error.log'
        lock_file_path = mpi_path / file_mpi_lock
        cls.rank_log_path = mpi_path / dict_rank_log
        cls.rank_log_path.mkdir(parents=True, exist_ok=True)
        cls.lock = InterProcessLock(str(lock_file_path))
        if cls.log is not None:
            raise MpiError('cls.log is not None', cls.error_file_path)
        cls.log = MpiLog(0, Get_processor_name(), cls.rank_log_path)
        cls.log.write(f'prepare: rank=0 tag={cls.tag}', cls.tag, stdout=True)

    @classmethod
    def abort(cls) -> None:
        COMM_WORLD.Abort()

    @classmethod
    def run_main(cls) -> None:
        if cls.executor is None:
            cls.executor = MPIPoolExecutor()

    @classmethod
    def run_worker(cls) -> None:
        MPIPoolExecutor()

    @classmethod
    def submit(cls, command: list[str], trial_id: int, gpu_mode: bool, silent: bool = True) -> tuple[str, int]:
        if cls.log is None:
            raise MpiError('cls.lock is None')
        if cls.lock is None:
            raise MpiError('cls.lock is None')
        if not cls.lock.acquire(timeout=file_mpi_lock_timeout):
            raise MpiError('Failed to lock due to timeout.', cls.error_file_path)
        try:
            ret = cls._submit(command, trial_id, gpu_mode, silent)
        except Exception as e:
            cls.log.write(f'submit(): catch Exception as {e}', stdout=True)
            cls.lock.release()
            raise e
        retf = ret
        cls.lock.release()
        return retf

    @classmethod
    def _submit(cls, command: list[str], trial_id: int, gpu_mode: bool, silent: bool) -> tuple[str, int]:
        if cls.log is None:
            raise MpiError('cls.log is None')
        cls.tag += 1
        tag = cls.tag
        if cls.executor is not None:
            cls.executor.submit(cls._func, command, gpu_mode, tag, str(cls.rank_log_path))
        comm = COMM_WORLD
        list_ = comm.recv(tag=tag)
        cls.log.write(f'submit start: recv: tag={tag} trial_id={trial_id} list={list_}', cls.tag)
        rank = list_[0]
        processor = list_[1]
        cls.trial_id_list.append([trial_id, tag])
        if gpu_mode:
            gpu_num = cls._get_gpu_num(processor, tag)
            comm.send(gpu_num, rank)
            cls.log.write(f'send: rank={rank} tag={tag} gpu_num={gpu_num}')
            cls.log.write(f'info: gpu_list={cls.gpu_list}')
        return (processor, tag)

    @classmethod
    def _func(cls, command: list[str], gpu_mode: int, tag: int, rank_log_path_str: str) -> None:
        try:
            cls._func_sub(command, gpu_mode, tag, rank_log_path_str)
        except Exception as e:
            if cls.log is None:
                raise MpiError(f'cls.log is None. after catch Exception as {e}')
            cls.log.write(f'_func(): catch Exception as {e}', stdout=True)
            raise e

    @classmethod
    def _func_sub(cls, command: list[str], gpu_mode: int, tag: int, rank_log_path_str: str) -> None:
        comm = COMM_WORLD
        rank = comm.Get_rank()
        processor = Get_processor_name()
        if cls.log is None:
            cls.rank_log_path = Path(rank_log_path_str)
            cls.log = MpiLog(rank, processor, cls.rank_log_path)

        cls.log.write(f'_func_sub(): tag={tag} command={command}', tag)
        comm.send([rank, processor], 0, tag=tag)
        if gpu_mode:
            gpu_num = comm.recv(source=0)
            cls.log.write(f'start: recv: gpu_num={gpu_num}')
            cls._set_gpu(str(gpu_num))
        proc = Popen(
            command,
            stdout=PIPE,
            stderr=STDOUT
        )
        while True:
            if proc.stdout is None:
                cls.log.write(f'_func_sub(): end: tag={tag} error.', stdout=True)
                s = f'{cls.func_end_id} error.'
                comm.send(s, 0, tag=tag)
                break

            line = proc.stdout.readline().decode().strip()

            if line:
                comm.send(line, 0, tag=tag)
                cls.log.write(f'_func_sub(): debug: line={line}')  # for debug

            if not line and proc.poll() is not None:
                cls.log.write(f'_func_sub(): end: tag={tag} process finished.')
                o, e = proc.communicate()
                s = ''
                if o:
                    s += o.decode().strip()
                if e:
                    s += e.decode().strip()
                comm.send(s, 0, tag=tag)
                s = f'{cls.func_end_id} process finished.'
                comm.send(s, 0, tag=tag)
                break

    @classmethod
    def _get_gpu_num(cls, processor: str, tag: int) -> int:
        if cls.log is None:
            raise MpiError('cls.log is None')
        n = 0
        try:
            if processor not in cls.gpu_list:
                cls.gpu_list[processor] = [[n, tag]]
            else:
                n = cls._get_empty_num(cls.gpu_list[processor])
                cls.gpu_list[processor].append([n, tag])
        except Exception as e:
            cls.log.write(f'_get_gpu_num(): catch Exception as {e}', stdout=True)
            raise e
        return n

    @classmethod
    def rm_trial_id(cls, tag: int) -> None:
        if cls.log is None:
            raise MpiError('cls.log is None')
        if cls.lock is None:
            raise MpiError('cls.lock is None')
        if not cls.lock.acquire(timeout=file_mpi_lock_timeout):
            raise MpiError('Failed to lock due to timeout.', cls.error_file_path)
        try:
            flag = False
            for a in cls.trial_id_list:
                if a[1] == tag:
                    cls.trial_id_list.remove(a)
                    flag = True
                    break
            if not flag:
                raise MpiError(f'There is no element tag({tag}) in cls.trial_id_list({cls.trial_id_list}).')
        except Exception as e:
            cls.log.write(f'rm_trial_id(): catch Exception as {e}', stdout=True)
            cls.lock.release()
            raise e
        cls.lock.release()

    @classmethod
    def get_trial_id_list(cls) -> list[int]:
        if cls.log is None:
            raise MpiError('cls.log is None')
        if cls.lock is None:
            raise MpiError('cls.lock is None')
        if not cls.lock.acquire(timeout=file_mpi_lock_timeout):
            raise MpiError('Failed to lock due to timeout.', cls.error_file_path)
        ret = []
        try:
            for a in cls.trial_id_list:
                ret.append(a[0])
        except Exception as e:
            cls.log.write(f'get_trial_id_list(): catch Exception as {e}', stdout=True)
            cls.lock.release()
            raise e
        cls.lock.release()
        return ret
        
    @classmethod
    def rm_gpu_num(cls, processor: str, tag: int) -> None:
        if cls.log is None:
            raise MpiError('cls.log is None')
        if cls.lock is None:
            raise MpiError('cls.lock is None')
        if not cls.lock.acquire(timeout=file_mpi_lock_timeout):
            raise MpiError('Failed to lock due to timeout.', cls.error_file_path)
        try:
            cls._rm_gpu_num(processor, tag)
        except Exception as e:
            cls.log.write(f'rm_gpu_num(): catch Exception as {e}', stdout=True)
            cls.lock.release()
            raise e
        cls.lock.release()

    @classmethod
    def _rm_gpu_num(cls, processor: str, tag: int) -> None:
        a = cls.gpu_list[processor]
        for i in range(len(a)):
            if a[i][1] == tag:
                del a[i]
                break

    @classmethod
    def _get_empty_num(cls, list_: list[list[int]]) -> int:
        if len(list_) == cls.gpu_max:
            raise MpiError("Not enough gpu. An error that shouldn't have happened.", cls.error_file_path)
        list_.sort()
        i = 0
        for a in list_:
            if a[0] != i:
                return i
            i += 1
        if i < cls.gpu_max:
            return i
        raise MpiError("Not empty gpu. An error that shouldn't have happened.", cls.error_file_path)

    @classmethod
    def _set_gpu(cls, gpu: str) -> None:
        environ['CUDA_VISIBLE_DEVICES'] = gpu

    @classmethod
    def run_bat(cls, config: MpiConfig, logger: Logger) -> None:
        mpi_env = config.mpi_enviroment.get()
        if mpi_env.lower() != resource_type_abci:
            logger.error(f'{mpi_env}, the mpi_enviroment, is not supported.')
            return
        if config.mpi_bat_make_file.get():
            cls._make_bat_file(config, logger)
        cls._run_bat_file(config, logger)

    @classmethod
    def _run_bat_file(cls, config: MpiConfig, logger: Logger) -> None:
        qsub_file = config.mpi_bat_file.get()
        abci_group = config.abci_group.get()[1:-1]
        qsub_cmd = f'qsub -g {abci_group} {qsub_file}'
        proc = run(qsub_cmd.split(' '), stdout=PIPE, stderr=STDOUT)
        res = proc.stdout.decode('utf8')
        logger.info(f'{res} < {qsub_cmd}')

    @classmethod
    def _make_bat_file(cls, config: MpiConfig, logger: Logger) -> None:
        rt_type = config.mpi_bat_rt_type.get()
        rt_num = config.mpi_bat_rt_num.get()
        h_rt = config.mpi_bat_h_rt.get()
        num_node = config.num_node.get()
        # mpi_npernode = config.mpi_npernode.get()
        root_path = Path(config.mpi_bat_root_dir.get())
        venv_dir = str(root_path / config.mpi_bat_venv_dir.get())
        aiaccel_dir = str(root_path / config.mpi_bat_aiaccel_dir.get())
        config_path = root_path / config.mpi_bat_config_dir.get()
        config_dir = str(config_path)
        qsub_file_path = config_path / config.mpi_bat_file.get()
        hostfile = str(config_path / config.mpi_hostfile.get())
        qsub_str = f'''#!/bin/bash

#$ -l rt_{rt_type}={rt_num}
#$ -l h_rt={h_rt}
#$ -j y
#$ -cwd

source /etc/profile.d/modules.sh
module load python/3.11
module load hpcx-mt/2.12
source {venv_dir}/bin/activate
export PYTHONPATH={aiaccel_dir}/:$PYTHONPATH

cd {config_dir}

python -m aiaccel.experimental.mpi.cli.start --config config.yaml --make_hostfile

mpiexec -n {num_node+1} -hostfile {hostfile} \
python -m mpi4py.futures -m aiaccel.experimental.mpi.cli.start --config config.yaml --clean --from_mpi_bat

deactivate
'''
        # 'mpiexec -n {num_node+1} -npernode {mpi_npernode}'
        qsub_file_path = Path(os_path.expanduser(str(qsub_file_path)))
        qsub_file_path.write_text(qsub_str)

    @classmethod
    def make_hostfile(cls, config: MpiConfig, logger: Logger) -> None:
        try:
            cls._make_hostfile(config, logger)
        except Exception as e:
            print(e)
            raise e

    @classmethod
    def _make_hostfile(cls, config: MpiConfig, logger: Logger) -> None:
        root_path = Path(config.mpi_bat_root_dir.get())
        hostfile_path = root_path / config.mpi_bat_config_dir.get() / config.mpi_hostfile.get()
        rt_num = config.mpi_bat_rt_num.get()
        mpi_npernode = config.mpi_npernode.get()
        mpi_gpu_mode = config.mpi_gpu_mode.get()
        istr = Path(environ['SGE_JOB_HOSTLIST']).read_text()
        hostlist = istr.split(linesep)
        ostr = ''
        for i in range(rt_num):
            ostr += f'{hostlist[i]} slots='
            if i == 0 and mpi_gpu_mode:
                ostr += f'{mpi_npernode+1}'
            else:
                ostr += f'{mpi_npernode}'
            ostr += linesep
        hostfile_path = Path(os_path.expanduser(str(hostfile_path)))
        hostfile_path.write_text(ostr)


class MpiOutputHandler(Thread):
    def __init__(
        self,
        parent: AbstractScheduler,
        gpu_mode: bool,
        processor: str,
        tag: int,
        module_name: str,
        trial_id: int,
        storage: Storage | None = None
    ) -> None:
        super(MpiOutputHandler, self).__init__()
        self._parent = parent
        self._gpu_mode = gpu_mode
        self._processor = processor
        self._tag = tag
        self._module_name = module_name
        self._trial_id = trial_id
        self._storage: Storage | None = storage
        self._abort = False

    def abort(self) -> None:
        self._abort = True

    def run(self) -> None:
        self._parent.logger.debug(f'{self._module_name}(tag={self._tag}) process started.')
        while True:
            s = COMM_WORLD.recv(tag=self._tag)
            if s.find(Mpi.func_end_id) == 0:
                self._parent.logger.debug(s)
                break
            print(s, flush=True)

            if self._abort:
                break
        Mpi.rm_trial_id(self._tag)
        if self._gpu_mode:
            Mpi.rm_gpu_num(self._processor, self._tag)
