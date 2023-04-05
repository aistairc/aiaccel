from __future__ import annotations

import subprocess

from omegaconf.dictconfig import DictConfig

from aiaccel.common import dict_runner
from aiaccel.abci.qstat import parse_qstat
from aiaccel.master.abstract_master import AbstractMaster
from aiaccel.util.filesystem import get_dict_files
from aiaccel.util.retry import retry


class AbciMaster(AbstractMaster):
    """A master class running on ABCI environment.

    Args:
        options (dict[str, str | int | bool]): A dictionary containing
            command line options.

    Attributes:
        runner_files (list[Path]): A list of path of runner files.
        stats (list[dict]): A result string of 'qstat' command.
    """

    def __init__(self, config: DictConfig) -> None:
        """Initial method of AbciMaster.

        Args:
            config (DictConfig): A configuration object.
        """
        super().__init__(config)
        self.runner_files = []
        self.stats = []

    def pre_process(self) -> None:
        """Pre-procedure before executing processes.

        Returns:
            None
        """

        self.runner_files = get_dict_files(
            self.ws / dict_runner,
            "run_*.sh"
        )

    @retry(_MAX_NUM=60, _DELAY=1.0)
    def get_stats(self) -> None:
        """Get a current status and update.

        Returns:
            None
        """
        commands = 'qstat -xml'
        p = subprocess.Popen(commands, stdout=subprocess.PIPE, shell=True)

        try:
            stdout_data, _ = p.communicate(timeout=1)
        except subprocess.TimeoutExpired:
            p.kill()
            stdout_data, _ = p.communicate()

        stats = stdout_data.decode('utf-8')

        # Write qstat result
        lines = ''

        for line in stats:
            lines += line

        if len(stats) < 1:
            return

        self.stats = parse_qstat(stats)
