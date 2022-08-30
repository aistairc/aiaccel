from aiaccel.abci.qstat import parse_qstat
from aiaccel.master.abstract import AbstractMaster
from aiaccel.util.filesystem import get_dict_files
import aiaccel
import subprocess
from aiaccel.util.filesystem import retry


class AbciMaster(AbstractMaster):
    """A master class running on ABCI environment.

    Attributes:
        runner_files (List[Path]): A list of path of runner files.
        stats (Anystr): A result string of 'qstat' command.
    """

    def __init__(self, options: dict) -> None:
        """Initial method of AbciMaster.

        Args:
            config (str): A file name of a configuration.
        """
        super().__init__(options)
        self.runner_files = []
        self.stats = []

    def pre_process(self) -> None:
        """Pre-procedure before executing processes.

        Returns:
            None
        """
        super().pre_process()

        #
        # In job_thread.after_runner, use 'run_{}.sh'.
        # match this one too.
        #
        # self.runner_files = get_dict_files(
        #     self.ws / aiaccel.dict_runner,
        #     # self.config.runner_search_pattern().get
        # )
        self.runner_files = get_dict_files(
            self.ws / aiaccel.dict_runner,
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
            stats, errs = p.communicate(timeout=1)
        except subprocess.TimeoutExpired:
            p.kill()
            stats, errs = p.communicate()

        stats = stats.decode('utf-8')

        # Write qstat result
        lines = ''

        for line in stats:
            lines += line

        # TODO: Confirm whether this line is needed?
        if stats is None:
            return

        if len(stats) < 1:
            return

        self.stats = parse_qstat(self.config, stats)

    def inner_loop_post_process(self) -> bool:
        """Called after exiting a main loop process. This process is repeated
            every main loop.

        Returns:
            bool: The process succeeds or not. The main loop exits if failed.
        """
        self.get_stats()
        self.print_dict_state()

        return True

    def loop_post_process(self) -> None:
        """Called after exiting a main loop process.

        Returns:
            None
        """
        return None

    def check_error(self):
        """ Check to confirm if an error has occurred.

        Args:
            None

        Returns:
            True: no error | False: with error.
        """
        return True
