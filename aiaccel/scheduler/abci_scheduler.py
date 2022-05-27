import subprocess
from aiaccel.abci.abci_qstat import parse_qstat
from aiaccel.scheduler.abstract_scheduler import AbstractScheduler


class AbciScheduler(AbstractScheduler):
    """A scheduler class running on ABCI environment.
    """

    def get_stats(self) -> None:
        """Get a current status and update.

        Returns:
            None
        """
        super().get_stats()

        commands = 'qstat -xml'
        p = subprocess.Popen(commands, stdout=subprocess.PIPE, shell=True)
        stats = p.communicate()[0]
        stats = stats.decode('utf-8')

        if len(stats) < 1:
            return

        self.stats = parse_qstat(self.config, stats)

        for stat in self.stats:
            self.logger.info('stat job-ID: {}, name: {}, state: {}'.format(
                stat['job-ID'], stat['name'], stat['state']))
