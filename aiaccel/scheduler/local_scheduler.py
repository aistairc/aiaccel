from aiaccel.scheduler.abstract_scheduler import AbstractScheduler,\
    parse_hashname
from aiaccel.util.process import ps2joblist


class LocalScheduler(AbstractScheduler):
    """A scheduler class running on a local computer.

    """

    def get_stats(self) -> None:
        """Get a current status and update.

        Returns:
            None
        """
        super().get_stats()

        res = ps2joblist()
        command = self.config.job_command.get()
        self.stats = []
        job_hashnames = [job['hashname'] for job in self.jobs]

        for r in res:
            if command in r['name']:
                hashname = parse_hashname(r['name'])

                if hashname in job_hashnames:
                    self.stats.append(r)
                else:
                    self.logger.warning('**** Unknown process: {}'.format(r))
