from pathlib import Path
from typing import List, Optional

from aiaccel.config import Config


class AbstractSchedulingAlgorithm(object):
    """An abstract class for scheduling algorithms used for Scheduler.

    Attributes:
        config (ConfileWrapper): A configuration object.
    """

    def __init__(self, config: Config) -> None:
        """Initial method for AbstractSchedulingAlgorithm.

        Args:
            config (ConfileWrapper): A configuration object.
        """
        self.config = config

    def select_hp(self, hp_ready: List[Path], num: Optional[int] = 1) -> None:
        """Select multiple hyper parameters.

        Args:
            hp_ready (List[Path]): A list of path of ready hyper parameters.
            num (Optional[int]): A number to select hyper parameters.

        Returns:
            None

        Raises:
            NotImplementedError: Causes when the inherited class does not
                implement.
        """
        raise NotImplementedError
