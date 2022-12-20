import numpy as np
from pathlib import Path
from typing import List, Optional

from omegaconf.dictconfig import DictConfig


class AbstractSchedulingAlgorithm(object):
    """An abstract class for scheduling algorithms used for Scheduler.

    Attributes:
        config (ConfileWrapper): A configuration object.
    """

    def __init__(self, config: DictConfig) -> None:
        """Initial method for AbstractSchedulingAlgorithm.

        Args:
            config (ConfileWrapper): A configuration object.
        """
        self.config = config

    def select_hp(
        self,
        hp_ready: List[Path],
        num: Optional[int] = 1,
        rng: np.random.RandomState = None
    ) -> None:
        """Select multiple hyper parameters.

        Args:
            hp_ready (List[Path]): A list of path of ready hyper parameters.
            num (Optional[int]): A number to select hyper parameters.
            rng (np.random.RandomState): A reference to a random generator.

        Returns:
            None

        Raises:
            NotImplementedError: Causes when the inherited class does not
                implement.
        """
        raise NotImplementedError
