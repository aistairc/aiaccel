from __future__ import annotations
import numpy as np
from pathlib import Path

from omegaconf.dictconfig import DictConfig


class AbstractSchedulingAlgorithm(object):
    """An abstract class for scheduling algorithms used for Scheduler.

    Args:
        config (ConfileWrapper): A configuration object.

    Attributes:
        config (ConfileWrapper): A configuration object.
    """

    def __init__(self, config: DictConfig) -> None:
        self.config = config

    def select_hp(
        self,
        hp_ready: list[Path],
        num: int = 1,
        rng: np.random.RandomState | None = None
    ) -> None:
        """Select multiple hyper parameters.

        Args:
            hp_ready (list[Path]): A list of path of ready hyper parameters.
            num (int, optional): A number to select hyper parameters Defaults
            to 1.
            rng (np.random.RandomState | None, optional): A reference to a
            random generator. Defaults to None.

        Raises:
            NotImplementedError: Causes when the inherited class does not
                implement.
        """
        raise NotImplementedError
