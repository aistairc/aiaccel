import numpy as np
from pathlib import Path
from typing import List, Optional

from aiaccel.scheduler.algorithm.abstract_scheduling_algorithm import \
    AbstractSchedulingAlgorithm


class RandomSampling(AbstractSchedulingAlgorithm):
    def select_hp(
        self,
        hp_ready: List[Path],
        num: Optional[int] = 1,
        rng: np.random.RandomState = None
    ) -> List[Path]:
        """

        Args:
            hp_ready (List[Path]): A list of path of ready hyper parameters.
            num (Optional[int]): A number to select hyper parameters.
            rng (np.random.RandomState): A random generator.

        Returns:
            List[Path]:
        """

        arr = []
        samples = min(num, len(hp_ready))

        for _ in range(0, samples):
            arr.append(rng.choice(hp_ready, 1)[0])

        return arr


class SequentialSampling(AbstractSchedulingAlgorithm):
    def select_hp(
        self,
        hp_ready: List[Path],               # A list of path of ready hyper parameters.
        num: Optional[int] = 1,             # A number to select hyper parameters.
        rng: np.random.RandomState = None   # A random generator.
    ) -> List[Path]:
        """

        Args:
            hp_ready (List[Path]): A list of path of ready hyper parameters.
            num (Optional[int]): A number to select hyper parameters.
            rng (np.random.RandomState): A random generator.

        Returns:
            List[Path]:
        """

        arr = hp_ready
        return arr
