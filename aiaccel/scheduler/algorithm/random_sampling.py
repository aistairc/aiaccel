import random
from pathlib import Path
from typing import List, Optional

from aiaccel.scheduler.algorithm.abstract_scheduling_algorithm import \
    AbstractSchedulingAlgorithm


class RandomSamplingSchedulingAlgorithm(AbstractSchedulingAlgorithm):
    """An algorithm to select hyper parameters.

    """

    def select_hp(self, hp_ready: List[Path], num: Optional[int] = 1) ->\
            List[Path]:
        """Select multiple hyper parameters.

        Args:
            hp_ready (List[Path]): A list of path of ready hyper parameters.
            num (Optional[int]): A number to select hyper parameters.

        Returns:
            List[Path]: Selected hyper parameters.
        """
        ret = []
        samples = min(num, len(hp_ready))

        for i in range(0, samples):
            ret.append(random.choice(hp_ready))

        return ret
