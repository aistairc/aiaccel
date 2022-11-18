import numpy as np
from pathlib import Path
from typing import List, Optional

from aiaccel.scheduler.algorithm.abstract_scheduling_algorithm import \
    AbstractSchedulingAlgorithm


class RandomSamplingSchedulingAlgorithm(AbstractSchedulingAlgorithm):
    """An algorithm to select hyper parameters.

    """

    def select_hp(
        self,
        hp_ready:
        List[Path],
        num: Optional[int] = 1,
        rng: np.random.RandomState = None
    ) -> List[Path]:
        """Select multiple hyper parameters.

        Args:
            hp_ready (List[Path]): A list of path of ready hyper parameters.
            num (Optional[int]): A number to select hyper parameters.
            rng (np.random.RandomState): A reference to a random generator.

        Returns:
            List[Path]: Selected hyper parameters.
        """
        ret = []
        samples = min(num, len(hp_ready))

        for _ in range(0, samples):
            ret.append(rng.choice(hp_ready, 1)[0])

        return ret
