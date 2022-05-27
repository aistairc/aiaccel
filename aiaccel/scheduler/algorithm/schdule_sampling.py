from aiaccel.scheduler.algorithm.abstract_scheduling_algorithm import \
    AbstractSchedulingAlgorithm
from pathlib import Path
from typing import List, Optional
import random


class RamsomSampling(AbstractSchedulingAlgorithm):
    def select_hp(
        self,
        hp_ready: List[Path],   # A list of path of ready hyper parameters.
        num: Optional[int] = 1  # A number to select hyper parameters.
    ) -> List[Path]:

        arr = []
        samples = min(num, len(hp_ready))

        for i in range(0, samples):
            arr.append(random.choice(hp_ready))

        return arr


class SequencialSampling(AbstractSchedulingAlgorithm):
    def select_hp(
        self,
        hp_ready: List[Path],   # A list of path of ready hyper parameters.
        num: Optional[int] = 1  # A number to select hyper parameters.
    ) -> List[Path]:

        arr = hp_ready
        return arr
