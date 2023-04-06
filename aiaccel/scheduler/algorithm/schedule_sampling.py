from __future__ import annotations

from pathlib import Path
from typing import Any

from aiaccel.scheduler.algorithm import AbstractSchedulingAlgorithm


class RandomSampling(AbstractSchedulingAlgorithm):
    def select_hp(
        self,
        hp_ready: list[Path],
        num: int = 1,
        rng: Any = None
    ) -> list[Path]:
        """Select multiple hyper parameters.

        Args:
            hp_ready (list[Path]): A list of path of ready hyper parameters.
            num (int, optional): A number to select hyper parameters Defaults
                to 1.
            rng (np.random.RandomState | None, optional): A reference to a
                random generator. Defaults to None.

        Returns:
            list[Path]: Selected hyper parameters.
        """

        arr = []
        samples = min(num, len(hp_ready))

        for _ in range(0, samples):
            arr.append(rng.choice(hp_ready, 1)[0])

        return arr


class SequentialSampling(AbstractSchedulingAlgorithm):
    def select_hp(
        self,
        hp_ready: list[Path],               # A list of path of ready hyper parameters.
        num: int = 1,             # A number to select hyper parameters.
        rng: Any = None   # A random generator.
    ) -> list[Path]:
        """Select multiple hyper parameters.

        Args:
            hp_ready (list[Path]): A list of path of ready hyper parameters.
            num (int, optional): A number to select hyper parameters Defaults
                to 1.
            rng (np.random.RandomState | None, optional): A reference to a
                random generator. Defaults to None.

        Returns:
            list[Path]: Selected hyper parameters.
        """

        arr = hp_ready
        return arr
