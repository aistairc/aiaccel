from __future__ import annotations

from typing import Any

from aiaccel.master.verifier import AbstractVerifier


class MultiObjectiveVerifier(AbstractVerifier):
    '''Verifier for multi-objevtive optimization.

    Note:
        Verification is not available for multi-objective optimization.
    '''

    def __init__(self, options: dict[str, Any]) -> None:
        super().__init__(options)

    def verify(self) -> None:
        """Not implemented.
        """
        pass
