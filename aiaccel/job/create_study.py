from __future__ import annotations

from typing import Sequence

import optuna
from optuna import logging
from optuna.study import Study
from optuna.study._study_direction import StudyDirection

_logger = logging.get_logger(__name__)


def create_study(
    *,
    storage: str | optuna.storages.BaseStorage | None = None,
    sampler: optuna.samplers.BaseSampler | None = None,
    pruner: optuna.pruners.BasePruner | None = None,
    study_name: str | None = None,
    direction: str | StudyDirection | None = None,
    load_if_exists: bool = False,
    directions: Sequence[str | StudyDirection] | None = None,
) -> Study:
    return optuna.create_study(
        storage=storage,
        sampler=sampler,
        pruner=pruner,
        study_name=study_name,
        direction=direction,
        load_if_exists=load_if_exists,
        directions=directions,
    )
