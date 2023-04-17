from __future__ import annotations

from pathlib import Path
from typing import Any

from sqlalchemy.exc import SQLAlchemyError

from aiaccel.storage.abstract import Abstract
from aiaccel.storage.model import HpTable
from aiaccel.util import retry


class Hp(Abstract):
    def __init__(self, file_name: Path) -> None:
        super().__init__(file_name)

    @retry(_MAX_NUM=60, _DELAY=1.0)
    def set_any_trial_param(self, trial_id: int, param_name: str, param_value: Any, param_type: str) -> None:
        """Set the specified parameter information for an any trial.

        Args:
            trial_id (int)    : Any trial id
            param_name (str)  : Hyperparameter name.
            param_value (Any) : Hyperparameter value
            param_type (str)  : Hyperparameter data type

        Returns:
            TrialTable | None
        """
        with self.create_session() as session:
            try:
                p = HpTable(trial_id=trial_id, param_name=param_name, param_value=param_value, param_type=param_type)
                session.add(p)
                session.commit()
            except SQLAlchemyError as e:
                session.rollback()
                raise e

    @retry(_MAX_NUM=60, _DELAY=1.0)
    def set_any_trial_params(self, trial_id: int, params: list[dict[str, Any]]) -> None:
        with self.create_session() as session:
            try:
                hps = [
                    HpTable(
                        trial_id=trial_id, param_name=d["parameter_name"], param_value=d["value"], param_type=d["type"]
                    )
                    for d in params
                ]
                session.bulk_save_objects(hps)
                session.commit()
            except SQLAlchemyError as e:
                session.rollback()
                raise e

    @retry(_MAX_NUM=60, _DELAY=1.0)
    def get_any_trial_params(self, trial_id: Any) -> list[HpTable] | None:
        """Obtain the set parameter information for any given trial.

        Args:
            trial_id(int): Any trial id.

        Returns:
            list[HpTable] | None:
        """
        with self.create_session() as session:
            hp = session.query(HpTable).filter(HpTable.trial_id == trial_id).with_for_update(read=True).all()

        if len(hp) == 0:
            return None
        return hp

    def get_any_trial_params_dict(self, trial_id: int) -> dict[str, int | float | str] | None:
        """Obtain the set parameter information for any given trial.

        Args:
            trial_id(int): Any trial id.

        Returns:
            list[HpTable] | None:
        """
        params = self.get_any_trial_params(trial_id)
        if params is None:
            return None

        return {p.param_name: p.param_value for p in params}

    @retry(_MAX_NUM=60, _DELAY=1.0)
    def get_num_params(self) -> int:
        """Get number of generated parameters.

        Args:
            None

        Returns:
            int: Number of generated parameters.
        """
        with self.create_session() as session:
            hp = session.query(HpTable).with_for_update(read=True).all()

        return len(hp)

    @retry(_MAX_NUM=60, _DELAY=1.0)
    def all_delete(self) -> None:
        """Clear table

        Returns:
            None
        """
        with self.create_session() as session:
            try:
                session.query(HpTable).with_for_update(read=True).delete()
                session.commit()
            except SQLAlchemyError as e:
                session.rollback()
                raise e

    @retry(_MAX_NUM=60, _DELAY=1.0)
    def delete_any_trial_params(self, trial_id: int) -> None:
        """
        Returns:
            None
        """
        with self.create_session() as session:
            try:
                session.query(HpTable).filter(HpTable.trial_id == trial_id).delete()
                session.commit()
            except SQLAlchemyError as e:
                session.rollback()
                raise e
