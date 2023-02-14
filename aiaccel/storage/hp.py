from __future__ import annotations

from typing import Any
from pathlib import Path

from sqlalchemy.exc import SQLAlchemyError

from aiaccel.storage import Abstract
from aiaccel.storage import HpTable
from aiaccel.util import retry


class Hp(Abstract):
    def __init__(self, file_name: Path) -> None:
        super().__init__(file_name)

    @retry(_MAX_NUM=60, _DELAY=1.0)
    def set_any_trial_param(
        self,
        trial_id: int,
        param_name: str,
        param_value: Any,
        param_type: str
    ) -> None:
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
                p = HpTable(
                    trial_id=trial_id,
                    param_name=param_name,
                    param_value=param_value,
                    param_type=param_type
                )
                session.add(p)
                session.commit()
            except SQLAlchemyError as e:
                session.rollback()
                raise e

    @retry(_MAX_NUM=60, _DELAY=1.0)
    def set_any_trial_params(self, trial_id: int, params: list) -> None:
        with self.create_session() as session:
            try:
                hps = [
                    HpTable(
                        trial_id=trial_id,
                        param_name=d['parameter_name'],
                        param_value=d['value'],
                        param_type=d['type']
                    ) for d in params
                ]
                session.bulk_save_objects(hps)
                session.commit()
            except SQLAlchemyError as e:
                session.rollback()
                raise e

    @retry(_MAX_NUM=60, _DELAY=1.0)
    def get_any_trial_params(self, trial_id: int) -> list[HpTable] | None:
        """ Obtain the set parameter information for any given trial.

        Args:
            trial_id(int): Any trial id.

        Returns:
            list[HpTable] | None:
        """
        with self.create_session() as session:
            hp = (
                session.query(HpTable)
                .filter(HpTable.trial_id == trial_id)
                .with_for_update(read=True)
                .all()
            )

        if len(hp) == 0:
            return None
        return hp

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
