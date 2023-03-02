from __future__ import annotations

from pathlib import Path
from sqlalchemy.exc import SQLAlchemyError

from aiaccel.storage.abstract import Abstract
from aiaccel.storage.model import ReturnCodeTable
from aiaccel.util.retry import retry


class ReturnCode(Abstract):
    def __init__(self, file_name: Path) -> None:
        super().__init__(file_name)

    @retry(_MAX_NUM=60, _DELAY=1.0)
    def set_any_trial_return_code(self, trial_id: int, return_code: int) -> None:
        """Set any error message for any trial.

        Args:
            trial_id (int): Any trial id
            error_message(str): Any error message

        Returns:
            None
        """
        with self.create_session() as session:
            try:
                data = (
                    session.query(ReturnCodeTable)
                    .filter(ReturnCodeTable.trial_id == trial_id)
                    .with_for_update(read=True)
                    .one_or_none()
                )

                if data is None:
                    new_row = ReturnCodeTable(trial_id=trial_id, returncode=return_code)
                    session.add(new_row)
                    session.commit()
                else:
                    data.returncode = return_code

            except SQLAlchemyError as e:
                session.rollback()
                raise e

    @retry(_MAX_NUM=60, _DELAY=1.0)
    def get_any_trial_return_code(self, trial_id: int) -> str | None:
        """Get error messages for any trial.

        Args:
            trial_id (int): Any trial id

        Returns:
            str | None:
        """
        with self.create_session() as session:
            data = (
                session.query(ReturnCodeTable)
                .filter(ReturnCodeTable.trial_id == trial_id)
                .with_for_update(read=True)
                .one_or_none()
            )

        if data is None:
            return None
        return data.returncode

    @retry(_MAX_NUM=60, _DELAY=1.0)
    def get_return_code_trial_id(self) -> list:
        """Obtain a list of trial ids in which an error occurred.

        Returns:
            trial_ids(list): trial id list
        """
        with self.create_session() as session:
            data = (
                session.query(ReturnCodeTable)
                .with_for_update(read=True)
                .all()
            )

        if data is None or len(data) == 0:
            return []

        return [d.trial_id for d in data]

    @retry(_MAX_NUM=60, _DELAY=1.0)
    def all_delete(self) -> None:
        """Clear table

        Returns:
            None
        """
        with self.create_session() as session:
            try:
                session.query(ReturnCodeTable).with_for_update(read=True).delete()
                session.commit()
            except SQLAlchemyError as e:
                session.rollback()
                raise e

    @retry(_MAX_NUM=60, _DELAY=1.0)
    def delete_any_trial_return_code(self, trial_id: int) -> None:
        """
        Returns:
            None
        """
        with self.create_session() as session:
            try:
                session.query(ReturnCodeTable).filter(ReturnCodeTable.trial_id == trial_id).delete()
                session.commit()
            except SQLAlchemyError as e:
                session.rollback()
                raise e
