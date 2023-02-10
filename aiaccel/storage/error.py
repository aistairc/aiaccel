from __future__ import annotations

from pathlib import PosixPath
from sqlalchemy.exc import SQLAlchemyError

from aiaccel.storage.abstract import Abstract
from aiaccel.storage.model import ErrorTable
from aiaccel.util.retry import retry


class Error(Abstract):
    def __init__(self, file_name: PosixPath) -> None:
        super().__init__(file_name)

    @retry(_MAX_NUM=60, _DELAY=1.0)
    def set_any_trial_error(self, trial_id: int, error_message: str) -> None:
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
                    session.query(ErrorTable)
                    .filter(ErrorTable.trial_id == trial_id)
                    .with_for_update(read=True)
                    .one_or_none()
                )

                if data is None:
                    new_row = ErrorTable(trial_id=trial_id, error=error_message)
                    session.add(new_row)
                    session.commit()
                else:
                    data.error = error_message

            except SQLAlchemyError as e:
                session.rollback()
                raise e

    @retry(_MAX_NUM=60, _DELAY=1.0)
    def get_any_trial_error(self, trial_id: int) -> str | None:
        """Get error messages for any trial.

        Args:
            trial_id (int): Any trial id

        Returns:
            str | None:
        """
        with self.create_session() as session:
            data = (
                session.query(ErrorTable)
                .filter(ErrorTable.trial_id == trial_id)
                .with_for_update(read=True)
                .one_or_none()
            )

        if data is None:
            return None
        return data.error

    @retry(_MAX_NUM=60, _DELAY=1.0)
    def get_error_trial_id(self) -> list:
        """Obtain a list of trial ids in which an error occurred.

        Returns:
            trial_ids(list): trial id list
        """
        with self.create_session() as session:
            data = (
                session.query(ErrorTable)
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
                session.query(ErrorTable).with_for_update(read=True).delete()
                session.commit()
            except SQLAlchemyError as e:
                session.rollback()
                raise e

    @retry(_MAX_NUM=60, _DELAY=1.0)
    def delete_any_trial_error(self, trial_id: int) -> None:
        """
        Returns:
            None
        """
        with self.create_session() as session:
            try:
                session.query(ErrorTable).filter(ErrorTable.trial_id == trial_id).delete()
                session.commit()
            except SQLAlchemyError as e:
                session.rollback()
                raise e
