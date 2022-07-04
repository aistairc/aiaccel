from typing import Union
from sqlalchemy.exc import SQLAlchemyError
from aiaccel.storage.abstruct.db import Abstract
from aiaccel.storage.model.db import ErrorTable
from aiaccel.util.retry import retry


class Error(Abstract):
    def __init__(self, file_name) -> None:
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
        session = self.session()
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
            else:
                data.error = error_message

        except SQLAlchemyError as e:
            session.rollback()
            raise e

        finally:
            session.commit()
            session.expunge_all()
            self.engine.dispose()

    @retry(_MAX_NUM=60, _DELAY=1.0)
    def get_any_trial_error(self, trial_id: int) -> Union[None, str]:
        """Get error messages for any trial.

        Args:
            trial_id (int): Any trial id

        Returns:
            str | None
        """
        session = self.session()
        data = (
            session.query(ErrorTable)
            .filter(ErrorTable.trial_id == trial_id)
            .with_for_update(read=True)
            .one_or_none()
        )
        session.expunge_all()
        self.engine.dispose()

        if data is None:
            return None
        return data.error

    @retry(_MAX_NUM=60, _DELAY=1.0)
    def get_error_trial_id(self) -> list:
        """Obtain a list of trial ids in which an error occurred.

        Returns:
            trial_ids(list): trial id list
        """
        session = self.session()
        data = (
            session.query(ErrorTable)
            .with_for_update(read=True)
            .all()
        )
        session.expunge_all()
        self.engine.dispose()

        if data is None:
            return []

        return [d.trial_id for d in data]

    @retry(_MAX_NUM=60, _DELAY=1.0)
    def all_delete(self) -> None:
        """Clear table

        Returns:
            None
        """
        session = self.session()
        session.query(ErrorTable).with_for_update(read=True).delete()
        session.commit()
        session.expunge_all()
        self.engine.dispose()
