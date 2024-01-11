from __future__ import annotations

from pathlib import Path

from sqlalchemy.exc import SQLAlchemyError

from aiaccel.storage import Abstract, ReturnCodeTable
from aiaccel.util import retry


class ReturnCode(Abstract):
    def __init__(self, file_name: Path) -> None:
        super().__init__(file_name)

    @retry(_MAX_NUM=60, _DELAY=1.0)
    def set_any_trial_returncode(self, trial_id: int, returncode: int) -> None:
        """Set any returncode for any trial.

        Args:
            trial_id (int): Any trial id
            returncode(int): Any returncode

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
                    new_row = ReturnCodeTable(trial_id=trial_id, returncode=returncode)
                    session.add(new_row)
                else:
                    data.returncode = returncode
                session.commit()
            except SQLAlchemyError as e:
                session.rollback()
                raise e

    @retry(_MAX_NUM=60, _DELAY=1.0)
    def get_any_trial_returncode(self, trial_id: int) -> list[int | float | str] | None:
        """Obtain the results of an arbitrary trial.

        Args:
            trial_id (int): Any trial id

        Returns:
            int | float | None: Any returncode
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
    def get_all_trial_returncode(self) -> list[int | float | str] | None:
        """Obtain the results of all trials.

        Args:
            trial_id (int): Any trial id

        Returns:
            int | float | None:
        """
        with self.create_session() as session:
            data = session.query(ReturnCodeTable).with_for_update(read=True).all()

        if data is None:
            return None

        return [d.returncode for d in data]
