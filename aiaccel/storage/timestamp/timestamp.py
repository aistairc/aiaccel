from sqlalchemy.exc import SQLAlchemyError
from aiaccel.storage.abstruct.abstruct import Abstract
from aiaccel.storage.model.model import TimestampTable
from aiaccel.util.retry import retry


class TimeStamp(Abstract):
    def __init__(self, file_name) -> None:
        super().__init__(file_name)

    @retry(_MAX_NUM=60, _DELAY=1.0)
    def set_any_trial_start_time(self, trial_id: int, start_time: str) -> None:
        """Set the specified start time for the specified trial.

        Args:
            trial_id (int) : Any trial id
            start_time(str): "MM/DD/YYYY hh:mm:ss"

        Returns:
            None
        """
        with self.create_session() as session:
            try:
                data = (
                    session.query(TimestampTable)
                    .filter(TimestampTable.trial_id == trial_id)
                    .with_for_update(read=True)
                    .one_or_none()
                )
                if data is None:
                    new_row = TimestampTable(
                        trial_id=trial_id,
                        start_time=start_time,
                        end_time=""
                    )
                    session.add(new_row)
                else:
                    data.start_time = start_time
                session.commit()
            except SQLAlchemyError as e:
                session.rollback()
                raise e

    @retry(_MAX_NUM=60, _DELAY=1.0)
    def set_any_trial_end_time(self, trial_id: int, end_time: str) -> None:
        """Set the specified end time for the specified trial.

        Args:
            trial_id(int): Any trial id
            end_time(str): "MM/DD/YYYY hh:mm:ss"

        Returns:
            None
        """
        with self.create_session() as session:
            try:
                data = (
                    session.query(TimestampTable)
                    .filter(TimestampTable.trial_id == trial_id)
                    .with_for_update(read=True)
                    .one_or_none()
                )
                if data is None:
                    assert False
                data.end_time = end_time
                session.commit()
            except SQLAlchemyError as e:
                session.rollback()
                raise e

    @retry(_MAX_NUM=60, _DELAY=1.0)
    def get_any_trial_start_time(self, trial_id: int) -> str:
        """Obtains the start time of the specified trial.

        Args:
            trial_id(int): Any trial id

        Returns:
            start_time(str): "MM/DD/YYYY hh:mm:ss"
        """
        with self.create_session() as session:
            data = (
                session.query(TimestampTable)
                .filter(TimestampTable.trial_id == trial_id)
                .with_for_update(read=True)
                .one_or_none()
            )

        if data is None:
            return None
        if data.start_time == '':
            return None
        return data.start_time

    @retry(_MAX_NUM=60, _DELAY=1.0)
    def get_any_trial_end_time(self, trial_id: int) -> str:
        """Obtains the end time of the specified trial.

        Args:
            trial_id(int): Any trial id

        Returns:
            end_time(str): "MM/DD/YYYY hh:mm:ss"
        """
        with self.create_session() as session:
            data = (
                session.query(TimestampTable)
                .filter(TimestampTable.trial_id == trial_id)
                .with_for_update(read=True)
                .one_or_none()
            )

        if data is None:
            return None
        if data.end_time == '':
            return None
        return data.end_time

    @retry(_MAX_NUM=60, _DELAY=1.0)
    def all_delete(self) -> None:
        """Clear table

        Returns:
            None
        """
        with self.create_session() as session:
            try:
                session.query(TimestampTable).with_for_update(read=True).delete()
                session.commit()
            except SQLAlchemyError as e:
                session.rollback()
                raise e

    @retry(_MAX_NUM=60, _DELAY=1.0)
    def delete_any_trial_timestamp(self, trial_id) -> None:
        with self.create_session() as session:
            try:
                session.query(TimestampTable).filter(TimestampTable.trial_id == trial_id).delete()
                session.commit()
            except SQLAlchemyError as e:
                session.rollback()
                raise e
