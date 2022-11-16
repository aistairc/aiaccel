from typing import Union

from sqlalchemy.exc import SQLAlchemyError

from aiaccel.storage.abstruct import Abstract
from aiaccel.storage.model import JobStateTable
from aiaccel.util.retry import retry


class JobState(Abstract):
    def __init__(self, file_name) -> None:
        super().__init__(file_name)

    @retry(_MAX_NUM=60, _DELAY=1.0)
    def set_any_trial_jobstate(self, trial_id: int, state: str) -> None:
        """Set the specified jobstate to the specified trial.

        Args:
            trial_id (int): Any trial id
            state (str): Any jobstate

        Returns:
            None
        """
        with self.create_session() as session:
            try:
                data = (
                    session.query(JobStateTable)
                    .filter(JobStateTable.trial_id == trial_id)
                    .with_for_update(read=True)
                    .one_or_none()
                )
                if data is None:
                    new_row = JobStateTable(
                        trial_id=trial_id,
                        state=state
                    )
                    session.add(new_row)
                else:
                    data.state = state
                session.commit()
            except SQLAlchemyError as e:
                session.rollback()
                raise e

    @retry(_MAX_NUM=60, _DELAY=1.0)
    def set_any_trial_jobstates(self, states: list) -> None:
        """Set the specified jobstate to the specified trial.

        Args:
            trial_id (int): Any trial id
            state (str): Any jobstate

        Returns:
            None
        """
        with self.create_session() as session:
            try:
                data = [
                    JobStateTable(
                        trial_id=state['trial_id'],
                        state=state['jobstate']
                    ) for state in states
                ]
                session.bulk_save_objects(data)
                session.commit()
            except SQLAlchemyError as e:
                session.rollback()
                raise e

    @retry(_MAX_NUM=60, _DELAY=1.0)
    def get_any_trial_jobstate(self, trial_id: int) -> Union[None, str]:
        """Get the job status of any trial.

        Args:
            trial_id (int): Any trial id

        Returns:
            str: Some kind of jobstate
        """
        with self.create_session() as session:
            data = (
                session.query(JobStateTable)
                .filter(JobStateTable.trial_id == trial_id)
                .with_for_update(read=True)
                .one_or_none()
            )

        if data is None:
            return None
        return data.state

    @retry(_MAX_NUM=60, _DELAY=1.0)
    def get_all_trial_jobstate(self) -> list:
        with self.create_session() as session:
            data = (
                session.query(JobStateTable)
                .with_for_update(read=True)
                .all()
            )

        if len(data) == 0:
            return [{'trial_id': None, 'jobstate': None}]

        jobstates = [
            {'trial_id': d.trial_id, 'jobstate': d.state}
            for d in data
        ]
        return jobstates

    @retry(_MAX_NUM=60, _DELAY=1.0)
    def delete_any_trial_jobstate(self, trial_id: int) -> None:
        """
        Returns:
            None
        """
        with self.create_session() as session:
            try:
                session.query(JobStateTable).filter(JobStateTable.trial_id == trial_id).delete()
                session.commit()
            except SQLAlchemyError as e:
                session.rollback()
                raise e

    @retry(_MAX_NUM=60, _DELAY=1.0)
    def is_failure(self, trial_id: int) -> bool:
        """Whether the jobstate of the specified trial is Failuer or not.

        Args:
            trial_id (int): Any trial id

        Return:
            bool
        """
        with self.create_session() as session:
            data = (
                session.query(JobStateTable)
                .filter(JobStateTable.trial_id == trial_id)
                .with_for_update(read=True)
                .one_or_none()
            )

        if data is None:
            return False
        return "failure" in data.state.lower()
