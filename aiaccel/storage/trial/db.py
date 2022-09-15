from typing import Union
from sqlalchemy.exc import SQLAlchemyError
from aiaccel.storage.abstruct.db import Abstract
from aiaccel.storage.model.db import TrialTable
from aiaccel.util.retry import retry


class Trial(Abstract):
    def __init__(self, file_name) -> None:
        super().__init__(file_name)

    @retry(_MAX_NUM=60, _DELAY=1.0)
    def get_any_trial_state(self, trial_id: int) -> Union[None, str]:
        """Get any trials state.

        Args:
            trial_id (int): Any trial id

        Returns:
            trials state(str): ready, running, finished
        """
        with self.create_session() as session:
            trials = (
                session.query(TrialTable)
                .filter(TrialTable.trial_id == trial_id)
                .with_for_update(read=True)
                .one_or_none()
            )

        if trials is None:
            return None
        return trials.state

    @retry(_MAX_NUM=60, _DELAY=1.0)
    def get_any_state_list(self, state: str) -> Union[None, list]:
        """Get any trials numbers.

        Args:
            trials state(str): ready, running, finished

        Returns:
            trial ids(list[int])
        """
        with self.create_session() as session:
            trials = (
                session.query(TrialTable)
                .filter(TrialTable.state == state)
                .with_for_update(read=True)
            )

        if trials is None:
            return None
        return [d.trial_id for d in trials]

    @retry(_MAX_NUM=60, _DELAY=1.0)
    def set_any_trial_state(self, trial_id: int, state: str) -> None:
        """Set any trials numbers.

        Args:
            trial_id (int): Any trial id
            trials state(str): ready, running, finished

        Returns:
            None
        """
        with self.create_session() as session:
            try:
                trials = (
                    session.query(TrialTable)
                    .filter(TrialTable.trial_id == trial_id)
                    .with_for_update(read=True)
                    .one_or_none()
                )
                if trials is None:
                    new_row = TrialTable(
                        trial_id=trial_id,
                        state=state
                    )
                    session.add(new_row)
                else:
                    trials.state = state
                session.commit()
            except SQLAlchemyError as e:
                session.rollback()
                raise e

    @retry(_MAX_NUM=60, _DELAY=1.0)
    def all_delete(self) -> None:
        """Clear table

        Returns:
            None
        """
        with self.create_session() as session:
            try:
                session.query(TrialTable).with_for_update(read=True).delete()
                session.commit()
            except SQLAlchemyError as e:
                session.rollback()
                raise e

    @retry(_MAX_NUM=60, _DELAY=1.0)
    def delete_any_trial_state(self, trial_id: int) -> None:
        with self.create_session() as session:
            try:
                session.query(TrialTable).filter(TrialTable.trial_id == trial_id).delete()
                session.commit()
            except SQLAlchemyError as e:
                session.rollback()
                raise e

    @retry(_MAX_NUM=60, _DELAY=1.0)
    def get_ready(self) -> list:
        """Get the trial id whose status is ready.

        Returns:
            trial ids(list[int])
        """
        with self.create_session() as session:
            trials = (
                session.query(TrialTable)
                .filter(TrialTable.state == 'ready')
                .with_for_update(read=True)
                .all()
            )

        return [trial.trial_id for trial in trials]

    @retry(_MAX_NUM=60, _DELAY=1.0)
    def get_running(self) -> list:
        """Get the trial id whose status is running.

        Returns:
            trial ids(list[int])
        """
        with self.create_session() as session:
            trials = (
                session.query(TrialTable)
                .filter(TrialTable.state == 'running')
                .with_for_update(read=True)
                .all()
            )

        return [trial.trial_id for trial in trials]

    @retry(_MAX_NUM=60, _DELAY=1.0)
    def get_finished(self) -> list:
        """Get the trial id whose status is finished.

        Returns:
            trial ids(list[int])
        """
        with self.create_session() as session:
            trials = (
                session.query(TrialTable)
                .filter(TrialTable.state == 'finished')
                .with_for_update(read=True)
                .all()
            )

        return [trial.trial_id for trial in trials]

    @retry(_MAX_NUM=60, _DELAY=1.0)
    def get_all_trial_id(self) -> Union[None, list]:
        """
        Returns:
            trial ids(list[int])
        """
        with self.create_session() as session:
            trials = (
                session.query(TrialTable)
                .with_for_update(read=True)
            )

        if trials is None:
            return None
        return [trial.trial_id for trial in trials]
