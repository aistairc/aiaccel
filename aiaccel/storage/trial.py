from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from sqlalchemy.exc import SQLAlchemyError

from aiaccel.storage import Abstract, TrialTable
from aiaccel.util import retry


class Trial(Abstract):
    def __init__(self, file_name: Path) -> None:
        super().__init__(file_name)

    @retry(_MAX_NUM=60, _DELAY=1.0)
    def get_any_trial_state(self, trial_id: int) -> Literal["ready", "running", "finished"] | None:
        """Get any trials state.

        Args:
            trial_id (int): Any trial id

        Returns:
            Literal['ready', 'running', 'finished'] | None: Trial state.
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
    def get_any_state_list(self, state: Literal["ready", "running", "finished"]) -> list[int] | None:
        """Get any trials numbers.

        Args:
            state (Literal['ready', 'running', 'finished'): Trial state.

        Returns:
            list[int] | None: A list of trial ids. None if no trials match the
            specified state.
        """
        with self.create_session() as session:
            trials = session.query(TrialTable).filter(TrialTable.state == state).with_for_update(read=True).all()

        if trials is None or len(trials) == 0:
            return None

        return [d.trial_id for d in trials]

    @retry(_MAX_NUM=60, _DELAY=1.0)
    def set_any_trial_state(self, trial_id: int, state: Literal["ready", "running", "finished"]) -> None:
        """Set any trials numbers.

        Args:
            trial_id (int): Any trial id
            state (Literal['ready', 'running', 'finished']): Trial state.

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
                    new_row = TrialTable(trial_id=trial_id, state=state)
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
    def get_ready(self) -> list[int]:
        """Get the trial id whose status is ready.

        Returns:
            trial ids(list[int])
        """
        with self.create_session() as session:
            trials = session.query(TrialTable).filter(TrialTable.state == "ready").with_for_update(read=True).all()

        return [trial.trial_id for trial in trials]

    @retry(_MAX_NUM=60, _DELAY=1.0)
    def get_running(self) -> list[int]:
        """Get the trial id whose status is running.

        Returns:
            trial ids(list[int])
        """
        with self.create_session() as session:
            trials = session.query(TrialTable).filter(TrialTable.state == "running").with_for_update(read=True).all()

        return [trial.trial_id for trial in trials]

    @retry(_MAX_NUM=60, _DELAY=1.0)
    def get_finished(self) -> list[Any]:
        """Get the trial id whose status is finished.

        Returns:
            trial ids(list[int])
        """
        with self.create_session() as session:
            trials = session.query(TrialTable).filter(TrialTable.state == "finished").with_for_update(read=True).all()

        return [trial.trial_id for trial in trials]

    @retry(_MAX_NUM=60, _DELAY=1.0)
    def get_num_running_ready_finished(self) -> tuple[int, int, int]:
        """Get num_of_ready, num_of_running, num_of_finished.

        Returns:
            tuple(int, int, int)
        """
        with self.create_session() as session:
            num_of_ready = (
                session.query(TrialTable).filter(TrialTable.state == "ready").with_for_update(read=True).count()
            )
            num_of_running = (
                session.query(TrialTable).filter(TrialTable.state == "running").with_for_update(read=True).count()
            )
            num_of_finished = (
                session.query(TrialTable).filter(TrialTable.state == "finished").with_for_update(read=True).count()
            )
        return (num_of_ready, num_of_running, num_of_finished)

    @retry(_MAX_NUM=60, _DELAY=1.0)
    def get_all_trial_id(self) -> list[int] | None:
        """
        Returns:
            list[int] | None: A list of trial ids.
        """
        with self.create_session() as session:
            trials = session.query(TrialTable).with_for_update(read=True).all()

        if trials is None or len(trials) == 0:
            return None

        return [trial.trial_id for trial in trials]
