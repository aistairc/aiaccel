from __future__ import annotations
from typing import Any
from pathlib import Path

from sqlalchemy.exc import SQLAlchemyError

from aiaccel.storage.abstract import Abstract
from aiaccel.storage.model import ResultTable
from aiaccel.util.retry import retry


class Result(Abstract):
    def __init__(self, file_name: Path) -> None:
        super().__init__(file_name)

    @retry(_MAX_NUM=60, _DELAY=1.0)
    def set_any_trial_objective(self, trial_id: int, objective: Any) -> None:
        """Set any trial result value.

        Args:
            trial_id (int): Any trial id
            objective(Any): ready, running, finished

        Returns:
            None
        """
        with self.create_session() as session:
            try:
                data = (
                    session.query(ResultTable)
                    .filter(ResultTable.trial_id == trial_id)
                    .with_for_update(read=True)
                    .one_or_none()
                )
                if data is None:
                    new_row = ResultTable(
                        trial_id=trial_id,
                        data_type=str(type(objective)),
                        objective=objective
                    )
                    session.add(new_row)
                else:
                    data.objective = objective
                session.commit()
            except SQLAlchemyError as e:
                session.rollback()
                raise e

    @retry(_MAX_NUM=60, _DELAY=1.0)
    def get_any_trial_objective(self, trial_id: int) -> int | float | None:
        """Obtain the results of an arbitrary trial.

        Args:
            trial_id (int): Any trial id

        Returns:
            int | float | None:
        """
        with self.create_session() as session:
            data = (
                session.query(ResultTable)
                .filter(ResultTable.trial_id == trial_id)
                .with_for_update(read=True)
                .one_or_none()
            )

        if data is None:
            return None

        return data.objective

    @retry(_MAX_NUM=60, _DELAY=1.0)
    def get_all_result(self) -> list:
        """Get all results

        Returns:
            list
        """
        with self.create_session() as session:
            data = (
                session.query(ResultTable)
                .with_for_update(read=True)
            )

        # return [d.objective for d in data]
        return data

    def get_objectives(self) -> list:
        """Get all results in list.

        Returns:
           list: result values
        """
        data = self.get_all_result()

        return [d.objective for d in data]

    def get_bests(self, goal: str) -> list:
        """Obtains the sorted result.

        Returns:
            list: result values
        """
        goal = goal.lower()
        objectives = self.get_objectives()
        best_values = []

        if goal == "maximize":
            best_value = float("-inf")
            for objective in objectives:
                if best_value < objective:
                    best_value = objective
                best_values.append(best_value)
        elif goal == "minimize":
            best_value = float("inf")
            for objective in objectives:
                if best_value > objective:
                    best_value = objective
                best_values.append(best_value)
        else:
            return []

        return best_values

    @retry(_MAX_NUM=60, _DELAY=1.0)
    def get_result_trial_id_list(self) -> list | None:
        """Obtains the sorted result.

        Returns:
            list | None: result values
        """
        with self.create_session() as session:
            data = (
                session.query(ResultTable)
                .with_for_update(read=True)
                .all()
            )

        if data is None or len(data) == 0:
            return None

        return [d.trial_id for d in data]

    @retry(_MAX_NUM=60, _DELAY=1.0)
    def all_delete(self) -> None:
        """Clear table

        Returns:
            None
        """
        with self.create_session() as session:
            try:
                session.query(ResultTable).with_for_update(read=True).delete()
                session.commit()
            except SQLAlchemyError as e:
                session.rollback()
                raise e

    @retry(_MAX_NUM=60, _DELAY=1.0)
    def delete_any_trial_objective(self, trial_id: int) -> None:
        """_summary_

        Args:
            trial_id (int): _description_

        Raises:
            e: _description_
        """
        with self.create_session() as session:
            try:
                session.query(ResultTable).filter(ResultTable.trial_id == trial_id).delete()
                session.commit()
            except SQLAlchemyError as e:
                session.rollback()
                raise e
