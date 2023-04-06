from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy.exc import SQLAlchemyError

from aiaccel.storage import Abstract, ResultTable
from aiaccel.util import retry


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
    def get_any_trial_objective(self, trial_id: int) -> list[int | float | str] | None:
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
    def get_all_result(self) -> dict[int, list[Any]]:
        """Get all results

        Returns:
            dict[int, list[Any]]: trial_id and result values
        """
        with self.create_session() as session:
            data = (
                session.query(ResultTable)
                .with_for_update(read=True)
            )

        return {d.trial_id: d.objective for d in data}
        # return data

    def get_objectives(self) -> list[Any]:
        """Get all results in list.

        Returns:
           list: result values
        """
        data = self.get_all_result()

        return [data[trial_id] for trial_id in data.keys()]

    def get_bests(self, goals: list[str]) -> list[Any]:
        """Obtains the sorted result.

        Returns:
            list: result values
        """
        objectives = np.array(self.get_objectives())
        bests = np.zeros((len(goals), len(objectives[0])))

        for i in range(len(goals)):
            if goals[i].lower() == "maximize":
                bests[i, :] = np.max(objectives[:, i], axis=0)
            elif goals[i].lower() == "minimize":
                bests[i, :] = np.min(objectives[:, i], axis=0)
            else:
                raise ValueError("Invalid goal value.")

        return [row[0] for row in bests.tolist()]

    @retry(_MAX_NUM=60, _DELAY=1.0)
    def get_result_trial_id_list(self) -> list[Any] | None:
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

    def get_any_trial_objective_and_best_value(self, trial_id: int, goals: list[str]) -> tuple[list[Any], list[Any]]:
        """Obtain the results of an arbitrary trial.

        Args:
            trial_id (int): Any trial id
            column_idx (int): Any column index of objectives and best_values

        Returns:
            int | float | None:
        """
        objectives: list[Any] = self.get_any_trial_objective(trial_id)
        if objectives is None:
            return None, None

        best_values = self.get_bests(goals)

        return objectives, best_values

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
