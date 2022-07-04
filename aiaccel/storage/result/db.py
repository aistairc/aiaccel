from typing import Any
from typing import Union
from sqlalchemy.exc import SQLAlchemyError
from aiaccel.storage.abstruct.db import Abstract
from aiaccel.storage.model.db import ResultTable
from aiaccel.util.retry import retry


class Result(Abstract):
    def __init__(self, file_name) -> None:
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
        session = self.session()
        try:
            data = ResultTable(trial_id=trial_id, objective=objective)
            session.add(data)

        except SQLAlchemyError as e:
            session.rollback()
            raise e

        finally:
            session.commit()
            session.expunge_all()
            self.engine.dispose()

    @retry(_MAX_NUM=60, _DELAY=1.0)
    def get_any_trial_objective(self, trial_id) -> Union[None, int, float]:
        """Obtain the results of an arbitrary trial.

        Args:
            trial_id (int): Any trial id

        Returns:
            Any
        """
        session = self.session()
        data = (
            session.query(ResultTable)
            .filter(ResultTable.trial_id == trial_id)
            .with_for_update(read=True)
            .one_or_none()
        )
        session.expunge_all()
        self.engine.dispose()

        if data is None:
            return None
        return data.objective

    @retry(_MAX_NUM=60, _DELAY=1.0)
    def get_all_result(self) -> list:
        """Get all results

        Returns:
            Any
        """
        session = self.session()
        data = (
            session.query(ResultTable)
            .with_for_update(read=True)
        )
        session.expunge_all()
        self.engine.dispose()
        # return [d.objective for d in data]
        return data

    def get_objectives(self) -> list:
        """Get all results in list.

        Returns:
            objectives(list): result values
        """
        data = self.get_all_result()
        return [d.objective for d in data]

    def get_bests(self, goal: str) -> list:
        """Obtains the sorted result.

        Returns:
            objectives(list): result values
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
    def get_result_trial_id_list(self) -> Union[None, list]:
        """Obtains the sorted result.

        Returns:
            objectives(list): result values
        """
        session = self.session()
        data = session.query(ResultTable).with_for_update(read=True)
        session.expunge_all()
        self.engine.dispose()

        if data is None:
            return None
        return [d.trial_id for d in data]

    @retry(_MAX_NUM=60, _DELAY=1.0)
    def all_delete(self) -> None:
        """Clear table

        Returns:
            None
        """
        session = self.session()
        session.query(ResultTable).with_for_update(read=True).delete()
        session.commit()
        session.expunge_all()
        self.engine.dispose()
