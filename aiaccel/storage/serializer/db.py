from typing import Any
from sqlalchemy.exc import SQLAlchemyError
from aiaccel.storage.abstruct.db import Abstract
from aiaccel.storage.model.db import SerializeTable
from aiaccel.util.retry import retry


class Serializer(Abstract):
    def __init__(self, file_name) -> None:
        super().__init__(file_name)

    @retry(_MAX_NUM=60, _DELAY=1.0)
    def set_any_trial_serialize(
        self,
        trial_id: int,
        optimization_variable,
        process_name: str,
        native_random_state: tuple,
        numpy_random_state: tuple
    ) -> None:
        """Sets serialization data for a given trial.

        Args:
            trial_id (int): Any trial id
            optimization_variable: serialized data
            process_name (str): master, optimizer, scheduler

        Returns:
            None
        """
        session = self.session()
        try:
            data = (
                session.query(SerializeTable)
                .filter(SerializeTable.trial_id == trial_id)
                .filter(SerializeTable.process_name == process_name)
                .with_for_update(read=True)
                .one_or_none()
            )
            if data is None:
                new_row = SerializeTable(
                    trial_id=trial_id,
                    process_name=process_name,
                    optimization_variable=optimization_variable,
                    native_random_state=native_random_state,
                    numpy_random_state=numpy_random_state
                )
                session.add(new_row)
            else:
                session.expunge_all()
                self.engine.dispose()
                return

        except SQLAlchemyError as e:
            session.rollback()
            raise e

        finally:
            session.commit()
            session.expunge_all()
            self.engine.dispose()

    @retry(_MAX_NUM=60, _DELAY=1.0)
    def get_any_trial_serialize(self, trial_id: int, process_name: str) -> Any:
        """Obtain serialized data for a given trial.

        Args:
            trial_id (int): Any trial id
            process_name (str): master, optimizer, scheduler

        Returns:
            serialized data
        """
        session = self.session()
        data = (
            session.query(SerializeTable)
            .filter(SerializeTable.trial_id == trial_id)
            .filter(SerializeTable.process_name == process_name)
            .with_for_update(read=True)
            .one_or_none()
        )
        session.expunge_all()
        self.engine.dispose()

        if data is None:
            return None

        return (
            data.optimization_variable,
            data.native_random_state,
            data.numpy_random_state
        )

    def is_exists_any_trial(self, trial_id: int):
        process_names = [
            'master',
            'optimizer',
            'scheduler'
        ]
        for process_name in process_names:
            if (
                self.get_any_trial_serialize(
                    trial_id=trial_id,
                    process_name=process_name
                )
            ) is None:
                return False
        return True
