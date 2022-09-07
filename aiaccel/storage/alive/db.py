from typing import Union
from aiaccel.storage.abstruct.db import Abstract
from aiaccel.storage.model.db import AliveTable
from sqlalchemy.exc import SQLAlchemyError
from aiaccel.util.retry import retry


class Alive(Abstract):
    def __init__(self, file_name) -> None:
        super().__init__(file_name)

    def init_alive(self) -> None:
        """Initialize alive state.

        Returns:
            None
        """
        self.set_any_process_state('master', 0)
        self.set_any_process_state('optimizer', 0)
        self.set_any_process_state('scheduler', 0)

    def get_state(self) -> dict:
        """Get all alive state.

        Returns:
            dict: {'master':int[0|1], 'optimizer':int[0|1], 'scheduler':int[0|1]}
        """
        alives = {
            'master': self.get_any_process_state('master'),
            'optimizer': self.get_any_process_state('optimizer'),
            'scheduler': self.get_any_process_state('scheduler')
        }
        return alives

    @retry(_MAX_NUM=60, _DELAY=1.0)
    def set_any_process_state(self, process_name: str, alive_state: int) -> None:
        """Set the specified process state.

        Returns:
            None
        """
        with self.create_session() as session:
            try:
                data = (
                    session.query(AliveTable)
                    .filter(AliveTable.process_name == process_name)
                    .with_for_update(read=True)
                    .one_or_none()
                )
                if data is None:
                    new_row = AliveTable(process_name=process_name, alive_state=alive_state)
                    session.add(new_row)
                else:
                    data.alive_state = alive_state
                session.commit()

            except SQLAlchemyError as e:
                session.rollback()
                raise e

    @retry(_MAX_NUM=60, _DELAY=1.0)
    def get_any_process_state(self, process_name: str) -> Union[None, int]:
        """Get the state of a any process.

        Returns:
            None
        """
        with self.create_session() as session:
            data = (
                session.query(AliveTable)
                .filter(AliveTable.process_name == process_name)
                .with_for_update(read=True)
                .one_or_none()
            )

        if data is None:
            assert False
        return data.alive_state

    def stop_any_process(self, process_name: str) -> None:
        """Stop state of any process..

        Returns:
            None
        """
        process_name = process_name.lower()
        self.set_any_process_state(process_name, 0)

    def check_alive(self, process_name: str) -> bool:
        """Check if any process is alive.

        Returns:
            bool
        """
        alive = self.get_state()
        return True if alive[process_name] == 1 else False
