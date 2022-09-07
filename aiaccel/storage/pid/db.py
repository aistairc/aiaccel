from sqlalchemy.exc import SQLAlchemyError
from typing import Union
from aiaccel.storage.abstruct.db import Abstract
from aiaccel.storage.model.db import PidTable
from aiaccel.util.retry import retry


class Pid(Abstract):
    def __init__(self, file_name) -> None:
        super().__init__(file_name)

    @retry(_MAX_NUM=60, _DELAY=1.0)
    def set_any_process_pid(self, process_name: str, pid: int) -> None:
        """Set the specified pid to the specified trial.

        Args:
            process_name (str): master, optimizer, scheduler
            pid (int): Any pid

        Returns:
            None
        """
        with self.create_session() as session:
            try:
                data = (
                    session.query(PidTable)
                    .filter(PidTable.process_name == process_name)
                    .with_for_update(read=True)
                    .one_or_none()
                )
                if data is None:
                    new_row = PidTable(process_name=process_name, pid=pid)
                    session.add(new_row)
                else:
                    data.pid = pid
                session.commit()
            except SQLAlchemyError as e:
                session.rollback()
                raise e

    @retry(_MAX_NUM=60, _DELAY=1.0)
    def get_any_process_pid(self, process_name: str) -> Union[None, int]:
        """Obtains the PID of the specified process.

        Args:
            process_name (str): master, optimizer, scheduler

        Returns:
            pid (int): Any pid
        """
        with self.create_session() as session:
            data = (
                session.query(PidTable)
                .filter(PidTable.process_name == process_name)
                .with_for_update(read=True)
                .one_or_none()
            )

        if data is None:
            return None
        return data.pid

    @retry(_MAX_NUM=60, _DELAY=1.0)
    def all_delete(self) -> None:
        """Clear table

        Returns:
            None
        """
        with self.create_session() as session:
            try:
                session.query(PidTable).with_for_update(read=True).delete()
                session.commit()
            except SQLAlchemyError as e:
                session.rollback()
                raise e

    @retry(_MAX_NUM=60, _DELAY=1.0)
    def delete_any_process_pid(self, process_name: str) -> None:
        """
        Returns:
            None
        """
        with self.create_session() as session:
            try:
                session.query(PidTable).filter(PidTable.process_name == process_name).delete()
                session.commit()
            except SQLAlchemyError as e:
                session.rollback()
                raise e
