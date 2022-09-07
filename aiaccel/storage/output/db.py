from typing import Union
from sqlalchemy.exc import SQLAlchemyError
from aiaccel.storage.abstruct.db import Abstract
from aiaccel.storage.model.db import AbciOutputTable
from aiaccel.util.retry import retry


class AbciOutput(Abstract):
    def __init__(self, file_name) -> None:
        super().__init__(file_name)

    @retry(_MAX_NUM=60, _DELAY=1.0)
    def set_any_trial_abci_output(self, trial_id: int, message: str) -> None:
        """Sets the ABCI output message for the specified trial.

        Args:
            trial_id (int): Any trial id
            message (str)

        Returns:
            None
        """
        with self.create_session() as session:
            try:
                data = (
                    session.query(AbciOutputTable)
                    .filter(AbciOutputTable.trial_id == trial_id)
                    .with_for_update(read=True)
                    .one_or_none()
                )

                if data is None:
                    new_row = AbciOutputTable(trial_id=trial_id, message=message)
                    session.add(new_row)
                else:
                    data.message = message
                session.commit()

            except SQLAlchemyError as e:
                session.rollback()
                raise e

    @retry(_MAX_NUM=60, _DELAY=1.0)
    def get_any_trial_abci_output(self, trial_id: int) -> Union[None, str]:
        """Obtain ABCI output messages for a given trial.

        Args:
            trial_id (int): Any trial id

        Returns:
            Union[None, str]
        """
        with self.create_session() as session:
            data = (
                session.query(AbciOutputTable)
                .filter(AbciOutputTable.trial_id == trial_id)
                .with_for_update(read=True)
                .one_or_none()
            )

        if data is None:
            return None
        return data.message
