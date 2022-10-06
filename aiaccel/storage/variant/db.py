from typing import Any
from sqlalchemy.exc import SQLAlchemyError
from aiaccel.storage.abstruct.db import Abstract
from aiaccel.storage.model.db import VariableTable
from aiaccel.util.retry import retry
from typing import Any
import copy


class Variable(Abstract):
    def __init__(self, file_name) -> None:
        super().__init__(file_name)

    @retry(_MAX_NUM=60, _DELAY=1.0)
    def set_any_trial_variable(
        self,
        trial_id: int,
        process_name: str,
        label: str,
        value: Any
    ):
        with self.create_session() as session:
            try:
                data = (
                    session.query(VariableTable)
                    .filter(VariableTable.trial_id == trial_id)
                    .filter(VariableTable.process_name == process_name)
                    .filter(VariableTable.label == label)
                    .with_for_update(read=True)
                    .one_or_none()
                )
                if data is not None:
                    self.delete_any_trial_variable(
                        trial_id=trial_id,
                        process_name=process_name,
                        label=label
                    )

                new_row = VariableTable(
                    trial_id=trial_id,
                    process_name=process_name,
                    label=label,
                    value=value
                )
                session.add(new_row)
                session.commit()

            except SQLAlchemyError as e:
                session.rollback()
                raise e

    @retry(_MAX_NUM=60, _DELAY=1.0)
    def get_any_trial_variable(self, trial_id: int, process_name: str, label: str):
        with self.create_session() as session:
            data = (
                session.query(VariableTable)
                .filter(VariableTable.trial_id == trial_id)
                .filter(VariableTable.process_name == process_name)
                .filter(VariableTable.label == label)
                .with_for_update(read=True)
                .one_or_none()
            )

            if data is None:
                return None
            return data.value

    @retry(_MAX_NUM=60, _DELAY=1.0)
    def all_delete(self) -> None:
        """Clear table

        Returns:
            None
        """
        with self.create_session() as session:
            try:
                session.query(VariableTable).with_for_update(read=True).delete()
                session.commit()
            except SQLAlchemyError as e:
                session.rollback()
                raise e

    @retry(_MAX_NUM=60, _DELAY=1.0)
    def delete_any_trial_variable(self, trial_id: int, process_name: str, label: str) -> None:
        with self.create_session() as session:
            try:
                (session.query(VariableTable)
                    .filter(VariableTable.trial_id == trial_id)
                    .filter(VariableTable.process_name == process_name)
                    .filter(VariableTable.label == label)
                    .delete())
                session.commit()
            except SQLAlchemyError as e:
                session.rollback()
                raise e


class Value(Variable):
    def __init__(self, file_name, label: str):
        super().__init__(file_name)
        self.process_name = None
        self.label = label
        self.value = None

    def set_process_name(self, process_name: str):
        self.process_name = process_name

    def set(self, trial_id: int, value: Any):
        self.set_any_trial_variable(
            trial_id=trial_id,
            process_name=self.process_name,
            label=self.label,
            value=value
        )

    def get(self, trial_id: int):
        self.value = self.get_any_trial_variable(
            trial_id=trial_id,
            process_name=self.process_name,
            label=self.label
        )
        return copy.deepcopy(self.value)

    def delete(self, trial_id: int):
        self.delete_any_trial_variable(
            trial_id=trial_id,
            process_name=self.process_name,
            label=self.label
        )


class Serializer:
    def __init__(self, file_name: str):
        self.file_name = file_name
        self.d = {}

    def register(self, process_name: str, labels: list):
        for i in range(len(labels)):
            if labels[i] not in self.d.keys():
                self.d[labels[i]] = Value(self.file_name, labels[i])
                self.d[labels[i]].set_process_name(process_name)

    def delete_any_trial_variable(self, trial_id):
        for key in self.d.keys():
            self.d[key].delete(trial_id)

if __name__ == "__main__":
    from pathlib import Path
    import numpy as np
    file_name = Path('/tmp/storage.db')
    var = Serializer(file_name=file_name)
    var.register(process_name='optimizer', labels=['hoge', 'foo', 'bar'])

    var.d['hoge'].set(trial_id=1, value=1.1)
    var.d['hoge'].set(trial_id=2, value=2.2)
    var.d['hoge'].set(trial_id=3, value=3.3)

    assert var.d['hoge'].get(trial_id=1) == 1.1
    assert var.d['hoge'].get(trial_id=2) == 2.2
    assert var.d['hoge'].get(trial_id=3) == 3.3

    var.d['hoge'].set(trial_id=3, value=4.4)
    assert var.d['hoge'].get(trial_id=3) == 4.4

    var.register(process_name='optimizer', labels=['random_state'])
    rs = np.random.get_state()
    var.d['hoge'].set(trial_id=1, value=rs)
    var.d['hoge'].set(trial_id=1, value=rs)


    d = var.d['hoge'].get(trial_id=1)
    for i in range(len(d)):
        if type(d[i]) is np.ndarray:
            assert all(d[i] == rs[i])
        else:
            assert d[i] == rs[i]
