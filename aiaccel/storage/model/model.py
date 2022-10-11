from typing import Any
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.schema import Column
from sqlalchemy.types import Integer
from sqlalchemy.types import BigInteger
from sqlalchemy.types import Float
from sqlalchemy.types import Text
from sqlalchemy.types import String
from sqlalchemy.types import PickleType

Base: Any = declarative_base()


# models
class TrialTable(Base):
    __tablename__ = 'Trial'
    trial_id = Column(Integer, primary_key=True, nullable=False)
    state = Column(Text, nullable=True)


class ErrorTable(Base):
    __tablename__ = 'errors'
    trial_id = Column(Integer, primary_key=True, nullable=False)
    error = Column(Text, nullable=True)


class TimestampTable(Base):
    __tablename__ = 'timestamp'
    trial_id = Column(Integer, primary_key=True, nullable=False)
    start_time = Column(String, nullable=True)
    end_time = Column(String, nullable=True)


class HpTable(Base):
    __tablename__ = 'trial_params'
    param_id = Column(Integer, primary_key=True)
    trial_id = Column(Integer, nullable=False)
    param_name = Column(String(length=512), nullable=True)
    param_value = Column(String(length=512), nullable=True)
    param_type = Column(String(length=512), nullable=True)


class ResultTable(Base):
    __tablename__ = 'result'
    trial_id = Column(Integer, primary_key=True, nullable=False)
    data_type = Column(String(length=128), nullable=True)
    objective = Column(Float, nullable=True)


class AliveTable(Base):
    __tablename__ = 'alive'
    process_name = Column(String(length=128), primary_key=True, nullable=False)
    alive_state = Column(Integer, primary_key=False, nullable=False)


class PidTable(Base):
    __tablename__ = 'pid'
    process_name = Column(String(length=128), primary_key=True, nullable=False)
    pid = Column(BigInteger, nullable=False)


class JobStateTable(Base):
    __tablename__ = 'job_status'
    trial_id = Column(Integer, primary_key=True, nullable=False)
    state = Column(String(length=128), nullable=True)


class InfoTable(Base):
    __tablename__ = 'info'
    data_id = Column(Integer, primary_key=True)
    trial_number = Column(Integer)
    config_path = Column(String(length=512), nullable=False)


class ConfigTable(Base):
    __tablename__ = 'config'
    data_id = Column(Integer, primary_key=True)
    file_path = Column(String(length=512), nullable=False)


class SerializeTable(Base):
    __tablename__ = 'serialize'
    data_id = Column(Integer, primary_key=True)
    trial_id = Column(Integer, nullable=False)
    process_name = Column(String(length=128), nullable=False)
    optimization_variable = Column(PickleType, nullable=False)
    native_random_state = Column(PickleType, nullable=False)
    numpy_random_state = Column(PickleType, nullable=False)


class RandomStateTable(Base):
    __tablename__ = 'random_state'
    data_id = Column(Integer, primary_key=True)
    trial_id = Column(Integer, nullable=False)
    process_name = Column(String(length=128), nullable=False)
    native_random_state = Column(PickleType, nullable=False)
    numpy_random_state = Column(PickleType, nullable=False)


class AbciOutputTable(Base):
    __tablename__ = 'abci_output'
    trial_id = Column(Integer, primary_key=True)
    message = Column(Text, nullable=True)
