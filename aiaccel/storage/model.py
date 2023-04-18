from sqlalchemy.orm import declarative_base
from sqlalchemy.schema import Column
from sqlalchemy.types import (Integer, PickleType, String, Text)
from sqlalchemy.ext.declarative import DeclarativeMeta

Base: DeclarativeMeta = declarative_base()


# models
class TrialTable(Base):
    __tablename__ = 'Trial'
    trial_id = Column(Integer, primary_key=True, nullable=False)
    state = Column(Text, nullable=True)


class ErrorTable(Base):
    __tablename__ = 'errors'
    trial_id = Column(Integer, primary_key=True, nullable=False)
    error = Column(Text, nullable=True)
    exitcode = Column(Integer, nullable=True)


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
    param_value = Column(PickleType, nullable=True)
    param_type = Column(String(length=512), nullable=True)


class ResultTable(Base):
    __tablename__ = 'result'
    trial_id = Column(Integer, primary_key=True, nullable=False)
    data_type = Column(String(length=128), nullable=True)
    objective = Column(PickleType, nullable=True)


class JobStateTable(Base):
    __tablename__ = 'job_status'
    trial_id = Column(Integer, primary_key=True, nullable=False)
    state = Column(String(length=128), nullable=True)


class VariableTable(Base):
    __tablename__ = 'variable'
    data_id = Column(Integer, primary_key=True)
    trial_id = Column(Integer, nullable=False)
    process_name = Column(String(length=128), nullable=False)
    label = Column(String(length=128), nullable=False)
    value = Column(PickleType, nullable=False)

