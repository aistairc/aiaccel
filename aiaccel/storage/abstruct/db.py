from pathlib import Path
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy import MetaData
from sqlalchemy.pool import NullPool
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import scoped_session

from aiaccel.storage.model.db import Base
from aiaccel.util import retry


class Abstract:

    @retry(_MAX_NUM=6, _DELAY=1.0)
    def __init__(self, file_name):
        self.url = f'sqlite:///{file_name}'
        self.engine = create_engine(
            self.url,
            echo=False,
            poolclass=NullPool,
            connect_args={'check_same_thread': False, 'timeout': 60}
        )
        self.metadata = MetaData()
        self.metadata.bind = self.engine
        Base.metadata.create_all(self.engine)
        self.session = scoped_session(
            sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        )
        self.lock_file = Path(file_name).resolve().parent / "db_lock"

    @contextmanager
    def create_session(self):
        session = self.session()
        yield session
        session.close()
