from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from sqlalchemy import MetaData, create_engine
from sqlalchemy.orm import Session, scoped_session, sessionmaker
from sqlalchemy.pool import NullPool

from aiaccel.storage.model import Base
from aiaccel.util import retry


class Abstract:
    """Abstract class for storage.

    Args:
        file_name (Path): Path to the storage file.

    Attributes:
        url (str): URL to the storage file.
        engine (Engine): Engine to the storage file.
        metadata (MetaData): MetaData to the storage file.
        session (Session): Session to the storage file.
        lock_file (Path): Path to the lock file.
    """

    @retry(_MAX_NUM=6, _DELAY=1.0)
    def __init__(self, file_name: Path) -> None:
        self.url = f"sqlite:///{file_name}"
        self.engine = create_engine(self.url, echo=False, poolclass=NullPool, connect_args={"timeout": 60})
        self.metadata = MetaData()
        self.metadata.bind = self.engine
        Base.metadata.create_all(self.engine)
        self.session = scoped_session(sessionmaker(autocommit=False, autoflush=True, bind=self.engine))
        self.lock_file = Path(file_name).resolve().parent / "db_lock"

    @contextmanager
    def create_session(self) -> Generator[Session, None, None]:
        session = self.session()
        yield session
        session.close()
