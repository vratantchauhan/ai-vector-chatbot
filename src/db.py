from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager

from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from src.config.settings import DATABASE_URL
from src.models import Base

_engine = None
_SessionLocal: sessionmaker[Session] | None = None


def _create_engine():
    eng = create_engine(
        DATABASE_URL,
        echo=False,
        future=True,
        pool_pre_ping=True,
    )
    if DATABASE_URL.startswith("sqlite"):

        @event.listens_for(eng, "connect")
        def _sqlite_pragma(dbapi_connection, _connection_record) -> None:
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    return eng


def get_engine():
    global _engine
    if _engine is None:
        _engine = _create_engine()
    return _engine


def get_session_factory() -> sessionmaker[Session]:
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(
            bind=get_engine(),
            autoflush=False,
            autocommit=False,
            expire_on_commit=False,
            future=True,
        )
    return _SessionLocal


def init_db() -> None:
    Base.metadata.create_all(bind=get_engine())


@contextmanager
def session_scope() -> Generator[Session, None, None]:
    factory = get_session_factory()
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
