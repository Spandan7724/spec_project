"""Database session management."""
from contextlib import contextmanager
from typing import Generator
from sqlalchemy.orm import Session
from src.database.connection import get_session_factory


@contextmanager
def get_db() -> Generator[Session, None, None]:
    """
    Context manager for database sessions.
    
    Usage:
        with get_db() as db:
            db.query(Model).all()
    """
    SessionLocal = get_session_factory()
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_db_session() -> Generator[Session, None, None]:
    """
    Get a database session (for dependency injection).
    
    Usage in FastAPI:
        def endpoint(db: Session = Depends(get_db_session)):
            ...
    """
    SessionLocal = get_session_factory()
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()

