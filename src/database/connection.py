"""Database connection management."""
from pathlib import Path
from sqlalchemy import create_engine, Engine
from sqlalchemy.orm import sessionmaker
from src.config import get_config
from src.database.models import Base
import logging

logger = logging.getLogger(__name__)

_engine: Engine = None
_SessionLocal: sessionmaker = None


def get_engine() -> Engine:
    """Get or create database engine."""
    global _engine
    
    if _engine is None:
        config = get_config()
        db_path = config.database_path
        
        # Ensure directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Create SQLite engine
        database_url = f"sqlite:///{db_path}"
        _engine = create_engine(
            database_url,
            connect_args={"check_same_thread": False},  # SQLite specific
            echo=config.get('database.echo', False)
        )
        
        logger.info(f"Database engine created: {db_path}")
    
    return _engine


def get_session_factory() -> sessionmaker:
    """Get or create session factory."""
    global _SessionLocal
    
    if _SessionLocal is None:
        engine = get_engine()
        _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    return _SessionLocal


def create_tables():
    """Create all database tables."""
    engine = get_engine()
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created")


def drop_tables():
    """Drop all database tables (for testing)."""
    engine = get_engine()
    Base.metadata.drop_all(bind=engine)
    logger.info("Database tables dropped")

