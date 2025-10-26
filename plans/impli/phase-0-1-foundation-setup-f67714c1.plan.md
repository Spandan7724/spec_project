<!-- f67714c1-a54f-4e8d-9617-16955f212afc f07ee625-2ed3-4c47-9ff5-3b33fb167c3e -->
# Phase 0.2: Database Schema & ORM Setup

## Overview

Create SQLite database schema, SQLAlchemy ORM models, Alembic migration system, database connection management, and in-memory cache with TTL support.

## Implementation Steps

### Step 1: Create In-Memory Cache (src/cache.py)

Simple Python-based cache with TTL for market data:

```python
"""Simple in-memory cache with TTL."""
from datetime import datetime, timedelta
from typing import Any, Optional, Dict
import threading

class SimpleCache:
    """Thread-safe in-memory cache with TTL."""
    
    def __init__(self):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
    
    def set(self, key: str, value: Any, ttl_seconds: int = 300) -> None:
        """Store value with expiration."""
        expires_at = datetime.utcnow() + timedelta(seconds=ttl_seconds)
        with self._lock:
            self._cache[key] = {"value": value, "expires_at": expires_at}
    
    def get(self, key: str) -> Optional[Any]:
        """Retrieve value if not expired."""
        with self._lock:
            if key not in self._cache:
                return None
            item = self._cache[key]
            if datetime.utcnow() < item["expires_at"]:
                return item["value"]
            else:
                del self._cache[key]
                return None
    
    def delete(self, key: str) -> None:
        """Delete key."""
        with self._lock:
            self._cache.pop(key, None)
    
    def clear(self) -> None:
        """Clear all entries."""
        with self._lock:
            self._cache.clear()
    
    def cleanup_expired(self) -> int:
        """Remove expired entries. Returns count removed."""
        with self._lock:
            now = datetime.utcnow()
            expired = [k for k, v in self._cache.items() if now >= v["expires_at"]]
            for key in expired:
                del self._cache[key]
            return len(expired)

# Global cache instance
cache = SimpleCache()
```

### Step 2: Create Database Models (src/database/models.py)

SQLAlchemy ORM models for all tables:

```python
"""Database models for Currency Assistant."""
from datetime import datetime
from typing import Optional
from sqlalchemy import String, Float, Integer, DateTime, JSON, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

class Base(DeclarativeBase):
    """Base class for all models."""
    pass

class Conversation(Base):
    """Conversation history table."""
    __tablename__ = "conversations"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[str] = mapped_column(String(100), index=True)
    user_query: Mapped[str] = mapped_column(Text)
    response: Mapped[str] = mapped_column(Text)
    user_params: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)

class PredictionHistory(Base):
    """Prediction tracking table."""
    __tablename__ = "prediction_history"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    currency_pair: Mapped[str] = mapped_column(String(10), index=True)
    prediction_horizon: Mapped[int] = mapped_column(Integer)
    predicted_rate: Mapped[float] = mapped_column(Float)
    confidence: Mapped[float] = mapped_column(Float)
    actual_rate: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    prediction_time: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    evaluation_time: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

class AgentMetrics(Base):
    """Agent performance metrics table."""
    __tablename__ = "agent_metrics"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    agent_name: Mapped[str] = mapped_column(String(50), index=True)
    execution_time_ms: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(20))
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)

class SystemLog(Base):
    """System audit log table."""
    __tablename__ = "system_logs"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    request_id: Mapped[str] = mapped_column(String(100), index=True)
    log_level: Mapped[str] = mapped_column(String(20), index=True)
    message: Mapped[str] = mapped_column(Text)
    agent: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    extra_data: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
```

### Step 3: Create Database Connection (src/database/connection.py)

Connection management for SQLite:

```python
"""Database connection management."""
from pathlib import Path
from sqlalchemy import create_engine, Engine
from sqlalchemy.orm import sessionmaker, Session
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
```

### Step 4: Create Session Management (src/database/session.py)

Context manager for database sessions:

```python
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

def get_db_session() -> Session:
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
```

### Step 5: Initialize Alembic

Set up Alembic for database migrations:

```bash
# Run from project root
alembic init alembic
```

Then update `alembic.ini`:

- Set `sqlalchemy.url` to read from config (or use env var)

Update `alembic/env.py`:

```python
from src.database.models import Base
target_metadata = Base.metadata

# Update config.set_main_option to use our config
from src.config import load_config
app_config = load_config()
config.set_main_option('sqlalchemy.url', f"sqlite:///{app_config.database_path}")
```

### Step 6: Create Initial Migration

Create first migration with all tables:

```bash
alembic revision --autogenerate -m "Initial schema"
alembic upgrade head
```

### Step 7: Write Tests

**tests/unit/test_cache.py**:

```python
"""Tests for cache module."""
import pytest
import time
from src.cache import SimpleCache

def test_cache_set_get():
    """Test basic set and get."""
    cache = SimpleCache()
    cache.set("key1", "value1", ttl_seconds=10)
    assert cache.get("key1") == "value1"

def test_cache_expiration():
    """Test TTL expiration."""
    cache = SimpleCache()
    cache.set("key1", "value1", ttl_seconds=1)
    time.sleep(1.1)
    assert cache.get("key1") is None

def test_cache_delete():
    """Test delete."""
    cache = SimpleCache()
    cache.set("key1", "value1")
    cache.delete("key1")
    assert cache.get("key1") is None

def test_cache_clear():
    """Test clear all."""
    cache = SimpleCache()
    cache.set("key1", "value1")
    cache.set("key2", "value2")
    cache.clear()
    assert cache.get("key1") is None
    assert cache.get("key2") is None
```

**tests/unit/test_database.py**:

```python
"""Tests for database module."""
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.database.models import Base, Conversation, PredictionHistory
from datetime import datetime

@pytest.fixture
def db_session():
    """Create in-memory test database."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    yield session
    session.close()

def test_conversation_model(db_session):
    """Test Conversation model."""
    conv = Conversation(
        session_id="test-123",
        user_query="Test query",
        response="Test response",
        user_params={"risk": "moderate"}
    )
    db_session.add(conv)
    db_session.commit()
    
    result = db_session.query(Conversation).first()
    assert result.session_id == "test-123"
    assert result.user_params["risk"] == "moderate"

def test_prediction_history_model(db_session):
    """Test PredictionHistory model."""
    pred = PredictionHistory(
        currency_pair="USD/EUR",
        prediction_horizon=7,
        predicted_rate=0.85,
        confidence=0.75
    )
    db_session.add(pred)
    db_session.commit()
    
    result = db_session.query(PredictionHistory).first()
    assert result.currency_pair == "USD/EUR"
    assert result.predicted_rate == 0.85
```

### Step 8: Integration Test

Test full database workflow:

```python
# tests/integration/test_database_integration.py
def test_full_database_workflow():
    """Test complete database workflow."""
    from src.database.connection import create_tables, get_session_factory
    from src.database.models import Conversation
    
    # Create tables
    create_tables()
    
    # Insert data
    SessionLocal = get_session_factory()
    with SessionLocal() as session:
        conv = Conversation(
            session_id="integration-test",
            user_query="Test",
            response="Response"
        )
        session.add(conv)
        session.commit()
        
        # Query data
        result = session.query(Conversation).filter_by(
            session_id="integration-test"
        ).first()
        
        assert result is not None
        assert result.user_query == "Test"
```

## Files to Create

1. `src/cache.py` - In-memory cache implementation
2. `src/database/models.py` - SQLAlchemy ORM models
3. `src/database/connection.py` - Database connection management
4. `src/database/session.py` - Session context managers
5. `alembic.ini` - Alembic configuration
6. `alembic/env.py` - Alembic environment setup
7. `alembic/versions/xxx_initial_schema.py` - Initial migration
8. `tests/unit/test_cache.py` - Cache tests
9. `tests/unit/test_database.py` - Database model tests
10. `tests/integration/test_database_integration.py` - Integration test

## Success Criteria

- Cache set/get/delete/clear work correctly
- Cache TTL expiration works
- All database models can be created
- Can insert and query data from all tables
- Alembic migrations run successfully
- All tests pass
- Database file created at correct location

### To-dos

- [ ] Create complete directory structure for all project modules
- [ ] Extend config.yaml with app, database, cache, api, agents, and logging sections
- [ ] Create .env.example with all required environment variables
- [ ] Implement src/utils/errors.py with custom exception hierarchy
- [ ] Implement src/utils/logging.py with structured logging and correlation IDs
- [ ] Implement src/utils/validation.py with input validation functions
- [ ] Implement src/config.py with YAML and environment variable loading
- [ ] Write comprehensive tests for config, validation, and error modules
- [ ] Update pyproject.toml to remove unnecessary dependencies
- [ ] Run all tests and validate configuration loading works correctly