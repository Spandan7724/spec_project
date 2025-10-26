"""Tests for database module."""
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.database.models import Base, Conversation, PredictionHistory, AgentMetrics, SystemLog


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
    assert result.user_query == "Test query"
    assert result.user_params["risk"] == "moderate"
    assert result.timestamp is not None


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
    assert result.prediction_horizon == 7
    assert result.actual_rate is None


def test_agent_metrics_model(db_session):
    """Test AgentMetrics model."""
    metric = AgentMetrics(
        agent_name="market_data",
        execution_time_ms=150,
        status="success"
    )
    db_session.add(metric)
    db_session.commit()
    
    result = db_session.query(AgentMetrics).first()
    assert result.agent_name == "market_data"
    assert result.execution_time_ms == 150
    assert result.status == "success"


def test_system_log_model(db_session):
    """Test SystemLog model."""
    log = SystemLog(
        request_id="req-123",
        log_level="INFO",
        message="Test log message",
        agent="supervisor",
        extra_data={"key": "value"}
    )
    db_session.add(log)
    db_session.commit()
    
    result = db_session.query(SystemLog).first()
    assert result.request_id == "req-123"
    assert result.log_level == "INFO"
    assert result.extra_data["key"] == "value"

