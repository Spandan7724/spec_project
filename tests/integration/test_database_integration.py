"""Integration tests for database functionality."""
import tempfile
from pathlib import Path


def test_full_database_workflow():
    """Test complete database workflow with real SQLite file."""
    # Load config first
    from src.config import load_config
    load_config()
    
    from src.database.connection import create_tables, get_session_factory
    from src.database.models import Conversation, PredictionHistory
    
    # Use the configured database path
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        
        # Create tables
        create_tables()
        
        # Insert data
        SessionLocal = get_session_factory()
        session = SessionLocal()
        
        try:
            # Add conversation
            conv = Conversation(
                session_id="integration-test",
                user_query="Convert 1000 USD to EUR",
                response="Wait for better rate",
                user_params={"risk": "moderate", "urgency": "normal"}
            )
            session.add(conv)
            
            # Add prediction
            pred = PredictionHistory(
                currency_pair="USD/EUR",
                prediction_horizon=7,
                predicted_rate=0.85,
                confidence=0.72
            )
            session.add(pred)
            
            session.commit()
            
            # Query data
            conv_result = session.query(Conversation).filter_by(
                session_id="integration-test"
            ).first()
            
            assert conv_result is not None
            assert conv_result.user_query == "Convert 1000 USD to EUR"
            assert conv_result.user_params["risk"] == "moderate"
            
            pred_result = session.query(PredictionHistory).filter_by(
                currency_pair="USD/EUR"
            ).first()
            
            assert pred_result is not None
            assert pred_result.predicted_rate == 0.85
            assert pred_result.confidence == 0.72
            
        finally:
            session.close()


def test_database_with_context_manager():
    """Test database session context manager."""
    from src.config import load_config
    load_config()
    
    from src.database.connection import create_tables
    from src.database.session import get_db
    from src.database.models import AgentMetrics
    
    create_tables()
    
    # Use context manager
    with get_db() as db:
        metric = AgentMetrics(
            agent_name="test_agent",
            execution_time_ms=100,
            status="success"
        )
        db.add(metric)
    
    # Verify data was committed
    with get_db() as db:
        result = db.query(AgentMetrics).filter_by(agent_name="test_agent").first()
        assert result is not None
        assert result.execution_time_ms == 100

