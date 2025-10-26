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

