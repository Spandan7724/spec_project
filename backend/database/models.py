"""Backend database models for API layer."""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy import String, Float, Integer, DateTime, JSON, Text, Boolean
from sqlalchemy.orm import Mapped, mapped_column

from src.database.models import Base


class AnalysisRecord(Base):
    """Store analysis results with short-term retention (24h auto-cleanup)."""
    __tablename__ = "analysis_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    correlation_id: Mapped[str] = mapped_column(String(100), unique=True, index=True)

    # Request parameters
    currency_pair: Mapped[str] = mapped_column(String(10), index=True)
    base_currency: Mapped[str] = mapped_column(String(5))
    quote_currency: Mapped[str] = mapped_column(String(5))
    amount: Mapped[float] = mapped_column(Float)
    risk_tolerance: Mapped[str] = mapped_column(String(20))
    urgency: Mapped[str] = mapped_column(String(20))
    timeframe_days: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Analysis status
    status: Mapped[str] = mapped_column(String(20), index=True)  # pending, processing, completed, error
    progress: Mapped[int] = mapped_column(Integer, default=0)
    message: Mapped[str] = mapped_column(Text)

    # Full result data (JSON serialized)
    recommendation: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    # Evidence data for visualization
    market_data: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    intelligence: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    prediction: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    expires_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.utcnow() + timedelta(hours=24),
        index=True
    )

    def is_expired(self) -> bool:
        """Check if this record has expired."""
        return datetime.utcnow() > self.expires_at

    def to_dict(self) -> dict:
        """Convert to dictionary for API response."""
        return {
            "correlation_id": self.correlation_id,
            "status": self.status,
            "progress": self.progress,
            "message": self.message,
            "recommendation": self.recommendation,
            "market_data": self.market_data,
            "intelligence": self.intelligence,
            "prediction": self.prediction,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
