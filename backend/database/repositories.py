"""Data access layer for backend database operations."""
from __future__ import annotations

from datetime import datetime
from typing import List, Optional, Dict, Any

from sqlalchemy import select, delete
from sqlalchemy.orm import Session

from backend.database.models import AnalysisRecord
from src.database.connection import get_session_factory


class AnalysisRepository:
    """Repository for analysis records CRUD operations."""

    def __init__(self):
        self.SessionLocal = get_session_factory()

    def create(
        self,
        correlation_id: str,
        currency_pair: str,
        base_currency: str,
        quote_currency: str,
        amount: float,
        risk_tolerance: str,
        urgency: str,
        timeframe_days: Optional[int] = None,
    ) -> AnalysisRecord:
        """Create a new analysis record."""
        with self.SessionLocal() as session:
            record = AnalysisRecord(
                correlation_id=correlation_id,
                currency_pair=currency_pair,
                base_currency=base_currency,
                quote_currency=quote_currency,
                amount=amount,
                risk_tolerance=risk_tolerance,
                urgency=urgency,
                timeframe_days=timeframe_days,
                status="pending",
                progress=0,
                message="Analysis queued",
            )
            session.add(record)
            session.commit()
            session.refresh(record)
            return record

    def get_by_correlation_id(self, correlation_id: str) -> Optional[AnalysisRecord]:
        """Get analysis record by correlation ID."""
        with self.SessionLocal() as session:
            stmt = select(AnalysisRecord).where(
                AnalysisRecord.correlation_id == correlation_id
            )
            result = session.execute(stmt).scalar_one_or_none()
            if result:
                # Detach from session before returning
                session.expunge(result)
            return result

    def update_status(
        self,
        correlation_id: str,
        status: str,
        progress: int,
        message: str,
    ) -> bool:
        """Update analysis status and progress."""
        with self.SessionLocal() as session:
            stmt = select(AnalysisRecord).where(
                AnalysisRecord.correlation_id == correlation_id
            )
            record = session.execute(stmt).scalar_one_or_none()
            if not record:
                return False

            record.status = status
            record.progress = progress
            record.message = message
            record.updated_at = datetime.utcnow()
            session.commit()
            return True

    def update_result(
        self,
        correlation_id: str,
        recommendation: Dict[str, Any],
        market_data: Optional[Dict[str, Any]] = None,
        intelligence: Optional[Dict[str, Any]] = None,
        prediction: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Update analysis with complete results."""
        with self.SessionLocal() as session:
            stmt = select(AnalysisRecord).where(
                AnalysisRecord.correlation_id == correlation_id
            )
            record = session.execute(stmt).scalar_one_or_none()
            if not record:
                return False

            record.status = "completed"
            record.progress = 100
            record.message = "Analysis complete"
            record.recommendation = recommendation
            record.market_data = market_data
            record.intelligence = intelligence
            record.prediction = prediction
            record.updated_at = datetime.utcnow()
            session.commit()
            return True

    def update_error(self, correlation_id: str, error_message: str) -> bool:
        """Mark analysis as errored."""
        with self.SessionLocal() as session:
            stmt = select(AnalysisRecord).where(
                AnalysisRecord.correlation_id == correlation_id
            )
            record = session.execute(stmt).scalar_one_or_none()
            if not record:
                return False

            record.status = "error"
            record.progress = 0
            record.message = error_message
            record.updated_at = datetime.utcnow()
            session.commit()
            return True

    def list_all(
        self,
        status: Optional[str] = None,
        currency_pair: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[AnalysisRecord]:
        """List analysis records with optional filters."""
        with self.SessionLocal() as session:
            stmt = select(AnalysisRecord).order_by(
                AnalysisRecord.created_at.desc()
            )

            if status:
                stmt = stmt.where(AnalysisRecord.status == status)
            if currency_pair:
                stmt = stmt.where(AnalysisRecord.currency_pair == currency_pair)

            stmt = stmt.limit(limit).offset(offset)

            results = session.execute(stmt).scalars().all()
            # Detach from session
            for r in results:
                session.expunge(r)
            return list(results)

    def delete_expired(self) -> int:
        """Delete expired analysis records. Returns count deleted."""
        with self.SessionLocal() as session:
            stmt = delete(AnalysisRecord).where(
                AnalysisRecord.expires_at < datetime.utcnow()
            )
            result = session.execute(stmt)
            session.commit()
            return result.rowcount

    def delete_by_correlation_id(self, correlation_id: str) -> bool:
        """Delete a specific analysis record."""
        with self.SessionLocal() as session:
            stmt = delete(AnalysisRecord).where(
                AnalysisRecord.correlation_id == correlation_id
            )
            result = session.execute(stmt)
            session.commit()
            return result.rowcount > 0
