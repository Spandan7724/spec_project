"""
SQLAlchemy models for FX data storage.

This module defines the database models for storing foreign exchange rates
and provider-specific data in PostgreSQL with TimescaleDB extension.
"""

from datetime import datetime
from decimal import Decimal
from typing import Optional

from sqlalchemy import Column, String, DECIMAL, DateTime, BigInteger, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func

Base = declarative_base()


class FXRate(Base):
    """
    Time-series table for storing foreign exchange rates.
    
    This table will be converted to a TimescaleDB hypertable for
    efficient time-series queries and data retention policies.
    """
    __tablename__ = "fx_rates"
    
    # Composite primary key (time, currency_pair)
    time = Column(DateTime(timezone=True), primary_key=True, nullable=False)
    currency_pair = Column(String(7), primary_key=True, nullable=False)  # e.g., "USD/EUR"
    
    # Rate data
    rate = Column(DECIMAL(12, 6), nullable=False)  # Mid-market rate
    bid = Column(DECIMAL(12, 6), nullable=True)    # Bid price
    ask = Column(DECIMAL(12, 6), nullable=True)    # Ask price
    
    # Volume and metadata
    volume = Column(BigInteger, nullable=True)      # Trading volume if available
    provider = Column(String(50), nullable=False)  # Data source
    
    # Automatically set timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Indexes for efficient querying
    __table_args__ = (
        Index('idx_fx_rates_time_desc', 'time', postgresql_using='btree'),
        Index('idx_fx_rates_pair_time', 'currency_pair', 'time'),
        Index('idx_fx_rates_provider', 'provider'),
    )
    
    def __repr__(self) -> str:
        return f"<FXRate({self.currency_pair}={self.rate} at {self.time})>"
    
    @property 
    def spread(self) -> Optional[Decimal]:
        """Calculate bid-ask spread if both bid and ask are available."""
        if self.bid and self.ask:
            return self.ask - self.bid
        return None
    
    @property
    def spread_percentage(self) -> Optional[Decimal]:
        """Calculate spread as percentage of mid-market rate."""
        spread = self.spread
        if spread and self.rate:
            return (spread / self.rate) * 100
        return None


class ProviderRate(Base):
    """
    Table for storing provider-specific exchange rates and fees.
    
    This table tracks rates offered by different financial service providers
    including banks, fintechs, and money transfer services.
    """
    __tablename__ = "provider_rates"
    
    id = Column(String(36), primary_key=True)  # UUID
    provider = Column(String(50), nullable=False)
    currency_pair = Column(String(7), nullable=False)
    
    # Rate and fee information
    rate = Column(DECIMAL(12, 6), nullable=False)
    spread_percent = Column(DECIMAL(5, 4), nullable=True)    # Spread over mid-market
    fixed_fee = Column(DECIMAL(10, 2), nullable=True)        # Fixed transfer fee
    percentage_fee = Column(DECIMAL(5, 4), nullable=True)    # Percentage fee
    
    # Transfer limits and timing
    min_amount = Column(DECIMAL(15, 2), nullable=True)
    max_amount = Column(DECIMAL(15, 2), nullable=True)
    delivery_time_hours = Column(BigInteger, nullable=True)
    
    # Timestamps
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    expires_at = Column(DateTime(timezone=True), nullable=True)  # Rate expiration
    
    # Indexes
    __table_args__ = (
        Index('idx_provider_rates_pair', 'currency_pair'),
        Index('idx_provider_rates_provider', 'provider'),
        Index('idx_provider_rates_updated', 'updated_at'),
    )
    
    def __repr__(self) -> str:
        return f"<ProviderRate({self.provider}: {self.currency_pair}={self.rate})>"
    
    def calculate_total_cost(self, amount: Decimal) -> Decimal:
        """
        Calculate total cost for converting a given amount.
        
        Args:
            amount: Amount to convert in source currency
            
        Returns:
            Total cost including all fees
        """
        # Base conversion cost
        converted_amount = amount * self.rate
        
        # Add fixed fee if applicable
        if self.fixed_fee:
            converted_amount += self.fixed_fee
        
        # Add percentage fee if applicable  
        if self.percentage_fee:
            converted_amount += amount * (self.percentage_fee / 100)
        
        return converted_amount
    
    @property
    def is_expired(self) -> bool:
        """Check if the rate quote has expired."""
        if not self.expires_at:
            return False
        return datetime.utcnow() > self.expires_at