"""
Database connection and operations manager.

Handles SQLAlchemy async sessions, connection pooling, and TimescaleDB
hypertable setup for time-series data optimization.
"""

import asyncio
import logging
from typing import AsyncGenerator, Optional
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession, 
    create_async_engine,
    async_sessionmaker,
)
from sqlalchemy.sql import text
from sqlalchemy import exc

from .models import Base

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Manages database connections and TimescaleDB hypertable setup.
    
    Provides async session management, connection pooling, and
    TimescaleDB-specific operations for time-series data.
    """
    
    def __init__(
        self,
        database_url: str,
        pool_size: int = 10,
        max_overflow: int = 20,
        pool_timeout: int = 30,
    ):
        """
        Initialize database manager.
        
        Args:
            database_url: Async PostgreSQL connection string
            pool_size: Number of connections to maintain in pool
            max_overflow: Maximum additional connections allowed
            pool_timeout: Timeout for getting connection from pool
        """
        self.database_url = database_url
        self.engine: Optional[AsyncEngine] = None
        self.session_maker: Optional[async_sessionmaker] = None
        
        # Connection pool settings
        self.pool_size = pool_size
        self.max_overflow = max_overflow  
        self.pool_timeout = pool_timeout
        
    async def initialize(self) -> None:
        """Initialize database engine and session maker."""
        self.engine = create_async_engine(
            self.database_url,
            echo=False,  # Set to True for SQL logging
            pool_size=self.pool_size,
            max_overflow=self.max_overflow,
            pool_timeout=self.pool_timeout,
            pool_pre_ping=True,  # Verify connections before use
        )
        
        self.session_maker = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
        
        logger.info("Database manager initialized successfully")
    
    async def create_tables(self) -> None:
        """Create all tables defined in models."""
        if not self.engine:
            raise RuntimeError("Database not initialized. Call initialize() first.")
        
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        logger.info("Database tables created successfully")
    
    async def setup_timescaledb(self) -> None:
        """
        Set up TimescaleDB hypertables and indexes.
        
        Converts the fx_rates table to a hypertable partitioned by time
        for efficient time-series queries and automatic data management.
        """
        if not self.session_maker:
            raise RuntimeError("Database not initialized. Call initialize() first.")
        
        async with self.session_maker() as session:
            try:
                # Check if TimescaleDB extension is installed
                result = await session.execute(
                    text("SELECT 1 FROM pg_extension WHERE extname = 'timescaledb'")
                )
                if not result.fetchone():
                    logger.warning("TimescaleDB extension not found. Installing...")
                    await session.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb"))
                
                # Create hypertable for fx_rates
                await session.execute(
                    text("""
                        SELECT create_hypertable(
                            'fx_rates', 
                            'time',
                            chunk_time_interval => INTERVAL '1 hour',
                            if_not_exists => TRUE
                        )
                    """)
                )
                
                # Set up data retention policy (keep data for 2 years)
                await session.execute(
                    text("""
                        SELECT add_retention_policy(
                            'fx_rates',
                            INTERVAL '2 years',
                            if_not_exists => TRUE
                        )
                    """)
                )
                
                # Create continuous aggregates for common queries
                await session.execute(
                    text("""
                        CREATE MATERIALIZED VIEW IF NOT EXISTS fx_rates_hourly
                        WITH (timescaledb.continuous) AS
                        SELECT 
                            time_bucket('1 hour', time) AS bucket,
                            currency_pair,
                            first(rate, time) AS open_rate,
                            max(rate) AS high_rate,
                            min(rate) AS low_rate,
                            last(rate, time) AS close_rate,
                            avg(rate) AS avg_rate,
                            count(*) AS num_samples
                        FROM fx_rates
                        GROUP BY bucket, currency_pair
                        WITH NO DATA
                    """)
                )
                
                # Enable continuous aggregate refresh policy
                await session.execute(
                    text("""
                        SELECT add_continuous_aggregate_policy(
                            'fx_rates_hourly',
                            start_offset => INTERVAL '3 hours',
                            end_offset => INTERVAL '1 hour',
                            schedule_interval => INTERVAL '1 hour',
                            if_not_exists => TRUE
                        )
                    """)
                )
                
                await session.commit()
                logger.info("TimescaleDB hypertables and policies configured successfully")
                
            except exc.SQLAlchemyError as e:
                await session.rollback()
                logger.error(f"Failed to setup TimescaleDB: {e}")
                raise
    
    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Get an async database session.
        
        Usage:
            async with db_manager.get_session() as session:
                # Use session for queries
                pass
        """
        if not self.session_maker:
            raise RuntimeError("Database not initialized. Call initialize() first.")
        
        async with self.session_maker() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
    
    async def health_check(self) -> bool:
        """
        Check database connectivity and health.
        
        Returns:
            True if database is healthy, False otherwise
        """
        try:
            async with self.get_session() as session:
                await session.execute(text("SELECT 1"))
                return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False
    
    async def close(self) -> None:
        """Close database engine and cleanup resources."""
        if self.engine:
            await self.engine.dispose()
            logger.info("Database connections closed")


# Global database manager instance
db_manager = DatabaseManager(
    database_url="postgresql+asyncpg://user:password@localhost/currency_assistant"
)


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency function for FastAPI to get database sessions.
    
    Usage in FastAPI endpoints:
        @app.get("/rates")
        async def get_rates(session: AsyncSession = Depends(get_db_session)):
            # Use session
    """
    async with db_manager.get_session() as session:
        yield session