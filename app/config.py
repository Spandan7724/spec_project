"""
Configuration management for the currency assistant application.

Uses pydantic-settings for environment variable handling and validation.
"""

import os
from typing import List, Optional
from pydantic import Field, validator
from pydantic_settings import BaseSettings


class DatabaseSettings(BaseSettings):
    """Database configuration."""
    
    # PostgreSQL connection
    host: str = Field(default="localhost", env="DB_HOST")
    port: int = Field(default=5432, env="DB_PORT")
    username: str = Field(default="postgres", env="DB_USERNAME") 
    password: str = Field(default="password", env="DB_PASSWORD")
    database: str = Field(default="currency_assistant", env="DB_NAME")
    
    # Connection pool settings
    pool_size: int = Field(default=10, env="DB_POOL_SIZE")
    max_overflow: int = Field(default=20, env="DB_MAX_OVERFLOW")
    pool_timeout: int = Field(default=30, env="DB_POOL_TIMEOUT")
    
    @property
    def async_url(self) -> str:
        """Generate async PostgreSQL connection URL."""
        return f"postgresql+asyncpg://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
    
    @property 
    def sync_url(self) -> str:
        """Generate sync PostgreSQL connection URL for migrations."""
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"


class RedisSettings(BaseSettings):
    """Redis configuration."""
    
    host: str = Field(default="localhost", env="REDIS_HOST")
    port: int = Field(default=6379, env="REDIS_PORT")
    password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    db: int = Field(default=0, env="REDIS_DB")
    
    # Connection settings
    max_connections: int = Field(default=100, env="REDIS_MAX_CONNECTIONS")
    
    @property
    def url(self) -> str:
        """Generate Redis connection URL."""
        if self.password:
            return f"redis://:{self.password}@{self.host}:{self.port}/{self.db}"
        return f"redis://{self.host}:{self.port}/{self.db}"


class APIProviderSettings(BaseSettings):
    """External API provider configuration."""
    
    # Alpha Vantage (free tier: 25 requests/day)
    alphavantage_api_key: Optional[str] = Field(default=None, env="ALPHAVANTAGE_API_KEY")
    alphavantage_base_url: str = "https://www.alphavantage.co"
    
    # Fixer.io (free tier: 100 requests/month)  
    fixer_api_key: Optional[str] = Field(default=None, env="FIXER_API_KEY")
    fixer_base_url: str = "http://data.fixer.io/api"
    
    # Exchange Rates API (free tier: 1500 requests/month)
    exchangerates_api_key: Optional[str] = Field(default=None, env="EXCHANGERATES_API_KEY")
    exchangerates_base_url: str = "https://v6.exchangerate-api.com/v6"
    
    # Request timeout and retry settings
    request_timeout: int = Field(default=10, env="API_REQUEST_TIMEOUT")
    max_retries: int = Field(default=3, env="API_MAX_RETRIES")
    retry_delay: float = Field(default=1.0, env="API_RETRY_DELAY")


class DataCollectionSettings(BaseSettings):
    """Data collection configuration."""
    
    # Currency pairs to collect
    currency_pairs: List[str] = Field(
        default=["USD/EUR", "USD/GBP", "USD/JPY", "EUR/GBP", "GBP/JPY"],
        env="CURRENCY_PAIRS"
    )
    
    # Collection frequency (seconds)
    collection_interval: int = Field(default=30, env="COLLECTION_INTERVAL")
    
    # Data validation thresholds
    max_rate_change_percent: float = Field(default=5.0, env="MAX_RATE_CHANGE_PERCENT")
    min_rate_value: float = Field(default=0.001, env="MIN_RATE_VALUE")
    max_rate_value: float = Field(default=1000.0, env="MAX_RATE_VALUE")
    
    @validator('currency_pairs', pre=True)
    def parse_currency_pairs(cls, v):
        """Parse currency pairs from environment variable string."""
        if isinstance(v, str):
            return [pair.strip() for pair in v.split(',')]
        return v


class CacheSettings(BaseSettings):
    """Caching configuration."""
    
    # TTL values (seconds)
    current_rate_ttl: int = Field(default=30, env="CACHE_CURRENT_RATE_TTL")
    provider_rate_ttl: int = Field(default=14400, env="CACHE_PROVIDER_RATE_TTL")  # 4 hours
    prediction_ttl: int = Field(default=3600, env="CACHE_PREDICTION_TTL")  # 1 hour
    
    # Cache key prefixes
    rate_key_prefix: str = "rate"
    provider_key_prefix: str = "provider"
    prediction_key_prefix: str = "prediction"


class Settings(BaseSettings):
    """Main application settings."""
    
    # Application info
    app_name: str = "Currency Assistant"
    app_version: str = "0.1.0"
    debug: bool = Field(default=False, env="DEBUG")
    
    # API settings
    api_host: str = Field(default="127.0.0.1", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    
    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    
    # Sub-configurations
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    api_providers: APIProviderSettings = Field(default_factory=APIProviderSettings)
    data_collection: DataCollectionSettings = Field(default_factory=DataCollectionSettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings = Settings()