"""Custom exception classes for the Currency Assistant."""


class CurrencyAssistantError(Exception):
    """Base exception for all Currency Assistant errors."""
    pass


class ConfigurationError(CurrencyAssistantError):
    """Raised when there's a configuration error."""
    pass


class DataProviderError(CurrencyAssistantError):
    """Base exception for data provider errors."""
    pass


class RateLimitError(DataProviderError):
    """Raised when API rate limit is exceeded."""
    pass


class DataNotFoundError(DataProviderError):
    """Raised when requested data is not available."""
    pass


class ValidationError(CurrencyAssistantError):
    """Raised when data validation fails."""
    pass


class PredictionError(CurrencyAssistantError):
    """Raised when prediction fails."""
    pass


class CacheError(CurrencyAssistantError):
    """Raised when cache operations fail."""
    pass

