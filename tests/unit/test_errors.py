"""Tests for custom errors."""
from src.utils.errors import (
    CurrencyAssistantError,
    ConfigurationError,
    ValidationError,
    DataProviderError
)


def test_error_hierarchy():
    """Test error inheritance."""
    assert issubclass(ConfigurationError, CurrencyAssistantError)
    assert issubclass(ValidationError, CurrencyAssistantError)
    assert issubclass(DataProviderError, CurrencyAssistantError)


def test_error_messages():
    """Test error messages."""
    error = ConfigurationError("Test message")
    assert str(error) == "Test message"

