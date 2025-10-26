"""Tests for validation utilities."""
import pytest
from src.utils.validation import (
    validate_currency_pair,
    validate_amount,
    validate_risk_tolerance,
    validate_urgency,
    validate_timeframe
)
from src.utils.errors import ValidationError


def test_validate_currency_pair_slash():
    """Test currency pair validation with slash."""
    base, quote = validate_currency_pair("USD/EUR")
    assert base == "USD"
    assert quote == "EUR"


def test_validate_currency_pair_no_separator():
    """Test currency pair validation without separator."""
    base, quote = validate_currency_pair("USDEUR")
    assert base == "USD"
    assert quote == "EUR"


def test_validate_currency_pair_invalid():
    """Test invalid currency pair."""
    with pytest.raises(ValidationError):
        validate_currency_pair("INVALID")


def test_validate_amount_valid():
    """Test valid amount."""
    assert validate_amount(100.0) == 100.0
    assert validate_amount(0.01) == 0.01


def test_validate_amount_negative():
    """Test negative amount."""
    with pytest.raises(ValidationError):
        validate_amount(-100.0)


def test_validate_amount_too_large():
    """Test amount too large."""
    with pytest.raises(ValidationError):
        validate_amount(1e11)


def test_validate_risk_tolerance():
    """Test risk tolerance validation."""
    assert validate_risk_tolerance("conservative") == "conservative"
    assert validate_risk_tolerance("MODERATE") == "moderate"
    
    with pytest.raises(ValidationError):
        validate_risk_tolerance("invalid")


def test_validate_urgency():
    """Test urgency validation."""
    assert validate_urgency("urgent") == "urgent"
    assert validate_urgency("NORMAL") == "normal"
    
    with pytest.raises(ValidationError):
        validate_urgency("invalid")


def test_validate_timeframe():
    """Test timeframe validation."""
    assert validate_timeframe("immediate") == "immediate"
    assert validate_timeframe("1_DAY") == "1_day"
    
    with pytest.raises(ValidationError):
        validate_timeframe("invalid")

