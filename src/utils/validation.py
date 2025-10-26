"""Input validation utilities."""
from src.utils.errors import ValidationError


def validate_currency_pair(pair: str) -> tuple[str, str]:
    """
    Validate and parse currency pair.
    
    Args:
        pair: Currency pair string (e.g., "USD/EUR", "USD-EUR", "USDEUR")
    
    Returns:
        Tuple of (base_currency, quote_currency)
    
    Raises:
        ValidationError: If pair format is invalid
    """
    # Remove whitespace
    pair = pair.strip().upper()
    
    # Try different separators
    for sep in ['/', '-', '_']:
        if sep in pair:
            parts = pair.split(sep)
            if len(parts) == 2:
                base, quote = parts
                if len(base) == 3 and len(quote) == 3 and base.isalpha() and quote.isalpha():
                    return base, quote
    
    # Try without separator (USDEUR)
    if len(pair) == 6 and pair.isalpha():
        return pair[:3], pair[3:]
    
    raise ValidationError(f"Invalid currency pair format: {pair}")


def validate_amount(amount: float) -> float:
    """
    Validate conversion amount.
    
    Args:
        amount: Amount to validate
    
    Returns:
        Validated amount
    
    Raises:
        ValidationError: If amount is invalid
    """
    if amount <= 0:
        raise ValidationError(f"Amount must be positive, got: {amount}")
    
    if amount > 1e10:  # 10 billion
        raise ValidationError(f"Amount too large: {amount}")
    
    return amount


def validate_risk_tolerance(risk: str) -> str:
    """Validate risk tolerance level."""
    valid_levels = ["conservative", "moderate", "aggressive"]
    risk = risk.lower().strip()
    
    if risk not in valid_levels:
        raise ValidationError(
            f"Invalid risk tolerance: {risk}. Must be one of {valid_levels}"
        )
    
    return risk


def validate_urgency(urgency: str) -> str:
    """Validate urgency level."""
    valid_levels = ["urgent", "normal", "flexible"]
    urgency = urgency.lower().strip()
    
    if urgency not in valid_levels:
        raise ValidationError(
            f"Invalid urgency: {urgency}. Must be one of {valid_levels}"
        )
    
    return urgency


def validate_timeframe(timeframe: str) -> str:
    """Validate timeframe."""
    valid_frames = ["immediate", "1_day", "1_week", "1_month"]
    timeframe = timeframe.lower().strip()
    
    if timeframe not in valid_frames:
        raise ValidationError(
            f"Invalid timeframe: {timeframe}. Must be one of {valid_frames}"
        )
    
    return timeframe

