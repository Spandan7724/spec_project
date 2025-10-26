"""Provider base classes and data contracts for market data providers."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional

from src.utils.errors import ValidationError


@dataclass
class ProviderRate:
    """Normalized provider rate output used by the aggregator layer.

    Fields mirror the ProviderRate schema in plans/market-data-agent-plan.md.
    """

    source: str  # "exchange_rate_host" | "yfinance"
    rate: float  # mid price; must be > 0
    bid: Optional[float] = None  # optional bid; if present must be > 0
    ask: Optional[float] = None  # optional ask; if present must be > 0 and >= bid
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )  # UTC timestamp
    notes: List[str] = field(default_factory=list)

    def validate(self) -> None:
        if self.rate is None or self.rate <= 0:
            raise ValidationError(f"Invalid rate: {self.rate}")
        if self.bid is not None and self.bid <= 0:
            raise ValidationError(f"Invalid bid: {self.bid}")
        if self.ask is not None and self.ask <= 0:
            raise ValidationError(f"Invalid ask: {self.ask}")
        if self.bid is not None and self.ask is not None and self.bid > self.ask:
            raise ValidationError(
                f"Bid must be <= ask (bid={self.bid}, ask={self.ask})"
            )


class BaseProvider(ABC):
    """Abstract base class for data providers."""

    NAME: str = "base"

    @abstractmethod
    async def get_rate(self, base: str, quote: str) -> ProviderRate:
        """Fetch a normalized rate for a currency pair."""

    @abstractmethod
    async def health_check(self) -> bool:
        """Return True when the upstream service looks reachable/healthy."""

    @staticmethod
    def validate_currency_code(code: str) -> None:
        """Validate a 3-letter ISO currency code in uppercase."""
        if not code or len(code) != 3 or not code.isalpha() or code != code.upper():
            raise ValidationError(
                f"Invalid currency code: {code}. Expect 3-letter uppercase ISO code."
            )

    @classmethod
    def validate_pair(cls, base: str, quote: str) -> None:
        """Validate base/quote are distinct valid ISO codes."""
        cls.validate_currency_code(base)
        cls.validate_currency_code(quote)
        if base == quote:
            raise ValidationError("Base and quote currencies cannot be the same")

