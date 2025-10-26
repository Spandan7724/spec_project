"""Provider factory and exports."""

from .base import BaseProvider, ProviderRate
from .exchange_rate_host import ExchangeRateHostClient
from .yfinance_client import YFinanceClient


def get_provider(provider_name: str) -> BaseProvider:
    """Get provider by canonical name.

    Canonical names:
    - "exchange_rate_host"
    - "yfinance"
    """
    if provider_name == "exchange_rate_host":
        return ExchangeRateHostClient()
    if provider_name == "yfinance":
        return YFinanceClient()
    raise ValueError(f"Unknown provider: {provider_name}")


__all__ = [
    "BaseProvider",
    "ProviderRate",
    "ExchangeRateHostClient",
    "YFinanceClient",
    "get_provider",
]

