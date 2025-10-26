"""Validation and normalization helpers for Supervisor NLU."""
from __future__ import annotations

from typing import Dict, Optional, Set, Tuple


# Common currency codes
CURRENCY_CODES: Set[str] = {
    "USD",
    "EUR",
    "GBP",
    "JPY",
    "CHF",
    "CAD",
    "AUD",
    "NZD",
    "CNY",
    "INR",
    "BRL",
    "RUB",
    "MXN",
    "SGD",
    "HKD",
    "SEK",
    "NOK",
    "DKK",
    "PLN",
    "TRY",
    "ZAR",
    "KRW",
    "THB",
    "IDR",
}


# Currency names to codes
CURRENCY_NAMES: Dict[str, str] = {
    "dollar": "USD",
    "dollars": "USD",
    "usd": "USD",
    "euro": "EUR",
    "euros": "EUR",
    "eur": "EUR",
    "pound": "GBP",
    "pounds": "GBP",
    "gbp": "GBP",
    "sterling": "GBP",
    "yen": "JPY",
    "jpy": "JPY",
    "franc": "CHF",
    "chf": "CHF",
    "swiss franc": "CHF",
    "canadian dollar": "CAD",
    "cad": "CAD",
    "australian dollar": "AUD",
    "aud": "AUD",
    "yuan": "CNY",
    "renminbi": "CNY",
    "cny": "CNY",
    "rupee": "INR",
    "rupees": "INR",
    "inr": "INR",
}


RISK_LEVELS = {"conservative", "moderate", "aggressive"}
URGENCY_LEVELS = {"urgent", "normal", "flexible"}
TIMEFRAME_LEVELS = {"immediate", "1_day", "1_week", "1_month"}


def normalize_currency_code(code: Optional[str]) -> Optional[str]:
    if not code:
        return None
    code = code.strip().upper()
    return code if code in CURRENCY_CODES else None


def is_valid_currency_code(code: Optional[str]) -> bool:
    return normalize_currency_code(code) is not None


def validate_currency_pair(base: Optional[str], quote: Optional[str]) -> Tuple[bool, Optional[str]]:
    if not base or not quote:
        return False, "Missing base or quote currency"
    base_n = normalize_currency_code(base)
    quote_n = normalize_currency_code(quote)
    if not base_n:
        return False, f"Invalid base currency: {base}"
    if not quote_n:
        return False, f"Invalid quote currency: {quote}"
    if base_n == quote_n:
        return False, f"Base and quote currencies cannot be the same: {base_n}"
    return True, None


def timeframe_to_days(timeframe: Optional[str]) -> Optional[int]:
    if not timeframe:
        return None
    mapping = {
        "immediate": 1,
        "1_day": 1,
        "1_week": 7,
        "1_month": 30,
    }
    return mapping.get(timeframe)

