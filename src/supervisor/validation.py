"""Validation and normalization helpers for Supervisor NLU."""
from __future__ import annotations

import logging
import re
from typing import Dict, Optional, Set, Tuple

try:
    from thefuzz import process, fuzz
    FUZZY_AVAILABLE = True
except ImportError:
    FUZZY_AVAILABLE = False


logger = logging.getLogger(__name__)


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


# Currency names to codes - comprehensive mapping
CURRENCY_NAMES: Dict[str, str] = {
    # USD - US Dollar
    "dollar": "USD",
    "dollars": "USD",
    "usd": "USD",
    "us dollar": "USD",
    "us dollars": "USD",
    "american dollar": "USD",
    "american dollars": "USD",
    "united states dollar": "USD",
    "greenback": "USD",
    "buck": "USD",
    "bucks": "USD",
    "$": "USD",

    # EUR - Euro
    "euro": "EUR",
    "euros": "EUR",
    "eur": "EUR",
    "european euro": "EUR",
    "€": "EUR",

    # GBP - British Pound
    "pound": "GBP",
    "pounds": "GBP",
    "gbp": "GBP",
    "pound sterling": "GBP",
    "sterling": "GBP",
    "british pound": "GBP",
    "british pounds": "GBP",
    "uk pound": "GBP",
    "quid": "GBP",
    "£": "GBP",

    # JPY - Japanese Yen
    "yen": "JPY",
    "jpy": "JPY",
    "japanese yen": "JPY",
    "¥": "JPY",

    # CHF - Swiss Franc
    "franc": "CHF",
    "francs": "CHF",
    "chf": "CHF",
    "swiss franc": "CHF",
    "swiss francs": "CHF",

    # CAD - Canadian Dollar
    "cad": "CAD",
    "canadian dollar": "CAD",
    "canadian dollars": "CAD",
    "loonie": "CAD",
    "loonies": "CAD",

    # AUD - Australian Dollar
    "aud": "AUD",
    "australian dollar": "AUD",
    "australian dollars": "AUD",
    "aussie dollar": "AUD",
    "aussie dollars": "AUD",

    # NZD - New Zealand Dollar
    "nzd": "NZD",
    "new zealand dollar": "NZD",
    "new zealand dollars": "NZD",
    "kiwi": "NZD",
    "kiwi dollar": "NZD",

    # CNY - Chinese Yuan
    "yuan": "CNY",
    "cny": "CNY",
    "renminbi": "CNY",
    "rmb": "CNY",
    "chinese yuan": "CNY",
    "chinese renminbi": "CNY",

    # INR - Indian Rupee
    "rupee": "INR",
    "rupees": "INR",
    "inr": "INR",
    "indian rupee": "INR",
    "indian rupees": "INR",
    "₹": "INR",

    # BRL - Brazilian Real
    "brl": "BRL",
    "real": "BRL",
    "reais": "BRL",
    "brazilian real": "BRL",
    "brazilian reais": "BRL",
    "r$": "BRL",

    # RUB - Russian Ruble
    "rub": "RUB",
    "ruble": "RUB",
    "rubles": "RUB",
    "rouble": "RUB",
    "roubles": "RUB",
    "russian ruble": "RUB",
    "russian rubles": "RUB",
    "₽": "RUB",

    # MXN - Mexican Peso
    "mxn": "MXN",
    "peso": "MXN",
    "pesos": "MXN",
    "mexican peso": "MXN",
    "mexican pesos": "MXN",

    # SGD - Singapore Dollar
    "sgd": "SGD",
    "singapore dollar": "SGD",
    "singapore dollars": "SGD",

    # HKD - Hong Kong Dollar
    "hkd": "HKD",
    "hong kong dollar": "HKD",
    "hong kong dollars": "HKD",
    "hk dollar": "HKD",
    "hk dollars": "HKD",

    # SEK - Swedish Krona
    "sek": "SEK",
    "krona": "SEK",
    "kronor": "SEK",
    "swedish krona": "SEK",
    "swedish kronor": "SEK",

    # NOK - Norwegian Krone
    "nok": "NOK",
    "krone": "NOK",
    "kroner": "NOK",
    "norwegian krone": "NOK",
    "norwegian kroner": "NOK",

    # DKK - Danish Krone
    "dkk": "DKK",
    "danish krone": "DKK",
    "danish kroner": "DKK",

    # PLN - Polish Zloty
    "pln": "PLN",
    "zloty": "PLN",
    "zlotys": "PLN",
    "polish zloty": "PLN",

    # TRY - Turkish Lira
    "try": "TRY",
    "lira": "TRY",
    "liras": "TRY",
    "turkish lira": "TRY",
    "turkish liras": "TRY",
    "₺": "TRY",

    # ZAR - South African Rand
    "zar": "ZAR",
    "rand": "ZAR",
    "south african rand": "ZAR",

    # KRW - South Korean Won
    "krw": "KRW",
    "won": "KRW",
    "korean won": "KRW",
    "south korean won": "KRW",
    "₩": "KRW",

    # THB - Thai Baht
    "thb": "THB",
    "baht": "THB",
    "thai baht": "THB",
    "฿": "THB",

    # IDR - Indonesian Rupiah
    "idr": "IDR",
    "rupiah": "IDR",
    "indonesian rupiah": "IDR",

    # Common country name shortcuts
    "america": "USD",
    "us": "USD",
    "usa": "USD",
    "europe": "EUR",
    "eu": "EUR",
    "britain": "GBP",
    "uk": "GBP",
    "england": "GBP",
    "japan": "JPY",
    "switzerland": "CHF",
    "canada": "CAD",
    "australia": "AUD",
    "china": "CNY",
    "india": "INR",
    "brazil": "BRL",
    "russia": "RUB",
    "mexico": "MXN",
    "singapore": "SGD",
}


RISK_LEVELS = {"conservative", "moderate", "aggressive"}
URGENCY_LEVELS = {"urgent", "normal", "flexible"}
TIMEFRAME_LEVELS = {"immediate", "1_day", "1_week", "1_month"}


def _normalize_string(s: str) -> str:
    """Normalize string for matching: remove extra spaces, punctuation, Unicode variations."""
    # Remove extra whitespace
    s = re.sub(r'\s+', ' ', s.strip())
    # Remove common prefixes
    for prefix in ['the ', 'a ', 'an ']:
        if s.startswith(prefix):
            s = s[len(prefix):]
    return s


def _fuzzy_match_currency(query: str, threshold: int = 85) -> Optional[str]:
    """Use fuzzy matching to find best currency match.

    Args:
        query: The string to match
        threshold: Minimum similarity score (0-100)

    Returns:
        ISO currency code if match found above threshold, None otherwise
    """
    if not FUZZY_AVAILABLE:
        return None

    # Build list of all possible strings (both codes and names)
    all_currency_strings = list(CURRENCY_NAMES.keys()) + list(CURRENCY_CODES)

    # Find best match
    try:
        result = process.extractOne(query, all_currency_strings, scorer=fuzz.ratio)
        if result and result[1] >= threshold:
            match_str = result[0]
            # If matched a name, return its code
            if match_str in CURRENCY_NAMES:
                code = CURRENCY_NAMES[match_str]
                logger.info(f"Fuzzy matched '{query}' to '{match_str}' → {code} (score: {result[1]})")
                return code
            # If matched a code directly
            if match_str in CURRENCY_CODES:
                logger.info(f"Fuzzy matched '{query}' to code {match_str} (score: {result[1]})")
                return match_str
    except Exception as e:
        logger.warning(f"Fuzzy matching failed for '{query}': {e}")

    return None


def normalize_currency_code(code: Optional[str], fuzzy: bool = True, log_failures: bool = True) -> Optional[str]:
    """Normalize currency code or name to ISO 4217 3-letter code using multi-strategy approach.

    Tries strategies in order until a match is found:
    1. Exact name match (after normalization)
    2. Exact ISO code match
    3. Fuzzy name/code matching (if enabled)

    Args:
        code: Currency code, name, symbol, or country (case-insensitive)
        fuzzy: Enable fuzzy matching for typos (default: True)
        log_failures: Log when no match found (default: True)

    Returns:
        Uppercase 3-letter ISO code if valid, None otherwise

    Examples:
        normalize_currency_code("USD") -> "USD"
        normalize_currency_code("euro") -> "EUR"
        normalize_currency_code("dollars") -> "USD"
        normalize_currency_code("€") -> "EUR"
        normalize_currency_code("america") -> "USD"
        normalize_currency_code("eruo")  # typo -> "EUR" (with fuzzy matching)
        normalize_currency_code("quid") -> "GBP"
        normalize_currency_code("greenback") -> "USD"
    """
    if not code:
        return None

    # Normalize the input string
    original = code
    code_normalized = _normalize_string(code.lower())

    # Strategy 1: Exact name match (highest priority after normalization)
    if code_normalized in CURRENCY_NAMES:
        return CURRENCY_NAMES[code_normalized]

    # Strategy 2: Exact ISO code match
    code_upper = code_normalized.upper()
    if code_upper in CURRENCY_CODES:
        return code_upper

    # Strategy 3: Fuzzy matching for typos and close matches
    if fuzzy:
        fuzzy_result = _fuzzy_match_currency(code_normalized, threshold=80)
        if fuzzy_result:
            return fuzzy_result

    # No match found - log for improvement
    if log_failures:
        logger.warning(
            f"Failed to normalize currency: '{original}' (normalized: '{code_normalized}'). "
            f"Not found in {len(CURRENCY_NAMES)} names or {len(CURRENCY_CODES)} codes. "
            f"Consider adding this variation to CURRENCY_NAMES."
        )

    return None


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

