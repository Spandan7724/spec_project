<!-- f67714c1-a54f-4e8d-9617-16955f212afc 839135ff-6c7b-4385-bedd-b8df4e06d972 -->
# Phase 1.1: Market Data - Provider Clients

## Overview

Create the provider abstraction layer and implement two currency rate data sources. This phase focuses on reliable data fetching with proper error handling, retries, and health checks. **All schemas must match market-data-agent-plan.md exactly.**

## Implementation Steps

### Step 1: Base Provider Interface

**File**: `src/data_collection/providers/base.py`

Create an abstract base class that defines the contract for all data providers:

- Abstract method `get_rate(base: str, quote: str) -> ProviderRate`
- Abstract method `health_check() -> bool`
- Currency validation: Must match pattern `^[A-Z]{3}$`, reject if base == quote
- Error handling helpers
- Rate limiting support (using decorators from Phase 0.4)

**ProviderRate Schema** (from market-data-agent-plan.md):

```python
@dataclass
class ProviderRate:
    source: str  # "exchange_rate_host" or "yfinance"
    rate: float  # Must be > 0
    bid: Optional[float]  # If present, must be > 0
    ask: Optional[float]  # If present, must be > 0 and >= bid
    timestamp: datetime  # UTC ISO-8601 format
    notes: List[str]  # Warnings or informational messages

class BaseProvider(ABC):
    @abstractmethod
    async def get_rate(self, base: str, quote: str) -> ProviderRate
    
    @abstractmethod
    async def health_check(self) -> bool
    
    def validate_currency_code(self, code: str) -> bool:
        """Validate 3-letter uppercase currency code"""
        ...
```

### Step 2: ExchangeRate.host Client

**File**: `src/data_collection/providers/exchange_rate_host.py`

Implement client for ExchangeRate.host API following architecture specs:

- Use `httpx.AsyncClient` for requests
- Endpoint: `https://api.exchangerate.host/latest?base={base}&symbols={quote}` (from config.yaml)
- Apply `@retry(max_attempts=3, delay=1.0)` and `@timeout(10.0)` decorators
- Parse response and convert to `ProviderRate` with canonical source name: `"exchange_rate_host"`
- Handle API errors gracefully (network, rate limits, invalid responses)
- Use timeout from `config.yaml`: `api.exchange_rate_host.timeout` (10 seconds)
- Optional: Support `EXCHANGE_RATE_HOST_API_KEY` from environment
- Return UTC ISO-8601 timestamp (append 'Z' or '+00:00')
- Validate rate > 0 before returning

### Step 3: yfinance Client

**File**: `src/data_collection/providers/yfinance_client.py`

Implement client for yfinance library:

- Use yfinance's `Ticker` class
- Convert currency pair format: "USD/EUR" -> "USDEUR=X" for yfinance
- Fetch current price with bid/ask if available (yfinance provides these)
- Apply `@retry` and `@timeout` decorators for resilience
- Handle yfinance-specific errors (no data, delisted pairs, network timeouts)
- Source name: `"yfinance"` (canonical from market-data-agent-plan.md)
- Return bid/ask when available (yfinance provides this for forex)
- UTC timestamps required

**Important**: yfinance returns bid/ask for forex pairs, which is valuable for spread calculation in aggregation layer.

### Step 4: Provider Factory & Init

**File**: `src/data_collection/providers/__init__.py`

Create factory for instantiating providers and exports:

```python
def get_provider(provider_name: str) -> BaseProvider:
    """Get provider by canonical name."""
    if provider_name == "exchange_rate_host":
        return ExchangeRateHostClient()
    elif provider_name == "yfinance":
        return YFinanceClient()
    raise ValueError(f"Unknown provider: {provider_name}")

# Export main classes
__all__ = [
    "BaseProvider",
    "ProviderRate", 
    "ExchangeRateHostClient",
    "YFinanceClient",
    "get_provider"
]
```

### Step 5: Unit Tests

**Files**:

- `tests/unit/test_providers/__init__.py`
- `tests/unit/test_providers/test_exchange_rate_host.py`
- `tests/unit/test_providers/test_yfinance.py`

Test each provider with:

- **Mock HTTP responses** (using `pytest-httpx` or mock library)
- Valid rate responses with proper schema
- Currency code validation (reject invalid codes, identical base/quote)
- Error scenarios (network errors, invalid JSON, rate limits, timeouts)
- Validation of `ProviderRate` output (rate > 0, bid <= ask if present)
- Health check functionality
- Retry behavior (should retry on transient failures)
- Timestamp format validation (UTC ISO-8601)

**Test Coverage Requirements**: > 80% per Phase 0 standards

### Step 6: Integration Tests (Optional)

**File**: `tests/integration/test_providers_integration.py`

Test providers with real API calls (can be skipped in CI using pytest markers):

- Fetch real rates for common pairs (USD/EUR, GBP/USD, EUR/GBP)
- Verify data quality and timestamps
- Test provider health checks with live APIs
- Validate rate dispersion is within reasonable bounds

## Key Design Decisions

1. **Async by default**: All provider methods are async to support parallel fetching in Layer 1
2. **Decorator-based resilience**: Use retry/timeout decorators from Phase 0.4
3. **Canonical naming**: Source names MUST match market-data-agent-plan.md: `"exchange_rate_host"`, `"yfinance"`
4. **Schema compliance**: ProviderRate must exactly match the schema in market-data-agent-plan.md
5. **Configuration-driven**: Read API endpoints and timeouts from `config.yaml` (already configured)
6. **Graceful degradation**: Providers should never crash the system, always return errors gracefully
7. **UTC timestamps**: All timestamps must be UTC ISO-8601 format
8. **Validation**: Currency codes must be 3-letter uppercase, base != quote

## Files to Create

- `src/data_collection/providers/base.py` (abstract interface + ProviderRate)
- `src/data_collection/providers/exchange_rate_host.py` (client)
- `src/data_collection/providers/yfinance_client.py` (client)
- `src/data_collection/providers/__init__.py` (factory + exports)
- `tests/unit/test_providers/__init__.py`
- `tests/unit/test_providers/test_exchange_rate_host.py`
- `tests/unit/test_providers/test_yfinance.py`
- `tests/integration/test_providers_integration.py` (optional, with pytest.mark.integration)

## Dependencies

- `httpx` (already in pyproject.toml) - for HTTP requests
- `yfinance` (already in pyproject.toml) - for forex data
- Decorators from `src/utils/decorators.py` - retry, timeout, log_execution
- Config from `src/config.py` - already has api.exchange_rate_host settings
- Error classes from `src/utils/errors.py` - DataProviderError, etc.
- Logging from `src/utils/logging.py` - structured logging

## Configuration (Already in config.yaml)

```yaml
api:
  exchange_rate_host:
    base_url: "https://api.exchangerate.host"
    timeout: 10

agents:
  market_data:
    providers: ["exchange_rate_host", "yfinance"]
    cache_ttl: 5  # seconds
```

## Validation

Run tests and manual validation:

```python
from src.data_collection.providers import get_provider

# Test ExchangeRate.host
client = get_provider("exchange_rate_host")
rate = await client.get_rate("USD", "EUR")
print(f"Rate: {rate.rate}, Timestamp: {rate.timestamp}")
assert rate.source == "exchange_rate_host"
assert rate.rate > 0

# Test yfinance
yfin_client = get_provider("yfinance")
rate = await yfin_client.get_rate("USD", "EUR")
print(f"Rate: {rate.rate}, Bid: {rate.bid}, Ask: {rate.ask}")
assert rate.source == "yfinance"
if rate.bid and rate.ask:
    assert rate.bid <= rate.ask
```

## Success Criteria

- Both providers can fetch rates for major currency pairs (USD/EUR, GBP/USD, EUR/JPY)
- All unit tests pass with mocked responses (>80% coverage)
- ProviderRate schema matches market-data-agent-plan.md exactly
- Error handling works correctly (network failures, invalid data, timeouts)
- Retry logic functions as expected (3 attempts with exponential backoff)
- Health checks return accurate status
- Currency validation rejects invalid codes
- Timestamps are UTC ISO-8601 format
- Code follows existing patterns from Phase 0
- Canonical source names are used consistently

## Next Phase

After Phase 1.1 completes, proceed to Phase 1.2: Market Data Aggregation & Indicators, which will:

- Aggregate rates from multiple providers using median consensus
- Calculate dispersion in basis points
- Compute technical indicators (RSI, MACD, Bollinger Bands, etc.)
- Build the complete LiveSnapshot schema

### To-dos

- [ ] Create base provider interface with ProviderRate dataclass matching market-data-agent-plan.md schema and BaseProvider ABC with currency validation
- [ ] Implement ExchangeRate.host client with httpx, retry/timeout decorators, and canonical source name
- [ ] Implement yfinance client with proper currency pair conversion (USD/EUR -> USDEUR=X) and bid/ask support
- [ ] Create provider factory get_provider() and __init__.py exports
- [ ] Write unit tests for both providers with mocked responses, >80% coverage
- [ ] Write optional integration tests with real API calls using pytest markers