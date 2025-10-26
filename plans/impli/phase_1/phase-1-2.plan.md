<!-- f67714c1-a54f-4e8d-9617-16955f212afc 503d9606-2e9c-4b08-90ae-56c8498468fe -->
# Phase 1.2: Market Data - Aggregation & Indicators

## Overview

Build the aggregation layer that combines rates from multiple providers (Phase 1.1), computes technical indicators, classifies market regime, and produces the complete LiveSnapshot schema defined in market-data-agent-plan.md. This phase implements the core Market Data Agent functionality.

## Implementation Steps

### Step 1: Rate Aggregator

**File**: `src/data_collection/market_data/aggregator.py`

Create the rate aggregation logic that combines multiple provider rates:

**Core Functions**:

- `aggregate_rates(provider_rates: List[ProviderRate]) -> AggregatedRate`
- Consensus calculation: Use **median** of provider rates (not mean)
- If bid/ask available: `mid_rate = median((bid + ask) / 2 for each provider)`
- Otherwise: `mid_rate = median(rate for each provider)`
- Calculate dispersion in basis points: `((max - min) / min) * 10000`
- Determine freshness: Check if latest timestamp is within cache TTL (5 seconds from config)
 - Select best bid (highest) and best ask (lowest) across providers
 - Calculate spread: `spread = best_ask - best_bid` (will be >= 0 when bids/asks are present)

**Quality Metrics Schema**:

```python
@dataclass
class QualityMetrics:
    sources_success: int  # Number of providers that returned data
    sources_total: int    # Total providers attempted
    dispersion_bps: float # Rate dispersion in basis points
    fresh: bool           # True if within cache TTL
    notes: List[str]      # Warnings like "High dispersion", "Stale data"
```

**Validation Rules** (from market-data-agent-plan.md):

- `sources_success >= 1` (at least one provider must succeed)
- Discard outliers: Remove rates > X bps from median (configurable, default 100 bps)
- `rate_timestamp = max(provider timestamps)` (most recent)
- Add warning if `dispersion_bps > 50` (configurable threshold)
- Mark `fresh = False` if timestamp age > cache TTL

### Step 2: Technical Indicators

**File**: `src/data_collection/market_data/indicators.py`

Implement technical indicator calculations using historical OHLC data:

**Required Indicators** (from market-data-agent-plan.md):

- SMA (Simple Moving Average): 20-day, 50-day
- EMA (Exponential Moving Average): 12-day, 26-day
- RSI (Relative Strength Index): 14-day (range: 0-100)
- MACD: signal line, histogram
- Bollinger Bands: middle (20-day SMA), upper, lower, position (0-1)
- ATR (Average True Range): 14-day
- Realized Volatility: 30-day

**Implementation**:

```python
@dataclass
class Indicators:
    sma_20: Optional[float]
    sma_50: Optional[float]
    ema_12: Optional[float]
    ema_26: Optional[float]
    rsi_14: Optional[float]  # Must be in [0, 100]
    macd: Optional[float]
    macd_signal: Optional[float]
    macd_histogram: Optional[float]
    bb_middle: Optional[float]
    bb_upper: Optional[float]
    bb_lower: Optional[float]
    bb_position: Optional[float]  # Must be in [0, 1]
    atr_14: Optional[float]
    realized_vol_30d: Optional[float]

def calculate_indicators(
    historical_data: pd.DataFrame,
    min_history_days: int = 50
) -> Indicators:
    """Calculate all technical indicators from OHLC data."""
    ...
```

**Data Requirements**:

- Fetch historical OHLC from yfinance (already in Phase 1.1)
- Minimum 50 days history required (configurable via `INDICATOR_MIN_HISTORY_DAYS`)
- If insufficient data: Return indicators with all fields as `None`, add warning
- Validate: RSI in [0, 100], bb_position in [0, 1]

**Libraries**: Use pandas for data manipulation, numpy for calculations (avoid TA-Lib for simplicity)

### Step 3: Regime Classifier

**File**: `src/data_collection/market_data/regime.py`

Classify market regime based on technical indicators:

**Regime Schema** (from market-data-agent-plan.md):

```python
@dataclass
class Regime:
    trend_direction: Optional[str]  # "up", "down", "sideways", None
    bias: Optional[str]              # "bullish", "bearish", "neutral", None
```

**Classification Logic**:

- **Trend Direction**:
  - `"up"`: price > SMA_50 and SMA_20 > SMA_50
  - `"down"`: price < SMA_50 and SMA_20 < SMA_50
  - `"sideways"`: Otherwise or insufficient data

- **Bias** (momentum):
  - `"bullish"`: RSI > 60 and MACD > 0
  - `"bearish"`: RSI < 40 and MACD < 0
  - `"neutral"`: Otherwise

- Return `None` for fields if indicators unavailable

### Step 4: Snapshot Builder

**File**: `src/data_collection/market_data/snapshot.py`

Build the complete LiveSnapshot combining all components:

**LiveSnapshot Schema** (exact match to market-data-agent-plan.md):

```python
@dataclass
class LiveSnapshot:
    currency_pair: str          # Format: "USD/EUR"
    rate_timestamp: datetime    # Max of provider timestamps (UTC)
    mid_rate: float             # Median consensus rate
    bid: Optional[float]        # Best bid across providers
    ask: Optional[float]        # Best ask across providers
    spread: Optional[float]     # ask - bid
    
    provider_breakdown: List[ProviderRate]  # All provider rates
    quality: QualityMetrics     # Data quality info
    indicators: Indicators      # Technical indicators
    regime: Regime              # Market regime classification

async def build_snapshot(
    base: str,
    quote: str,
    providers: List[BaseProvider],
    cache: SimpleCache,
    lookback_days: int = 90
) -> LiveSnapshot:
    """
    Build complete market snapshot.
    
    Args:
        base: Base currency (e.g., "USD")
        quote: Quote currency (e.g., "EUR")
        providers: List of provider instances
        cache: Cache instance for storing snapshots
        lookback_days: Historical data lookback for indicators
    
    Returns:
        Complete LiveSnapshot
    """
    ...
```

**Implementation Flow**:

1. Validate currency pair format: `^[A-Z]{3}/[A-Z]{3}$`
2. Check cache for recent snapshot (TTL: 5s from config)
3. Fetch rates from all providers in parallel (already async)
4. Aggregate rates using median consensus
5. Fetch historical OHLC data from yfinance
6. Calculate technical indicators
7. Classify market regime
8. Build LiveSnapshot with all components
9. Cache snapshot with TTL
10. Return snapshot

**Error Handling**:

- If all providers fail: Return cached snapshot with `fresh=False` + warning
- If historical fetch fails: Return snapshot with `indicators=None` + warning
- Never crash: Always return best available data

### Step 5: Integration with Cache

**File**: `src/data_collection/market_data/cache.py` (optional wrapper)

**Note**: We already have `src/cache.py` from Phase 0.2. This step is about integrating it properly:

- Use existing `SimpleCache` from `src/cache.py`
- Cache key format: `"market_snapshot:{currency_pair}"` (e.g., `"market_snapshot:USD/EUR"`)
- TTL: 5 seconds (from `config.yaml`: `agents.market_data.cache_ttl`)
- Cache invalidation: Automatic via TTL
- Cache bypass: Add parameter for forced refresh

**Optional**: Create a thin wrapper if needed for market data specific caching logic.

### Step 6: Unit Tests

**Files**:

- `tests/unit/test_market_data/__init__.py`
- `tests/unit/test_market_data/test_aggregator.py`
- `tests/unit/test_market_data/test_indicators.py`
- `tests/unit/test_market_data/test_regime.py`
- `tests/unit/test_market_data/test_snapshot.py`

**Test Coverage**:

- **Aggregator**: Test median calculation, outlier removal, dispersion calculation, quality metrics
- **Indicators**: Test each indicator with known data, handle insufficient history
- **Regime**: Test classification logic for all combinations
- **Snapshot**: Test complete flow with mocked providers and historical data
- **Caching**: Test cache hit/miss, TTL expiration, staleness handling

**Mock Data**: Create fixtures for provider rates and historical OHLC

**Coverage Target**: > 80%

### Step 7: Integration Tests

**File**: `tests/integration/test_market_data_integration.py`

Test the complete market data pipeline:

- Fetch live data from real providers (Phase 1.1)
- Build complete LiveSnapshot
- Verify all fields populated correctly
- Test caching behavior
- Test error scenarios (provider failures, network issues)
- Validate output against schema

**Use pytest markers**: `@pytest.mark.integration` to skip in CI if needed

## Key Design Decisions

1. **Median consensus**: More robust than mean, less affected by outliers
2. **Outlier removal**: Discard rates > 100 bps from median (configurable)
3. **Parallel provider fetching**: Use asyncio.gather for speed
4. **Graceful degradation**: Return partial data if some components fail
5. **Schema compliance**: Exact match to market-data-agent-plan.md
6. **Caching strategy**: 5-second TTL balances freshness vs. API costs
7. **UTC timestamps**: All timestamps must be UTC ISO-8601
8. **Validation**: All numeric constraints enforced (RSI [0,100], bb_position [0,1], rates > 0)

## Files to Create

- `src/data_collection/market_data/__init__.py`
- `src/data_collection/market_data/aggregator.py`
- `src/data_collection/market_data/indicators.py`
- `src/data_collection/market_data/regime.py`
- `src/data_collection/market_data/snapshot.py`
- `src/data_collection/market_data/cache.py` (optional wrapper)
- `tests/unit/test_market_data/__init__.py`
- `tests/unit/test_market_data/test_aggregator.py`
- `tests/unit/test_market_data/test_indicators.py`
- `tests/unit/test_market_data/test_regime.py`
- `tests/unit/test_market_data/test_snapshot.py`
- `tests/integration/test_market_data_integration.py`

## Dependencies

- Providers from Phase 1.1: `src/data_collection/providers`
- Cache from Phase 0.2: `src/cache.py`
- Config from Phase 0.1: `src/config.py`
- Utilities from Phase 0: errors, logging, validation, decorators
- Libraries: pandas, numpy (already in pyproject.toml)
- yfinance: For historical OHLC data (already in Phase 1.1)

## Configuration (Already in config.yaml)

```yaml
agents:
  market_data:
    cache_ttl: 5  # seconds
    providers: ["exchange_rate_host", "yfinance"]
```

**Additional env vars** (optional):

- `HISTORICAL_LOOKBACK_DAYS`: Default 90
- `INDICATOR_MIN_HISTORY_DAYS`: Default 50
- `OUTLIER_THRESHOLD_BPS`: Default 100

## Validation

Run tests and manual validation:

```python
from src.data_collection.market_data import get_market_snapshot
from src.cache import cache

# Build complete snapshot
snapshot = await get_market_snapshot("USD", "EUR")

# Verify schema
assert snapshot.currency_pair == "USD/EUR"
assert snapshot.mid_rate > 0
assert snapshot.quality.sources_success >= 1
assert len(snapshot.provider_breakdown) >= 1

# Check indicators
if snapshot.indicators.rsi_14 is not None:
    assert 0 <= snapshot.indicators.rsi_14 <= 100
    
if snapshot.indicators.bb_position is not None:
    assert 0 <= snapshot.indicators.bb_position <= 1

# Check regime
print(f"Trend: {snapshot.regime.trend_direction}, Bias: {snapshot.regime.bias}")

# Test caching
snapshot2 = await get_market_snapshot("USD", "EUR")
assert snapshot.rate_timestamp == snapshot2.rate_timestamp  # Cache hit
```

## Success Criteria

- LiveSnapshot schema exactly matches market-data-agent-plan.md
- Median aggregation works correctly with multiple providers
- Outlier detection removes bad data appropriately
- All technical indicators calculated correctly
- Regime classification logic works as specified
- Quality metrics accurately reflect data quality
- Caching reduces redundant API calls (>80% cache hit rate for repeated queries)
- All unit tests pass (>80% coverage)
- Integration test successfully builds snapshot with real data
- Error handling works for all failure modes
- Code follows patterns from Phase 0
- Timestamps are UTC ISO-8601 format
- Validation enforces all constraints (RSI, bb_position, rates > 0)

## Next Phase

After Phase 1.2 completes, proceed to **Phase 1.3: Market Data LangGraph Node**, which will:

- Create the Market Data agent node for LangGraph
- Integrate snapshot builder with LangGraph state
- Add error handling and fallbacks
- Add performance logging and metrics
- Write node integration tests

### To-dos

- [ ] Create rate aggregator with median consensus, outlier removal, and quality metrics calculation
- [ ] Implement all technical indicators (SMA, EMA, RSI, MACD, Bollinger, ATR, volatility)
- [ ] Implement market regime classification logic (trend direction and bias)
- [ ] Build complete LiveSnapshot combining aggregation, indicators, and regime with caching
- [ ] Write comprehensive unit tests for aggregator, indicators, regime, and snapshot (>80% coverage)
- [ ] Write integration tests for complete market data pipeline with real providers
