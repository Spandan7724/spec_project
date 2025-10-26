<!-- f67714c1-a54f-4e8d-9617-16955f212afc 9582d905-f858-4d5c-90f6-3e9ccf27a58d -->
# Phase 1.3: Market Data - LangGraph Node

## Overview

Integrate the Market Data snapshot builder (Phase 1.2) into the LangGraph workflow by creating an agent node that fits into the multi-agent system. This node will be executed in parallel with Market Intelligence in Layer 1 of the workflow.

## Implementation Steps

### Step 1: Market Data Node Implementation

**File**: `src/agentic/nodes/market_data.py`

Create the LangGraph node that integrates the snapshot builder:

**Core Functionality**:

```python
from typing import Dict, Any
from src.agentic.state import AgentState
from src.data_collection.market_data.snapshot import build_snapshot
from src.data_collection.providers import get_provider
from src.cache import cache
from src.config import get_config
from src.utils.logging import get_logger
from src.utils.decorators import log_execution, timeout
import time

logger = get_logger(__name__)

@timeout(10.0)  # 10 second timeout for Market Data agent
@log_execution(log_args=False, log_result=False)
async def market_data_node(state: AgentState) -> Dict[str, Any]:
    """
    Market Data agent node for LangGraph.
    
    Fetches live market snapshot with rates, indicators, and regime.
    Updates state with market data or error information.
    
    Args:
        state: Current agent state
    
    Returns:
        Dictionary with market_snapshot and status fields
    """
    start_time = time.time()
    correlation_id = state.get("correlation_id", "unknown")
    
    try:
        # Extract currency pair from state
        base = state.get("base_currency")
        quote = state.get("quote_currency")
        
        if not base or not quote:
            logger.warning(
                "Missing currency codes in state",
                extra={"correlation_id": correlation_id}
            )
            return {
                "market_data_status": "error",
                "market_data_error": "Missing base or quote currency",
                "market_snapshot": None
            }
        
        # Get configured providers
        config = get_config()
        provider_names = config.agents.market_data.providers  # ["exchange_rate_host", "yfinance"]
        providers = [get_provider(name) for name in provider_names]
        
        # Build snapshot
        logger.info(
            f"Fetching market snapshot for {base}/{quote}",
            extra={"correlation_id": correlation_id, "base": base, "quote": quote}
        )
        
        snapshot = await build_snapshot(
            base=base,
            quote=quote,
            providers=providers,
            cache=cache,
            lookback_days=90  # From config or default
        )
        
        # Calculate execution time
        execution_time_ms = int((time.time() - start_time) * 1000)
        
        # Convert snapshot to dict for state storage
        snapshot_dict = {
            "currency_pair": snapshot.currency_pair,
            "rate_timestamp": snapshot.rate_timestamp.isoformat(),
            "mid_rate": snapshot.mid_rate,
            "bid": snapshot.bid,
            "ask": snapshot.ask,
            "spread": snapshot.spread,
            "provider_breakdown": [
                {
                    "source": p.source,
                    "rate": p.rate,
                    "bid": p.bid,
                    "ask": p.ask,
                    "timestamp": p.timestamp.isoformat(),
                    "notes": p.notes
                }
                for p in snapshot.provider_breakdown
            ],
            "quality": {
                "sources_success": snapshot.quality.sources_success,
                "sources_total": snapshot.quality.sources_total,
                "dispersion_bps": snapshot.quality.dispersion_bps,
                "fresh": snapshot.quality.fresh,
                "notes": snapshot.quality.notes
            },
            "indicators": {
                "sma_20": snapshot.indicators.sma_20,
                "sma_50": snapshot.indicators.sma_50,
                "ema_12": snapshot.indicators.ema_12,
                "ema_26": snapshot.indicators.ema_26,
                "rsi_14": snapshot.indicators.rsi_14,
                "macd": snapshot.indicators.macd,
                "macd_signal": snapshot.indicators.macd_signal,
                "macd_histogram": snapshot.indicators.macd_histogram,
                "bb_middle": snapshot.indicators.bb_middle,
                "bb_upper": snapshot.indicators.bb_upper,
                "bb_lower": snapshot.indicators.bb_lower,
                "bb_position": snapshot.indicators.bb_position,
                "atr_14": snapshot.indicators.atr_14,
                "realized_vol_30d": snapshot.indicators.realized_vol_30d
            },
            "regime": {
                "trend_direction": snapshot.regime.trend_direction,
                "bias": snapshot.regime.bias
            }
        }
        
        logger.info(
            f"Market snapshot fetched successfully",
            extra={
                "correlation_id": correlation_id,
                "mid_rate": snapshot.mid_rate,
                "sources_success": snapshot.quality.sources_success,
                "execution_time_ms": execution_time_ms
            }
        )
        
        # Return state updates (only fields this node is responsible for)
        return {
            "market_snapshot": snapshot_dict,
            "market_data_status": "success",
            "market_data_error": None
        }
        
    except TimeoutError as e:
        execution_time_ms = int((time.time() - start_time) * 1000)
        logger.error(
            "Market Data agent timed out",
            extra={"correlation_id": correlation_id, "execution_time_ms": execution_time_ms}
        )
        return {
            "market_data_status": "error",
            "market_data_error": f"Timeout after 10 seconds: {str(e)}",
            "market_snapshot": None
        }
        
    except Exception as e:
        execution_time_ms = int((time.time() - start_time) * 1000)
        logger.error(
            f"Market Data agent failed: {str(e)}",
            extra={"correlation_id": correlation_id, "error": str(e), "execution_time_ms": execution_time_ms}
        )
        
        # Try to return cached data as fallback
        cache_key = f"market_snapshot:{base}/{quote}"
        cached_snapshot = cache.get(cache_key)
        
        if cached_snapshot:
            logger.info(
                "Returning cached snapshot as fallback",
                extra={"correlation_id": correlation_id}
            )
            return {
                "market_data_status": "partial",
                "market_data_error": f"Live fetch failed, using cache: {str(e)}",
                "market_snapshot": cached_snapshot
            }
        
        return {
            "market_data_status": "error",
            "market_data_error": str(e),
            "market_snapshot": None
        }
```

**Key Features**:

- Integrates with LangGraph state from Phase 0.3
- Uses timeout decorator (10s per IMPLEMENTATION_ROADMAP.md)
- Proper error handling with fallback to cached data
- Structured logging with correlation IDs
- Performance tracking (execution time)
- Returns only the fields this node updates (for parallel execution)

### Step 2: Update Graph Definition

**File**: `src/agentic/graph.py`

Replace the placeholder `market_data_node` with the real implementation:

```python
from src.agentic.nodes.market_data import market_data_node

# In create_graph():
workflow.add_node("market_data", market_data_node)
```

**Note**: The graph structure already supports parallel execution from Phase 0.3. This node will run in parallel with `market_intelligence` node.

### Step 3: Add Error Recovery Logic

**File**: `src/agentic/nodes/market_data.py` (extend)

Add helper functions for error recovery:

```python
async def get_fallback_snapshot(base: str, quote: str) -> Optional[Dict[str, Any]]:
    """
    Attempt to get fallback snapshot from cache or return minimal data.
    
    Args:
        base: Base currency
        quote: Quote currency
    
    Returns:
        Cached snapshot or None
    """
    cache_key = f"market_snapshot:{base}/{quote}"
    return cache.get(cache_key)


def create_minimal_snapshot(base: str, quote: str, rate: float = None) -> Dict[str, Any]:
    """
    Create minimal snapshot with just the essentials when all else fails.
    
    Used as absolute last resort when cache is empty and providers fail.
    """
    return {
        "currency_pair": f"{base}/{quote}",
        "rate_timestamp": datetime.utcnow().isoformat(),
        "mid_rate": rate,
        "quality": {
            "sources_success": 0,
            "sources_total": 2,
            "fresh": False,
            "notes": ["Emergency fallback - no data available"]
        },
        "indicators": None,
        "regime": None
    }
```

### Step 4: Unit Tests

**File**: `tests/unit/test_agentic/test_nodes/test_market_data.py`

Create comprehensive unit tests:

**Test Cases**:

```python
import pytest
from unittest.mock import patch, AsyncMock
from src.agentic.nodes.market_data import market_data_node
from src.agentic.state import initialize_state

@pytest.mark.asyncio
async def test_market_data_node_success():
    """Test successful market data fetch."""
    state = initialize_state(
        "Convert USD to EUR",
        base_currency="USD",
        quote_currency="EUR"
    )
    
    with patch("src.agentic.nodes.market_data.build_snapshot") as mock_snapshot:
        # Mock successful snapshot
        mock_snapshot.return_value = create_mock_snapshot()
        
        result = await market_data_node(state)
        
        assert result["market_data_status"] == "success"
        assert result["market_snapshot"] is not None
        assert result["market_snapshot"]["mid_rate"] > 0


@pytest.mark.asyncio
async def test_market_data_node_missing_currencies():
    """Test handling of missing currency codes."""
    state = initialize_state("Convert to EUR")  # Missing base
    
    result = await market_data_node(state)
    
    assert result["market_data_status"] == "error"
    assert "Missing" in result["market_data_error"]


@pytest.mark.asyncio
async def test_market_data_node_timeout():
    """Test timeout handling."""
    state = initialize_state(
        "Convert USD to EUR",
        base_currency="USD",
        quote_currency="EUR"
    )
    
    with patch("src.agentic.nodes.market_data.build_snapshot") as mock_snapshot:
        # Simulate slow operation
        async def slow_build(*args, **kwargs):
            await asyncio.sleep(15)  # Exceeds 10s timeout
        
        mock_snapshot.side_effect = slow_build
        
        result = await market_data_node(state)
        
        assert result["market_data_status"] == "error"
        assert "Timeout" in result["market_data_error"]


@pytest.mark.asyncio
async def test_market_data_node_fallback_cache():
    """Test fallback to cached data on error."""
    state = initialize_state(
        "Convert USD to EUR",
        base_currency="USD",
        quote_currency="EUR"
    )
    
    # Pre-populate cache
    cache.set("market_snapshot:USD/EUR", {"mid_rate": 0.85}, ttl_seconds=60)
    
    with patch("src.agentic.nodes.market_data.build_snapshot") as mock_snapshot:
        mock_snapshot.side_effect = Exception("Provider failure")
        
        result = await market_data_node(state)
        
        assert result["market_data_status"] == "partial"
        assert result["market_snapshot"] is not None
        assert "cache" in result["market_data_error"].lower()
```

**Coverage Target**: > 80%

### Step 5: Integration Tests

**File**: `tests/integration/test_agentic/test_market_data_integration.py`

Test the node in the full LangGraph context:

```python
import pytest
from src.agentic.graph import create_graph
from src.agentic.state import initialize_state

@pytest.mark.integration
@pytest.mark.asyncio
async def test_market_data_node_in_graph():
    """Test Market Data node within complete LangGraph workflow."""
    graph = create_graph()
    
    initial_state = initialize_state(
        "Should I convert 5000 USD to EUR now?",
        base_currency="USD",
        quote_currency="EUR",
        amount=5000
    )
    
    # Execute graph (will run Market Data in parallel with Market Intelligence)
    result = await graph.ainvoke(initial_state)
    
    # Verify Market Data executed successfully
    assert result["market_data_status"] in ["success", "partial"]
    assert result["market_snapshot"] is not None
    assert result["market_snapshot"]["mid_rate"] > 0
    
    # Verify state has required fields for downstream agents
    assert "quality" in result["market_snapshot"]
    assert "indicators" in result["market_snapshot"]
    assert "regime" in result["market_snapshot"]
```

### Step 6: Performance Monitoring

**File**: `src/agentic/nodes/market_data.py` (extend)

Add performance metrics recording (aligned with current DB helpers/models):

```python
from typing import Optional

def record_performance_metrics(
    state: AgentState,
    execution_time_ms: int,
    status: str,
    error_message: Optional[str] = None
):
    """Record performance metrics to database."""
    from src.database.models import AgentMetrics
    from src.database.session import get_db

    with get_db() as session:
        metric = AgentMetrics(
            agent_name="market_data",
            execution_time_ms=execution_time_ms,
            status=status,
            error_message=error_message
        )
        session.add(metric)
```

Note: If you want to persist extra context (e.g., correlation_id, currency_pair, cache hit), either extend `AgentMetrics` with a JSON column later or write an additional `SystemLog` entry with `extra_data`.

## Key Design Decisions

1. **Parallel execution compatible**: Returns only specific state fields to avoid conflicts
2. **Timeout enforcement**: 10-second timeout via decorator (matches IMPLEMENTATION_ROADMAP.md)
3. **Graceful degradation**: Falls back to cached data on errors
4. **Structured logging**: All operations logged with correlation IDs
5. **Performance tracking**: Execution time recorded for monitoring
6. **Error categorization**: "success", "partial", or "error" status
7. **State format**: Converts dataclasses to dicts for JSON serialization in state

## Files to Create

- `src/agentic/nodes/__init__.py`
- `src/agentic/nodes/market_data.py`
- `tests/unit/test_agentic/__init__.py`
- `tests/unit/test_agentic/test_nodes/__init__.py`
- `tests/unit/test_agentic/test_nodes/test_market_data.py`
- `tests/integration/test_agentic/__init__.py`
- `tests/integration/test_agentic/test_market_data_integration.py`

## Dependencies

- Phase 0.3: LangGraph state (`src/agentic/state.py`)
- Phase 0.3: Graph structure (`src/agentic/graph.py`)
- Phase 0.4: Decorators (`src/utils/decorators.py`)
- Phase 0.2: Cache (`src/cache.py`)
- Phase 0.2: Database models (`src/database/models.py`)
- Phase 1.1: Providers (`src/data_collection/providers`)
- Phase 1.2: Snapshot builder (`src/data_collection/market_data/snapshot.py`)

## Validation

Manual testing with graph execution:

```python
from src.agentic.graph import create_graph
from src.agentic.state import initialize_state

# Create graph
graph = create_graph()

# Initialize state
state = initialize_state(
    "Convert 5000 USD to EUR",
    base_currency="USD",
    quote_currency="EUR",
    amount=5000
)

# Execute graph
result = await graph.ainvoke(state)

# Verify Market Data executed
print(f"Status: {result['market_data_status']}")
print(f"Mid Rate: {result['market_snapshot']['mid_rate']}")
print(f"RSI: {result['market_snapshot']['indicators']['rsi_14']}")
print(f"Regime: {result['market_snapshot']['regime']['trend_direction']}")
```

## Success Criteria

- Node executes successfully in LangGraph workflow
- Parallel execution works with Market Intelligence node (no state conflicts)
- Timeout enforcement prevents hanging (10-second limit)
- Fallback to cache works when providers fail
- All error scenarios handled gracefully
- Performance metrics recorded correctly
- Structured logging includes correlation IDs
- Unit tests pass with >80% coverage
- Integration test verifies node works in full graph
- State updates match expected schema from Phase 0.3
- No blocking of other parallel agents

## Next Phase

After Phase 1.3 completes, proceed to **Phase 1.4: Market Intelligence - Serper Integration**, which will:

- Create Serper API client for news and economic calendar search
- Implement search query builders for currency-specific information
- Add retry and error handling for API calls
- Write comprehensive tests

### To-dos

- [ ] Create Market Data agent node with proper LangGraph integration
- [ ] Implement timeout enforcement (10 seconds) and error handling
- [ ] Add fallback to cached data when providers fail
- [ ] Integrate snapshot builder with LangGraph state management
- [ ] Add performance logging and metrics recording
- [ ] Write comprehensive unit tests for the market data node (>80% coverage)
- [ ] Write integration tests for node within complete LangGraph workflow
- [ ] Update graph definition to use real market data node implementation
