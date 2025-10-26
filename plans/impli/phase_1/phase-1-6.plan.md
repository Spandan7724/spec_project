<!-- f67714c1-a54f-4e8d-9617-16955f212afc b92de828-1081-48a5-ae9d-dcd6423ccde3 -->
# Phase 1.6: Market Intelligence - Aggregation & Node

## Overview

Integrate calendar events and news sentiment into a unified Market Intelligence service, compute policy bias and event timing, and create the LangGraph node. This completes the Market Intelligence agent implementation, making it ready to run in parallel with Market Data in Layer 1.

## Implementation Steps

### Step 1: News Aggregator

**File**: `src/data_collection/market_intelligence/aggregator.py`

Aggregate news sentiment across articles for a currency pair:

```python
from dataclasses import dataclass
from typing import List, Dict, Any
from datetime import datetime, timezone
import numpy as np
from src.data_collection.market_intelligence.models import NewsClassification
from src.data_collection.market_intelligence.news_collector import NewsCollector
from src.data_collection.market_intelligence.extractors.news_classifier import NewsClassifier as Classifier
from src.utils.logging import get_logger

logger = get_logger(__name__)

@dataclass
class PairNewsSnapshot:
    """Aggregated news sentiment for a currency pair"""
    pair: str
    ts_utc: datetime
    sent_base: float  # -1 to +1
    sent_quote: float  # -1 to +1
    pair_bias: float  # sent_base - sent_quote
    confidence: str  # "high" | "medium" | "low"
    n_articles_used: int
    top_evidence: List[Dict[str, Any]]  # Top 5 articles

class NewsAggregator:
    """Aggregate news sentiment for currency pairs"""
    
    MIN_RELEVANCE_THRESHOLD = 0.3  # Filter articles by relevance
    
    def __init__(self, news_collector: NewsCollector, classifier: Classifier):
        self.collector = news_collector
        self.classifier = classifier
    
    async def get_pair_snapshot(
        self, 
        base: str, 
        quote: str,
        hours_back: int = 24
    ) -> PairNewsSnapshot:
        """
        Get aggregated news sentiment for a currency pair.
        
        Args:
            base: Base currency (e.g., "USD")
            quote: Quote currency (e.g., "EUR")
            hours_back: Hours of news to fetch (default 24)
        
        Returns:
            Aggregated sentiment snapshot
        """
        logger.info(
            f"Aggregating news for {base}/{quote}",
            extra={"base": base, "quote": quote, "hours_back": hours_back}
        )
        
        # 1. Fetch news for both currencies
        pair_news = await self.collector.collect_pair_news(base, quote, hours_back)
        
        all_articles = pair_news["base"] + pair_news["quote"]
        
        # 2. Deduplicate by URL
        seen_urls = set()
        unique_articles = []
        for article in all_articles:
            if article.url not in seen_urls:
                seen_urls.add(article.url)
                unique_articles.append(article)
        
        logger.info(
            f"Found {len(unique_articles)} unique articles",
            extra={"base": base, "quote": quote, "total": len(all_articles), "unique": len(unique_articles)}
        )
        
        # 3. Classify each article with gpt-5-mini
        classifications = await self.classifier.classify_batch(
            unique_articles,
            currencies=[base, quote]
        )
        
        # 4. Aggregate sentiment
        sent_base, sent_quote, confidence, top_evidence = self._aggregate_sentiment(
            classifications, base, quote
        )
        
        # 5. Calculate pair bias
        pair_bias = sent_base - sent_quote
        
        snapshot = PairNewsSnapshot(
            pair=f"{base}/{quote}",
            ts_utc=datetime.now(timezone.utc),
            sent_base=sent_base,
            sent_quote=sent_quote,
            pair_bias=pair_bias,
            confidence=confidence,
            n_articles_used=len(classifications),
            top_evidence=top_evidence
        )
        
        logger.info(
            f"News aggregation complete",
            extra={
                "pair": f"{base}/{quote}",
                "pair_bias": pair_bias,
                "confidence": confidence,
                "n_articles": len(classifications)
            }
        )
        
        return snapshot
    
    def _aggregate_sentiment(
        self, 
        classifications: List[NewsClassification], 
        base: str, 
        quote: str
    ) -> tuple:
        """
        Calculate weighted mean of per-article sentiment.
        
        Returns:
            Tuple of (sent_base, sent_quote, confidence, top_evidence)
        """
        if not classifications:
            return 0.0, 0.0, "low", []
        
        base_sentiments = []
        quote_sentiments = []
        
        # Filter by relevance threshold
        for c in classifications:
            if c.relevance.get(base, 0) >= self.MIN_RELEVANCE_THRESHOLD:
                base_sentiments.append(c.sentiment.get(base, 0))
            
            if c.relevance.get(quote, 0) >= self.MIN_RELEVANCE_THRESHOLD:
                quote_sentiments.append(c.sentiment.get(quote, 0))
        
        # Calculate means
        sent_base = float(np.mean(base_sentiments)) if base_sentiments else 0.0
        sent_quote = float(np.mean(quote_sentiments)) if quote_sentiments else 0.0
        
        # Calculate confidence based on article count and variance
        all_sentiments = base_sentiments + quote_sentiments
        variance = float(np.var(all_sentiments)) if all_sentiments else 1.0
        n_articles = len(classifications)
        
        # Confidence heuristic
        if n_articles >= 10 and variance < 0.3:
            confidence = "high"
        elif n_articles >= 5 and variance < 0.5:
            confidence = "medium"
        else:
            confidence = "low"
        
        # Top evidence (by max relevance across currencies)
        top = sorted(
            classifications,
            key=lambda c: max(c.relevance.values()),
            reverse=True
        )[:5]
        
        top_evidence = [
            {
                "title": c.title,
                "url": c.url,
                "source": c.source,
                "relevance": c.relevance,
                "sentiment": c.sentiment
            }
            for c in top
        ]
        
        return sent_base, sent_quote, confidence, top_evidence
```

### Step 2: Policy Bias Calculator

**File**: `src/data_collection/market_intelligence/bias_calculator.py`

Calculate policy bias from economic events:

```python
from typing import List
from src.data_collection.market_intelligence.models import EconomicEvent
from src.utils.logging import get_logger

logger = get_logger(__name__)

class PolicyBiasCalculator:
    """Calculate policy bias from economic events"""
    
    # Event keywords that suggest hawkish (rate hike) policy
    HAWKISH_KEYWORDS = [
        "rate hike", "interest rate increase", "hawkish", 
        "tightening", "inflation target", "rate decision"
    ]
    
    # Event keywords that suggest dovish (rate cut) policy
    DOVISH_KEYWORDS = [
        "rate cut", "interest rate decrease", "dovish",
        "easing", "stimulus", "quantitative easing"
    ]
    
    def calculate_policy_bias(
        self,
        events: List[EconomicEvent],
        currency: str
    ) -> float:
        """
        Calculate policy bias from events.
        
        Args:
            events: List of economic events
            currency: Currency to analyze
        
        Returns:
            Bias score: +1.0 (hawkish) to -1.0 (dovish)
        """
        # Filter to high-importance events for this currency
        relevant_events = [
            e for e in events
            if e.currency == currency and e.importance == "high"
        ]
        
        if not relevant_events:
            return 0.0
        
        bias_scores = []
        
        for event in relevant_events:
            event_text = event.event.lower()
            
            # Check for hawkish signals
            if any(kw in event_text for kw in self.HAWKISH_KEYWORDS):
                # Weight by proximity (more weight to imminent events)
                if event.days_until <= 1:
                    bias_scores.append(0.8)
                elif event.days_until <= 7:
                    bias_scores.append(0.5)
                else:
                    bias_scores.append(0.3)
            
            # Check for dovish signals
            elif any(kw in event_text for kw in self.DOVISH_KEYWORDS):
                if event.days_until <= 1:
                    bias_scores.append(-0.8)
                elif event.days_until <= 7:
                    bias_scores.append(-0.5)
                else:
                    bias_scores.append(-0.3)
            
            # Neutral event (just presence of high-importance event)
            else:
                if event.days_until <= 1:
                    bias_scores.append(0.0)
        
        # Average bias
        if bias_scores:
            import numpy as np
            bias = float(np.mean(bias_scores))
            return np.clip(bias, -1.0, 1.0)
        
        return 0.0
```

### Step 3: Market Intelligence Service

**File**: `src/data_collection/market_intelligence/intelligence_service.py`

Unified service integrating calendar + news:

```python
from typing import Dict, Any, Optional
from datetime import datetime, timezone
from src.data_collection.market_intelligence.serper_client import SerperClient
from src.data_collection.market_intelligence.calendar_collector import CalendarCollector
from src.data_collection.market_intelligence.news_collector import NewsCollector
from src.data_collection.market_intelligence.extractors.calendar_extractor import CalendarExtractor
from src.data_collection.market_intelligence.extractors.news_classifier import NewsClassifier
from src.data_collection.market_intelligence.extractors.narrative_generator import NarrativeGenerator
from src.data_collection.market_intelligence.aggregator import NewsAggregator
from src.data_collection.market_intelligence.bias_calculator import PolicyBiasCalculator
from src.data_collection.market_intelligence.models import EconomicEvent
from src.utils.logging import get_logger

logger = get_logger(__name__)

class MarketIntelligenceService:
    """Unified service for economic calendar + news sentiment"""
    
    def __init__(self, serper_api_key: str, llm_manager):
        """Initialize all components"""
        # Serper client
        self.serper = SerperClient(serper_api_key)
        
        # Collectors
        self.calendar_collector = CalendarCollector(self.serper)
        self.news_collector = NewsCollector(self.serper)
        
        # Extractors
        self.calendar_extractor = CalendarExtractor(llm_manager)
        self.news_classifier = NewsClassifier(llm_manager)
        self.narrative_gen = NarrativeGenerator(llm_manager)
        
        # Aggregators
        self.news_aggregator = NewsAggregator(self.news_collector, self.news_classifier)
        self.bias_calculator = PolicyBiasCalculator()
    
    async def get_pair_intelligence(
        self, 
        base: str, 
        quote: str,
        days_ahead: int = 7
    ) -> Dict[str, Any]:
        """
        Get complete intelligence snapshot for a currency pair.
        
        Args:
            base: Base currency
            quote: Quote currency
            days_ahead: Days ahead for calendar events
        
        Returns:
            Complete intelligence report
        """
        logger.info(
            f"Getting intelligence for {base}/{quote}",
            extra={"base": base, "quote": quote}
        )
        
        # 1. Get news sentiment
        news_snapshot = await self.news_aggregator.get_pair_snapshot(base, quote)
        
        # 2. Get calendar events for both currencies
        import asyncio
        
        base_urls, quote_urls = await asyncio.gather(
            self.calendar_collector.collect_calendar_urls(base),
            self.calendar_collector.collect_calendar_urls(quote)
        )
        
        base_events, quote_events = await asyncio.gather(
            self.calendar_extractor.extract_events_batch(base_urls, base),
            self.calendar_extractor.extract_events_batch(quote_urls, quote)
        )
        
        all_events = base_events + quote_events
        
        # Filter to date range
        upcoming_events = [
            e for e in all_events
            if 0 <= e.days_until <= days_ahead
        ]
        
        # 3. Find next high-impact event
        high_impact_events = [e for e in upcoming_events if e.importance == "high"]
        next_high_event = None
        
        if high_impact_events:
            # Sort by proximity
            high_impact_events.sort(key=lambda e: e.proximity_minutes)
            # Get next upcoming event (positive proximity)
            upcoming_high = [e for e in high_impact_events if e.proximity_minutes > 0]
            if upcoming_high:
                next_high_event = upcoming_high[0]
        
        # 4. Calculate policy bias
        base_bias = self.bias_calculator.calculate_policy_bias(upcoming_events, base)
        quote_bias = self.bias_calculator.calculate_policy_bias(upcoming_events, quote)
        overall_bias = base_bias - quote_bias  # Positive = bullish base
        
        # 5. Generate narrative
        narrative = await self.narrative_gen.generate_news_narrative(
            pair=news_snapshot.pair,
            sent_base=news_snapshot.sent_base,
            sent_quote=news_snapshot.sent_quote,
            pair_bias=news_snapshot.pair_bias,
            confidence=news_snapshot.confidence,
            n_articles=news_snapshot.n_articles_used,
            top_evidence=news_snapshot.top_evidence
        )
        
        # 6. Build unified report
        report = {
            "pair": news_snapshot.pair,
            "ts_utc": news_snapshot.ts_utc.isoformat(),
            "overall_bias": overall_bias,  # Combined news + policy bias
            "news": {
                "sent_base": news_snapshot.sent_base,
                "sent_quote": news_snapshot.sent_quote,
                "pair_bias": news_snapshot.pair_bias,
                "confidence": news_snapshot.confidence,
                "n_articles_used": news_snapshot.n_articles_used,
                "top_evidence": news_snapshot.top_evidence,
                "narrative": narrative
            },
            "calendar": {
                "next_high_event": {
                    "when_utc": next_high_event.when_utc.isoformat(),
                    "currency": next_high_event.currency,
                    "event": next_high_event.event,
                    "source_url": next_high_event.source_url,
                    "proximity_minutes": next_high_event.proximity_minutes,
                    "is_imminent": next_high_event.is_imminent,
                    "days_until": next_high_event.days_until
                } if next_high_event else None,
                "total_high_impact_events_7d": len(high_impact_events),
                "base_policy_bias": base_bias,
                "quote_policy_bias": quote_bias
            }
        }
        
        logger.info(
            f"Intelligence complete",
            extra={
                "pair": f"{base}/{quote}",
                "overall_bias": overall_bias,
                "n_articles": news_snapshot.n_articles_used,
                "n_events": len(upcoming_events)
            }
        )
        
        return report
```

### Step 4: Market Intelligence Node

**File**: `src/agentic/nodes/market_intelligence.py`

Create the LangGraph node:

```python
from typing import Dict, Any
from src.agentic.state import AgentState
from src.data_collection.market_intelligence.intelligence_service import MarketIntelligenceService
from src.config import get_config
from src.utils.logging import get_logger
from src.utils.decorators import log_execution, timeout
import time
import os

logger = get_logger(__name__)

@timeout(15.0)  # 15 second timeout for Market Intelligence (more than Market Data due to LLM calls)
@log_execution(log_args=False, log_result=False)
async def market_intelligence_node(state: AgentState) -> Dict[str, Any]:
    """
    Market Intelligence agent node for LangGraph.
    
    Fetches economic calendar events and news sentiment.
    Updates state with intelligence report.
    
    Args:
        state: Current agent state
    
    Returns:
        Dictionary with intelligence_report and status fields
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
                "intelligence_status": "error",
                "intelligence_error": "Missing base or quote currency",
                "intelligence_report": None
            }
        
        # Initialize service
        serper_api_key = os.getenv("SERPER_API_KEY")
        if not serper_api_key:
            raise ValueError("SERPER_API_KEY not found in environment")
        
        from src.llm.manager import LLMManager
        llm_manager = LLMManager()
        
        service = MarketIntelligenceService(serper_api_key, llm_manager)
        
        # Get intelligence report
        logger.info(
            f"Fetching intelligence for {base}/{quote}",
            extra={"correlation_id": correlation_id, "base": base, "quote": quote}
        )
        
        report = await service.get_pair_intelligence(base, quote, days_ahead=7)
        
        # Calculate execution time
        execution_time_ms = int((time.time() - start_time) * 1000)
        
        logger.info(
            f"Intelligence fetched successfully",
            extra={
                "correlation_id": correlation_id,
                "overall_bias": report["overall_bias"],
                "n_articles": report["news"]["n_articles_used"],
                "execution_time_ms": execution_time_ms
            }
        )
        
        # Return state updates (only fields this node is responsible for)
        return {
            "intelligence_report": report,
            "intelligence_status": "success",
            "intelligence_error": None
        }
        
    except TimeoutError as e:
        execution_time_ms = int((time.time() - start_time) * 1000)
        logger.error(
            "Market Intelligence agent timed out",
            extra={"correlation_id": correlation_id, "execution_time_ms": execution_time_ms}
        )
        return {
            "intelligence_status": "error",
            "intelligence_error": f"Timeout after 15 seconds: {str(e)}",
            "intelligence_report": None
        }
        
    except Exception as e:
        execution_time_ms = int((time.time() - start_time) * 1000)
        logger.error(
            f"Market Intelligence agent failed: {str(e)}",
            extra={"correlation_id": correlation_id, "error": str(e), "execution_time_ms": execution_time_ms}
        )
        
        # Return minimal report as fallback
        return {
            "intelligence_status": "error",
            "intelligence_error": str(e),
            "intelligence_report": None
        }
```

### Step 5: Update Graph Definition

**File**: `src/agentic/graph.py`

Replace placeholder with real implementation:

```python
from src.agentic.nodes.market_intelligence import market_intelligence_node

# In create_graph():
workflow.add_node("market_intelligence", market_intelligence_node)
```

### Step 6: Unit Tests

**File**: `tests/unit/test_agentic/test_nodes/test_market_intelligence.py`

Test the node with mocked services:

```python
import pytest
from unittest.mock import patch, AsyncMock
from src.agentic.nodes.market_intelligence import market_intelligence_node
from src.agentic.state import initialize_state

@pytest.mark.asyncio
async def test_market_intelligence_node_success():
    """Test successful intelligence fetch."""
    state = initialize_state(
        "Convert USD to EUR",
        base_currency="USD",
        quote_currency="EUR"
    )
    
    mock_report = {
        "pair": "USD/EUR",
        "overall_bias": 0.3,
        "news": {
            "pair_bias": 0.2,
            "confidence": "high",
            "n_articles_used": 15
        },
        "calendar": {
            "next_high_event": None,
            "total_high_impact_events_7d": 2
        }
    }
    
    with patch("src.agentic.nodes.market_intelligence.MarketIntelligenceService") as mock_service:
        mock_instance = AsyncMock()
        mock_instance.get_pair_intelligence.return_value = mock_report
        mock_service.return_value = mock_instance
        
        result = await market_intelligence_node(state)
        
        assert result["intelligence_status"] == "success"
        assert result["intelligence_report"] is not None
        assert result["intelligence_report"]["overall_bias"] == 0.3


@pytest.mark.asyncio
async def test_market_intelligence_node_missing_currencies():
    """Test handling of missing currency codes."""
    state = initialize_state("Convert to EUR")  # Missing base
    
    result = await market_intelligence_node(state)
    
    assert result["intelligence_status"] == "error"
    assert "Missing" in result["intelligence_error"]


@pytest.mark.asyncio
async def test_market_intelligence_node_timeout():
    """Test timeout handling."""
    state = initialize_state(
        "Convert USD to EUR",
        base_currency="USD",
        quote_currency="EUR"
    )
    
    with patch("src.agentic.nodes.market_intelligence.MarketIntelligenceService") as mock_service:
        mock_instance = AsyncMock()
        
        async def slow_intelligence(*args, **kwargs):
            await asyncio.sleep(20)  # Exceeds 15s timeout
        
        mock_instance.get_pair_intelligence.side_effect = slow_intelligence
        mock_service.return_value = mock_instance
        
        result = await market_intelligence_node(state)
        
        assert result["intelligence_status"] == "error"
        assert "Timeout" in result["intelligence_error"]
```

**Coverage Target**: >80%

### Step 7: Integration Tests

**File**: `tests/integration/test_agentic/test_market_intelligence_integration.py`

Test the complete flow:

```python
import pytest
from src.agentic.graph import create_graph
from src.agentic.state import initialize_state

@pytest.mark.integration
@pytest.mark.asyncio
async def test_market_intelligence_node_in_graph():
    """Test Market Intelligence node within complete LangGraph workflow."""
    graph = create_graph()
    
    initial_state = initialize_state(
        "Should I convert 5000 USD to EUR now?",
        base_currency="USD",
        quote_currency="EUR",
        amount=5000
    )
    
    # Execute graph (will run Market Intelligence in parallel with Market Data)
    result = await graph.ainvoke(initial_state)
    
    # Verify Market Intelligence executed
    assert result["intelligence_status"] in ["success", "partial", "error"]
    
    if result["intelligence_status"] == "success":
        assert result["intelligence_report"] is not None
        assert "overall_bias" in result["intelligence_report"]
        assert "news" in result["intelligence_report"]
        assert "calendar" in result["intelligence_report"]
```

## Key Design Decisions

1. **Unified intelligence report**: Combines news sentiment + calendar events into single coherent output
2. **Policy bias calculation**: Weight events by importance and proximity
3. **Overall bias score**: Combines news sentiment and policy bias for holistic view
4. **15-second timeout**: Longer than Market Data (10s) due to multiple LLM calls
5. **Parallel execution compatible**: Returns only specific state fields
6. **Graceful degradation**: Returns error status but doesn't crash system
7. **Structured logging**: All operations logged with correlation IDs
8. **Service composition**: Clean separation of concerns (collectors, extractors, aggregators, service)

## Files to Create

- `src/data_collection/market_intelligence/aggregator.py`
- `src/data_collection/market_intelligence/bias_calculator.py`
- `src/data_collection/market_intelligence/intelligence_service.py`
- `src/agentic/nodes/market_intelligence.py`
- `tests/unit/test_market_intelligence/test_aggregator.py`
- `tests/unit/test_market_intelligence/test_bias_calculator.py`
- `tests/unit/test_agentic/test_nodes/test_market_intelligence.py`
- `tests/integration/test_agentic/test_market_intelligence_integration.py`

## Dependencies

- Phase 1.4: Serper client, collectors
- Phase 1.5: Extractors (calendar, news, narrative)
- Phase 0.3: LangGraph state and graph
- Phase 0.4: Decorators (timeout, retry, log_execution)
- Phase 0.1: Config and logging
- Phase 0 (LLM): LLMManager, agent_helpers
- Python: numpy (for aggregation)

## Configuration

Environment variables:

```bash
SERPER_API_KEY=your_key
COPILOT_ACCESS_TOKEN=your_token
```

## Validation

Manual testing:

```python
from src.data_collection.market_intelligence.intelligence_service import MarketIntelligenceService
from src.llm.manager import LLMManager
import os

# Initialize
llm_manager = LLMManager()
service = MarketIntelligenceService(os.getenv("SERPER_API_KEY"), llm_manager)

# Get intelligence
report = await service.get_pair_intelligence("USD", "EUR")

print(f"Overall Bias: {report['overall_bias']:+.2f}")
print(f"News Sentiment: {report['news']['pair_bias']:+.2f} ({report['news']['confidence']})")
print(f"Articles: {report['news']['n_articles_used']}")
print(f"Next High Event: {report['calendar']['next_high_event']}")
print(f"\nNarrative: {report['news']['narrative']}")
```

## Success Criteria

- News aggregator calculates sentiment correctly with confidence scoring
- Policy bias calculator weights events by importance and proximity
- Intelligence service combines calendar + news into unified report
- Overall bias combines news and policy signals
- Market Intelligence node executes successfully in LangGraph
- Parallel execution works with Market Data node (no state conflicts)
- Timeout enforcement prevents hanging (15-second limit)
- All error scenarios handled gracefully
- All unit tests pass with mocked components (>80% coverage)
- Integration test verifies node works in full graph
- Structured logging tracks all operations
- Code follows Phase 0 patterns

## Cost Summary

**Total cost per agent call (Phases 1.4 + 1.5 + 1.6)**:

- Serper (2-3 news + 2-3 calendar): ~$0.02-0.03
- LLM classification (10-15 articles): ~$0.0015
- LLM calendar extraction (3-5 snippets): ~$0.001
- LLM narrative generation (1 call): ~$0.01
- **Total**: ~$0.04-0.05 per agent call

**Monthly cost (30 calls/day)**: ~$45-50/month

## Next Phase

After Phase 1.6 completes, **Layer 1 (Market Data + Market Intelligence) is complete**!

Proceed to **Phase 2: Price Prediction Agent**, which will:

- Create data pipeline for historical OHLC data
- Implement feature engineering (technical indicators + optional intelligence features)
- Build model registry (JSON + pickle)
- Implement LightGBM backend with SHAP explainability
- Create fallback heuristics
- Add prediction caching
- Create Price Prediction LangGraph node

### To-dos

- [ ] Create news aggregator for sentiment calculation across articles
- [ ] Implement policy bias calculator from economic events
- [ ] Create unified Market Intelligence service integrating calendar + news
- [ ] Implement Market Intelligence LangGraph node with 15-second timeout
- [ ] Add performance logging and error handling for the node
- [ ] Write comprehensive unit tests for aggregator, bias calculator, and service (>80% coverage)
- [ ] Write integration tests for complete Market Intelligence flow
- [ ] Update graph definition to use real Market Intelligence node implementation