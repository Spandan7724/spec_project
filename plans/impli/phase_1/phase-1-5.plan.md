<!-- f67714c1-a54f-4e8d-9617-16955f212afc 861da746-ccf0-46e6-a175-0cc22fce6fb8 -->
# Phase 1.5: Market Intelligence - LLM Extraction

## Overview

Add LLM-powered extraction and classification to transform Serper search results into structured data. This phase implements calendar event extraction, news sentiment classification, and narrative generation using task-optimized models.

**Key Design Decision**: Use gpt-5-mini for high-volume extraction/classification (cheap, fast) and gpt-4o for narrative generation (quality).

## Implementation Steps

### Step 1: Economic Event Models

**File**: `src/data_collection/market_intelligence/models.py`

Define structured data contracts for economic events:

```python
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

@dataclass
class EconomicEvent:
    """Standardized economic calendar event"""
    when_utc: datetime
    when_local: datetime
    timezone: str
    country: str
    currency: str
    event: str
    importance: str  # "high" | "medium" | "low"
    source: str
    source_url: str
    
    @property
    def proximity_minutes(self) -> int:
        """Minutes until event (negative if past)"""
        now = datetime.now(timezone.utc)
        return int((self.when_utc - now).total_seconds() / 60)
    
    @property
    def is_imminent(self) -> bool:
        """Event within 60 minutes"""
        return 0 <= self.proximity_minutes <= 60
    
    @property
    def is_today(self) -> bool:
        """Event is today"""
        return self.when_utc.date() == datetime.now(timezone.utc).date()
    
    @property
    def days_until(self) -> int:
        """Days until event (negative if past)"""
        now = datetime.now(timezone.utc)
        delta = self.when_utc.date() - now.date()
        return delta.days

@dataclass
class NewsClassification:
    """Classification result for a single article"""
    article_id: str
    url: str
    source: str
    title: str
    published_utc: datetime
    relevance: dict[str, float]  # {"USD": 0.9, "EUR": 0.2}
    sentiment: dict[str, float]  # {"USD": 0.4, "EUR": -0.1} range: -1 to +1
    quality_flags: dict[str, bool]  # {clickbait, rumor_speculative, non_econ}
```

### Step 2: Calendar Event Extractor

**File**: `src/data_collection/market_intelligence/extractors/calendar_extractor.py`

Extract structured events from calendar search results using gpt-5-mini:

````python
import json
from typing import List
from datetime import datetime, timezone
from src.llm.agent_helpers import chat_with_model, get_recommended_model_for_task
from src.data_collection.market_intelligence.models import EconomicEvent
from src.data_collection.market_intelligence.serper_client import SerperSearchResult
from src.utils.logging import get_logger
from src.utils.decorators import retry, log_execution

logger = get_logger(__name__)

class CalendarExtractor:
    """Extract economic events from search results using LLM"""
    
    CURRENCY_TO_COUNTRY = {
        "USD": "US",
        "EUR": "EA",  # Euro Area
        "GBP": "GB",
        "JPY": "JP",
        "CHF": "CH",
        "CAD": "CA",
        "AUD": "AU",
        "NZD": "NZ"
    }
    
    def __init__(self, llm_manager):
        self.llm_manager = llm_manager
    
    @retry(max_attempts=2, delay=1.0)
    @log_execution(log_args=False, log_result=False)
    async def extract_events_from_snippet(
        self, 
        result: SerperSearchResult,
        currency: str
    ) -> List[EconomicEvent]:
        """
        Extract structured events from a search result snippet.
        
        Args:
            result: Serper search result
            currency: Currency code for context
        
        Returns:
            List of extracted economic events
        """
        prompt = f'''Extract economic calendar events from this search result.

Title: {result.title}
Snippet: {result.snippet}
Currency: {currency}

Return a JSON array of events with fields:
- date: ISO format (YYYY-MM-DD)
- time: ISO time (HH:MM) or null if not specified
- event: Event name/description
- importance: "high", "medium", or "low"
  * high = central bank decisions, GDP, CPI, NFP, employment, interest rates, FOMC, ECB decisions
  * medium = PMI, ISM, retail sales, industrial production
  * low = other economic data

Only extract events that mention specific dates. If no events found, return empty array.

Example:
[
  {{"date": "2025-10-30", "time": "12:15", "event": "ECB Interest Rate Decision", "importance": "high"}},
  {{"date": "2025-11-01", "time": null, "event": "US Jobs Report", "importance": "high"}}
]
'''
        
        messages = [
            {"role": "system", "content": "Extract economic events from text. Return only valid JSON array."},
            {"role": "user", "content": prompt}
        ]
        
        # Use gpt-5-mini for fast extraction
        model = get_recommended_model_for_task("data_extraction")
        
        try:
            response = await chat_with_model(messages, model, self.llm_manager)
            content = response.content.strip()
            
            # Clean markdown code blocks if present
            if content.startswith("```"):
                content = content.strip("`").strip()
                if content.lower().startswith("json"):
                    content = content[4:].strip()
            
            events_json = json.loads(content)
            
            # Convert to EconomicEvent objects
            events = []
            for e in events_json:
                try:
                    event_obj = self._parse_event_json(e, currency, result.url)
                    if event_obj:
                        events.append(event_obj)
                except Exception as parse_err:
                    logger.warning(f"Failed to parse event: {parse_err}", extra={"event": e})
                    continue
            
            logger.info(
                f"Extracted {len(events)} events from snippet",
                extra={"currency": currency, "url": result.url, "count": len(events)}
            )
            
            return events
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON from LLM: {e}", extra={"response": content[:200]})
            return []
        except Exception as e:
            logger.error(f"Event extraction failed: {e}", extra={"currency": currency})
            return []
    
    def _parse_event_json(self, event_dict: dict, currency: str, source_url: str) -> Optional[EconomicEvent]:
        """Parse JSON event dict into EconomicEvent object"""
        date_str = event_dict.get("date")
        time_str = event_dict.get("time")
        
        if not date_str:
            return None
        
        # Parse datetime
        if time_str:
            dt_str = f"{date_str}T{time_str}:00Z"
        else:
            dt_str = f"{date_str}T00:00:00Z"
        
        try:
            when_utc = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
        except ValueError:
            return None
        
        return EconomicEvent(
            when_utc=when_utc,
            when_local=when_utc,  # TODO: Add timezone conversion
            timezone="UTC",
            country=self.CURRENCY_TO_COUNTRY.get(currency, "UNKNOWN"),
            currency=currency,
            event=event_dict.get("event", ""),
            importance=event_dict.get("importance", "medium"),
            source=source_url,
            source_url=source_url
        )
    
    async def extract_events_batch(
        self,
        results: List[SerperSearchResult],
        currency: str
    ) -> List[EconomicEvent]:
        """
        Extract events from multiple search results.
        
        Args:
            results: List of search results
            currency: Currency code
        
        Returns:
            Combined list of events, deduplicated
        """
        import asyncio
        
        # Extract from top 5 results
        tasks = [
            self.extract_events_from_snippet(result, currency)
            for result in results[:5]
        ]
        
        event_lists = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Flatten and filter exceptions
        all_events = []
        for events in event_lists:
            if isinstance(events, list):
                all_events.extend(events)
        
        # Deduplicate by date + event name
        return self._deduplicate_events(all_events)
    
    def _deduplicate_events(self, events: List[EconomicEvent]) -> List[EconomicEvent]:
        """Remove duplicate events based on date + event name"""
        seen = set()
        unique = []
        
        for event in events:
            key = f"{event.when_utc.date()}_{event.event.lower()}"
            if key not in seen:
                seen.add(key)
                unique.append(event)
        
        return sorted(unique, key=lambda e: e.when_utc)
````

### Step 3: News Sentiment Classifier

**File**: `src/data_collection/market_intelligence/extractors/news_classifier.py`

Classify news articles for relevance and sentiment using gpt-5-mini:

````python
import json
import hashlib
from datetime import datetime, timezone
from typing import List
from src.llm.agent_helpers import chat_with_model, get_recommended_model_for_task
from src.data_collection.market_intelligence.models import NewsClassification
from src.data_collection.market_intelligence.serper_client import SerperNewsResult
from src.utils.logging import get_logger
from src.utils.decorators import retry, log_execution

logger = get_logger(__name__)

class NewsClassifier:
    """Classify news articles using gpt-5-mini"""
    
    def __init__(self, llm_manager):
        self.llm_manager = llm_manager
    
    @retry(max_attempts=2, delay=1.0)
    @log_execution(log_args=False, log_result=False)
    async def classify_article(
        self, 
        article: SerperNewsResult,
        currencies: List[str]
    ) -> NewsClassification:
        """
        Classify a single article for currency relevance and sentiment.
        
        Args:
            article: News article from Serper
            currencies: List of currencies to analyze (e.g., ["USD", "EUR"])
        
        Returns:
            NewsClassification with relevance, sentiment, and quality flags
        """
        prompt = f'''Analyze this financial news article for currency sentiment.

Title: {article.title}
Snippet: {article.snippet}
Source: {article.source}
Currencies: {", ".join(currencies)}

Return JSON with these fields:

1. relevance: For each currency, score 0.0-1.0
   - 1.0 = Directly about this currency's economy/policy/central bank
   - 0.5 = Indirectly related (global factors)
   - 0.0 = Not related

2. sentiment: For each currency, score -1.0 to +1.0
   - Positive = Bullish factors (strong growth, hawkish policy, rate hikes)
   - Negative = Bearish factors (weak data, dovish policy, rate cuts)
   - Zero = Neutral or unclear

3. quality_flags:
   - clickbait: Is this a clickbait headline?
   - rumor_speculative: Is this speculation vs hard news?
   - non_econ: Is this non-economic news (politics, sports, etc.)?

Example:
{{
  "relevance": {{"USD": 0.9, "EUR": 0.3}},
  "sentiment": {{"USD": 0.5, "EUR": -0.2}},
  "quality_flags": {{"clickbait": false, "rumor_speculative": false, "non_econ": false}}
}}
'''
        
        messages = [
            {"role": "system", "content": "You are a financial news classifier. Return only valid JSON."},
            {"role": "user", "content": prompt}
        ]
        
        # Use gpt-5-mini for fast classification
        model = get_recommended_model_for_task("classification")
        
        try:
            response = await chat_with_model(messages, model, self.llm_manager)
            content = response.content.strip()
            
            # Clean markdown code blocks
            if content.startswith("```"):
                content = content.strip("`").strip()
                if content.lower().startswith("json"):
                    content = content[4:].strip()
            
            classification_data = json.loads(content)
            
            # Generate article ID from URL
            article_id = hashlib.sha256(article.url.encode()).hexdigest()[:16]
            
            return NewsClassification(
                article_id=article_id,
                url=article.url,
                source=article.source,
                title=article.title,
                published_utc=datetime.now(timezone.utc),  # TODO: Parse from date string
                relevance=classification_data.get("relevance", {}),
                sentiment=classification_data.get("sentiment", {}),
                quality_flags=classification_data.get("quality_flags", {})
            )
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse classification JSON: {e}", extra={"article": article.title})
            # Return neutral classification
            return self._create_neutral_classification(article, currencies)
        except Exception as e:
            logger.error(f"Classification failed: {e}", extra={"article": article.title})
            return self._create_neutral_classification(article, currencies)
    
    def _create_neutral_classification(
        self, 
        article: SerperNewsResult,
        currencies: List[str]
    ) -> NewsClassification:
        """Create neutral classification as fallback"""
        article_id = hashlib.sha256(article.url.encode()).hexdigest()[:16]
        
        return NewsClassification(
            article_id=article_id,
            url=article.url,
            source=article.source,
            title=article.title,
            published_utc=datetime.now(timezone.utc),
            relevance={c: 0.0 for c in currencies},
            sentiment={c: 0.0 for c in currencies},
            quality_flags={"clickbait": False, "rumor_speculative": True, "non_econ": False}
        )
    
    async def classify_batch(
        self,
        articles: List[SerperNewsResult],
        currencies: List[str]
    ) -> List[NewsClassification]:
        """
        Classify multiple articles in parallel.
        
        Args:
            articles: List of news articles
            currencies: Currencies to analyze
        
        Returns:
            List of classifications
        """
        import asyncio
        
        tasks = [
            self.classify_article(article, currencies)
            for article in articles
        ]
        
        classifications = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        valid = [c for c in classifications if isinstance(c, NewsClassification)]
        
        # Filter out low quality
        filtered = [
            c for c in valid
            if not c.quality_flags.get("non_econ", False)
        ]
        
        logger.info(
            f"Classified {len(filtered)} articles out of {len(articles)}",
            extra={"total": len(articles), "valid": len(filtered)}
        )
        
        return filtered
````

### Step 4: Narrative Generator

**File**: `src/data_collection/market_intelligence/extractors/narrative_generator.py`

Generate human-readable summaries using gpt-4o:

```python
from src.llm.agent_helpers import chat_with_model, get_recommended_model_for_task
from src.utils.logging import get_logger
from src.utils.decorators import retry, log_execution

logger = get_logger(__name__)

class NarrativeGenerator:
    """Generate human-readable narratives using gpt-4o"""
    
    def __init__(self, llm_manager):
        self.llm_manager = llm_manager
    
    @retry(max_attempts=2, delay=1.0)
    @log_execution(log_args=False, log_result=False)
    async def generate_news_narrative(
        self, 
        pair: str,
        sent_base: float,
        sent_quote: float,
        pair_bias: float,
        confidence: str,
        n_articles: int,
        top_evidence: list
    ) -> str:
        """
        Generate 1-2 sentence summary of news sentiment.
        
        Args:
            pair: Currency pair (e.g., "USD/EUR")
            sent_base: Base currency sentiment (-1 to +1)
            sent_quote: Quote currency sentiment (-1 to +1)
            pair_bias: sent_base - sent_quote
            confidence: "high" | "medium" | "low"
            n_articles: Number of articles analyzed
            top_evidence: Top headlines list
        
        Returns:
            Professional narrative summary
        """
        pair_parts = pair.split('/')
        base = pair_parts[0]
        quote = pair_parts[1]
        
        bias_text = "bullish" if pair_bias > 0.2 else "bearish" if pair_bias < -0.2 else "neutral"
        
        # Build evidence string
        headlines_str = "\n".join(f"- {e['title']}" for e in top_evidence[:3])
        
        prompt = f'''Generate a concise 1-2 sentence summary of current news sentiment for {pair}.

Data:
- {base} sentiment: {sent_base:+.2f}
- {quote} sentiment: {sent_quote:+.2f}
- Pair bias: {pair_bias:+.2f} ({bias_text} {base})
- Confidence: {confidence}
- Based on {n_articles} articles

Top headlines:
{headlines_str}

Write a professional, concise summary suitable for a financial analysis report.
Focus on what the news means for the currency pair direction.
'''
        
        messages = [
            {"role": "system", "content": "You are a financial analyst summarizing market sentiment."},
            {"role": "user", "content": prompt}
        ]
        
        # Use gpt-4o for narrative quality
        model = get_recommended_model_for_task("summarization")
        
        try:
            response = await chat_with_model(messages, model, self.llm_manager)
            narrative = response.content.strip()
            
            logger.info(
                "Generated narrative",
                extra={"pair": pair, "length": len(narrative)}
            )
            
            return narrative
            
        except Exception as e:
            logger.error(f"Narrative generation failed: {e}", extra={"pair": pair})
            # Return fallback narrative
            return f"Market sentiment for {pair} is {bias_text} ({confidence} confidence based on {n_articles} articles)."
```

### Step 5: Update Module Exports

**File**: `src/data_collection/market_intelligence/__init__.py`

Add extractor exports:

```python
"""Market Intelligence data collection using Serper API + LLM extraction."""

from src.data_collection.market_intelligence.serper_client import (
    SerperClient,
    SerperNewsResult,
    SerperSearchResult
)
from src.data_collection.market_intelligence.calendar_collector import CalendarCollector
from src.data_collection.market_intelligence.news_collector import NewsCollector
from src.data_collection.market_intelligence.models import (
    EconomicEvent,
    NewsClassification
)
from src.data_collection.market_intelligence.extractors.calendar_extractor import CalendarExtractor
from src.data_collection.market_intelligence.extractors.news_classifier import NewsClassifier
from src.data_collection.market_intelligence.extractors.narrative_generator import NarrativeGenerator

__all__ = [
    "SerperClient",
    "SerperNewsResult",
    "SerperSearchResult",
    "CalendarCollector",
    "NewsCollector",
    "EconomicEvent",
    "NewsClassification",
    "CalendarExtractor",
    "NewsClassifier",
    "NarrativeGenerator"
]
```

### Step 6: Unit Tests

**Files**:

- `tests/unit/test_market_intelligence/test_extractors.py`

Test extractors with mocked LLM responses:

```python
import pytest
from unittest.mock import patch, AsyncMock
from src.data_collection.market_intelligence.extractors import (
    CalendarExtractor,
    NewsClassifier,
    NarrativeGenerator
)
from src.data_collection.market_intelligence import SerperSearchResult, SerperNewsResult

@pytest.mark.asyncio
async def test_calendar_extractor():
    """Test calendar event extraction with mocked LLM."""
    llm_manager = MockLLMManager()
    extractor = CalendarExtractor(llm_manager)
    
    result = SerperSearchResult(
        title="ECB Meeting October 2025",
        url="https://ecb.europa.eu/calendar",
        snippet="ECB will announce interest rate decision on October 30, 2025 at 12:15 UTC.",
        position=1
    )
    
    # Mock LLM response
    mock_json = '''[
        {"date": "2025-10-30", "time": "12:15", "event": "ECB Interest Rate Decision", "importance": "high"}
    ]'''
    
    with patch.object(llm_manager, 'chat_with_model') as mock_chat:
        mock_chat.return_value = AsyncMock(content=mock_json)
        
        events = await extractor.extract_events_from_snippet(result, "EUR")
        
        assert len(events) == 1
        assert events[0].event == "ECB Interest Rate Decision"
        assert events[0].importance == "high"
        assert events[0].currency == "EUR"


@pytest.mark.asyncio
async def test_news_classifier():
    """Test news classification with mocked LLM."""
    llm_manager = MockLLMManager()
    classifier = NewsClassifier(llm_manager)
    
    article = SerperNewsResult(
        title="Fed Raises Rates 0.25%",
        url="https://reuters.com/fed-rates",
        source="Reuters",
        snippet="The Federal Reserve raised interest rates...",
        date="1 hour ago",
        position=1
    )
    
    # Mock LLM response
    mock_json = '''{
        "relevance": {"USD": 0.95, "EUR": 0.3},
        "sentiment": {"USD": 0.6, "EUR": -0.1},
        "quality_flags": {"clickbait": false, "rumor_speculative": false, "non_econ": false}
    }'''
    
    with patch.object(llm_manager, 'chat_with_model') as mock_chat:
        mock_chat.return_value = AsyncMock(content=mock_json)
        
        classification = await classifier.classify_article(article, ["USD", "EUR"])
        
        assert classification.relevance["USD"] == 0.95
        assert classification.sentiment["USD"] == 0.6
        assert not classification.quality_flags["non_econ"]


@pytest.mark.asyncio
async def test_narrative_generator():
    """Test narrative generation with mocked LLM."""
    llm_manager = MockLLMManager()
    generator = NarrativeGenerator(llm_manager)
    
    mock_narrative = "Recent news shows bullish sentiment for USD driven by strong economic data and hawkish Fed signals."
    
    with patch.object(llm_manager, 'chat_with_model') as mock_chat:
        mock_chat.return_value = AsyncMock(content=mock_narrative)
        
        narrative = await generator.generate_news_narrative(
            pair="USD/EUR",
            sent_base=0.4,
            sent_quote=-0.1,
            pair_bias=0.5,
            confidence="high",
            n_articles=15,
            top_evidence=[{"title": "Fed Signals More Hikes"}]
        )
        
        assert len(narrative) > 0
        assert "USD" in narrative or "bullish" in narrative.lower()
```

**Coverage Target**: >80%

### Step 7: Integration Tests

**File**: `tests/integration/test_market_intelligence/test_extraction_integration.py`

Test extraction with real LLM (optional):

```python
import pytest
import os
from src.llm.manager import LLMManager
from src.data_collection.market_intelligence.extractors import CalendarExtractor, NewsClassifier

@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_calendar_extraction():
    """Test calendar extraction with real LLM."""
    if not os.getenv("COPILOT_ACCESS_TOKEN"):
        pytest.skip("COPILOT_ACCESS_TOKEN not set")
    
    llm_manager = LLMManager()
    extractor = CalendarExtractor(llm_manager)
    
    # Real search result example
    result = SerperSearchResult(
        title="ECB Economic Calendar 2025",
        url="https://ecb.europa.eu",
        snippet="Upcoming ECB meeting on October 30, 2025. Interest rate decision expected.",
        position=1
    )
    
    events = await extractor.extract_events_from_snippet(result, "EUR")
    
    # Should extract at least one event
    assert len(events) > 0
    print(f"Extracted {len(events)} events")
    if events:
        print(f"Sample: {events[0].event} on {events[0].when_utc}")
```

## Key Design Decisions

1. **Task-optimized model selection**: gpt-5-mini for high-volume extraction/classification, gpt-4o for narrative quality
2. **Structured JSON extraction**: All LLM outputs are JSON-validated with fallbacks
3. **Retry logic**: 2 attempts for LLM calls (transient failures)
4. **Parallel processing**: Batch extraction/classification for speed
5. **Quality filtering**: Remove clickbait and non-economic news
6. **Deduplication**: Remove duplicate events by date + event name
7. **Graceful degradation**: Return neutral classifications on LLM failures
8. **Markdown code block handling**: Clean LLM responses that include ```json``` wrappers

## Files to Create

- `src/data_collection/market_intelligence/models.py`
- `src/data_collection/market_intelligence/extractors/__init__.py`
- `src/data_collection/market_intelligence/extractors/calendar_extractor.py`
- `src/data_collection/market_intelligence/extractors/news_classifier.py`
- `src/data_collection/market_intelligence/extractors/narrative_generator.py`
- `tests/unit/test_market_intelligence/test_extractors.py`
- `tests/integration/test_market_intelligence/test_extraction_integration.py`

## Dependencies

- Phase 1.4: Serper client and collectors
- Phase 0 (LLM): `src/llm/agent_helpers.py` - chat_with_model, get_recommended_model_for_task
- Phase 0 (LLM): `src/llm/manager.py` - LLMManager
- Phase 0.4: Decorators - retry, timeout, log_execution
- Phase 0.1: Logging - structured logging
- Python: json, hashlib, datetime

## Configuration

Model usage (from market-intelligence.md):

- **gpt-5-mini**: Calendar extraction, news classification (fast, cheap, high volume)
- **gpt-4o**: Narrative generation (quality, low volume)

Cost per agent call (LLM only):

- News classification: 10-15 calls × ~$0.0001 = $0.0015
- Calendar extraction: 3-5 calls × ~$0.0002 = $0.001
- Narrative generation: 1 call × ~$0.01 = $0.01
- **Total LLM cost**: ~$0.01-0.02 per agent call

**Combined with Phase 1.4 Serper cost** (~$0.02-0.03): **Total Phase 1.4+1.5: ~$0.04-0.05 per agent call**

## Validation

Manual testing:

```python
from src.llm.manager import LLMManager
from src.data_collection.market_intelligence import SerperClient, CalendarCollector
from src.data_collection.market_intelligence.extractors import CalendarExtractor

# Initialize
llm_manager = LLMManager()
serper = SerperClient()
collector = CalendarCollector(serper)
extractor = CalendarExtractor(llm_manager)

# Get calendar URLs
results = await collector.collect_calendar_urls("USD")

# Extract events
events = await extractor.extract_events_batch(results, "USD")
print(f"Extracted {len(events)} events")

for event in events[:5]:
    print(f"- {event.event} on {event.when_utc.date()} ({event.importance})")
```

## Success Criteria

- Calendar extractor produces valid EconomicEvent objects
- Event importance classification matches high/medium/low rules
- News classifier returns relevance and sentiment scores in valid ranges
- Quality flags identify clickbait and non-economic articles
- Narrative generator produces coherent 1-2 sentence summaries
- All LLM calls use task-appropriate models (gpt-5-mini vs gpt-4o)
- JSON parsing handles markdown code blocks
- Retry logic handles LLM failures gracefully
- Batch processing works in parallel
- All unit tests pass with mocked LLM (>80% coverage)
- Integration tests work with real LLM (optional)
- Code follows Phase 0 patterns

## Next Phase

After Phase 1.5 completes, proceed to **Phase 1.6: Market Intelligence - Aggregation & Node**, which will:

- Create intelligence aggregator combining calendar + news
- Compute policy bias from events
- Calculate next high-impact event ETA
- Create Market Intelligence LangGraph node
- Write integration tests for complete flow

### To-dos

- [ ] Create economic event models (EconomicEvent, NewsClassification) with proper schemas
- [ ] Implement calendar event extractor using gpt-5-mini for fast extraction
- [ ] Implement news sentiment classifier using gpt-5-mini for high-volume classification
- [ ] Implement narrative generator using gpt-4o for quality summaries
- [ ] Add JSON schema validation and markdown code block handling
- [ ] Implement retry logic and graceful degradation for LLM failures
- [ ] Write comprehensive unit tests with mocked LLM responses (>80% coverage)
- [ ] Write optional integration tests with real LLM using pytest markers