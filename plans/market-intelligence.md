<!-- 537cf47b-219b-4e1b-b98e-4b6d556a014a 2fba26dd-97a5-46ff-9241-5628992cfd81 -->
# Serper API Integration Plan (Revised & Updated)

## Project Status

**What Remains:**
- ✅ Core infrastructure (models, rate collectors, exchange rate providers)
- ✅ LLM infrastructure (multi-model support already configured)
- ✅ Working Serper test examples
- ✅ Cache manager and decision engine

### ✅ Phase 2: LLM Multi-Model Setup (COMPLETED)

**Configured Providers:**

```yaml
llm:
  default_provider: "copilot"
  providers:
    copilot:
      model: "gpt-4o-2024-11-20"  # For narratives, complex reasoning
      enabled: true
    copilot_mini:
      model: "gpt-5-mini"  # For fast classification, extraction
      enabled: true
    copilot_claude:
      model: "claude-3.5-sonnet"  # For deep analysis
      enabled: true
```

**Task-Specific Model Recommendations:**

| Task | Model | Reason |
|------|-------|--------|
| News Classification | gpt-5-mini | Fast, cheap, high volume |
| Economic Data Extraction | gpt-5-mini | Simple structured extraction |
| HTML Calendar Parsing | gpt-5-mini | Pattern recognition |
| Narrative Generation | gpt-4o-2024-11-20 | Better coherence |
| Deep Analysis | claude-3.5-sonnet | Superior reasoning |

**Helper Functions Available:**

```python
from src.llm.agent_helpers import chat_with_model, get_recommended_model_for_task

# Use specific model
response = await chat_with_model(messages, "gpt-5-mini", llm_manager)

# Use recommended model for task type
model = get_recommended_model_for_task("sentiment_analysis")  # Returns "gpt-5-mini"
response = await chat_with_model(messages, model, llm_manager)
```

### ✅ Phase 3: Proof of Concept Testing (COMPLETED)

**Test Files Created:**

1. ✅ `test_calendar_generalized.py` - Serper economic calendar extraction demo
2. ✅ `test_news_sentiment.py` - Serper news sentiment analysis demo
3. ✅ `test_comparison_crawl_vs_serper.py` - Objective comparison (deleted after decision)

**Key Findings from Tests:**

**Economic Calendar (test_calendar_generalized.py):**
```python
# Generalized search approach works better than site-specific scraping
# Example output:
{
  "when_utc": "2025-10-30T12:15:00Z",
  "event": "ECB Monetary Policy Statement",
  "importance": "high",
  "currency": "EUR",
  "country": "EA",
  "source": "ecb.europa.eu"
}
```

**News Sentiment (test_news_sentiment.py):**
```python
# Successfully demonstrated:
# 1. Serper search with 2 queries (USD, EUR)
# 2. gpt-5-mini classification per article
# 3. Aggregated sentiment output
{
  "pair": "USD/EUR",
  "sentiment": {"USD": 0.15, "EUR": -0.05},
  "pair_bias": 0.20,  # Bullish USD
  "confidence": "medium",
  "n_articles": 16
}
```

**Comparison Results:**
- crawl4ai: 6 "articles" (actually navigation links) ❌
- Serper: 16 real articles from multiple sources ✅
- **Decision: Use Serper**

---

## 🎯 Phase 4: Implementation Plan (CURRENT)

### Overview

Implement a production-ready data collection system using:

- **Serper API** for news discovery and calendar URL discovery
- **gpt-5-mini** for fast classification, extraction (high volume, cheap)
- **gpt-4o-2024-11-20** for narrative generation (low volume, quality)
- **Structured data contracts** with confidence scoring
- **Cost optimization** through strategic model selection

### Cost Estimates (Per Agent Call)

| Component | Model | Calls | Cost per Call | Subtotal |
|-----------|-------|-------|---------------|----------|
| Serper news search | N/A | 2-3 | $0.005 | $0.015 |
| News classification | gpt-5-mini | 10-15 | ~$0.0001 | $0.0015 |
| Calendar search | N/A | 2-3 | $0.005 | $0.015 |
| Calendar extraction | gpt-5-mini | 3-5 | ~$0.0002 | $0.001 |
| Narrative generation | gpt-4o | 1 | ~$0.01 | $0.01 |
| **TOTAL** | | | | **~$0.04-0.05** |

**Monthly Cost (30 runs/day):** ~$45-50/month
**Value:** Production-quality, reliable data with zero maintenance

---

## Part A: Economic Calendar

### Strategy Revision

**Original Plan:** Direct scraping of official sites (ECB, Fed, BLS, BEA)
**Reality Check:** 
- BLS returned 403 errors
- BEA ICS link returned HTML instead of .ics
- Fed uses custom HTML structures
- **Too brittle!**

**New Approach: Generalized Search**

Instead of site-specific scraping, use Serper to find relevant calendar pages for any currency:

```python
# Query pattern that works
query = f'("{currency}" OR "{currency_name}") economic calendar events {month} {year}'
# Example: '("USD" OR "US Dollar") economic calendar events October 2025'
```

### 1. Calendar Data Contract

**File: `src/data_collection/economic/models.py`** (NEW)

```python
from dataclasses import dataclass
from datetime import datetime
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
    
    # Computed fields
    @property
    def proximity_minutes(self) -> int:
        """Minutes until event (negative if past)"""
        return int((self.when_utc - datetime.now(timezone.utc)).total_seconds() / 60)
    
    @property
    def is_imminent(self) -> bool:
        """Event within 60 minutes"""
        return 0 <= self.proximity_minutes <= 60
    
    @property
    def is_today(self) -> bool:
        """Event is today"""
        return self.when_utc.date() == datetime.now(timezone.utc).date()
```

### 2. Serper Calendar Collector

**File: `src/data_collection/economic/serper_calendar_collector.py`** (NEW)

```python
import httpx
from typing import List, Dict
from datetime import datetime, timezone
from src.llm.agent_helpers import chat_with_model, get_recommended_model_for_task

class SerperCalendarCollector:
    """Collect economic calendar events via Serper search + LLM extraction"""
    
    def __init__(self, serper_api_key: str, llm_manager):
        self.serper_api_key = serper_api_key
        self.llm_manager = llm_manager
        self.base_url = "https://google.serper.dev/search"
    
    async def collect_events(self, currency: str, days_ahead: int = 14) -> List[EconomicEvent]:
        """Collect economic events for a currency"""
        
        # 1. Build search query
        currency_names = {
            "USD": "US Dollar",
            "EUR": "Euro",
            "GBP": "British Pound",
            "JPY": "Japanese Yen"
        }
        
        query = f'("{currency}" OR "{currency_names.get(currency, currency)}") economic calendar events'
        
        # 2. Search with Serper
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                self.base_url,
                headers={
                    "X-API-KEY": self.serper_api_key,
                    "Content-Type": "application/json"
                },
                json={
                    "q": query,
                    "num": 10
                }
            )
            
            if response.status_code != 200:
                raise Exception(f"Serper error: {response.status_code}")
            
            search_results = response.json()
        
        # 3. Extract events with gpt-5-mini
        events = []
        
        for result in search_results.get("organic", [])[:5]:
            title = result.get("title", "")
            snippet = result.get("snippet", "")
            url = result.get("link", "")
            
            # Use gpt-5-mini for extraction
            extracted = await self._extract_events_from_snippet(
                title, snippet, url, currency
            )
            events.extend(extracted)
        
        # 4. Filter to date range
        now = datetime.now(timezone.utc)
        filtered = [
            e for e in events
            if 0 <= (e.when_utc - now).days <= days_ahead
        ]
        
        # 5. Deduplicate
        return self._deduplicate(filtered)
    
    async def _extract_events_from_snippet(
        self, title: str, snippet: str, url: str, currency: str
    ) -> List[EconomicEvent]:
        """Extract structured events from search result using gpt-5-mini"""
        
        prompt = f'''Extract economic calendar events from this search result.

Title: {title}
Snippet: {snippet}
Currency: {currency}

Return a JSON array of events with fields:
- date: ISO format (YYYY-MM-DD)
- time: ISO time (HH:MM) or null if not specified
- event: Event name/description
- importance: "high", "medium", or "low" (high = central bank decisions, GDP, CPI, NFP, employment)

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
        response = await chat_with_model(messages, model, self.llm_manager)
        
        # Parse JSON
        import json
        try:
            events_json = json.loads(response.content)
            
            # Convert to EconomicEvent objects
            events = []
            for e in events_json:
                # Parse date/time
                date_str = e.get("date")
                time_str = e.get("time")
                
                if time_str:
                    dt_str = f"{date_str}T{time_str}:00Z"
                    when_utc = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
                else:
                    dt_str = f"{date_str}T00:00:00Z"
                    when_utc = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
                
                event = EconomicEvent(
                    when_utc=when_utc,
                    when_local=when_utc,  # TODO: Add timezone conversion
                    timezone="UTC",
                    country=self._currency_to_country(currency),
                    currency=currency,
                    event=e.get("event", ""),
                    importance=e.get("importance", "medium"),
                    source=url,
                    source_url=url
                )
                events.append(event)
            
            return events
        except Exception as e:
            # Extraction failed, return empty
            return []
    
    def _currency_to_country(self, currency: str) -> str:
        """Map currency to country code"""
        mapping = {
            "USD": "US",
            "EUR": "EA",  # Euro Area
            "GBP": "GB",
            "JPY": "JP"
        }
        return mapping.get(currency, "UNKNOWN")
    
    def _deduplicate(self, events: List[EconomicEvent]) -> List[EconomicEvent]:
        """Remove duplicate events based on date + event name"""
        seen = set()
        unique = []
        
        for event in events:
            key = f"{event.when_utc.date()}_{event.event.lower()}"
            if key not in seen:
                seen.add(key)
                unique.append(event)
        
        return sorted(unique, key=lambda e: e.when_utc)
```

---

## Part B: News Sentiment

### 3. Serper Client

**File: `src/services/serper_client.py`** (NEW)

```python
import httpx
from typing import List, Dict, Optional
from dataclasses import dataclass

@dataclass
class SerperNewsResult:
    """Single news result from Serper"""
    title: str
    url: str
    source: str
    snippet: str
    date: str
    
class SerperClient:
    """Client for Serper.dev API"""
    
    WHITELISTED_DOMAINS = [
        "reuters.com", "bloomberg.com", "ft.com", "wsj.com",
        "apnews.com", "bbc.com", "cnbc.com", "marketwatch.com",
        "economist.com", "fxstreet.com", "forex.com"
    ]
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://google.serper.dev"
    
    async def search_news(
        self, 
        query: str, 
        time_range: str = "qdr:d",
        num_results: int = 20
    ) -> List[SerperNewsResult]:
        """Search news with /news endpoint"""
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{self.base_url}/news",
                headers={
                    "X-API-KEY": self.api_key,
                    "Content-Type": "application/json"
                },
                json={
                    "q": query,
                    "tbs": time_range,  # qdr:d = last day, qdr:w = last week
                    "num": num_results
                }
            )
            
            if response.status_code != 200:
                raise Exception(f"Serper error: {response.status_code}")
            
            data = response.json()
            articles = data.get("news", [])
            
            # Convert to SerperNewsResult and filter by domain
            results = []
            for article in articles:
                url = article.get("link", "")
                
                # Filter by whitelist
                if any(domain in url for domain in self.WHITELISTED_DOMAINS):
                    results.append(SerperNewsResult(
                        title=article.get("title", ""),
                        url=url,
                        source=article.get("source", ""),
                        snippet=article.get("snippet", ""),
                        date=article.get("date", "")
                    ))
            
            return results
```

### 4. News Classification

**File: `src/data_collection/news/news_classifier.py`** (NEW)

```python
from dataclasses import dataclass
from typing import Dict, List
import json
import hashlib
from datetime import datetime, timezone

from src.llm.agent_helpers import chat_with_model, get_recommended_model_for_task

@dataclass
class NewsClassification:
    """Classification result for a single article"""
    article_id: str
    url: str
    source: str
    title: str
    published_utc: datetime
    relevance: Dict[str, float]  # {"USD": 0.9, "EUR": 0.2}
    sentiment: Dict[str, float]  # {"USD": 0.4, "EUR": -0.1} range: -1 to +1
    quality_flags: Dict[str, bool]  # {clickbait, rumor, duplicate, non_econ}

class NewsClassifier:
    """Classify news articles using gpt-5-mini"""
    
    def __init__(self, llm_manager):
        self.llm_manager = llm_manager
    
    async def classify_article(
        self, 
        title: str, 
        snippet: str, 
        url: str,
        source: str,
        currencies: List[str]
    ) -> NewsClassification:
        """Classify a single article for currency relevance and sentiment"""
        
        prompt = f'''Analyze this financial news article for currency sentiment.

Title: {title}
Snippet: {snippet}
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
        response = await chat_with_model(messages, model, self.llm_manager)
        
        # Parse JSON response
        content = response.content.strip()
        
        # Clean markdown code blocks if present
        if content.startswith("```"):
            content = content.strip("`").strip()
            if content.lower().startswith("json"):
                content = content[4:].strip()
        
        classification_data = json.loads(content)
        
        # Generate article ID from URL
        article_id = hashlib.sha256(url.encode()).hexdigest()[:16]
        
        return NewsClassification(
            article_id=article_id,
            url=url,
            source=source,
            title=title,
            published_utc=datetime.now(timezone.utc),  # TODO: Parse from date string
            relevance=classification_data.get("relevance", {}),
            sentiment=classification_data.get("sentiment", {}),
            quality_flags=classification_data.get("quality_flags", {})
        )
```

### 5. News Aggregator

**File: `src/data_collection/news/news_aggregator.py`** (NEW)

```python
from dataclasses import dataclass
from typing import List, Dict, Any
from datetime import datetime, timezone
import numpy as np

from src.services.serper_client import SerperClient
from src.data_collection.news.news_classifier import NewsClassifier, NewsClassification

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
    
    def __init__(self, serper_client: SerperClient, classifier: NewsClassifier):
        self.serper = serper_client
        self.classifier = classifier
    
    async def get_pair_snapshot(
        self, 
        base: str, 
        quote: str,
        hours_back: int = 24
    ) -> PairNewsSnapshot:
        """Get aggregated news sentiment for a currency pair"""
        
        # 1. Build search queries
        currency_names = {
            "USD": "US Dollar",
            "EUR": "Euro",
            "GBP": "British Pound",
            "JPY": "Japanese Yen"
        }
        
        base_name = currency_names.get(base, base)
        quote_name = currency_names.get(quote, quote)
        
        queries = [
            f'("{base}" OR "{base_name}") (currency OR forex OR rates OR economy)',
            f'("{quote}" OR "{quote_name}") (currency OR forex OR rates OR economy)',
        ]
        
        # 2. Fetch from Serper
        all_articles = []
        for query in queries:
            results = await self.serper.search_news(query, time_range="qdr:d", num_results=10)
            all_articles.extend(results)
        
        # 3. Deduplicate by URL
        seen_urls = set()
        unique_articles = []
        for article in all_articles:
            if article.url not in seen_urls:
                seen_urls.add(article.url)
                unique_articles.append(article)
        
        # 4. Classify each with gpt-5-mini
        classifications = []
        for article in unique_articles:
            try:
                classification = await self.classifier.classify_article(
                    title=article.title,
                    snippet=article.snippet,
                    url=article.url,
                    source=article.source,
                    currencies=[base, quote]
                )
                
                # Filter out low quality
                if not classification.quality_flags.get("non_econ", False):
                    classifications.append(classification)
            except Exception as e:
                # Skip failed classifications
                continue
        
        # 5. Aggregate sentiment
        sent_base, sent_quote, confidence, top_evidence = self._aggregate_sentiment(
            classifications, base, quote
        )
        
        # 6. Calculate pair bias
        pair_bias = sent_base - sent_quote
        
        return PairNewsSnapshot(
            pair=f"{base}/{quote}",
            ts_utc=datetime.now(timezone.utc),
            sent_base=sent_base,
            sent_quote=sent_quote,
            pair_bias=pair_bias,
            confidence=confidence,
            n_articles_used=len(classifications),
            top_evidence=top_evidence
        )
    
    def _aggregate_sentiment(
        self, 
        classifications: List[NewsClassification], 
        base: str, 
        quote: str
    ) -> tuple:
        """Weighted mean of per-article sentiment"""
        
        if not classifications:
            return 0.0, 0.0, "low", []
        
        base_sentiments = []
        quote_sentiments = []
        
        for c in classifications:
            # Filter by relevance threshold
            if c.relevance.get(base, 0) >= 0.3:
                base_sentiments.append(c.sentiment.get(base, 0))
            
            if c.relevance.get(quote, 0) >= 0.3:
                quote_sentiments.append(c.sentiment.get(quote, 0))
        
        # Calculate means
        sent_base = float(np.mean(base_sentiments)) if base_sentiments else 0.0
        sent_quote = float(np.mean(quote_sentiments)) if quote_sentiments else 0.0
        
        # Calculate confidence
        all_sentiments = base_sentiments + quote_sentiments
        variance = float(np.var(all_sentiments)) if all_sentiments else 1.0
        n_articles = len(classifications)
        
        if n_articles >= 10 and variance < 0.3:
            confidence = "high"
        elif n_articles >= 5 and variance < 0.5:
            confidence = "medium"
        else:
            confidence = "low"
        
        # Top evidence (by relevance)
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

### 6. Narrative Generator

**File: `src/data_collection/news/narrative_generator.py`** (NEW)

```python
from src.llm.agent_helpers import chat_with_model, get_recommended_model_for_task

class NarrativeGenerator:
    """Generate human-readable narratives using gpt-4o"""
    
    def __init__(self, llm_manager):
        self.llm_manager = llm_manager
    
    async def generate_narrative(self, snapshot) -> str:
        """Generate 1-2 sentence summary of news sentiment"""
        
        pair_parts = snapshot.pair.split('/')
        base = pair_parts[0]
        quote = pair_parts[1]
        
        bias_text = "bullish" if snapshot.pair_bias > 0.2 else "bearish" if snapshot.pair_bias < -0.2 else "neutral"
        
        prompt = f'''Generate a concise 1-2 sentence summary of current news sentiment for {snapshot.pair}.

Data:
- {base} sentiment: {snapshot.sent_base:+.2f}
- {quote} sentiment: {snapshot.sent_quote:+.2f}
- Pair bias: {snapshot.pair_bias:+.2f} ({bias_text} {base})
- Confidence: {snapshot.confidence}
- Based on {snapshot.n_articles_used} articles

Top headlines:
{chr(10).join(f"- {e['title']}" for e in snapshot.top_evidence[:3])}

Write a professional, concise summary suitable for a financial analysis report.
Focus on what the news means for the currency pair direction.
'''
        
        messages = [
            {"role": "system", "content": "You are a financial analyst summarizing market sentiment."},
            {"role": "user", "content": prompt}
        ]
        
        # Use gpt-4o for narrative quality
        model = get_recommended_model_for_task("summarization")
        response = await chat_with_model(messages, model, self.llm_manager)
        
        return response.content.strip()
```

---

## Part C: Integrated Service

### 7. Market Intelligence Service

**File: `src/data_collection/market_intelligence_service.py`** (NEW)

```python
from typing import Dict, Any, Optional
from datetime import datetime, timezone

from src.services.serper_client import SerperClient
from src.data_collection.economic.serper_calendar_collector import SerperCalendarCollector
from src.data_collection.news.news_classifier import NewsClassifier
from src.data_collection.news.news_aggregator import NewsAggregator, PairNewsSnapshot
from src.data_collection.news.narrative_generator import NarrativeGenerator

class MarketIntelligenceService:
    """Unified service for economic calendar + news sentiment"""
    
    def __init__(self, serper_api_key: str, llm_manager):
        # Initialize components
        self.serper_client = SerperClient(serper_api_key)
        self.calendar_collector = SerperCalendarCollector(serper_api_key, llm_manager)
        self.news_classifier = NewsClassifier(llm_manager)
        self.news_aggregator = NewsAggregator(self.serper_client, self.news_classifier)
        self.narrative_gen = NarrativeGenerator(llm_manager)
    
    async def get_pair_intelligence(
        self, 
        base: str, 
        quote: str
    ) -> Dict[str, Any]:
        """Get complete intelligence snapshot for a currency pair"""
        
        # 1. Get news sentiment
        snapshot = await self.news_aggregator.get_pair_snapshot(base, quote)
        
        # 2. Get calendar events for both currencies
        base_events = await self.calendar_collector.collect_events(base, days_ahead=7)
        quote_events = await self.calendar_collector.collect_events(quote, days_ahead=7)
        
        all_events = base_events + quote_events
        
        # 3. Find next high-impact event
        high_impact_events = [e for e in all_events if e.importance == "high"]
        next_high_event = None
        
        if high_impact_events:
            # Sort by proximity
            high_impact_events.sort(key=lambda e: e.proximity_minutes)
            # Get next upcoming event (positive proximity)
            upcoming = [e for e in high_impact_events if e.proximity_minutes > 0]
            if upcoming:
                next_high_event = upcoming[0]
        
        # 4. Generate narrative
        narrative = await self.narrative_gen.generate_narrative(snapshot)
        
        # 5. Build response
        return {
            "pair": snapshot.pair,
            "ts_utc": snapshot.ts_utc.isoformat(),
            "news": {
                "sent_base": snapshot.sent_base,
                "sent_quote": snapshot.sent_quote,
                "pair_bias": snapshot.pair_bias,
                "confidence": snapshot.confidence,
                "n_articles_used": snapshot.n_articles_used,
                "top_evidence": snapshot.top_evidence,
                "narrative": narrative
            },
            "calendar": {
                "next_high_event": {
                    "when_utc": next_high_event.when_utc.isoformat(),
                    "currency": next_high_event.currency,
                    "event": next_high_event.event,
                    "source_url": next_high_event.source_url,
                    "proximity_minutes": next_high_event.proximity_minutes,
                    "is_imminent": next_high_event.is_imminent
                } if next_high_event else None,
                "total_high_impact_events_7d": len(high_impact_events)
            }
        }
```

---

## Part D: Integration

### 8. Update Economic Agent

**File: `src/agentic/nodes/economic.py`** (MODIFY)

```python
from src.data_collection.market_intelligence_service import MarketIntelligenceService

class EconomicAnalysisAgent:
    def __init__(self, market_intel: Optional[MarketIntelligenceService] = None):
        self.market_intel = market_intel or self._create_default_service()
    
    def _create_default_service(self):
        """Create market intelligence service with config"""
        import os
        from src.llm.manager import LLMManager
        
        serper_api_key = os.getenv("SERPER_API_KEY")
        llm_manager = LLMManager()
        
        return MarketIntelligenceService(serper_api_key, llm_manager)
    
    async def __call__(self, state: AgentGraphState) -> AgentGraphState:
        request = state.request
        
        # Get integrated intelligence
        intel = await self.market_intel.get_pair_intelligence(
            request.base_currency,
            request.quote_currency
        )
        
        # Build economic analysis
        overall_bias = "neutral"
        if intel["news"]["pair_bias"] > 0.2:
            overall_bias = "bullish"
        elif intel["news"]["pair_bias"] < -0.2:
            overall_bias = "bearish"
        
        high_impact_events = []
        if intel["calendar"]["next_high_event"]:
            high_impact_events = [intel["calendar"]["next_high_event"]]
        
        economic = EconomicAnalysis(
            summary=intel["news"]["narrative"],
            overall_bias=overall_bias,
            high_impact_events=high_impact_events,
            news_sentiment_score=intel["news"]["pair_bias"],
            confidence=intel["news"]["confidence"],
            data_source_notes=[
                f"Based on {intel['news']['n_articles_used']} news articles",
                f"{intel['calendar']['total_high_impact_events_7d']} high-impact events in next 7 days"
            ]
        )
        
        return state.with_economic(economic)
```

### 9. Configuration

**File: `config.yaml`** (ADD)

```yaml
serper:
  api_key_env: "SERPER_API_KEY"
  news_time_range: "qdr:d"  # Last day
  max_results_per_query: 20
  domain_whitelist:
    - reuters.com
    - bloomberg.com
    - ft.com
    - wsj.com
    - apnews.com
    - bbc.com
    - cnbc.com
    - marketwatch.com
    - economist.com
    - fxstreet.com
    - forex.com

calendar:
  days_ahead: 14
  importance_keywords:
    high:
      - "policy decision"
      - "FOMC"
      - "ECB decision"
      - "CPI"
      - "NFP"
      - "GDP"
      - "interest rate"
      - "employment"
    medium:
      - "PMI"
      - "ISM"
      - "retail sales"
      - "industrial production"
    low: ["*"]

news_aggregation:
  min_relevance_threshold: 0.3
  premium_sources:
    - reuters.com
    - bloomberg.com
```

---

## Implementation Checklist

### Phase 1: Core Services ✅
- [x] Cleanup old implementations
- [x] Configure multi-model LLM support
- [x] Create proof-of-concept tests

### Phase 2: Data Collection (CURRENT)
- [ ] Create `SerperClient` (`src/services/serper_client.py`)
- [ ] Create `EconomicEvent` model (`src/data_collection/economic/models.py`)
- [ ] Create `SerperCalendarCollector` (`src/data_collection/economic/serper_calendar_collector.py`)
- [ ] Create `NewsClassifier` (`src/data_collection/news/news_classifier.py`)
- [ ] Create `NewsAggregator` (`src/data_collection/news/news_aggregator.py`)
- [ ] Create `NarrativeGenerator` (`src/data_collection/news/narrative_generator.py`)

### Phase 3: Integration
- [ ] Create `MarketIntelligenceService` (`src/data_collection/market_intelligence_service.py`)
- [ ] Update `EconomicAnalysisAgent` (`src/agentic/nodes/economic.py`)
- [ ] Add Serper config to `config.yaml`

### Phase 4: Testing
- [ ] Create integration tests (`test_market_intelligence.py`)
- [ ] Create usage examples (`example_market_intelligence.py`)
- [ ] Test cost tracking

### Phase 5: Monitoring
- [ ] Create `MarketDataMonitor` (`src/monitoring/market_data_monitor.py`)
- [ ] Add Serper API usage tracking
- [ ] Add data quality alerts

---

## Dependencies to Add

```toml
# Add to pyproject.toml dependencies
dependencies = [
    # ... existing ...
    "beautifulsoup4>=4.12.0",  # For HTML parsing if needed
    "numpy>=2.3.2",  # Already present
]
```

---

## Environment Variables

```bash
# Required
SERPER_API_KEY=your_key_here

# Already configured
COPILOT_ACCESS_TOKEN=your_token_here
```

=

## Key Design Decisions

1. **Generalized Search Over Site-Specific Scraping**
   - More robust to website changes
   - Scales to new currencies easily
   - No bot detection issues

2. **Task-Specific Model Selection**
   - gpt-5-mini for high-volume classification
   - gpt-4o-2024-11-20 for low-volume narratives
   - Optimizes cost vs quality

3. **Structured Data Contracts**
   - Clear JSON outputs at each layer
   - Easy to test and validate
   - Confidence scoring built-in

4. **Modular Architecture**
   - SerperClient: API wrapper
   - Classifiers: LLM extraction
   - Aggregators: Business logic
   - Service: Integration layer
   - Easy to test and maintain
