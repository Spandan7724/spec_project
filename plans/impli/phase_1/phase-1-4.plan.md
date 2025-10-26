<!-- f67714c1-a54f-4e8d-9617-16955f212afc a09e35d2-3594-4729-9d81-17eed9bb7327 -->
# Phase 1.4: Market Intelligence - Serper Integration

## Overview

Implement Serper API client for discovering news articles and economic calendar events related to currencies. This establishes the data collection foundation for the Market Intelligence agent, which will be enhanced with LLM extraction in Phase 1.5.

**Key Design Decision**: Use generalized search over site-specific scraping for robustness and scalability across currencies.

## Implementation Steps

### Step 1: Serper Client Base

**File**: `src/data_collection/market_intelligence/serper_client.py`

Create the core Serper API client with news and general search capabilities:

**Core Functionality**:

```python
import httpx
from typing import List, Dict, Optional
from dataclasses import dataclass
from src.utils.decorators import retry, timeout, log_execution
from src.utils.logging import get_logger
from src.config import get_config

logger = get_logger(__name__)

@dataclass
class SerperNewsResult:
    """Single news result from Serper /news endpoint"""
    title: str
    url: str
    source: str
    snippet: str
    date: str  # Date string from Serper
    position: int  # Result position

@dataclass
class SerperSearchResult:
    """General search result from Serper /search endpoint"""
    title: str
    url: str
    snippet: str
    position: int

class SerperClient:
    """Client for Serper.dev API with news and search endpoints"""
    
    # Trusted financial news sources
    WHITELISTED_DOMAINS = [
        "reuters.com", "bloomberg.com", "ft.com", "wsj.com",
        "apnews.com", "bbc.com", "cnbc.com", "marketwatch.com",
        "economist.com", "fxstreet.com", "forex.com",
        "investing.com", "tradingeconomics.com"
    ]
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Serper client.
        
        Args:
            api_key: Serper API key (if None, loads from env)
        """
        self.api_key = api_key or self._get_api_key_from_env()
        self.base_url = "https://google.serper.dev"
    
    def _get_api_key_from_env(self) -> str:
        """Get API key from environment"""
        import os
        api_key = os.getenv("SERPER_API_KEY")
        if not api_key:
            raise ValueError("SERPER_API_KEY not found in environment")
        return api_key
    
    @retry(max_attempts=3, delay=1.0, exceptions=(httpx.HTTPError, httpx.TimeoutException))
    @timeout(30.0)
    @log_execution(log_args=False, log_result=False)
    async def search_news(
        self, 
        query: str, 
        time_range: str = "qdr:d",  # Last day
        num_results: int = 20
    ) -> List[SerperNewsResult]:
        """
        Search news articles using Serper /news endpoint.
        
        Args:
            query: Search query string
            time_range: Time range (qdr:d = last day, qdr:w = last week)
            num_results: Number of results to return (max 100)
        
        Returns:
            List of news results, filtered by whitelist
        """
        logger.info(
            f"Searching news: {query}",
            extra={"query": query, "time_range": time_range}
        )
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{self.base_url}/news",
                headers={
                    "X-API-KEY": self.api_key,
                    "Content-Type": "application/json"
                },
                json={
                    "q": query,
                    "tbs": time_range,
                    "num": num_results
                }
            )
            
            response.raise_for_status()
            data = response.json()
            
            articles = data.get("news", [])
            
            # Convert to SerperNewsResult and filter by whitelist
            results = []
            for idx, article in enumerate(articles):
                url = article.get("link", "")
                
                # Filter by domain whitelist
                if any(domain in url for domain in self.WHITELISTED_DOMAINS):
                    results.append(SerperNewsResult(
                        title=article.get("title", ""),
                        url=url,
                        source=article.get("source", ""),
                        snippet=article.get("snippet", ""),
                        date=article.get("date", ""),
                        position=idx + 1
                    ))
            
            logger.info(
                f"Found {len(results)} whitelisted articles out of {len(articles)} total",
                extra={"query": query, "filtered": len(results), "total": len(articles)}
            )
            
            return results
    
    @retry(max_attempts=3, delay=1.0, exceptions=(httpx.HTTPError, httpx.TimeoutException))
    @timeout(30.0)
    @log_execution(log_args=False, log_result=False)
    async def search(
        self, 
        query: str, 
        num_results: int = 10
    ) -> List[SerperSearchResult]:
        """
        General search using Serper /search endpoint (for calendar URLs).
        
        Args:
            query: Search query string
            num_results: Number of results to return
        
        Returns:
            List of search results
        """
        logger.info(
            f"Searching general: {query}",
            extra={"query": query}
        )
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{self.base_url}/search",
                headers={
                    "X-API-KEY": self.api_key,
                    "Content-Type": "application/json"
                },
                json={
                    "q": query,
                    "num": num_results
                }
            )
            
            response.raise_for_status()
            data = response.json()
            
            results = []
            for idx, result in enumerate(data.get("organic", [])):
                results.append(SerperSearchResult(
                    title=result.get("title", ""),
                    url=result.get("link", ""),
                    snippet=result.get("snippet", ""),
                    position=idx + 1
                ))
            
            return results
    
    async def health_check(self) -> bool:
        """
        Check if Serper API is reachable.
        
        Returns:
            True if API is healthy
        """
        try:
            # Simple test query
            _ = await self.search("test", num_results=1)
            return True  # API responded without error
        except Exception as e:
            logger.error(f"Serper health check failed: {e}")
            return False
```

**Key Features**:

- Retry decorator for network resilience (3 attempts)
- Timeout protection (30 seconds)
- Domain whitelisting for trusted sources
- Structured logging with query tracking
- Separate endpoints for news vs general search
- Health check for monitoring

### Step 2: Calendar Collector

**File**: `src/data_collection/market_intelligence/calendar_collector.py`

Build queries and collect economic calendar search results:

```python
from typing import List, Dict, Optional
from datetime import datetime
from src.data_collection.market_intelligence.serper_client import SerperClient, SerperSearchResult
from src.utils.logging import get_logger

logger = get_logger(__name__)

class CalendarCollector:
    """Collect economic calendar URLs using Serper search"""
    
    # Currency to country/region name mapping
    CURRENCY_NAMES = {
        "USD": "US Dollar",
        "EUR": "Euro",
        "GBP": "British Pound",
        "JPY": "Japanese Yen",
        "CHF": "Swiss Franc",
        "CAD": "Canadian Dollar",
        "AUD": "Australian Dollar",
        "NZD": "New Zealand Dollar"
    }
    
    def __init__(self, serper_client: SerperClient):
        self.serper = serper_client
    
    async def collect_calendar_urls(
        self, 
        currency: str,
        month: Optional[str] = None,
        year: Optional[int] = None
    ) -> List[SerperSearchResult]:
        """
        Search for economic calendar pages for a currency.
        
        Args:
            currency: Currency code (e.g., "USD", "EUR")
            month: Month name (e.g., "October"), if None uses current
            year: Year, if None uses current
        
        Returns:
            List of search results with calendar URLs
        """
        # Default to current month/year
        now = datetime.now()
        if not month:
            month = now.strftime("%B")  # Full month name
        if not year:
            year = now.year
        
        # Get currency full name
        currency_name = self.CURRENCY_NAMES.get(currency, currency)
        
        # Build generalized search query (from market-intelligence.md)
        query = f'("{currency}" OR "{currency_name}") economic calendar events {month} {year}'
        
        logger.info(
            f"Collecting calendar for {currency}",
            extra={"currency": currency, "month": month, "year": year, "query": query}
        )
        
        # Search with Serper
        results = await self.serper.search(query, num_results=10)
        
        logger.info(
            f"Found {len(results)} calendar URLs for {currency}",
            extra={"currency": currency, "count": len(results)}
        )
        
        return results
```

### Step 3: News Collector

**File**: `src/data_collection/market_intelligence/news_collector.py`

Build currency-specific news queries:

```python
from typing import List
from datetime import datetime
from src.data_collection.market_intelligence.serper_client import SerperClient, SerperNewsResult
from src.utils.logging import get_logger

logger = get_logger(__name__)

class NewsCollector:
    """Collect currency-related news using Serper"""
    
    CURRENCY_NAMES = {
        "USD": "US Dollar",
        "EUR": "Euro",
        "GBP": "British Pound",
        "JPY": "Japanese Yen",
        "CHF": "Swiss Franc",
        "CAD": "Canadian Dollar",
        "AUD": "Australian Dollar",
        "NZD": "New Zealand Dollar"
    }
    
    def __init__(self, serper_client: SerperClient):
        self.serper = serper_client
    
    async def collect_currency_news(
        self, 
        currency: str,
        hours_back: int = 24
    ) -> List[SerperNewsResult]:
        """
        Collect news articles related to a currency.
        
        Args:
            currency: Currency code (e.g., "USD")
            hours_back: How many hours of news to fetch (24, 48, 168=week)
        
        Returns:
            List of news articles
        """
        currency_name = self.CURRENCY_NAMES.get(currency, currency)
        
        # Build search query (from market-intelligence.md pattern)
        query = f'("{currency}" OR "{currency_name}") (currency OR forex OR rates OR economy)'
        
        # Determine time range
        if hours_back <= 24:
            time_range = "qdr:d"  # Last day
        elif hours_back <= 168:
            time_range = "qdr:w"  # Last week
        else:
            time_range = "qdr:m"  # Last month
        
        logger.info(
            f"Collecting news for {currency}",
            extra={"currency": currency, "hours_back": hours_back, "query": query}
        )
        
        results = await self.serper.search_news(
            query=query,
            time_range=time_range,
            num_results=20
        )
        
        logger.info(
            f"Found {len(results)} news articles for {currency}",
            extra={"currency": currency, "count": len(results)}
        )
        
        return results
    
    async def collect_pair_news(
        self, 
        base: str, 
        quote: str,
        hours_back: int = 24
    ) -> Dict[str, List[SerperNewsResult]]:
        """
        Collect news for both currencies in a pair.
        
        Args:
            base: Base currency
            quote: Quote currency
            hours_back: Hours of news to fetch
        
        Returns:
            Dict with 'base' and 'quote' keys containing news lists
        """
        # Fetch news for both currencies in parallel
        import asyncio
        
        base_news, quote_news = await asyncio.gather(
            self.collect_currency_news(base, hours_back),
            self.collect_currency_news(quote, hours_back)
        )
        
        return {
            "base": base_news,
            "quote": quote_news
        }
```

### Step 4: Module Exports

**File**: `src/data_collection/market_intelligence/__init__.py`

```python
"""Market Intelligence data collection using Serper API."""

from src.data_collection.market_intelligence.serper_client import (
    SerperClient,
    SerperNewsResult,
    SerperSearchResult
)
from src.data_collection.market_intelligence.calendar_collector import CalendarCollector
from src.data_collection.market_intelligence.news_collector import NewsCollector

__all__ = [
    "SerperClient",
    "SerperNewsResult",
    "SerperSearchResult",
    "CalendarCollector",
    "NewsCollector"
]
```

### Step 5: Unit Tests

**Files**:

- `tests/unit/test_market_intelligence/__init__.py`
- `tests/unit/test_market_intelligence/test_serper_client.py`
- `tests/unit/test_market_intelligence/test_collectors.py`

**Test Coverage**:

```python
import pytest
from unittest.mock import patch, AsyncMock
from src.data_collection.market_intelligence import SerperClient, NewsCollector

@pytest.mark.asyncio
async def test_serper_news_search():
    """Test Serper news search with mocked response."""
    client = SerperClient(api_key="test_key")
    
    mock_response = {
        "news": [
            {
                "title": "Fed Raises Rates",
                "link": "https://reuters.com/article1",
                "source": "Reuters",
                "snippet": "The Federal Reserve...",
                "date": "1 day ago"
            }
        ]
    }
    
    with patch("httpx.AsyncClient.post") as mock_post:
        mock_post.return_value = AsyncMock(
            status_code=200,
            json=lambda: mock_response
        )
        
        results = await client.search_news("USD economy")
        
        assert len(results) == 1
        assert results[0].title == "Fed Raises Rates"
        assert "reuters.com" in results[0].url


@pytest.mark.asyncio
async def test_serper_domain_filtering():
    """Test that non-whitelisted domains are filtered out."""
    client = SerperClient(api_key="test_key")
    
    mock_response = {
        "news": [
            {"title": "Article 1", "link": "https://reuters.com/a1", "source": "Reuters", "snippet": "...", "date": "1d"},
            {"title": "Article 2", "link": "https://spam-site.com/a2", "source": "Spam", "snippet": "...", "date": "1d"}
        ]
    }
    
    with patch("httpx.AsyncClient.post") as mock_post:
        mock_post.return_value = AsyncMock(
            status_code=200,
            json=lambda: mock_response
        )
        
        results = await client.search_news("test")
        
        # Only whitelisted domain should be returned
        assert len(results) == 1
        assert "reuters.com" in results[0].url


@pytest.mark.asyncio
async def test_serper_retry_on_failure():
    """Test retry logic on network failure."""
    client = SerperClient(api_key="test_key")
    
    with patch("httpx.AsyncClient.post") as mock_post:
        # First 2 calls fail, 3rd succeeds
        mock_post.side_effect = [
            httpx.HTTPError("Network error"),
            httpx.HTTPError("Network error"),
            AsyncMock(status_code=200, json=lambda: {"news": []})
        ]
        
        results = await client.search_news("test")
        
        # Should succeed after retries
        assert isinstance(results, list)
        assert mock_post.call_count == 3


@pytest.mark.asyncio
async def test_news_collector_currency_query():
    """Test NewsCollector builds correct queries."""
    client = SerperClient(api_key="test_key")
    collector = NewsCollector(client)
    
    with patch.object(client, 'search_news') as mock_search:
        mock_search.return_value = []
        
        await collector.collect_currency_news("USD")
        
        # Verify query includes currency and keywords
        call_args = mock_search.call_args
        query = call_args[1]['query']
        assert "USD" in query or "US Dollar" in query
        assert any(kw in query for kw in ["currency", "forex", "rates", "economy"])
```

**Coverage Target**: >80%

### Step 6: Integration Tests

**File**: `tests/integration/test_market_intelligence/test_serper_integration.py`

Test with real Serper API (optional, use pytest markers):

```python
import pytest
import os
from src.data_collection.market_intelligence import SerperClient, NewsCollector, CalendarCollector

@pytest.mark.integration
@pytest.mark.asyncio
async def test_serper_real_api():
    """Test with real Serper API (requires SERPER_API_KEY)."""
    api_key = os.getenv("SERPER_API_KEY")
    if not api_key:
        pytest.skip("SERPER_API_KEY not set")
    
    client = SerperClient(api_key=api_key)
    
    # Test news search
    results = await client.search_news("USD economy", num_results=5)
    
    assert len(results) > 0
    assert all(hasattr(r, 'title') for r in results)
    assert all(hasattr(r, 'url') for r in results)
    
    # Verify whitelisting
    for result in results:
        assert any(domain in result.url for domain in client.WHITELISTED_DOMAINS)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_news_collector_real():
    """Test NewsCollector with real API."""
    api_key = os.getenv("SERPER_API_KEY")
    if not api_key:
        pytest.skip("SERPER_API_KEY not set")
    
    client = SerperClient(api_key=api_key)
    collector = NewsCollector(client)
    
    # Collect USD news
    news = await collector.collect_currency_news("USD", hours_back=24)
    
    assert len(news) > 0
    print(f"Found {len(news)} USD articles")
    
    # Print sample
    if news:
        print(f"Sample: {news[0].title}")
```

## Key Design Decisions

1. **Generalized search over site-specific scraping**: More robust to website changes, scales to new currencies
2. **Domain whitelisting**: Filter for trusted financial news sources only
3. **Retry with exponential backoff**: Handle transient API failures gracefully
4. **Timeout protection**: Prevent hanging on slow API responses
5. **Structured results**: Dataclasses for type safety and clarity
6. **Separate news/search endpoints**: Different use cases (news articles vs calendar URLs)
7. **Parallel collection**: Fetch base and quote currency news simultaneously
8. **Configuration-driven**: API key from environment, domains from config

## Files to Create

- `src/data_collection/market_intelligence/__init__.py`
- `src/data_collection/market_intelligence/serper_client.py`
- `src/data_collection/market_intelligence/calendar_collector.py`
- `src/data_collection/market_intelligence/news_collector.py`
- `tests/unit/test_market_intelligence/__init__.py`
- `tests/unit/test_market_intelligence/test_serper_client.py`
- `tests/unit/test_market_intelligence/test_collectors.py`
- `tests/integration/test_market_intelligence/__init__.py`
- `tests/integration/test_market_intelligence/test_serper_integration.py`

## Dependencies

- `httpx` (already in pyproject.toml) - for async HTTP requests
- Phase 0.4: Decorators (`src/utils/decorators.py`) - retry, timeout, log_execution
- Phase 0.1: Config (`src/config.py`) - for configuration loading
- Phase 0.1: Logging (`src/utils/logging.py`) - for structured logging
- Environment: `SERPER_API_KEY` required

## Configuration

Update `.env.example`:

```bash
# Market Intelligence
SERPER_API_KEY=your_serper_key_here
```

## Validation

Manual testing:

```python
from src.data_collection.market_intelligence import SerperClient, NewsCollector

# Initialize
client = SerperClient()  # Loads from env

# Test news search
news = await client.search_news("USD economy", time_range="qdr:d")
print(f"Found {len(news)} articles")
for article in news[:3]:
    print(f"- {article.title} ({article.source})")

# Test collector
collector = NewsCollector(client)
pair_news = await collector.collect_pair_news("USD", "EUR")
print(f"USD: {len(pair_news['base'])} articles")
print(f"EUR: {len(pair_news['quote'])} articles")

# Test health check
is_healthy = await client.health_check()
print(f"API Health: {'OK' if is_healthy else 'FAIL'}")
```

## Success Criteria

- Serper client successfully fetches news articles
- Domain whitelisting filters to trusted sources only
- Retry logic handles transient failures (3 attempts)
- Timeout prevents hanging (30-second limit)
- Calendar collector builds correct search queries
- News collector handles both single currency and pairs
- All unit tests pass with mocked API (>80% coverage)
- Integration tests work with real API (optional)
- Health check accurately reflects API status
- Structured logging tracks all API calls
- Code follows Phase 0 patterns

## Cost Estimate

Per Phase 1.4 operations (Serper API only, no LLM yet):

- News search: 2-3 queries × $0.005 = $0.010-0.015
- Calendar search: 2-3 queries × $0.005 = $0.010-0.015
- **Total per agent call**: ~$0.02-0.03

**Note**: LLM extraction costs will be added in Phase 1.5 (~$0.02 additional).

## Next Phase

After Phase 1.4 completes, proceed to **Phase 1.5: Market Intelligence - LLM Extraction**, which will:

- Create calendar event extractor using gpt-5-mini
- Create news sentiment classifier using gpt-5-mini
- Create narrative generator using gpt-4o
- Implement JSON schema validation
- Add LLM error handling and retries
- Write extraction tests with mock LLM

### To-dos

- [ ] Create Serper API client with news and general search endpoints
- [ ] Implement domain whitelisting for trusted financial news sources
- [ ] Create calendar collector for economic calendar URL discovery
- [ ] Create news collector for currency-specific news gathering
- [ ] Add retry logic and timeout protection for API calls
- [ ] Write comprehensive unit tests with mocked API responses (>80% coverage)
- [ ] Write optional integration tests with real Serper API using pytest markers
