"""
Real News Data Providers for Market Intelligence.

Integrates with multiple news APIs to fetch real financial news,
central bank announcements, and economic analysis for currency markets.
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import httpx
from dataclasses import dataclass
import re

logger = logging.getLogger(__name__)


@dataclass
class NewsArticle:
    """Structured news article data."""
    title: str
    content: str
    source: str
    url: str
    published_at: datetime
    relevance_score: float = 0.0
    sentiment_score: float = 0.0
    currencies_mentioned: List[str] = None
    
    def __post_init__(self):
        if self.currencies_mentioned is None:
            self.currencies_mentioned = []


class NewsAPIProvider:
    """
    Free news provider using NewsAPI.org.
    
    Note: Free tier allows 100 requests/day, 1000/month
    No API key required for developer endpoints in some cases.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.base_url = "https://newsapi.org/v2"
        self.api_key = api_key
        self.session = None
    
    async def __aenter__(self):
        self.session = httpx.AsyncClient(timeout=30.0)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.aclose()
    
    async def fetch_currency_news(self, 
                                currency_pair: str,
                                hours_back: int = 24,
                                max_articles: int = 20) -> List[NewsArticle]:
        """
        Fetch currency-related news from NewsAPI.
        
        Args:
            currency_pair: Currency pair to search for
            hours_back: How many hours back to search
            max_articles: Maximum number of articles to return
            
        Returns:
            List of relevant news articles
        """
        if not self.session:
            raise RuntimeError("Provider not properly initialized - use async context manager")
        
        try:
            # Build search query for currency pair
            base_currency, quote_currency = currency_pair.split('/')
            
            # Create search terms
            search_terms = [
                f"{base_currency} {quote_currency}",
                f"{base_currency}/{quote_currency}",
                f"{base_currency} currency",
                f"{quote_currency} exchange rate"
            ]
            
            query = " OR ".join(f'"{term}"' for term in search_terms[:2])  # Limit query complexity
            
            # Calculate date range
            from_date = (datetime.utcnow() - timedelta(hours=hours_back)).isoformat()
            
            # Build API request
            params = {
                'q': query,
                'from': from_date,
                'sortBy': 'relevancy',
                'pageSize': min(max_articles, 100),
                'language': 'en',
                'domains': 'reuters.com,bloomberg.com,cnbc.com,marketwatch.com,ft.com'
            }
            
            if self.api_key:
                params['apiKey'] = self.api_key
            
            # Make request
            response = await self.session.get(
                f"{self.base_url}/everything",
                params=params
            )
            
            if response.status_code == 200:
                data = response.json()
                articles = []
                
                for article_data in data.get('articles', []):
                    try:
                        article = self._parse_article(article_data, currency_pair)
                        if article and article.relevance_score > 0.1:  # Filter low relevance
                            articles.append(article)
                    except Exception as e:
                        logger.warning(f"Failed to parse article: {e}")
                        continue
                
                logger.info(f"Fetched {len(articles)} relevant articles for {currency_pair}")
                return articles[:max_articles]
            
            else:
                logger.warning(f"NewsAPI request failed: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Failed to fetch news from NewsAPI: {e}")
            return []
    
    def _parse_article(self, article_data: Dict[str, Any], currency_pair: str) -> Optional[NewsArticle]:
        """Parse article data from NewsAPI response."""
        title = article_data.get('title', '')
        description = article_data.get('description', '')
        content = article_data.get('content', description)  # Use description if content not available
        
        if not title or not content:
            return None
        
        # Parse published date
        published_str = article_data.get('publishedAt', '')
        try:
            published_at = datetime.fromisoformat(published_str.replace('Z', '+00:00'))
        except:
            published_at = datetime.utcnow()
        
        # Calculate relevance based on currency mentions
        relevance_score = self._calculate_relevance(title + ' ' + content, currency_pair)
        
        # Extract mentioned currencies
        currencies_mentioned = self._extract_currencies(title + ' ' + content)
        
        return NewsArticle(
            title=title,
            content=content,
            source=article_data.get('source', {}).get('name', 'Unknown'),
            url=article_data.get('url', ''),
            published_at=published_at,
            relevance_score=relevance_score,
            currencies_mentioned=currencies_mentioned
        )
    
    def _calculate_relevance(self, text: str, currency_pair: str) -> float:
        """Calculate relevance score for currency pair."""
        text_lower = text.lower()
        base_currency, quote_currency = currency_pair.split('/')
        
        # Count mentions
        base_mentions = text_lower.count(base_currency.lower())
        quote_mentions = text_lower.count(quote_currency.lower())
        pair_mentions = text_lower.count(f"{base_currency.lower()}/{quote_currency.lower()}")
        
        # Currency-related keywords
        fx_keywords = ['exchange rate', 'currency', 'forex', 'fx', 'central bank', 'monetary policy']
        keyword_matches = sum(1 for keyword in fx_keywords if keyword in text_lower)
        
        # Calculate relevance score (0.0 to 1.0)
        relevance = (base_mentions + quote_mentions + pair_mentions * 2 + keyword_matches) / 10.0
        return min(1.0, relevance)
    
    def _extract_currencies(self, text: str) -> List[str]:
        """Extract currency codes from text."""
        # Find currency codes (3-letter patterns)
        currency_pattern = r'\b[A-Z]{3}\b'
        potential_currencies = re.findall(currency_pattern, text.upper())
        
        # Filter to known currencies
        known_currencies = {
            'USD', 'EUR', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD', 'NZD',
            'CNY', 'INR', 'KRW', 'SGD', 'HKD', 'NOK', 'SEK', 'DKK'
        }
        
        return list(set(curr for curr in potential_currencies if curr in known_currencies))


class RSSNewsProvider:
    """
    RSS feed provider for financial news.
    Free alternative that doesn't require API keys.
    """
    
    def __init__(self):
        # Updated RSS feeds with more reliable sources
        self.rss_feeds = {
            'cnn_money': 'https://rss.cnn.com/rss/money_latest.rss',
            'reuters_business': 'https://feeds.reuters.com/reuters/businessNews',
            'investing_com': 'https://www.investing.com/rss/news.rss',
            'forex_live': 'https://www.forexlive.com/feed/',
            'reuters_world': 'https://feeds.reuters.com/reuters/worldNews'
        }
        self.session = None
    
    async def __aenter__(self):
        self.session = httpx.AsyncClient(
            timeout=30.0,
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'},
            follow_redirects=True  # Handle redirects automatically
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.aclose()
    
    async def fetch_currency_news(self, 
                                currency_pair: str,
                                hours_back: int = 24,
                                max_articles: int = 20) -> List[NewsArticle]:
        """Fetch currency news from RSS feeds."""
        if not self.session:
            raise RuntimeError("Provider not properly initialized")
        
        all_articles = []
        
        # Fetch from multiple feeds with better error handling
        for feed_name, feed_url in self.rss_feeds.items():
            try:
                articles = await self._fetch_rss_feed_with_retry(feed_url, currency_pair, feed_name)
                all_articles.extend(articles)
                logger.debug(f"Fetched {len(articles)} articles from {feed_name}")
            except Exception as e:
                logger.warning(f"Failed to fetch from {feed_name}: {e}")
                continue
        
        # Filter by time and relevance
        cutoff_time = datetime.utcnow() - timedelta(hours=hours_back)
        recent_articles = [
            article for article in all_articles
            if article.published_at > cutoff_time and article.relevance_score > 0.1
        ]
        
        # Sort by relevance and recency
        recent_articles.sort(key=lambda x: (x.relevance_score * 0.7 + 
                                          (1 - (datetime.utcnow() - x.published_at).total_seconds() / (24*3600)) * 0.3), 
                           reverse=True)
        
        return recent_articles[:max_articles]
    
    async def _fetch_rss_feed_with_retry(self, feed_url: str, currency_pair: str, feed_name: str, max_retries: int = 2) -> List[NewsArticle]:
        """Fetch RSS feed with retry logic."""
        for attempt in range(max_retries + 1):
            try:
                return await self._fetch_rss_feed(feed_url, currency_pair, feed_name)
            except Exception as e:
                if attempt == max_retries:
                    raise e
                logger.debug(f"Retry {attempt + 1} for {feed_name}: {e}")
                await asyncio.sleep(1 * (attempt + 1))  # Exponential backoff
        return []
    
    async def _fetch_rss_feed(self, feed_url: str, currency_pair: str, feed_name: str) -> List[NewsArticle]:
        """Fetch and parse RSS feed."""
        try:
            response = await self.session.get(feed_url)
            if response.status_code != 200:
                logger.warning(f"RSS feed {feed_name} returned {response.status_code}")
                return []
            
            # Simple RSS parsing (without external dependencies)
            content = response.text
            articles = []
            
            # Extract items using regex (basic RSS parsing)
            item_pattern = r'<item>(.*?)</item>'
            items = re.findall(item_pattern, content, re.DOTALL)
            
            for item in items:
                title_match = re.search(r'<title><!\[CDATA\[(.*?)\]\]></title>|<title>(.*?)</title>', item)
                desc_match = re.search(r'<description><!\[CDATA\[(.*?)\]\]></description>|<description>(.*?)</description>', item)
                date_match = re.search(r'<pubDate>(.*?)</pubDate>', item)
                link_match = re.search(r'<link>(.*?)</link>', item)
                
                if title_match:
                    title = title_match.group(1) or title_match.group(2) or ''
                    description = desc_match.group(1) or desc_match.group(2) if desc_match else ''
                    link = link_match.group(1) if link_match else ''
                    
                    # Parse date
                    try:
                        date_str = date_match.group(1) if date_match else ''
                        published_at = self._parse_rss_date(date_str)
                    except:
                        published_at = datetime.utcnow()
                    
                    # Calculate relevance
                    relevance = self._calculate_relevance(title + ' ' + description, currency_pair)
                    
                    if relevance > 0.05:  # Only include somewhat relevant articles
                        article = NewsArticle(
                            title=title.strip(),
                            content=description.strip(),
                            source=feed_name,
                            url=link.strip(),
                            published_at=published_at,
                            relevance_score=relevance,
                            currencies_mentioned=self._extract_currencies(title + ' ' + description)
                        )
                        articles.append(article)
            
            return articles
            
        except Exception as e:
            logger.error(f"Failed to parse RSS from {feed_url}: {e}")
            return []
    
    def _parse_rss_date(self, date_str: str) -> datetime:
        """Parse RSS date string.""" 
        if not date_str:
            return datetime.utcnow()
        
        # Common RSS date formats
        formats = [
            '%a, %d %b %Y %H:%M:%S %z',
            '%a, %d %b %Y %H:%M:%S %Z',
            '%Y-%m-%dT%H:%M:%S%z',
            '%Y-%m-%dT%H:%M:%SZ'
        ]
        
        for fmt in formats:
            try:
                if date_str.endswith('Z'):
                    date_str = date_str[:-1] + '+00:00'
                return datetime.strptime(date_str, fmt)
            except:
                continue
        
        return datetime.utcnow()
    
    def _calculate_relevance(self, text: str, currency_pair: str) -> float:
        """Calculate relevance score (same as NewsAPIProvider)."""
        text_lower = text.lower()
        base_currency, quote_currency = currency_pair.split('/')
        
        base_mentions = text_lower.count(base_currency.lower())
        quote_mentions = text_lower.count(quote_currency.lower())
        pair_mentions = text_lower.count(f"{base_currency.lower()}/{quote_currency.lower()}")
        
        fx_keywords = ['exchange rate', 'currency', 'forex', 'fx', 'central bank', 'monetary policy']
        keyword_matches = sum(1 for keyword in fx_keywords if keyword in text_lower)
        
        relevance = (base_mentions + quote_mentions + pair_mentions * 2 + keyword_matches) / 8.0
        return min(1.0, relevance)
    
    def _extract_currencies(self, text: str) -> List[str]:
        """Extract currency codes from text."""
        currency_pattern = r'\b[A-Z]{3}\b'
        potential_currencies = re.findall(currency_pattern, text.upper())
        
        known_currencies = {
            'USD', 'EUR', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD', 'NZD',
            'CNY', 'INR', 'KRW', 'SGD', 'HKD', 'NOK', 'SEK', 'DKK'
        }
        
        return list(set(curr for curr in potential_currencies if curr in known_currencies))


class RedditNewsProvider:
    """
    Reddit financial news provider using public Reddit JSON API.
    No API key required for read-only access.
    """
    
    def __init__(self):
        self.base_url = "https://www.reddit.com"
        self.session = None
    
    async def __aenter__(self):
        self.session = httpx.AsyncClient(
            timeout=30.0,
            headers={'User-Agent': 'CurrencyAssistant/1.0'}
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.aclose()
    
    async def fetch_currency_news(self, 
                                currency_pair: str,
                                hours_back: int = 24,
                                max_articles: int = 15) -> List[NewsArticle]:
        """Fetch currency discussions from relevant subreddits."""
        if not self.session:
            raise RuntimeError("Provider not properly initialized")
        
        # Relevant subreddits for FX news
        subreddits = ['forex', 'investing', 'economics', 'CurrencyTrading', 'finance']
        
        all_posts = []
        
        for subreddit in subreddits:
            try:
                posts = await self._fetch_subreddit_posts(subreddit, currency_pair)
                all_posts.extend(posts)
            except Exception as e:
                logger.warning(f"Failed to fetch from r/{subreddit}: {e}")
                continue
        
        # Filter by time and relevance
        cutoff_time = datetime.utcnow() - timedelta(hours=hours_back)
        recent_posts = [
            post for post in all_posts
            if post.published_at > cutoff_time and post.relevance_score > 0.1
        ]
        
        # Sort by relevance
        recent_posts.sort(key=lambda x: x.relevance_score, reverse=True)
        return recent_posts[:max_articles]
    
    async def _fetch_subreddit_posts(self, subreddit: str, currency_pair: str) -> List[NewsArticle]:
        """Fetch posts from a specific subreddit."""
        try:
            url = f"{self.base_url}/r/{subreddit}/hot.json?limit=25"
            response = await self.session.get(url)
            
            if response.status_code != 200:
                return []
            
            data = response.json()
            posts = []
            
            for post_data in data.get('data', {}).get('children', []):
                post = post_data.get('data', {})
                
                title = post.get('title', '')
                selftext = post.get('selftext', '')
                url = f"https://reddit.com{post.get('permalink', '')}"
                
                # Calculate relevance
                relevance = self._calculate_relevance(title + ' ' + selftext, currency_pair)
                
                if relevance > 0.05:  # Filter low relevance posts
                    # Convert Reddit timestamp
                    created_utc = post.get('created_utc', 0)
                    published_at = datetime.fromtimestamp(created_utc) if created_utc else datetime.utcnow()
                    
                    article = NewsArticle(
                        title=title,
                        content=selftext[:500],  # Limit content length
                        source=f"r/{subreddit}",
                        url=url,
                        published_at=published_at,
                        relevance_score=relevance,
                        currencies_mentioned=self._extract_currencies(title + ' ' + selftext)
                    )
                    posts.append(article)
            
            return posts
            
        except Exception as e:
            logger.error(f"Failed to fetch from r/{subreddit}: {e}")
            return []
    
    def _calculate_relevance(self, text: str, currency_pair: str) -> float:
        """Calculate relevance score for Reddit posts."""
        text_lower = text.lower()
        base_currency, quote_currency = currency_pair.split('/')
        
        # Count currency mentions
        base_mentions = text_lower.count(base_currency.lower())
        quote_mentions = text_lower.count(quote_currency.lower())
        pair_mentions = text_lower.count(f"{base_currency.lower()}{quote_currency.lower()}")
        
        # FX-related terms
        fx_terms = ['exchange', 'rate', 'forex', 'fx', 'currency', 'fed', 'ecb', 'boe', 'central bank']
        term_matches = sum(1 for term in fx_terms if term in text_lower)
        
        relevance = (base_mentions + quote_mentions + pair_mentions * 2 + term_matches * 0.5) / 6.0
        return min(1.0, relevance)
    
    def _extract_currencies(self, text: str) -> List[str]:
        """Extract currency codes from text."""
        currency_pattern = r'\b[A-Z]{3}\b'
        potential_currencies = re.findall(currency_pattern, text.upper())
        
        known_currencies = {
            'USD', 'EUR', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD', 'NZD',
            'CNY', 'INR', 'KRW', 'SGD', 'HKD', 'NOK', 'SEK', 'DKK'
        }
        
        return list(set(curr for curr in potential_currencies if curr in known_currencies))


class NewsAggregator:
    """Aggregates news from multiple providers."""
    
    def __init__(self, newsapi_key: Optional[str] = None):
        self.newsapi_key = newsapi_key
        self.providers = {}
    
    async def fetch_comprehensive_news(self,
                                     currency_pair: str,
                                     hours_back: int = 24,
                                     max_articles: int = 30) -> List[NewsArticle]:
        """
        Fetch news from all available providers.
        
        Args:
            currency_pair: Currency pair to analyze
            hours_back: Hours of historical news
            max_articles: Maximum total articles
            
        Returns:
            Aggregated and deduplicated news articles
        """
        all_articles = []
        
        # Fetch from RSS feeds (always available)
        try:
            async with RSSNewsProvider() as rss_provider:
                rss_articles = await rss_provider.fetch_currency_news(
                    currency_pair, hours_back, max_articles // 2
                )
                all_articles.extend(rss_articles)
                logger.info(f"Fetched {len(rss_articles)} articles from RSS feeds")
        except Exception as e:
            logger.error(f"RSS news fetch failed: {e}")
        
        # Fetch from NewsAPI if key available
        if self.newsapi_key:
            try:
                async with NewsAPIProvider(self.newsapi_key) as news_provider:
                    news_articles = await news_provider.fetch_currency_news(
                        currency_pair, hours_back, max_articles // 2
                    )
                    all_articles.extend(news_articles)
                    logger.info(f"Fetched {len(news_articles)} articles from NewsAPI")
            except Exception as e:
                logger.error(f"NewsAPI fetch failed: {e}")
        
        # Deduplicate by title similarity
        deduplicated = self._deduplicate_articles(all_articles)
        
        # Sort by relevance and recency
        deduplicated.sort(key=lambda x: (x.relevance_score * 0.6 + 
                                       self._recency_score(x.published_at) * 0.4),
                         reverse=True)
        
        return deduplicated[:max_articles]
    
    def _deduplicate_articles(self, articles: List[NewsArticle]) -> List[NewsArticle]:
        """Remove duplicate articles based on title similarity."""
        if not articles:
            return []
        
        unique_articles = []
        seen_titles = set()
        
        for article in articles:
            # Simple deduplication by title (could be improved with fuzzy matching)
            title_key = article.title.lower().strip()
            title_words = set(title_key.split())
            
            # Check if this title is too similar to existing ones
            is_duplicate = False
            for seen_title in seen_titles:
                seen_words = set(seen_title.split())
                similarity = len(title_words.intersection(seen_words)) / max(len(title_words), len(seen_words))
                
                if similarity > 0.8:  # 80% word overlap = duplicate
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_articles.append(article)
                seen_titles.add(title_key)
        
        return unique_articles
    
    def _recency_score(self, published_at: datetime) -> float:
        """Calculate recency score (1.0 = very recent, 0.0 = old)."""
        hours_ago = (datetime.utcnow() - published_at).total_seconds() / 3600
        # Linear decay over 48 hours
        return max(0.0, 1.0 - (hours_ago / 48.0))


# Convenience functions
async def fetch_real_news_data(currency_pair: str, 
                             hours_back: int = 24,
                             newsapi_key: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Convenience function to fetch real news data.
    
    Args:
        currency_pair: Currency pair to analyze
        hours_back: Hours of historical news
        newsapi_key: Optional NewsAPI key for additional sources
        
    Returns:
        List of news articles in dictionary format
    """
    aggregator = NewsAggregator(newsapi_key)
    articles = await aggregator.fetch_comprehensive_news(currency_pair, hours_back, 25)
    
    # Convert to dictionary format for agent consumption
    return [
        {
            "title": article.title,
            "content": article.content,
            "source": article.source,
            "url": article.url,
            "timestamp": article.published_at.isoformat(),
            "relevance_score": article.relevance_score,
            "currencies_mentioned": article.currencies_mentioned
        }
        for article in articles
    ]


if __name__ == "__main__":
    # Test the news providers
    async def test_news_providers():
        print("📰 Testing Real News Data Sources...")
        
        # Test RSS provider
        async with RSSNewsProvider() as rss:
            articles = await rss.fetch_currency_news("USD/EUR", 48, 10)
            print(f"RSS Articles: {len(articles)}")
            for article in articles[:3]:
                print(f"  - {article.title[:60]}... (relevance: {article.relevance_score:.2f})")
        
        # Test aggregator
        news_data = await fetch_real_news_data("USD/EUR", 24)
        print(f"Total Aggregated Articles: {len(news_data)}")
    
    asyncio.run(test_news_providers())