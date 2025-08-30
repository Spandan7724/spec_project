"""
Financial news web scraper for forex and currency market news.
"""

import logging
import re
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from crawl4ai import AsyncWebCrawler

from .news_models import NewsArticle, NewsSentiment

logger = logging.getLogger(__name__)


class FinancialNewsScraper:
    """
    Web scraper for financial news from forex and currency market sources.
    
    Scrapes news from:
    - FXStreet for forex analysis and market commentary
    - Investing.com for currency news and central bank updates
    - Filters for currency-relevant content only
    """
    
    def __init__(self):
        self.sources = {
            "fxstreet": "https://www.fxstreet.com/news",
            "investing": "https://www.investing.com/news/forex-news"
        }
        self.crawler: Optional[AsyncWebCrawler] = None
        
        # Currency terms for relevance filtering
        self.currency_terms = {
            "currencies": ["USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD", "INR"],
            "pairs": ["USD/EUR", "EUR/USD", "GBP/USD", "USD/JPY", "USD/CAD", "AUD/USD"],
            "central_banks": ["Fed", "ECB", "BOE", "BOJ", "RBI", "SNB", "RBA"],
            "keywords": ["forex", "currency", "exchange rate", "monetary policy", "interest rate", 
                        "inflation", "central bank", "dollar", "euro", "pound", "yen"]
        }
    
    async def __aenter__(self):
        """Initialize crawler."""
        self.crawler = AsyncWebCrawler(verbose=False)
        await self.crawler.__aenter__()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Close crawler."""
        if self.crawler:
            await self.crawler.__aexit__(exc_type, exc_val, exc_tb)
    
    async def get_latest_news(self, hours_back: int = 24) -> Optional[List[NewsArticle]]:
        """
        Get latest financial news articles relevant to currencies.
        
        Args:
            hours_back: How many hours back to look for news
            
        Returns:
            List of NewsArticle objects or None if failed
        """
        if not self.crawler:
            raise RuntimeError("Scraper not initialized - use async context manager")
        
        try:
            all_articles = []
            
            # Scrape FXStreet
            fxstreet_articles = await self._scrape_fxstreet()
            if fxstreet_articles:
                all_articles.extend(fxstreet_articles)
                logger.info(f"Scraped {len(fxstreet_articles)} articles from FXStreet")
            
            # Scrape Investing.com
            investing_articles = await self._scrape_investing()
            if investing_articles:
                all_articles.extend(investing_articles)
                logger.info(f"Scraped {len(investing_articles)} articles from Investing.com")
            
            # Filter for currency relevance and recency
            relevant_articles = self._filter_relevant_articles(all_articles, hours_back)
            
            # Add sentiment analysis
            for article in relevant_articles:
                article.sentiment = self._analyze_sentiment(article.title)
                article.affected_currencies = self._extract_currencies(article.title)
            
            logger.info(f"Found {len(relevant_articles)} relevant currency news articles")
            return relevant_articles
            
        except Exception as e:
            logger.error(f"News scraping error: {e}")
            return None
    
    async def _scrape_fxstreet(self) -> Optional[List[NewsArticle]]:
        """Scrape news articles from FXStreet."""
        try:
            result = await self.crawler.arun(
                url=self.sources["fxstreet"],
                word_count_threshold=10
            )
            
            if result.status_code != 200:
                logger.error(f"FXStreet scraping failed: {result.status_code}")
                return None
            
            articles = []
            content = result.markdown
            
            # Extract article links and titles
            # Look for forex-related headlines
            lines = content.split('\n')
            
            for line in lines:
                # Look for actual news article links (not navigation)
                link_match = re.search(r'\[([^\]]+)\]\((https://www\.fxstreet\.com/[^\)]+202508[^\)]+)\)', line)
                if link_match:
                    title = link_match.group(1)
                    url = link_match.group(2)
                    
                    # Filter out navigation and ensure it's actual news
                    if (self._is_currency_relevant(title) and 
                        len(title) > 10 and 
                        not any(nav in title.lower() for nav in ['chart', 'rates', 'technical', 'detector']) and
                        '202508' in url):  # Ensure it's current news with date in URL
                        
                        article_id = hashlib.md5(url.encode()).hexdigest()[:12]
                        
                        article = NewsArticle(
                            article_id=article_id,
                            title=title,
                            url=url,
                            source="FXStreet",
                            published_date=datetime.utcnow(),  # Approximate
                            category="forex_analysis"
                        )
                        articles.append(article)
            
            return articles
            
        except Exception as e:
            logger.error(f"FXStreet scraping error: {e}")
            return None
    
    async def _scrape_investing(self) -> Optional[List[NewsArticle]]:
        """Scrape news articles from Investing.com."""
        try:
            result = await self.crawler.arun(
                url=self.sources["investing"],
                word_count_threshold=10
            )
            
            if result.status_code != 200:
                logger.error(f"Investing.com scraping failed: {result.status_code}")
                return None
            
            articles = []
            content = result.markdown
            
            # Extract headlines from Investing.com format
            lines = content.split('\n')
            
            for line in lines:
                # Look for actual news headlines with Investing.com format
                if line.strip().startswith("Investing.com -") and " has " in line:
                    title = line.replace("Investing.com -", "").strip()
                    
                    # Only real news headlines, not fragments
                    if (self._is_currency_relevant(title) and 
                        len(title) > 30 and 
                        any(word in title.lower() for word in ['dollar', 'currency', 'fed', 'bank', 'rate']) and
                        not title.lower().startswith('the ') == False):  # Ensure complete sentences
                        
                        article_id = hashlib.md5(title.encode()).hexdigest()[:12]
                        
                        article = NewsArticle(
                            article_id=article_id,
                            title=title,
                            url=self.sources["investing"],  # Base URL for now
                            source="Investing.com",
                            published_date=datetime.utcnow(),  # Approximate
                            category="forex_news"
                        )
                        articles.append(article)
            
            return articles
            
        except Exception as e:
            logger.error(f"Investing.com scraping error: {e}")
            return None
    
    def _is_currency_relevant(self, text: str) -> bool:
        """Check if text is relevant to currency markets."""
        text_lower = text.lower()
        
        # Check for currency codes
        for currency in self.currency_terms["currencies"]:
            if currency.lower() in text_lower:
                return True
        
        # Check for currency pairs
        for pair in self.currency_terms["pairs"]:
            if pair.lower() in text_lower:
                return True
        
        # Check for central banks
        for bank in self.currency_terms["central_banks"]:
            if bank.lower() in text_lower:
                return True
        
        # Check for forex keywords
        for keyword in self.currency_terms["keywords"]:
            if keyword in text_lower:
                return True
        
        return False
    
    def _filter_relevant_articles(self, articles: List[NewsArticle], hours_back: int) -> List[NewsArticle]:
        """Filter articles for currency relevance and recency."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours_back)
        
        relevant = []
        for article in articles:
            # Check currency relevance
            if not self._is_currency_relevant(article.title):
                continue
            
            # Check recency (approximate since we don't have exact timestamps)
            if article.published_date and article.published_date < cutoff_time:
                continue
            
            relevant.append(article)
        
        return relevant
    
    def _extract_currencies(self, text: str) -> List[str]:
        """Extract currency codes mentioned in text."""
        currencies = []
        text_upper = text.upper()
        
        for currency in self.currency_terms["currencies"]:
            if currency in text_upper:
                currencies.append(currency)
        
        return list(set(currencies))  # Remove duplicates
    
    def _analyze_sentiment(self, title: str) -> NewsSentiment:
        """Basic sentiment analysis based on keywords."""
        title_lower = title.lower()
        
        # Positive indicators
        positive_words = ["rise", "gain", "surge", "rally", "strengthen", "boost", 
                         "optimism", "bullish", "positive", "up", "higher", "climb"]
        
        # Negative indicators  
        negative_words = ["fall", "drop", "decline", "plunge", "weaken", "crash",
                         "pessimism", "bearish", "negative", "down", "lower", "slide"]
        
        positive_count = sum(1 for word in positive_words if word in title_lower)
        negative_count = sum(1 for word in negative_words if word in title_lower)
        
        if positive_count > negative_count:
            return NewsSentiment.POSITIVE
        elif negative_count > positive_count:
            return NewsSentiment.NEGATIVE
        else:
            return NewsSentiment.NEUTRAL