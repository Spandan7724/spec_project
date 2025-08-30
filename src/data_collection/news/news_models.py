"""
Data models for financial news and sentiment analysis.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import List, Optional


class NewsSentiment(Enum):
    """News sentiment classification."""
    POSITIVE = "positive"
    NEGATIVE = "negative" 
    NEUTRAL = "neutral"


@dataclass
class NewsArticle:
    """
    Financial news article with metadata and sentiment.
    """
    article_id: str
    title: str
    url: str
    source: str  # "FXStreet", "Investing.com", etc.
    
    # Content
    summary: Optional[str] = None
    content: Optional[str] = None
    
    # Timing
    published_date: Optional[datetime] = None
    scraped_date: datetime = None
    
    # Currency relevance
    affected_currencies: List[str] = None  # ["USD", "EUR", etc.]
    currency_pairs: List[str] = None  # ["USD/EUR", "GBP/USD", etc.]
    
    # Sentiment and impact
    sentiment: Optional[NewsSentiment] = None
    sentiment_score: Optional[float] = None  # -1.0 to 1.0
    market_impact: Optional[str] = None  # "high", "medium", "low"
    
    # Classification
    category: Optional[str] = None  # "monetary_policy", "economic_data", "geopolitical"
    tags: List[str] = None
    
    # Metadata
    author: Optional[str] = None
    read_time: Optional[int] = None  # Minutes
    
    def __post_init__(self):
        """Initialize default values."""
        if self.affected_currencies is None:
            self.affected_currencies = []
        if self.currency_pairs is None:
            self.currency_pairs = []
        if self.tags is None:
            self.tags = []
        if self.scraped_date is None:
            self.scraped_date = datetime.utcnow()
    
    @property
    def is_recent(self) -> bool:
        """Check if article is from last 24 hours."""
        if not self.published_date:
            return False
        return (datetime.utcnow() - self.published_date).days == 0
    
    @property
    def is_currency_relevant(self) -> bool:
        """Check if article affects currencies."""
        return len(self.affected_currencies) > 0 or len(self.currency_pairs) > 0
    
    @property
    def is_bullish(self) -> bool:
        """Check if sentiment is bullish/positive."""
        return self.sentiment == NewsSentiment.POSITIVE
    
    @property
    def is_bearish(self) -> bool:
        """Check if sentiment is bearish/negative."""
        return self.sentiment == NewsSentiment.NEGATIVE
    
    def affects_pair(self, currency_pair: str) -> bool:
        """Check if article affects a specific currency pair."""
        # Handle both formats: 'USD/EUR' and 'USDEUR'
        if '/' in currency_pair:
            base_currency, quote_currency = currency_pair.split('/')
        else:
            if len(currency_pair) == 6:
                base_currency = currency_pair[:3]
                quote_currency = currency_pair[3:]
            else:
                return False
        
        return (base_currency in self.affected_currencies or 
                quote_currency in self.affected_currencies or
                currency_pair in self.currency_pairs)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'article_id': self.article_id,
            'title': self.title,
            'url': self.url,
            'source': self.source,
            'summary': self.summary,
            'published_date': self.published_date.isoformat() if self.published_date else None,
            'scraped_date': self.scraped_date.isoformat(),
            'affected_currencies': self.affected_currencies,
            'currency_pairs': self.currency_pairs,
            'sentiment': self.sentiment.value if self.sentiment else None,
            'sentiment_score': self.sentiment_score,
            'market_impact': self.market_impact,
            'category': self.category,
            'tags': self.tags,
            'author': self.author,
            'is_recent': self.is_recent,
            'is_currency_relevant': self.is_currency_relevant
        }