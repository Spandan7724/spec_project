"""
Market Analysis Tools for Currency Intelligence.

Provides tools and utilities for market data analysis, sentiment tracking,
and economic event processing to support the MarketIntelligenceAgent.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import json
import re
from decimal import Decimal
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class NewsItem:
    """Structured representation of a news item."""
    title: str
    content: str
    source: str
    timestamp: datetime
    relevance_score: float = 0.0
    sentiment_score: float = 0.0
    impact_level: str = "low"  # low, medium, high


@dataclass
class EconomicEvent:
    """Structured representation of an economic calendar event."""
    name: str
    date: datetime
    currency: str
    importance: str  # low, medium, high
    previous_value: Optional[str] = None
    forecast_value: Optional[str] = None
    actual_value: Optional[str] = None
    impact_direction: str = "neutral"  # positive, negative, neutral


@dataclass
class TechnicalIndicator:
    """Technical analysis indicator data."""
    name: str
    value: float
    signal: str  # bullish, bearish, neutral
    confidence: float
    timeframe: str


class MarketDataProcessor:
    """Processes raw market data into structured insights."""
    
    def __init__(self):
        self.sentiment_keywords = {
            'positive': ['strong', 'bullish', 'growth', 'positive', 'recovery', 'rising', 'gains'],
            'negative': ['weak', 'bearish', 'decline', 'negative', 'falling', 'losses', 'crisis']
        }
    
    def analyze_text_sentiment(self, text: str) -> Tuple[float, str]:
        """
        Analyze sentiment of text using keyword-based approach.
        
        Args:
            text: Text to analyze
            
        Returns:
            Tuple of (sentiment_score, explanation)
        """
        if not text:
            return 0.0, "No text provided"
            
        text_lower = text.lower()
        positive_count = sum(1 for word in self.sentiment_keywords['positive'] if word in text_lower)
        negative_count = sum(1 for word in self.sentiment_keywords['negative'] if word in text_lower)
        
        total_words = len(text.split())
        if total_words == 0:
            return 0.0, "Empty text"
        
        # Calculate sentiment score
        sentiment_diff = positive_count - negative_count
        sentiment_score = max(-1.0, min(1.0, sentiment_diff / max(1, total_words / 10)))
        
        explanation = f"Found {positive_count} positive and {negative_count} negative indicators"
        
        return sentiment_score, explanation
    
    def extract_currency_mentions(self, text: str) -> List[str]:
        """Extract currency codes and currency-related terms from text."""
        # Common currency patterns
        currency_codes = re.findall(r'\b[A-Z]{3}/[A-Z]{3}\b|\b[A-Z]{3}\b(?=\s*(?:dollar|euro|yen|pound|yuan|franc))', text)
        
        # Currency names
        currency_names = {
            'dollar': 'USD', 'usd': 'USD', 'euro': 'EUR', 'eur': 'EUR',
            'yen': 'JPY', 'jpn': 'JPY', 'pound': 'GBP', 'gbp': 'GBP',
            'yuan': 'CNY', 'cny': 'CNY', 'franc': 'CHF', 'chf': 'CHF'
        }
        
        text_lower = text.lower()
        for name, code in currency_names.items():
            if name in text_lower and code not in currency_codes:
                currency_codes.append(code)
        
        return list(set(currency_codes))
    
    def categorize_economic_event(self, event_name: str) -> Tuple[str, str]:
        """
        Categorize economic event by type and expected impact.
        
        Args:
            event_name: Name of the economic event
            
        Returns:
            Tuple of (category, impact_level)
        """
        event_lower = event_name.lower()
        
        # High impact events
        high_impact_keywords = [
            'central bank', 'fed meeting', 'ecb meeting', 'boe meeting',
            'interest rate', 'gdp', 'inflation', 'cpi', 'ppi',
            'employment', 'nonfarm payrolls', 'unemployment'
        ]
        
        # Medium impact events
        medium_impact_keywords = [
            'retail sales', 'industrial production', 'pmi',
            'consumer confidence', 'trade balance', 'housing'
        ]
        
        # Determine impact level
        impact_level = "low"
        for keyword in high_impact_keywords:
            if keyword in event_lower:
                impact_level = "high"
                break
        
        if impact_level == "low":
            for keyword in medium_impact_keywords:
                if keyword in event_lower:
                    impact_level = "medium"
                    break
        
        # Determine category
        category = "other"
        if any(word in event_lower for word in ['central bank', 'fed', 'ecb', 'boe', 'interest']):
            category = "monetary_policy"
        elif any(word in event_lower for word in ['gdp', 'growth', 'industrial']):
            category = "growth_indicators"
        elif any(word in event_lower for word in ['inflation', 'cpi', 'ppi']):
            category = "inflation_data"
        elif any(word in event_lower for word in ['employment', 'payrolls', 'unemployment']):
            category = "employment_data"
        elif any(word in event_lower for word in ['trade', 'balance', 'exports', 'imports']):
            category = "trade_data"
        
        return category, impact_level


class NewsAnalyzer:
    """Analyzes news data for currency market intelligence."""
    
    def __init__(self):
        self.processor = MarketDataProcessor()
    
    def process_news_batch(self, news_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process a batch of news items for market intelligence.
        
        Args:
            news_items: List of news dictionaries with title, content, source, timestamp
            
        Returns:
            Aggregated analysis results
        """
        if not news_items:
            return self._empty_news_result()
        
        processed_items = []
        total_sentiment = 0.0
        currency_mentions = {}
        
        for item in news_items:
            # Process individual news item
            title = item.get('title', '')
            content = item.get('content', '')
            combined_text = f"{title} {content}"
            
            # Analyze sentiment
            sentiment_score, sentiment_reason = self.processor.analyze_text_sentiment(combined_text)
            
            # Extract currency mentions
            currencies = self.processor.extract_currency_mentions(combined_text)
            
            # Calculate relevance score based on currency mentions
            relevance_score = len(currencies) / 10.0  # Simple relevance metric
            
            processed_item = NewsItem(
                title=title,
                content=content,
                source=item.get('source', 'Unknown'),
                timestamp=datetime.fromisoformat(item.get('timestamp', datetime.utcnow().isoformat())),
                relevance_score=min(1.0, relevance_score),
                sentiment_score=sentiment_score,
                impact_level="high" if abs(sentiment_score) > 0.5 else "medium" if abs(sentiment_score) > 0.2 else "low"
            )
            
            processed_items.append(processed_item)
            total_sentiment += sentiment_score
            
            # Count currency mentions
            for currency in currencies:
                currency_mentions[currency] = currency_mentions.get(currency, 0) + 1
        
        # Calculate aggregated metrics
        avg_sentiment = total_sentiment / len(processed_items) if processed_items else 0.0
        high_impact_count = sum(1 for item in processed_items if item.impact_level == "high")
        
        return {
            "total_items": len(processed_items),
            "average_sentiment": avg_sentiment,
            "sentiment_distribution": self._calculate_sentiment_distribution(processed_items),
            "currency_mentions": currency_mentions,
            "high_impact_news_count": high_impact_count,
            "most_mentioned_currencies": sorted(currency_mentions.items(), key=lambda x: x[1], reverse=True)[:5],
            "recent_items": [item.__dict__ for item in processed_items[-5:]],  # Last 5 items
            "confidence": min(1.0, len(processed_items) / 10.0)  # More items = higher confidence
        }
    
    def _calculate_sentiment_distribution(self, items: List[NewsItem]) -> Dict[str, int]:
        """Calculate distribution of sentiment across news items."""
        distribution = {"very_positive": 0, "positive": 0, "neutral": 0, "negative": 0, "very_negative": 0}
        
        for item in items:
            score = item.sentiment_score
            if score > 0.5:
                distribution["very_positive"] += 1
            elif score > 0.1:
                distribution["positive"] += 1
            elif score > -0.1:
                distribution["neutral"] += 1
            elif score > -0.5:
                distribution["negative"] += 1
            else:
                distribution["very_negative"] += 1
        
        return distribution
    
    def _empty_news_result(self) -> Dict[str, Any]:
        """Return empty result when no news data is available."""
        return {
            "total_items": 0,
            "average_sentiment": 0.0,
            "sentiment_distribution": {"very_positive": 0, "positive": 0, "neutral": 0, "negative": 0, "very_negative": 0},
            "currency_mentions": {},
            "high_impact_news_count": 0,
            "most_mentioned_currencies": [],
            "recent_items": [],
            "confidence": 0.0
        }


class EconomicCalendarAnalyzer:
    """Analyzes economic calendar events for market impact."""
    
    def __init__(self):
        self.processor = MarketDataProcessor()
    
    def process_economic_events(self, events: List[Dict[str, Any]], 
                              target_currencies: List[str],
                              timeframe_days: int = 7) -> Dict[str, Any]:
        """
        Process economic calendar events for market intelligence.
        
        Args:
            events: List of economic event dictionaries
            target_currencies: Currencies to focus analysis on
            timeframe_days: Days ahead to consider
            
        Returns:
            Processed economic event analysis
        """
        if not events:
            return self._empty_calendar_result()
        
        end_date = datetime.utcnow() + timedelta(days=timeframe_days)
        relevant_events = []
        
        for event in events:
            event_date = datetime.fromisoformat(event.get('date', datetime.utcnow().isoformat()))
            
            # Filter by timeframe
            if event_date > end_date:
                continue
            
            # Process event
            event_name = event.get('name', '')
            currency = event.get('currency', 'USD')
            
            # Check relevance to target currencies
            if currency not in target_currencies:
                continue
            
            category, impact_level = self.processor.categorize_economic_event(event_name)
            
            processed_event = EconomicEvent(
                name=event_name,
                date=event_date,
                currency=currency,
                importance=impact_level,
                previous_value=event.get('previous'),
                forecast_value=event.get('forecast'),
                actual_value=event.get('actual'),
                impact_direction=self._determine_impact_direction(event)
            )
            
            relevant_events.append(processed_event)
        
        # Analyze upcoming events
        return self._analyze_event_impact(relevant_events, target_currencies)
    
    def _determine_impact_direction(self, event: Dict[str, Any]) -> str:
        """Determine if event is likely positive, negative, or neutral for currency."""
        # Simple heuristic based on event type and values
        event_name = event.get('name', '').lower()
        
        if 'interest rate' in event_name or 'fed meeting' in event_name:
            # Rate decisions are context-dependent
            return "neutral"
        elif any(word in event_name for word in ['gdp', 'employment', 'payrolls']):
            # Higher values generally positive for currency
            return "positive"
        elif any(word in event_name for word in ['inflation', 'cpi']):
            # Complex - high inflation can be positive or negative
            return "neutral"
        
        return "neutral"
    
    def _analyze_event_impact(self, events: List[EconomicEvent], 
                            target_currencies: List[str]) -> Dict[str, Any]:
        """Analyze the collective impact of economic events."""
        if not events:
            return self._empty_calendar_result()
        
        # Count by importance and currency
        importance_counts = {"high": 0, "medium": 0, "low": 0}
        currency_events = {}
        
        for event in events:
            importance_counts[event.importance] += 1
            if event.currency not in currency_events:
                currency_events[event.currency] = []
            currency_events[event.currency].append(event.__dict__)
        
        # Calculate overall impact score
        impact_score = (
            importance_counts["high"] * 0.8 +
            importance_counts["medium"] * 0.5 +
            importance_counts["low"] * 0.2
        ) / max(1, len(events))
        
        return {
            "total_events": len(events),
            "impact_score": min(1.0, impact_score),
            "importance_breakdown": importance_counts,
            "events_by_currency": currency_events,
            "next_high_impact": self._find_next_high_impact_event(events),
            "days_until_next_major": self._days_until_next_major_event(events),
            "confidence": min(1.0, len(events) / 5.0)
        }
    
    def _find_next_high_impact_event(self, events: List[EconomicEvent]) -> Optional[Dict[str, Any]]:
        """Find the next high-impact economic event."""
        high_impact_events = [e for e in events if e.importance == "high"]
        if not high_impact_events:
            return None
        
        # Sort by date
        next_event = min(high_impact_events, key=lambda e: e.date)
        return {
            "name": next_event.name,
            "date": next_event.date.isoformat(),
            "currency": next_event.currency,
            "days_away": (next_event.date - datetime.utcnow()).days
        }
    
    def _days_until_next_major_event(self, events: List[EconomicEvent]) -> int:
        """Calculate days until next major economic event."""
        major_events = [e for e in events if e.importance in ["high", "medium"]]
        if not major_events:
            return 30  # Default assumption
        
        next_major = min(major_events, key=lambda e: e.date)
        return max(0, (next_major.date - datetime.utcnow()).days)
    
    def _empty_calendar_result(self) -> Dict[str, Any]:
        """Return empty result when no calendar data is available."""
        return {
            "total_events": 0,
            "impact_score": 0.0,
            "importance_breakdown": {"high": 0, "medium": 0, "low": 0},
            "events_by_currency": {},
            "next_high_impact": None,
            "days_until_next_major": 30,
            "confidence": 0.0
        }


class TechnicalAnalyzer:
    """Provides technical analysis capabilities."""
    
    def calculate_moving_average_signal(self, prices: List[float], 
                                      short_period: int = 20,
                                      long_period: int = 50) -> TechnicalIndicator:
        """Calculate moving average crossover signal."""
        if len(prices) < long_period:
            return TechnicalIndicator(
                name="Moving Average",
                value=0.0,
                signal="neutral",
                confidence=0.0,
                timeframe=f"{short_period}/{long_period}D"
            )
        
        # Calculate moving averages
        short_ma = sum(prices[-short_period:]) / short_period
        long_ma = sum(prices[-long_period:]) / long_period
        
        # Determine signal
        if short_ma > long_ma * 1.01:  # 1% threshold
            signal = "bullish"
            confidence = min(1.0, (short_ma / long_ma - 1) * 10)
        elif short_ma < long_ma * 0.99:  # 1% threshold
            signal = "bearish"
            confidence = min(1.0, (1 - short_ma / long_ma) * 10)
        else:
            signal = "neutral"
            confidence = 0.5
        
        return TechnicalIndicator(
            name="Moving Average Crossover",
            value=short_ma / long_ma,
            signal=signal,
            confidence=confidence,
            timeframe=f"{short_period}/{long_period}D"
        )
    
    def calculate_rsi(self, prices: List[float], period: int = 14) -> TechnicalIndicator:
        """Calculate Relative Strength Index."""
        if len(prices) < period + 1:
            return TechnicalIndicator(
                name="RSI",
                value=50.0,
                signal="neutral",
                confidence=0.0,
                timeframe=f"{period}D"
            )
        
        # Calculate price changes
        changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        gains = [max(0, change) for change in changes[-period:]]
        losses = [max(0, -change) for change in changes[-period:]]
        
        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period
        
        if avg_loss == 0:
            rsi = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi = 100.0 - (100.0 / (1.0 + rs))
        
        # Determine signal
        if rsi > 70:
            signal = "bearish"  # Overbought
            confidence = min(1.0, (rsi - 70) / 30)
        elif rsi < 30:
            signal = "bullish"  # Oversold
            confidence = min(1.0, (30 - rsi) / 30)
        else:
            signal = "neutral"
            confidence = 0.5
        
        return TechnicalIndicator(
            name="RSI",
            value=rsi,
            signal=signal,
            confidence=confidence,
            timeframe=f"{period}D"
        )
    
    def analyze_support_resistance(self, prices: List[float], 
                                 lookback_periods: int = 20) -> Dict[str, Any]:
        """Identify key support and resistance levels."""
        if len(prices) < lookback_periods:
            return {"support": None, "resistance": None, "confidence": 0.0}
        
        recent_prices = prices[-lookback_periods:]
        current_price = prices[-1]
        
        # Find local minima (support) and maxima (resistance)
        support_levels = []
        resistance_levels = []
        
        for i in range(1, len(recent_prices) - 1):
            # Local minimum (support)
            if recent_prices[i] < recent_prices[i-1] and recent_prices[i] < recent_prices[i+1]:
                support_levels.append(recent_prices[i])
            
            # Local maximum (resistance)
            if recent_prices[i] > recent_prices[i-1] and recent_prices[i] > recent_prices[i+1]:
                resistance_levels.append(recent_prices[i])
        
        # Find nearest levels
        nearest_support = max(support_levels) if support_levels else None
        
        # Filter resistance levels above current price
        resistance_above = [r for r in resistance_levels if r > current_price]
        nearest_resistance = min(resistance_above) if resistance_above else None
        
        confidence = len(support_levels + resistance_levels) / lookback_periods
        
        return {
            "support": nearest_support,
            "resistance": nearest_resistance,
            "confidence": min(1.0, confidence),
            "all_support_levels": support_levels,
            "all_resistance_levels": resistance_levels
        }


class MarketRegimeDetector:
    """Detects current market regime (trending, ranging, volatile)."""
    
    def detect_regime(self, prices: List[float], 
                     volume: Optional[List[float]] = None,
                     period: int = 20) -> Dict[str, Any]:
        """
        Detect the current market regime.
        
        Args:
            prices: Recent price data
            volume: Recent volume data (optional)
            period: Analysis period
            
        Returns:
            Market regime analysis
        """
        if len(prices) < period:
            return self._default_regime_result()
        
        recent_prices = prices[-period:]
        
        # Calculate volatility
        returns = [(recent_prices[i] / recent_prices[i-1] - 1) for i in range(1, len(recent_prices))]
        volatility = (sum(r**2 for r in returns) / len(returns)) ** 0.5
        
        # Calculate trend strength
        start_price = recent_prices[0]
        end_price = recent_prices[-1]
        total_return = (end_price / start_price) - 1
        
        # Determine regime
        regime = "ranging"
        confidence = 0.5
        
        if abs(total_return) > 0.05 and volatility < 0.02:  # 5% move with low volatility
            regime = "trending_up" if total_return > 0 else "trending_down"
            confidence = min(1.0, abs(total_return) / 0.1)
        elif volatility > 0.03:  # High volatility
            regime = "volatile"
            confidence = min(1.0, volatility / 0.05)
        else:  # Low movement, low volatility
            regime = "ranging"
            confidence = 1.0 - abs(total_return) / 0.05
        
        return {
            "regime": regime,
            "confidence": confidence,
            "volatility": volatility,
            "trend_strength": abs(total_return),
            "direction": "up" if total_return > 0 else "down" if total_return < 0 else "sideways",
            "analysis_period": period
        }
    
    def _default_regime_result(self) -> Dict[str, Any]:
        """Default regime result when insufficient data."""
        return {
            "regime": "unknown",
            "confidence": 0.0,
            "volatility": 0.0,
            "trend_strength": 0.0,
            "direction": "sideways",
            "analysis_period": 0
        }


class MarketIntelligenceToolkit:
    """Main toolkit for market intelligence operations."""
    
    def __init__(self):
        self.news_analyzer = NewsAnalyzer()
        self.calendar_analyzer = EconomicCalendarAnalyzer()
        self.technical_analyzer = TechnicalAnalyzer()
        self.regime_detector = MarketRegimeDetector()
    
    async def comprehensive_market_analysis(self, 
                                          currency_pair: str,
                                          price_data: Optional[List[float]] = None,
                                          news_data: Optional[List[Dict[str, Any]]] = None,
                                          calendar_data: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Perform comprehensive market analysis combining all available data.
        
        Args:
            currency_pair: Currency pair to analyze
            price_data: Historical price data
            news_data: Recent news items
            calendar_data: Economic calendar events
            
        Returns:
            Comprehensive market intelligence
        """
        base_currency, quote_currency = currency_pair.split('/')
        target_currencies = [base_currency, quote_currency]
        
        analysis_results = {}
        
        # News analysis
        if news_data:
            news_analysis = self.news_analyzer.process_news_batch(news_data)
            analysis_results["news"] = news_analysis
        else:
            analysis_results["news"] = self.news_analyzer._empty_news_result()
        
        # Economic calendar analysis
        if calendar_data:
            calendar_analysis = self.calendar_analyzer.process_economic_events(
                calendar_data, target_currencies
            )
            analysis_results["economic_events"] = calendar_analysis
        else:
            analysis_results["economic_events"] = self.calendar_analyzer._empty_calendar_result()
        
        # Technical analysis
        if price_data and len(price_data) > 20:
            # Moving averages
            ma_signal = self.technical_analyzer.calculate_moving_average_signal(price_data)
            analysis_results["moving_averages"] = ma_signal.__dict__
            
            # RSI
            rsi_signal = self.technical_analyzer.calculate_rsi(price_data)
            analysis_results["rsi"] = rsi_signal.__dict__
            
            # Support/Resistance
            sr_analysis = self.technical_analyzer.analyze_support_resistance(price_data)
            analysis_results["support_resistance"] = sr_analysis
            
            # Market regime
            regime_analysis = self.regime_detector.detect_regime(price_data)
            analysis_results["market_regime"] = regime_analysis
        else:
            # Default technical analysis when no price data
            analysis_results["moving_averages"] = {"signal": "neutral", "confidence": 0.0}
            analysis_results["rsi"] = {"value": 50.0, "signal": "neutral", "confidence": 0.0}
            analysis_results["support_resistance"] = {"support": None, "resistance": None, "confidence": 0.0}
            analysis_results["market_regime"] = self.regime_detector._default_regime_result()
        
        # Calculate overall confidence
        confidence_scores = [
            analysis_results["news"]["confidence"],
            analysis_results["economic_events"]["confidence"],
            analysis_results.get("market_regime", {}).get("confidence", 0.0)
        ]
        overall_confidence = sum(confidence_scores) / len(confidence_scores)
        
        return {
            "currency_pair": currency_pair,
            "analysis_timestamp": datetime.utcnow().isoformat(),
            "overall_confidence": overall_confidence,
            "components": analysis_results,
            "summary": self._create_analysis_summary(analysis_results, currency_pair)
        }
    
    def _create_analysis_summary(self, results: Dict[str, Any], currency_pair: str) -> Dict[str, Any]:
        """Create a summary of the analysis results."""
        news = results.get("news", {})
        events = results.get("economic_events", {})
        regime = results.get("market_regime", {})
        
        # Determine overall sentiment
        overall_sentiment = "neutral"
        sentiment_score = news.get("average_sentiment", 0.0)
        
        if sentiment_score > 0.2:
            overall_sentiment = "positive"
        elif sentiment_score < -0.2:
            overall_sentiment = "negative"
        
        # Determine timing recommendation
        timing_rec = "unclear"
        high_impact_events = events.get("high_impact_news_count", 0)
        days_to_major = events.get("days_until_next_major", 30)
        
        if high_impact_events > 2 and days_to_major <= 3:
            timing_rec = "wait_for_clarity"
        elif regime.get("regime") == "trending_up" and overall_sentiment == "positive":
            timing_rec = "immediate"
        elif regime.get("volatility", 0) > 0.03:
            timing_rec = "wait_1_3_days"
        
        return {
            "overall_sentiment": overall_sentiment,
            "sentiment_score": sentiment_score,
            "regime_type": regime.get("regime", "unknown"),
            "volatility_level": "high" if regime.get("volatility", 0) > 0.03 else "low",
            "timing_recommendation": timing_rec,
            "key_factors": [
                f"News sentiment: {overall_sentiment}",
                f"Market regime: {regime.get('regime', 'unknown')}",
                f"Major events in {days_to_major} days"
            ]
        }


# Convenience functions for agent integration
def create_market_toolkit() -> MarketIntelligenceToolkit:
    """Factory function to create market intelligence toolkit."""
    return MarketIntelligenceToolkit()


def mock_market_data_for_testing(currency_pair: str) -> Dict[str, Any]:
    """Generate mock market data for testing purposes."""
    import random
    
    # Mock price data (simulated 30 days) with more realistic price movements
    base_price = 0.85 if currency_pair == "USD/EUR" else 1.0
    price_data = []
    for i in range(30):
        # Add some trend and noise
        trend_factor = 0.001 * (i - 15)  # Slight trend over time
        noise = random.uniform(-0.01, 0.01)
        base_price = base_price * (1 + trend_factor + noise)
        price_data.append(base_price)
    
    # Mock news data
    news_data = [
        {
            "title": f"{currency_pair} shows strong performance amid economic optimism",
            "content": "Markets are showing positive sentiment following recent economic indicators...",
            "source": "Financial Times",
            "timestamp": (datetime.utcnow() - timedelta(hours=2)).isoformat()
        },
        {
            "title": f"Central bank signals potential policy changes affecting {currency_pair}",
            "content": "Recent statements suggest monetary policy adjustments may be coming...",
            "source": "Reuters",
            "timestamp": (datetime.utcnow() - timedelta(hours=8)).isoformat()
        }
    ]
    
    # Mock economic events
    calendar_data = [
        {
            "name": "GDP Growth Rate",
            "date": (datetime.utcnow() + timedelta(days=2)).isoformat(),
            "currency": currency_pair.split('/')[0],
            "importance": "high",
            "previous": "2.1%",
            "forecast": "2.3%"
        },
        {
            "name": "Consumer Price Index",
            "date": (datetime.utcnow() + timedelta(days=5)).isoformat(),
            "currency": currency_pair.split('/')[1],
            "importance": "medium",
            "previous": "3.2%",
            "forecast": "3.1%"
        }
    ]
    
    return {
        "currency_pair": currency_pair,
        "price_data": price_data,
        "news_data": news_data,
        "calendar_data": calendar_data
    }