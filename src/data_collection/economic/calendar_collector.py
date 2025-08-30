"""
Economic calendar data collection and management.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class EventImpact(Enum):
    """Economic event impact classification."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class EventStatus(Enum):
    """Economic event status."""
    SCHEDULED = "scheduled"
    RELEASED = "released"
    REVISED = "revised"
    CANCELLED = "cancelled"


@dataclass
class EconomicEvent:
    """
    Individual economic event with all metadata.
    """
    event_id: str  # Unique identifier
    title: str
    country: str  # ISO country code (US, GB, EU, etc.)
    currency: str  # Currency code (USD, GBP, EUR, etc.)
    
    # Timing
    release_date: datetime
    period: str  # "Q3 2025", "August 2025", etc.
    
    # Impact classification
    impact: EventImpact
    affected_currencies: List[str]  # Currencies this event affects
    
    # Values
    previous_value: Optional[float] = None
    forecast_value: Optional[float] = None
    actual_value: Optional[float] = None
    
    # Status and source
    status: EventStatus = EventStatus.SCHEDULED
    source: str = "unknown"  # FRED, ECB, BOE, etc.
    category: str = "economic"  # economic, employment, inflation, etc.
    
    # Additional metadata
    unit: Optional[str] = None  # %, millions, index points, etc.
    frequency: Optional[str] = None  # monthly, quarterly, annual
    description: Optional[str] = None
    
    @property
    def is_upcoming(self) -> bool:
        """Check if event is scheduled for the future."""
        return self.release_date > datetime.utcnow() and self.status == EventStatus.SCHEDULED
    
    @property
    def is_high_impact(self) -> bool:
        """Check if event has high market impact."""
        return self.impact == EventImpact.HIGH
    
    @property
    def surprise_factor(self) -> Optional[float]:
        """Calculate surprise factor (actual vs forecast) if available."""
        if self.actual_value is None or self.forecast_value is None:
            return None
        
        if self.forecast_value == 0:
            return None  # Avoid division by zero
        
        # Return percentage difference
        return ((self.actual_value - self.forecast_value) / abs(self.forecast_value)) * 100
    
    @property
    def relevance_score(self) -> float:
        """Calculate relevance score 0-1 based on impact and currency importance."""
        base_score = {
            EventImpact.LOW: 0.3,
            EventImpact.MEDIUM: 0.6, 
            EventImpact.HIGH: 1.0
        }[self.impact]
        
        # Boost for major currencies
        major_currencies = {'USD', 'EUR', 'GBP', 'JPY'}
        if self.currency in major_currencies:
            base_score *= 1.2
        
        # Boost for multi-currency impact
        if len(self.affected_currencies) > 1:
            base_score *= 1.1
        
        return min(1.0, base_score)
    
    def affects_pair(self, currency_pair: str) -> bool:
        """Check if this event affects a specific currency pair."""
        # Handle both formats: 'USD/EUR' and 'USDEUR'
        if '/' in currency_pair:
            base_currency, quote_currency = currency_pair.split('/')
        else:
            # Split 6-char format like 'USDEUR' -> 'USD', 'EUR'
            if len(currency_pair) == 6:
                base_currency = currency_pair[:3]
                quote_currency = currency_pair[3:]
            else:
                return False
        
        return (self.currency in [base_currency, quote_currency] or 
                base_currency in self.affected_currencies or
                quote_currency in self.affected_currencies)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'event_id': self.event_id,
            'title': self.title,
            'country': self.country,
            'currency': self.currency,
            'release_date': self.release_date.isoformat(),
            'period': self.period,
            'impact': self.impact.value,
            'affected_currencies': self.affected_currencies,
            'previous_value': self.previous_value,
            'forecast_value': self.forecast_value,
            'actual_value': self.actual_value,
            'status': self.status.value,
            'source': self.source,
            'category': self.category,
            'unit': self.unit,
            'frequency': self.frequency,
            'description': self.description,
            'surprise_factor': self.surprise_factor,
            'relevance_score': self.relevance_score,
            'is_upcoming': self.is_upcoming,
            'is_high_impact': self.is_high_impact
        }


@dataclass
class EconomicCalendar:
    """
    Complete economic calendar with events and metadata.
    """
    events: List[EconomicEvent]
    start_date: datetime
    end_date: datetime
    last_updated: datetime
    sources: List[str]
    
    def __post_init__(self):
        """Sort events by release date after initialization."""
        self.events.sort(key=lambda e: e.release_date)
    
    def get_upcoming_events(self, days_ahead: int = 14) -> List[EconomicEvent]:
        """Get upcoming events within specified days."""
        cutoff_date = datetime.utcnow() + timedelta(days=days_ahead)
        
        return [event for event in self.events 
                if event.is_upcoming and event.release_date <= cutoff_date]
    
    def get_events_by_impact(self, impact: EventImpact) -> List[EconomicEvent]:
        """Get events by impact level."""
        return [event for event in self.events if event.impact == impact]
    
    def get_events_by_currency(self, currency: str) -> List[EconomicEvent]:
        """Get events affecting a specific currency."""
        return [event for event in self.events 
                if event.currency == currency or currency in event.affected_currencies]
    
    def get_events_for_pair(self, currency_pair: str) -> List[EconomicEvent]:
        """Get events affecting a currency pair."""
        return [event for event in self.events if event.affects_pair(currency_pair)]
    
    def get_high_impact_upcoming(self, days_ahead: int = 7) -> List[EconomicEvent]:
        """Get high impact events coming up."""
        upcoming = self.get_upcoming_events(days_ahead)
        return [event for event in upcoming if event.is_high_impact]
    
    @property
    def event_count(self) -> int:
        """Total number of events."""
        return len(self.events)
    
    @property
    def upcoming_count(self) -> int:
        """Number of upcoming events."""
        return len([e for e in self.events if e.is_upcoming])
    
    @property 
    def high_impact_count(self) -> int:
        """Number of high impact events."""
        return len([e for e in self.events if e.is_high_impact])


class EconomicCalendarCollector:
    """
    Collects and manages economic calendar data from multiple sources.
    """
    
    def __init__(self):
        self._calendar_cache: Optional[EconomicCalendar] = None
        self._cache_timestamp: Optional[datetime] = None
        self._cache_ttl = timedelta(hours=4)  # Cache calendar for 4 hours
        
        # Event deduplication tracking
        self._seen_events: Set[str] = set()
    
    async def get_economic_calendar(self, 
                                  days_ahead: int = 14,
                                  force_refresh: bool = False) -> Optional[EconomicCalendar]:
        """
        Get economic calendar with events from all sources.
        
        Args:
            days_ahead: Number of days ahead to fetch events for
            force_refresh: Whether to force refresh cached calendar
            
        Returns:
            EconomicCalendar or None if failed
        """
        # Check cache first
        if not force_refresh and self._is_cached_valid():
            logger.debug("Using cached economic calendar")
            return self._calendar_cache
        
        logger.info(f"Fetching economic calendar for next {days_ahead} days")
        
        try:
            # Get events from all sources
            all_events = []
            sources_used = []
            
            # FRED (US economic data)
            fred_events = await self._fetch_fred_events(days_ahead)
            if fred_events:
                all_events.extend(fred_events)
                sources_used.append("FRED")
                logger.info(f"Collected {len(fred_events)} events from FRED")
            
            # ECB (European economic data)
            ecb_events = await self._fetch_ecb_events(days_ahead)
            if ecb_events:
                all_events.extend(ecb_events)
                sources_used.append("ECB")
                logger.info(f"Collected {len(ecb_events)} events from ECB")
            
            # BOE (UK economic data)  
            boe_events = await self._fetch_boe_events(days_ahead)
            if boe_events:
                all_events.extend(boe_events)
                sources_used.append("BOE")
                logger.info(f"Collected {len(boe_events)} events from BOE")
            
            # RBI (Indian economic data) - using web scraper
            rbi_events = await self._fetch_rbi_events(days_ahead)
            if rbi_events:
                all_events.extend(rbi_events)
                sources_used.append("RBI")
                logger.info(f"Collected {len(rbi_events)} events from RBI")
            
            if not all_events:
                logger.warning("No economic events collected from any source")
                return None
            
            # Deduplicate events
            unique_events = self._deduplicate_events(all_events)
            logger.info(f"Deduplicated to {len(unique_events)} unique events")
            
            # Create calendar
            calendar = EconomicCalendar(
                events=unique_events,
                start_date=datetime.utcnow(),
                end_date=datetime.utcnow() + timedelta(days=days_ahead),
                last_updated=datetime.utcnow(),
                sources=sources_used
            )
            
            # Cache the result
            self._calendar_cache = calendar
            self._cache_timestamp = datetime.utcnow()
            
            logger.info(f"Economic calendar created: {calendar.event_count} total events, "
                       f"{calendar.upcoming_count} upcoming, {calendar.high_impact_count} high impact")
            
            return calendar
            
        except Exception as e:
            logger.error(f"Failed to create economic calendar: {e}")
            return None
    
    def _is_cached_valid(self) -> bool:
        """Check if cached calendar is still valid."""
        if self._calendar_cache is None or self._cache_timestamp is None:
            return False
        
        age = datetime.utcnow() - self._cache_timestamp
        return age < self._cache_ttl
    
    async def _fetch_fred_events(self, days_ahead: int) -> Optional[List[EconomicEvent]]:
        """Fetch events from FRED API."""
        try:
            from .fred_provider import FREDProvider
            
            import os
            api_key = os.getenv('FRED_API_KEY')
            if not api_key:
                logger.debug("No FRED API key available")
                return None
            
            async with FREDProvider(api_key) as provider:
                events = await provider.get_upcoming_releases(days_ahead)
                return events or []
                
        except Exception as e:
            logger.error(f"FRED events fetch error: {e}")
            return None
    
    async def _fetch_ecb_events(self, days_ahead: int) -> Optional[List[EconomicEvent]]:
        """Fetch events from ECB API."""
        try:
            from .ecb_provider import ECBProvider
            
            async with ECBProvider() as provider:
                events = await provider.get_upcoming_releases(days_ahead)
                return events or []
                
        except Exception as e:
            logger.error(f"ECB events fetch error: {e}")
            return None
    
    async def _fetch_boe_events(self, days_ahead: int) -> Optional[List[EconomicEvent]]:
        """BOE provider removed due to API access issues."""
        logger.debug("BOE provider not available - API access blocked")
        return None
    
    async def _fetch_rbi_events(self, days_ahead: int) -> Optional[List[EconomicEvent]]:
        """Fetch events from RBI using web scraper."""
        try:
            from .rbi_scraper import RBIScraper
            
            async with RBIScraper() as scraper:
                events = await scraper.get_upcoming_releases(days_ahead)
                return events or []
                
        except Exception as e:
            logger.error(f"RBI scraping error: {e}")
            return None
    
    def _deduplicate_events(self, events: List[EconomicEvent]) -> List[EconomicEvent]:
        """Remove duplicate events based on title, date, and country."""
        seen_keys = set()
        unique_events = []
        
        for event in events:
            # Create unique key from title, date, and country
            event_key = f"{event.title.lower()}_{event.release_date.date()}_{event.country}"
            
            if event_key not in seen_keys:
                seen_keys.add(event_key)
                unique_events.append(event)
            else:
                # If duplicate, keep the one with more complete data
                existing_event = None
                for i, existing in enumerate(unique_events):
                    existing_key = f"{existing.title.lower()}_{existing.release_date.date()}_{existing.country}"
                    if existing_key == event_key:
                        existing_event = existing
                        existing_index = i
                        break
                
                if existing_event and self._is_more_complete(event, existing_event):
                    unique_events[existing_index] = event
        
        return unique_events
    
    def _is_more_complete(self, event1: EconomicEvent, event2: EconomicEvent) -> bool:
        """Check if event1 has more complete data than event2."""
        event1_score = (
            (1 if event1.forecast_value is not None else 0) +
            (1 if event1.previous_value is not None else 0) +
            (1 if event1.description else 0) +
            (1 if event1.unit else 0)
        )
        
        event2_score = (
            (1 if event2.forecast_value is not None else 0) +
            (1 if event2.previous_value is not None else 0) +
            (1 if event2.description else 0) +
            (1 if event2.unit else 0)
        )
        
        return event1_score > event2_score
    
    def clear_cache(self) -> None:
        """Clear cached calendar data."""
        self._calendar_cache = None
        self._cache_timestamp = None
        self._seen_events.clear()
        logger.info("Cleared economic calendar cache")


# Convenience functions
async def get_upcoming_events(currency_pair: Optional[str] = None, 
                            days_ahead: int = 7) -> Optional[List[EconomicEvent]]:
    """
    Get upcoming economic events, optionally filtered by currency pair.
    
    Args:
        currency_pair: Currency pair to filter for (e.g., 'USD/EUR')
        days_ahead: Number of days ahead to look
        
    Returns:
        List of upcoming events or None if failed
    """
    collector = EconomicCalendarCollector()
    calendar = await collector.get_economic_calendar(days_ahead)
    
    if not calendar:
        return None
    
    upcoming = calendar.get_upcoming_events(days_ahead)
    
    if currency_pair:
        upcoming = [event for event in upcoming if event.affects_pair(currency_pair)]
    
    return upcoming


async def get_events_by_impact(impact: EventImpact, 
                             days_ahead: int = 14) -> Optional[List[EconomicEvent]]:
    """
    Get events filtered by impact level.
    
    Args:
        impact: Event impact level to filter for
        days_ahead: Number of days ahead to look
        
    Returns:
        List of events or None if failed
    """
    collector = EconomicCalendarCollector()
    calendar = await collector.get_economic_calendar(days_ahead)
    
    if not calendar:
        return None
    
    return calendar.get_events_by_impact(impact)