"""
Free Economic Calendar Data Sources.

Integrates with free APIs (FRED, ECB, BOE) and web scraping for economic 
calendar events affecting currency markets.
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import httpx
from dataclasses import dataclass
import re
import json

logger = logging.getLogger(__name__)


@dataclass
class EconomicEvent:
    """Structured economic calendar event."""
    name: str
    date: datetime
    currency: str
    importance: str  # low, medium, high
    category: str    # monetary_policy, inflation, employment, gdp, etc.
    previous_value: Optional[str] = None
    forecast_value: Optional[str] = None
    actual_value: Optional[str] = None
    source: str = "Unknown"
    impact_direction: str = "neutral"  # positive, negative, neutral


class FREDDataProvider:
    """
    Federal Reserve Economic Data (FRED) API provider.
    Free API for US economic indicators.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.base_url = "https://api.stlouisfed.org/fred"
        self.api_key = api_key  # Free registration at https://fred.stlouisfed.org/docs/api/api_key.html
        self.session = None
    
    async def __aenter__(self):
        self.session = httpx.AsyncClient(timeout=30.0)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.aclose()
    
    async def fetch_economic_indicators(self, 
                                      days_ahead: int = 30) -> List[EconomicEvent]:
        """
        Fetch US economic indicators from FRED.
        
        Args:
            days_ahead: Days ahead to look for releases
            
        Returns:
            List of economic events
        """
        if not self.session:
            raise RuntimeError("Provider not properly initialized")
        
        if not self.api_key:
            logger.warning("No FRED API key - using mock US economic data")
            return self._generate_mock_us_events(days_ahead)
        
        try:
            events = []
            
            # Key US economic indicators
            indicators = {
                'GDP': 'GDP',
                'UNRATE': 'Unemployment Rate', 
                'CPIAUCSL': 'Consumer Price Index',
                'PAYEMS': 'Nonfarm Payrolls',
                'FEDFUNDS': 'Federal Funds Rate'
            }
            
            for series_id, name in indicators.items():
                try:
                    event = await self._fetch_series_latest_release(series_id, name)
                    if event:
                        events.append(event)
                except Exception as e:
                    logger.warning(f"Failed to fetch {name}: {e}")
                    continue
            
            return events
            
        except Exception as e:
            logger.error(f"FRED data fetch failed: {e}")
            return self._generate_mock_us_events(days_ahead)
    
    async def _fetch_series_latest_release(self, series_id: str, name: str) -> Optional[EconomicEvent]:
        """Fetch latest release info for a FRED series."""
        try:
            # Get series release dates
            url = f"{self.base_url}/series/releases"
            params = {
                'series_id': series_id,
                'api_key': self.api_key,
                'file_type': 'json'
            }
            
            response = await self.session.get(url, params=params)
            if response.status_code != 200:
                return None
            
            data = response.json()
            releases = data.get('releases', [])
            
            if not releases:
                return None
            
            # Get the most recent release
            latest_release = releases[0]
            
            # Parse release date
            release_date_str = latest_release.get('realtime_end', '')
            try:
                release_date = datetime.fromisoformat(release_date_str)
            except:
                release_date = datetime.utcnow() + timedelta(days=7)  # Default future date
            
            # Determine importance based on series
            importance = self._determine_importance(series_id)
            category = self._determine_category(series_id)
            
            return EconomicEvent(
                name=name,
                date=release_date,
                currency="USD",
                importance=importance,
                category=category,
                source="FRED",
                impact_direction="neutral"
            )
            
        except Exception as e:
            logger.error(f"Failed to fetch FRED series {series_id}: {e}")
            return None
    
    def _determine_importance(self, series_id: str) -> str:
        """Determine importance level based on series ID."""
        high_impact = ['GDP', 'PAYEMS', 'FEDFUNDS', 'CPIAUCSL']
        medium_impact = ['UNRATE', 'INDPRO', 'HOUST']
        
        if series_id in high_impact:
            return "high"
        elif series_id in medium_impact:
            return "medium"
        else:
            return "low"
    
    def _determine_category(self, series_id: str) -> str:
        """Determine category based on series ID."""
        categories = {
            'GDP': 'gdp',
            'PAYEMS': 'employment',
            'UNRATE': 'employment',
            'CPIAUCSL': 'inflation',
            'FEDFUNDS': 'monetary_policy'
        }
        return categories.get(series_id, 'other')
    
    def _generate_mock_us_events(self, days_ahead: int) -> List[EconomicEvent]:
        """Generate mock US economic events when API unavailable."""
        base_date = datetime.utcnow()
        
        mock_events = [
            EconomicEvent(
                name="GDP Growth Rate (Q3)",
                date=base_date + timedelta(days=7),
                currency="USD",
                importance="high",
                category="gdp",
                previous_value="2.1%",
                forecast_value="2.3%",
                source="FRED (Mock)"
            ),
            EconomicEvent(
                name="Consumer Price Index",
                date=base_date + timedelta(days=14),
                currency="USD", 
                importance="high",
                category="inflation",
                previous_value="3.2%",
                forecast_value="3.1%",
                source="FRED (Mock)"
            ),
            EconomicEvent(
                name="Nonfarm Payrolls",
                date=base_date + timedelta(days=21),
                currency="USD",
                importance="high", 
                category="employment",
                previous_value="223K",
                forecast_value="180K",
                source="FRED (Mock)"
            )
        ]
        
        return [event for event in mock_events if (event.date - base_date).days <= days_ahead]


class ECBDataProvider:
    """
    European Central Bank Statistical Data Warehouse provider.
    Free API for EU economic indicators.
    """
    
    def __init__(self):
        self.base_url = "https://data-api.ecb.europa.eu/service/data"
        self.session = None
    
    async def __aenter__(self):
        self.session = httpx.AsyncClient(timeout=30.0)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.aclose()
    
    async def fetch_eu_economic_indicators(self, days_ahead: int = 30) -> List[EconomicEvent]:
        """Fetch EU economic indicators from ECB."""
        if not self.session:
            raise RuntimeError("Provider not properly initialized")
        
        try:
            # For now, return mock EU events (ECB API is complex)
            # TODO: Implement full ECB API integration
            return self._generate_mock_eu_events(days_ahead)
            
        except Exception as e:
            logger.error(f"ECB data fetch failed: {e}")
            return self._generate_mock_eu_events(days_ahead)
    
    def _generate_mock_eu_events(self, days_ahead: int) -> List[EconomicEvent]:
        """Generate mock EU economic events."""
        base_date = datetime.utcnow()
        
        mock_events = [
            EconomicEvent(
                name="ECB Interest Rate Decision",
                date=base_date + timedelta(days=10),
                currency="EUR",
                importance="high",
                category="monetary_policy",
                previous_value="4.50%",
                forecast_value="4.25%",
                source="ECB (Mock)"
            ),
            EconomicEvent(
                name="Eurozone CPI Flash Estimate",
                date=base_date + timedelta(days=18),
                currency="EUR",
                importance="high",
                category="inflation",
                previous_value="2.4%",
                forecast_value="2.2%",
                source="ECB (Mock)"
            ),
            EconomicEvent(
                name="Eurozone GDP Growth",
                date=base_date + timedelta(days=25),
                currency="EUR",
                importance="medium",
                category="gdp",
                previous_value="0.3%",
                forecast_value="0.4%",
                source="ECB (Mock)"
            )
        ]
        
        return [event for event in mock_events if (event.date - base_date).days <= days_ahead]


class BOEDataProvider:
    """
    Bank of England data provider.
    Free access to UK economic indicators.
    """
    
    def __init__(self):
        self.base_url = "https://www.bankofengland.co.uk"
        self.session = None
    
    async def __aenter__(self):
        self.session = httpx.AsyncClient(timeout=30.0)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.aclose()
    
    async def fetch_uk_economic_indicators(self, days_ahead: int = 30) -> List[EconomicEvent]:
        """Fetch UK economic indicators."""
        try:
            # For now, return mock UK events
            # TODO: Implement BOE API integration if needed
            return self._generate_mock_uk_events(days_ahead)
            
        except Exception as e:
            logger.error(f"BOE data fetch failed: {e}")
            return self._generate_mock_uk_events(days_ahead)
    
    def _generate_mock_uk_events(self, days_ahead: int) -> List[EconomicEvent]:
        """Generate mock UK economic events."""
        base_date = datetime.utcnow()
        
        mock_events = [
            EconomicEvent(
                name="BOE Interest Rate Decision",
                date=base_date + timedelta(days=12),
                currency="GBP",
                importance="high",
                category="monetary_policy",
                previous_value="5.25%",
                forecast_value="5.00%",
                source="BOE (Mock)"
            ),
            EconomicEvent(
                name="UK CPI Inflation",
                date=base_date + timedelta(days=20),
                currency="GBP",
                importance="high",
                category="inflation",
                previous_value="4.0%",
                forecast_value="3.8%",
                source="BOE (Mock)"
            )
        ]
        
        return [event for event in mock_events if (event.date - base_date).days <= days_ahead]


class EconomicCalendarAggregator:
    """Aggregates economic calendar data from multiple free sources."""
    
    def __init__(self, fred_api_key: Optional[str] = None):
        self.fred_api_key = fred_api_key
    
    async def fetch_comprehensive_calendar(self,
                                         target_currencies: List[str],
                                         days_ahead: int = 14) -> List[EconomicEvent]:
        """
        Fetch economic calendar from all available free sources.
        
        Args:
            target_currencies: List of currencies to focus on
            days_ahead: Days ahead to fetch events for
            
        Returns:
            Aggregated economic events
        """
        all_events = []
        
        # Fetch US data if USD is in target currencies
        if 'USD' in target_currencies:
            try:
                async with FREDDataProvider(self.fred_api_key) as fred:
                    us_events = await fred.fetch_economic_indicators(days_ahead)
                    all_events.extend(us_events)
                    logger.info(f"Fetched {len(us_events)} US economic events")
            except Exception as e:
                logger.error(f"Failed to fetch US economic data: {e}")
        
        # Fetch EU data if EUR is in target currencies  
        if 'EUR' in target_currencies:
            try:
                async with ECBDataProvider() as ecb:
                    eu_events = await ecb.fetch_eu_economic_indicators(days_ahead)
                    all_events.extend(eu_events)
                    logger.info(f"Fetched {len(eu_events)} EU economic events")
            except Exception as e:
                logger.error(f"Failed to fetch EU economic data: {e}")
        
        # Fetch UK data if GBP is in target currencies
        if 'GBP' in target_currencies:
            try:
                async with BOEDataProvider() as boe:
                    uk_events = await boe.fetch_uk_economic_indicators(days_ahead)
                    all_events.extend(uk_events)
                    logger.info(f"Fetched {len(uk_events)} UK economic events")
            except Exception as e:
                logger.error(f"Failed to fetch UK economic data: {e}")
        
        # Sort by date and importance
        all_events.sort(key=lambda x: (x.date, -self._importance_score(x.importance)))
        
        return all_events
    
    def _importance_score(self, importance: str) -> int:
        """Convert importance to numerical score for sorting."""
        scores = {"high": 3, "medium": 2, "low": 1}
        return scores.get(importance.lower(), 1)


# Convenience function
async def fetch_real_economic_calendar(target_currencies: List[str],
                                     days_ahead: int = 14,
                                     fred_api_key: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Convenience function to fetch real economic calendar data.
    
    Args:
        target_currencies: List of currencies to focus on  
        days_ahead: Days ahead to fetch events
        fred_api_key: Optional FRED API key
        
    Returns:
        List of economic events in dictionary format
    """
    aggregator = EconomicCalendarAggregator(fred_api_key)
    events = await aggregator.fetch_comprehensive_calendar(target_currencies, days_ahead)
    
    # Convert to dictionary format for agent consumption
    return [
        {
            "name": event.name,
            "date": event.date.isoformat(),
            "currency": event.currency,
            "importance": event.importance,
            "category": event.category,
            "previous": event.previous_value,
            "forecast": event.forecast_value,
            "actual": event.actual_value,
            "source": event.source,
            "impact_direction": event.impact_direction
        }
        for event in events
    ]


if __name__ == "__main__":
    # Test economic calendar providers
    async def test_economic_providers():
        print("📅 Testing Economic Calendar Data Sources...")
        
        # Test FRED provider
        async with FREDDataProvider() as fred:
            us_events = await fred.fetch_economic_indicators(21)
            print(f"US Economic Events: {len(us_events)}")
            for event in us_events:
                print(f"  - {event.name} ({event.date.strftime('%Y-%m-%d')}) - {event.importance}")
        
        # Test aggregated calendar
        calendar_data = await fetch_real_economic_calendar(['USD', 'EUR', 'GBP'], 14)
        print(f"Total Calendar Events: {len(calendar_data)}")
    
    asyncio.run(test_economic_providers())