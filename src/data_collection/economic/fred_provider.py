"""
FRED (Federal Reserve Economic Data) API provider for US economic events.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import httpx

from .calendar_collector import EconomicEvent, EventImpact, EventStatus

logger = logging.getLogger(__name__)


class FREDProvider:
    """
    Provider for US economic data via FRED (Federal Reserve Economic Data) API.
    
    FRED provides extensive US economic data including:
    - Employment data (unemployment, job reports)
    - Inflation data (CPI, PPI, PCE)
    - GDP and economic growth
    - Fed policy and interest rates
    """
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.stlouisfed.org/fred"
        self.client: Optional[httpx.AsyncClient] = None
        
        # Key US economic series IDs that affect USD
        self.key_series = {
            # Employment
            'UNRATE': {
                'title': 'Unemployment Rate',
                'impact': EventImpact.HIGH,
                'category': 'employment',
                'frequency': 'monthly'
            },
            'PAYEMS': {
                'title': 'Nonfarm Payrolls',
                'impact': EventImpact.HIGH,
                'category': 'employment', 
                'frequency': 'monthly'
            },
            'CIVPART': {
                'title': 'Labor Force Participation Rate',
                'impact': EventImpact.MEDIUM,
                'category': 'employment',
                'frequency': 'monthly'
            },
            
            # Inflation
            'CPIAUCSL': {
                'title': 'Consumer Price Index',
                'impact': EventImpact.HIGH,
                'category': 'inflation',
                'frequency': 'monthly'
            },
            'CPILFESL': {
                'title': 'Core CPI',
                'impact': EventImpact.HIGH,
                'category': 'inflation',
                'frequency': 'monthly'
            },
            'PCEPI': {
                'title': 'PCE Price Index',
                'impact': EventImpact.HIGH,
                'category': 'inflation',
                'frequency': 'monthly'
            },
            'PCEPILFE': {
                'title': 'Core PCE Price Index',
                'impact': EventImpact.HIGH,
                'category': 'inflation',
                'frequency': 'monthly'
            },
            
            # Economic Growth
            'GDP': {
                'title': 'Gross Domestic Product',
                'impact': EventImpact.HIGH,
                'category': 'growth',
                'frequency': 'quarterly'
            },
            'GDPC1': {
                'title': 'Real GDP',
                'impact': EventImpact.HIGH,
                'category': 'growth',
                'frequency': 'quarterly'
            },
            
            # Fed Policy
            'FEDFUNDS': {
                'title': 'Federal Funds Rate',
                'impact': EventImpact.HIGH,
                'category': 'policy',
                'frequency': 'monthly'
            },
            'DFF': {
                'title': 'Federal Funds Effective Rate',
                'impact': EventImpact.HIGH,
                'category': 'policy',
                'frequency': 'daily'
            },
            
            # Consumer & Business
            'UMCSENT': {
                'title': 'Consumer Sentiment',
                'impact': EventImpact.MEDIUM,
                'category': 'sentiment',
                'frequency': 'monthly'
            },
            'INDPRO': {
                'title': 'Industrial Production',
                'impact': EventImpact.MEDIUM,
                'category': 'production',
                'frequency': 'monthly'
            },
            'HOUST': {
                'title': 'Housing Starts',
                'impact': EventImpact.MEDIUM,
                'category': 'housing',
                'frequency': 'monthly'
            }
        }
    
    async def __aenter__(self):
        """Initialize HTTP client."""
        self.client = httpx.AsyncClient(timeout=30.0)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Close HTTP client."""
        if self.client:
            await self.client.aclose()
    
    async def get_upcoming_releases(self, days_ahead: int = 14) -> Optional[List[EconomicEvent]]:
        """
        Get upcoming US economic data releases.
        
        Note: FRED doesn't provide release calendars, so we estimate based on
        typical release patterns for each series.
        """
        if not self.client:
            raise RuntimeError("Provider not initialized - use async context manager")
        
        try:
            events = []
            
            # For each key series, estimate next release date and get recent data
            for series_id, series_info in self.key_series.items():
                try:
                    event = await self._create_event_for_series(series_id, series_info, days_ahead)
                    if event:
                        events.append(event)
                        
                except Exception as e:
                    logger.debug(f"Failed to create event for {series_id}: {e}")
                    continue
            
            logger.info(f"Created {len(events)} FRED events")
            return events
            
        except Exception as e:
            logger.error(f"FRED upcoming releases error: {e}")
            return None
    
    async def _create_event_for_series(self, series_id: str, series_info: Dict, 
                                     days_ahead: int) -> Optional[EconomicEvent]:
        """Create an economic event for a FRED series."""
        try:
            # Get series metadata
            series_data = await self._get_series_info(series_id)
            if not series_data:
                return None
            
            # Get recent observations to determine pattern
            recent_data = await self._get_recent_observations(series_id, limit=12)
            if not recent_data or len(recent_data) < 2:
                return None
            
            # Estimate next release date based on frequency
            next_release = self._estimate_next_release_date(
                series_info['frequency'], 
                recent_data
            )
            
            if not next_release or next_release > datetime.utcnow() + timedelta(days=days_ahead):
                return None
            
            # Get previous value (most recent)
            previous_value = None
            if recent_data:
                try:
                    previous_value = float(recent_data[0]['value'])
                except (ValueError, TypeError):
                    previous_value = None
            
            # Create event
            event = EconomicEvent(
                event_id=f"fred_{series_id}_{next_release.date()}",
                title=series_info['title'],
                country='US',
                currency='USD',
                release_date=next_release,
                period=self._format_period(next_release, series_info['frequency']),
                impact=series_info['impact'],
                affected_currencies=['USD'],
                previous_value=previous_value,
                forecast_value=None,  # FRED doesn't provide forecasts
                actual_value=None,
                status=EventStatus.SCHEDULED,
                source='FRED',
                category=series_info['category'],
                unit=series_data.get('units_short', ''),
                frequency=series_info['frequency'],
                description=series_data.get('title', series_info['title'])
            )
            
            return event
            
        except Exception as e:
            logger.debug(f"Error creating event for {series_id}: {e}")
            return None
    
    async def _get_series_info(self, series_id: str) -> Optional[Dict]:
        """Get metadata for a FRED series."""
        try:
            url = f"{self.base_url}/series"
            params = {
                'series_id': series_id,
                'api_key': self.api_key,
                'file_type': 'json'
            }
            
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            if 'seriess' in data and data['seriess']:
                return data['seriess'][0]
            
            return None
            
        except Exception as e:
            logger.debug(f"FRED series info error for {series_id}: {e}")
            return None
    
    async def _get_recent_observations(self, series_id: str, limit: int = 12) -> Optional[List[Dict]]:
        """Get recent observations for a FRED series."""
        try:
            url = f"{self.base_url}/series/observations"
            params = {
                'series_id': series_id,
                'api_key': self.api_key,
                'file_type': 'json',
                'limit': limit,
                'sort_order': 'desc'  # Most recent first
            }
            
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            if 'observations' in data:
                return data['observations']
            
            return None
            
        except Exception as e:
            logger.debug(f"FRED observations error for {series_id}: {e}")
            return None
    
    def _estimate_next_release_date(self, frequency: str, recent_data: List[Dict]) -> Optional[datetime]:
        """Estimate next release date based on frequency and recent data patterns."""
        if not recent_data:
            return None
        
        try:
            # Get the most recent data point date
            latest_date_str = recent_data[0]['date']
            latest_date = datetime.strptime(latest_date_str, '%Y-%m-%d')
            
            # Estimate next release based on frequency
            if frequency == 'monthly':
                # Monthly data typically released with 1-2 month lag
                next_data_period = latest_date + timedelta(days=30)
                # Add typical release delay
                next_release = next_data_period + timedelta(days=45)
                
            elif frequency == 'quarterly':
                # Quarterly data typically released with 1-3 month lag
                next_data_period = latest_date + timedelta(days=90)
                next_release = next_data_period + timedelta(days=60)
                
            elif frequency == 'daily':
                # Daily data typically released next day
                next_release = datetime.utcnow() + timedelta(days=1)
                
            else:
                # Default: assume monthly pattern
                next_release = latest_date + timedelta(days=75)
            
            # Ensure it's in the future
            if next_release <= datetime.utcnow():
                next_release = datetime.utcnow() + timedelta(days=1)
            
            return next_release
            
        except Exception as e:
            logger.debug(f"Error estimating release date: {e}")
            return None
    
    def _format_period(self, release_date: datetime, frequency: str) -> str:
        """Format the period string for the event."""
        if frequency == 'monthly':
            return release_date.strftime('%B %Y')
        elif frequency == 'quarterly':
            quarter = (release_date.month - 1) // 3 + 1
            return f"Q{quarter} {release_date.year}"
        elif frequency == 'annual':
            return str(release_date.year)
        else:
            return release_date.strftime('%Y-%m-%d')
    
    async def get_historical_data(self, series_id: str, 
                                start_date: datetime, 
                                end_date: datetime) -> Optional[List[Dict]]:
        """
        Get historical data for a FRED series.
        
        This can be used for backtesting and analysis.
        """
        if not self.client:
            raise RuntimeError("Provider not initialized")
        
        try:
            url = f"{self.base_url}/series/observations"
            params = {
                'series_id': series_id,
                'api_key': self.api_key,
                'file_type': 'json',
                'start_date': start_date.strftime('%Y-%m-%d'),
                'end_date': end_date.strftime('%Y-%m-%d'),
                'sort_order': 'asc'
            }
            
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            if 'observations' in data:
                return data['observations']
            
            return None
            
        except Exception as e:
            logger.error(f"FRED historical data error for {series_id}: {e}")
            return None