"""
ECB (European Central Bank) data provider for European economic events.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import httpx

from .calendar_collector import EconomicEvent, EventImpact, EventStatus

logger = logging.getLogger(__name__)


class ECBProvider:
    """
    Provider for European economic data via ECB Statistical Data Warehouse API.
    
    ECB provides Euro area economic data including:
    - Inflation data (HICP)
    - GDP and economic indicators
    - Employment statistics
    - ECB policy rates and decisions
    """
    
    def __init__(self):
        self.base_url = "https://data-api.ecb.europa.eu/service/data"
        self.client: Optional[httpx.AsyncClient] = None
        
        # Key ECB data series that affect EUR
        self.key_series = {
            # Inflation
            'ICP.M.U2.N.000000.4.ANR': {
                'title': 'Euro Area HICP - All Items',
                'impact': EventImpact.HIGH,
                'category': 'inflation',
                'frequency': 'monthly',
                'description': 'Harmonised Index of Consumer Prices'
            },
            'ICP.M.U2.N.XEF000.4.ANR': {
                'title': 'Euro Area Core HICP',
                'impact': EventImpact.HIGH,
                'category': 'inflation',
                'frequency': 'monthly',
                'description': 'Core HICP excluding energy and food'
            },
            
            # GDP
            'MNA.Q.N.I8.W2.S1.S1.B.B1GQ._Z._Z._Z.EUR.LR.N': {
                'title': 'Euro Area GDP',
                'impact': EventImpact.HIGH,
                'category': 'growth',
                'frequency': 'quarterly',
                'description': 'Gross Domestic Product'
            },
            
            # Employment  
            'LFSI.M.I8.S.UNEHRT.TOTAL0.15_74.T': {
                'title': 'Euro Area Unemployment Rate',
                'impact': EventImpact.HIGH,
                'category': 'employment',
                'frequency': 'monthly',
                'description': 'Unemployment rate, total'
            },
            
            # ECB Policy
            'FM.D.U2.EUR.4F.KR.MRR_FR.LEV': {
                'title': 'ECB Main Refinancing Rate',
                'impact': EventImpact.HIGH,
                'category': 'policy',
                'frequency': 'daily',
                'description': 'ECB key interest rate'
            },
            
            # Industrial Production
            'STS.M.I8.N.PROD.NS0010.4.000': {
                'title': 'Euro Area Industrial Production',
                'impact': EventImpact.MEDIUM,
                'category': 'production',
                'frequency': 'monthly',
                'description': 'Industrial production index'
            },
            
            # PMI-like indicators (if available)
            'ESI.M.I8.N.INDU.BS.4.VAL': {
                'title': 'Euro Area Industrial Confidence',
                'impact': EventImpact.MEDIUM,
                'category': 'sentiment',
                'frequency': 'monthly',
                'description': 'Industrial confidence indicator'
            }
        }
    
    async def __aenter__(self):
        """Initialize HTTP client."""
        self.client = httpx.AsyncClient(
            timeout=30.0,
            headers={
                # Use application/json which works with ECB API
                'Accept': 'application/json',
                'User-Agent': 'CurrencyAssistant/1.0'
            }
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Close HTTP client."""
        if self.client:
            await self.client.aclose()
    
    async def get_upcoming_releases(self, days_ahead: int = 14) -> Optional[List[EconomicEvent]]:
        """
        Get upcoming European economic data releases.
        
        Note: ECB API doesn't provide release calendars, so we estimate based on
        typical European statistical release patterns.
        """
        if not self.client:
            raise RuntimeError("Provider not initialized - use async context manager")
        
        try:
            events = []
            
            # For each key series, try to estimate next release
            for series_key, series_info in self.key_series.items():
                try:
                    event = await self._create_event_for_series(series_key, series_info, days_ahead)
                    if event:
                        events.append(event)
                        
                except Exception as e:
                    logger.debug(f"Failed to create ECB event for {series_key}: {e}")
                    continue
                
                # Add small delay to be respectful to ECB API
                await asyncio.sleep(0.1)
            
            logger.info(f"Created {len(events)} ECB events")
            return events
            
        except Exception as e:
            logger.error(f"ECB upcoming releases error: {e}")
            return None
    
    async def _create_event_for_series(self, series_key: str, series_info: Dict, 
                                     days_ahead: int) -> Optional[EconomicEvent]:
        """Create an economic event for an ECB series."""
        try:
            # Get recent data to understand release pattern
            recent_data = await self._get_recent_data(series_key, last_n_periods=6)
            if not recent_data:
                return None
            
            # Estimate next release date
            next_release = self._estimate_next_release_date(
                series_info['frequency'], 
                recent_data,
                series_info['category']
            )
            
            if not next_release or next_release > datetime.utcnow() + timedelta(days=days_ahead):
                return None
            
            # Get previous value (most recent)
            previous_value = None
            if recent_data and recent_data[0]:
                try:
                    previous_value = float(recent_data[0]['value'])
                except (ValueError, TypeError, KeyError):
                    previous_value = None
            
            # Create event
            event = EconomicEvent(
                event_id=f"ecb_{hash(series_key)}_{next_release.date()}",
                title=series_info['title'],
                country='EU',
                currency='EUR',
                release_date=next_release,
                period=self._format_period(next_release, series_info['frequency']),
                impact=series_info['impact'],
                affected_currencies=['EUR'],
                previous_value=previous_value,
                forecast_value=None,  # ECB doesn't provide forecasts
                actual_value=None,
                status=EventStatus.SCHEDULED,
                source='ECB',
                category=series_info['category'],
                frequency=series_info['frequency'],
                description=series_info['description']
            )
            
            return event
            
        except Exception as e:
            logger.debug(f"Error creating ECB event for {series_key}: {e}")
            return None
    
    async def _get_recent_data(self, series_key: str, last_n_periods: int = 6) -> Optional[List[Dict]]:
        """Get recent data points for an ECB series."""
        try:
            # ECB API endpoint format requires dataflow and key separated by '/'
            url = self._build_series_url(series_key)
            
            # Get data for last 2 years to ensure we have recent points
            start_date = (datetime.utcnow() - timedelta(days=730)).strftime('%Y-%m')
            end_date = datetime.utcnow().strftime('%Y-%m')
            
            params = {
                'startPeriod': start_date,
                'endPeriod': end_date,
                'lastNObservations': last_n_periods
            }
            
            response = await self.client.get(url, params=params)
            # No fallback needed - application/json works
            
            # ECB API can return 404 for some series, that's OK
            if response.status_code == 404:
                logger.debug(f"ECB series {series_key} not found")
                return None
            
            response.raise_for_status()
            data = response.json()
            
            # Parse ECB SDMX JSON format
            observations = self._parse_ecb_observations(data)
            return observations
            
        except Exception as e:
            logger.debug(f"ECB data fetch error for {series_key}: {e}")
            return None
    
    def _parse_ecb_observations(self, data: Dict) -> List[Dict]:
        """Parse ECB JSON format to extract observations."""
        observations = []
        
        try:
            if 'dataSets' in data and data['dataSets']:
                dataset = data['dataSets'][0]
                
                if 'series' in dataset:
                    # Get the first (and usually only) series
                    series_key = list(dataset['series'].keys())[0]
                    series_data = dataset['series'][series_key]
                    
                    if 'observations' in series_data:
                        # Get time dimension from structure
                        structure = data['structure']
                        time_values = structure['dimensions']['observation'][0]['values']
                        
                        # Map observations to time periods
                        for obs_key, obs_data in series_data['observations'].items():
                            if isinstance(obs_data, list) and len(obs_data) > 0:
                                time_index = int(obs_key)
                                
                                if time_index < len(time_values):
                                    period = time_values[time_index]['name']
                                    value = obs_data[0]  # First element is the value
                                    
                                    observations.append({
                                        'period': period,
                                        'value': value
                                    })
            
            return observations[::-1]  # Reverse to get most recent first
            
        except Exception as e:
            logger.debug(f"Error parsing ECB observations: {e}")
            return []
    
    def _estimate_next_release_date(self, frequency: str, recent_data: List[Dict], 
                                  category: str) -> Optional[datetime]:
        """Estimate next release date for European statistics."""
        if not recent_data:
            return None
        
        try:
            # European statistics have typical release schedules
            now = datetime.utcnow()
            
            if frequency == 'monthly':
                if category == 'inflation':
                    # HICP typically released around 17th of following month
                    next_month = now.replace(day=1) + timedelta(days=32)
                    next_release = next_month.replace(day=17)
                elif category == 'employment':
                    # Employment data typically 2-3 months delay
                    next_release = now + timedelta(days=60)
                else:
                    # Other monthly data: ~6 weeks delay
                    next_release = now + timedelta(days=42)
                    
            elif frequency == 'quarterly':
                # GDP and quarterly data: ~2-3 months after quarter end
                # Next quarter end
                current_quarter = (now.month - 1) // 3
                next_quarter_month = (current_quarter + 1) * 3 + 1
                if next_quarter_month > 12:
                    next_quarter_month = 1
                    next_year = now.year + 1
                else:
                    next_year = now.year
                    
                quarter_end = datetime(next_year, next_quarter_month, 1) - timedelta(days=1)
                next_release = quarter_end + timedelta(days=75)  # ~2.5 month delay
                
            elif frequency == 'daily':
                # Policy rates: next ECB meeting (roughly every 6 weeks)
                next_release = now + timedelta(days=42)
                
            else:
                # Default estimation
                next_release = now + timedelta(days=30)
            
            # Ensure it's in business hours and future
            if next_release <= now:
                next_release = now + timedelta(days=1)
            
            # Adjust to business day (avoid weekends)
            if next_release.weekday() >= 5:  # Saturday or Sunday
                next_release += timedelta(days=8 - next_release.weekday())
            
            return next_release
            
        except Exception as e:
            logger.debug(f"Error estimating ECB release date: {e}")
            return None
    
    def _format_period(self, release_date: datetime, frequency: str) -> str:
        """Format the period string for European data."""
        if frequency == 'monthly':
            return release_date.strftime('%B %Y')
        elif frequency == 'quarterly':
            quarter = (release_date.month - 1) // 3 + 1
            return f"Q{quarter} {release_date.year}"
        elif frequency == 'annual':
            return str(release_date.year)
        else:
            return release_date.strftime('%Y-%m-%d')
    
    async def get_historical_data(self, series_key: str, 
                                start_date: datetime, 
                                end_date: datetime) -> Optional[List[Dict]]:
        """Get historical data for an ECB series."""
        if not self.client:
            raise RuntimeError("Provider not initialized")
        
        try:
            url = self._build_series_url(series_key)
            params = {
                'startPeriod': start_date.strftime('%Y-%m'),
                'endPeriod': end_date.strftime('%Y-%m')
            }
            
            response = await self.client.get(url, params=params)
            # application/json header works fine
            if response.status_code == 404:
                return None
                
            response.raise_for_status()
            data = response.json()
            
            observations = self._parse_ecb_observations(data)
            return observations
            
        except Exception as e:
            logger.error(f"ECB historical data error for {series_key}: {e}")
            return None

    def _build_series_url(self, series_key: str) -> str:
        """Build correct ECB SDMX series URL from dot-separated key.

        Expected series_key like: "ICP.M.U2.N.000000.4.ANR" where
        dataflow = "ICP" and the remaining is the key. The API path must be
        "/service/data/{dataflow}/{key}".
        """
        try:
            parts = series_key.split('.', 1)
            if len(parts) == 2:
                flow, key = parts[0], parts[1]
                return f"{self.base_url}/{flow}/{key}"
            # If not in expected format, fall back to original (may 404)
            return f"{self.base_url}/{series_key}"
        except Exception:
            return f"{self.base_url}/{series_key}"
