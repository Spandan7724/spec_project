"""
RBI (Reserve Bank of India) data provider for Indian economic events.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import httpx

from .calendar_collector import EconomicEvent, EventImpact, EventStatus

logger = logging.getLogger(__name__)


class RBIProvider:
    """
    Provider for Indian economic data via RBI APIs and statistical sources.
    
    RBI and Indian statistical agencies provide data on:
    - RBI Monetary Policy Committee (MPC) decisions
    - Indian inflation (CPI, WPI)
    - GDP and economic indicators
    - Current account balance and forex reserves
    - Industrial production and manufacturing
    """
    
    def __init__(self):
        self.rbi_base_url = "https://rbi.org.in"
        self.client: Optional[httpx.AsyncClient] = None
        
        # Key Indian economic events that affect INR
        self.scheduled_events = {
            # RBI Policy Events (scheduled dates - updated quarterly)
            'rbi_mpc': {
                'title': 'RBI Monetary Policy Committee Decision',
                'impact': EventImpact.HIGH,
                'category': 'policy',
                'frequency': 'bi-monthly',
                'description': 'RBI policy rate and monetary policy stance',
                'typical_dates': [  # 2025 MPC meeting dates (approximate)
                    '2025-02-07', '2025-04-05', '2025-06-07', 
                    '2025-08-09', '2025-10-11', '2025-12-06'
                ]
            },
            
            # Economic Data Releases (estimated based on typical patterns)
            'india_cpi': {
                'title': 'India Consumer Price Index',
                'impact': EventImpact.HIGH,
                'category': 'inflation',
                'frequency': 'monthly',
                'description': 'Indian inflation rate - CPI combined',
                'release_delay_days': 12  # CPI typically released ~12 days after month end
            },
            
            'india_wpi': {
                'title': 'India Wholesale Price Index',
                'impact': EventImpact.MEDIUM,
                'category': 'inflation',
                'frequency': 'monthly',
                'description': 'Wholesale price inflation',
                'release_delay_days': 14
            },
            
            'india_gdp': {
                'title': 'India GDP Growth',
                'impact': EventImpact.HIGH,
                'category': 'growth',
                'frequency': 'quarterly',
                'description': 'Quarterly GDP growth rate',
                'release_delay_days': 60  # GDP typically released ~2 months after quarter
            },
            
            'india_iip': {
                'title': 'India Industrial Production (IIP)',
                'impact': EventImpact.MEDIUM,
                'category': 'production',
                'frequency': 'monthly',
                'description': 'Index of Industrial Production',
                'release_delay_days': 45  # IIP typically 6 weeks delay
            },
            
            'india_pmi_mfg': {
                'title': 'India Manufacturing PMI',
                'impact': EventImpact.MEDIUM,
                'category': 'manufacturing',
                'frequency': 'monthly',
                'description': 'Manufacturing Purchasing Managers Index',
                'release_delay_days': 1  # PMI typically released on 1st of next month
            },
            
            'india_pmi_services': {
                'title': 'India Services PMI',
                'impact': EventImpact.MEDIUM,
                'category': 'services',
                'frequency': 'monthly',
                'description': 'Services Purchasing Managers Index',
                'release_delay_days': 3
            },
            
            'india_trade_balance': {
                'title': 'India Trade Balance',
                'impact': EventImpact.MEDIUM,
                'category': 'trade',
                'frequency': 'monthly',
                'description': 'Monthly trade deficit/surplus',
                'release_delay_days': 15
            },
            
            'india_forex_reserves': {
                'title': 'India Forex Reserves',
                'impact': EventImpact.MEDIUM,
                'category': 'reserves',
                'frequency': 'weekly',
                'description': 'Foreign exchange reserves',
                'release_delay_days': 7  # Released weekly on Fridays
            },
            
            'india_current_account': {
                'title': 'India Current Account Balance',
                'impact': EventImpact.HIGH,
                'category': 'balance_of_payments',
                'frequency': 'quarterly',
                'description': 'Current account deficit/surplus',
                'release_delay_days': 90  # Typically 3 months delay
            }
        }
    
    async def __aenter__(self):
        """Initialize HTTP client."""
        self.client = httpx.AsyncClient(
            timeout=30.0,
            headers={
                'User-Agent': 'CurrencyAssistant/1.0',
                'Accept': 'application/json, text/html'
            }
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Close HTTP client."""
        if self.client:
            await self.client.aclose()
    
    async def get_upcoming_releases(self, days_ahead: int = 14) -> Optional[List[EconomicEvent]]:
        """
        Get upcoming Indian economic data releases.
        
        Since RBI doesn't provide a comprehensive API calendar, we estimate
        based on typical Indian statistical release patterns.
        """
        if not self.client:
            raise RuntimeError("Provider not initialized - use async context manager")
        
        try:
            events = []
            
            # Create events for each type of Indian economic data
            for event_key, event_info in self.scheduled_events.items():
                try:
                    estimated_events = await self._create_events_for_type(
                        event_key, event_info, days_ahead
                    )
                    if estimated_events:
                        events.extend(estimated_events)
                        
                except Exception as e:
                    logger.debug(f"Failed to create Indian events for {event_key}: {e}")
                    continue
            
            # Try to get actual RBI MPC dates if available
            try:
                rbi_mpc_events = await self._get_rbi_mpc_schedule(days_ahead)
                if rbi_mpc_events:
                    # Replace estimated MPC events with actual ones
                    events = [e for e in events if 'MPC' not in e.title]
                    events.extend(rbi_mpc_events)
            except Exception as e:
                logger.debug(f"Could not fetch actual RBI MPC dates: {e}")
            
            logger.info(f"Created {len(events)} Indian economic events")
            return events
            
        except Exception as e:
            logger.error(f"RBI upcoming releases error: {e}")
            return None
    
    async def _create_events_for_type(self, event_key: str, event_info: Dict, 
                                    days_ahead: int) -> Optional[List[EconomicEvent]]:
        """Create events for a specific type of Indian economic data."""
        events = []
        
        try:
            if event_key == 'rbi_mpc':
                # Handle RBI MPC scheduled dates
                events.extend(await self._create_mpc_events(event_info, days_ahead))
            else:
                # Handle regular economic data releases
                next_releases = self._estimate_next_releases(event_info, days_ahead)
                
                for release_date in next_releases:
                    event = EconomicEvent(
                        event_id=f"rbi_{event_key}_{release_date.date()}",
                        title=event_info['title'],
                        country='IN',
                        currency='INR',
                        release_date=release_date,
                        period=self._format_period(release_date, event_info['frequency']),
                        impact=event_info['impact'],
                        affected_currencies=['INR'],
                        previous_value=None,  # Would need separate API calls to get historical data
                        forecast_value=None,
                        actual_value=None,
                        status=EventStatus.SCHEDULED,
                        source='RBI',
                        category=event_info['category'],
                        frequency=event_info['frequency'],
                        description=event_info['description']
                    )
                    events.append(event)
            
            return events
            
        except Exception as e:
            logger.debug(f"Error creating events for {event_key}: {e}")
            return None
    
    async def _create_mpc_events(self, event_info: Dict, days_ahead: int) -> List[EconomicEvent]:
        """Create RBI MPC events based on scheduled dates."""
        events = []
        cutoff_date = datetime.utcnow() + timedelta(days=days_ahead)
        
        for date_str in event_info.get('typical_dates', []):
            try:
                event_date = datetime.strptime(date_str, '%Y-%m-%d')
                
                # Only include future dates within our range
                if datetime.utcnow() < event_date <= cutoff_date:
                    event = EconomicEvent(
                        event_id=f"rbi_mpc_{event_date.date()}",
                        title=event_info['title'],
                        country='IN',
                        currency='INR',
                        release_date=event_date.replace(hour=14, minute=30),  # Typical RBI announcement time
                        period=self._format_period(event_date, event_info['frequency']),
                        impact=event_info['impact'],
                        affected_currencies=['INR', 'USD'],  # MPC decisions affect USD/INR heavily
                        previous_value=None,
                        forecast_value=None,
                        actual_value=None,
                        status=EventStatus.SCHEDULED,
                        source='RBI',
                        category=event_info['category'],
                        frequency=event_info['frequency'],
                        description=event_info['description']
                    )
                    events.append(event)
                    
            except ValueError:
                continue
        
        return events
    
    def _estimate_next_releases(self, event_info: Dict, days_ahead: int) -> List[datetime]:
        """Estimate next release dates based on frequency and typical delays."""
        releases = []
        now = datetime.utcnow()
        cutoff_date = now + timedelta(days=days_ahead)
        
        frequency = event_info['frequency']
        delay_days = event_info.get('release_delay_days', 30)
        
        if frequency == 'monthly':
            # Monthly data: typically released for previous month
            current_month = now.replace(day=1)
            
            # Generate next few monthly releases
            for i in range(3):  # Check next 3 months
                data_month = current_month + timedelta(days=32 * i)
                data_month = data_month.replace(day=1)  # First of month
                
                # Add typical delay
                release_date = data_month + timedelta(days=delay_days)
                
                if now < release_date <= cutoff_date:
                    # Adjust to business day
                    release_date = self._adjust_to_business_day(release_date)
                    releases.append(release_date)
        
        elif frequency == 'quarterly':
            # Quarterly data
            current_quarter = (now.month - 1) // 3
            
            for i in range(2):  # Check next 2 quarters
                quarter = (current_quarter + i) % 4
                year = now.year + ((current_quarter + i) // 4)
                quarter_end_month = quarter * 3 + 3
                
                quarter_end = datetime(year, quarter_end_month, 1) - timedelta(days=1)
                release_date = quarter_end + timedelta(days=delay_days)
                
                if now < release_date <= cutoff_date:
                    release_date = self._adjust_to_business_day(release_date)
                    releases.append(release_date)
        
        elif frequency == 'weekly':
            # Weekly data (like forex reserves)
            next_friday = now + timedelta(days=(4 - now.weekday()) % 7 + 7)  # Next Friday
            
            while next_friday <= cutoff_date:
                releases.append(next_friday.replace(hour=17, minute=0))  # 5 PM release
                next_friday += timedelta(days=7)
        
        return releases
    
    def _adjust_to_business_day(self, date: datetime) -> datetime:
        """Adjust date to nearest business day (avoid weekends)."""
        if date.weekday() >= 5:  # Weekend
            # Move to next Monday
            days_to_add = 7 - date.weekday()
            date += timedelta(days=days_to_add)
        
        # Set to typical business hours (11 AM IST for most releases)
        return date.replace(hour=11, minute=30)
    
    def _format_period(self, release_date: datetime, frequency: str) -> str:
        """Format the period string for Indian data."""
        if frequency == 'monthly':
            prev_month = release_date - timedelta(days=30)
            return prev_month.strftime('%B %Y')
        elif frequency == 'quarterly':
            # Determine which quarter this release is for
            quarter_month = release_date.month - 2  # Approximate
            quarter = max(1, (quarter_month - 1) // 3 + 1)
            return f"Q{quarter} FY{release_date.year}"
        elif frequency == 'weekly':
            return f"Week of {release_date.strftime('%d %B %Y')}"
        elif frequency == 'bi-monthly':
            return release_date.strftime('%B %Y')
        else:
            return release_date.strftime('%Y-%m-%d')
    
    async def _get_rbi_mpc_schedule(self, days_ahead: int) -> Optional[List[EconomicEvent]]:
        """
        Try to fetch actual RBI MPC meeting dates from RBI website.
        
        This is best-effort - RBI website structure may change.
        """
        try:
            # RBI typically publishes MPC calendar on their website
            # This is a simplified approach - real implementation might need web scraping
            
            # For now, return None - we'll use the estimated dates from typical_dates
            # In production, this could scrape https://rbi.org.in/Scripts/MonetaryPolicyCommittee.aspx
            # or use any RSS feeds RBI provides
            
            return None
            
        except Exception as e:
            logger.debug(f"Could not fetch RBI MPC schedule: {e}")
            return None
    
    async def get_historical_data(self, event_type: str, 
                                start_date: datetime, 
                                end_date: datetime) -> Optional[List[Dict]]:
        """
        Get historical data for Indian economic indicators.
        
        This would typically involve:
        1. RBI Database on Indian Economy (DBIE)
        2. Ministry of Statistics APIs
        3. Third-party data providers
        """
        # Placeholder for historical data fetching
        # In production, this would connect to:
        # - RBI DBIE (Database on Indian Economy)
        # - data.gov.in APIs
        # - Ministry of Statistics APIs
        
        logger.debug(f"Historical data for {event_type} not implemented yet")
        return None
    
    async def get_forex_reserves(self) -> Optional[Dict]:
        """
        Get latest Indian forex reserves data.
        
        This is often a key indicator for INR strength.
        """
        try:
            # RBI publishes weekly forex reserves data
            # This would typically fetch from RBI's weekly statistical supplement
            
            # Placeholder - would need actual RBI API integration
            logger.debug("Forex reserves data fetch not implemented yet")
            return None
            
        except Exception as e:
            logger.error(f"Error fetching forex reserves: {e}")
            return None