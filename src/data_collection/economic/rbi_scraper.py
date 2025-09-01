"""
RBI web scraper for real Indian economic data and events.
"""

import logging
import re
from datetime import datetime, timedelta
from typing import List, Optional
from crawl4ai import AsyncWebCrawler

from .calendar_collector import EconomicEvent, EventImpact, EventStatus

logger = logging.getLogger(__name__)


class RBIScraper:
    """
    Web scraper for RBI (Reserve Bank of India) economic data and events.
    
    Scrapes real data from:
    - RBI press releases for MPC decisions and policy announcements
    - RBI statistics page for economic data release schedules
    - Treasury bill auctions and monetary operations
    """
    
    def __init__(self):
        self.rbi_press_url = "https://www.rbi.org.in/Scripts/BS_PressReleaseDisplay.aspx"
        self.rbi_stats_url = "https://www.rbi.org.in/scripts/statistics.aspx"
        self.crawler: Optional[AsyncWebCrawler] = None
    
    async def __aenter__(self):
        """Initialize crawler."""
        self.crawler = AsyncWebCrawler(verbose=False)
        await self.crawler.__aenter__()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Close crawler."""
        if self.crawler:
            await self.crawler.__aexit__(exc_type, exc_val, exc_tb)
    
    async def get_upcoming_releases(self, days_ahead: int = 14) -> Optional[List[EconomicEvent]]:
        """
        Get upcoming Indian economic events by scraping RBI website.
        
        Args:
            days_ahead: Number of days ahead to look for events
            
        Returns:
            List of EconomicEvent objects or None if failed
        """
        if not self.crawler:
            raise RuntimeError("Scraper not initialized - use async context manager")
        
        try:
            events = []
            
            # Scrape RBI press releases for policy events
            press_events = await self._scrape_press_releases(days_ahead)
            if press_events:
                events.extend(press_events)
            
            # Scrape statistics page for data release schedules
            stats_events = await self._scrape_statistics_schedule(days_ahead)
            if stats_events:
                events.extend(stats_events)
            
            logger.info(f"RBI scraper found {len(events)} upcoming events")
            return events
            
        except Exception as e:
            logger.error(f"RBI scraping error: {e}")
            return None
    
    async def _scrape_press_releases(self, days_ahead: int) -> Optional[List[EconomicEvent]]:
        """Scrape RBI press releases for MPC meetings and policy events."""
        try:
            result = await self.crawler.arun(
                url=self.rbi_press_url,
                word_count_threshold=10
            )
            
            if result.status_code != 200:
                logger.error(f"Failed to fetch RBI press releases: {result.status_code}")
                return None
            
            content = result.markdown
            events = []
            
            # Extract MPC meeting events
            mpc_events = self._extract_mpc_events(content, days_ahead)
            events.extend(mpc_events)
            
            # Extract inflation survey events
            inflation_events = self._extract_inflation_events(content, days_ahead)
            events.extend(inflation_events)
            
            # Extract monetary policy events
            policy_events = self._extract_policy_events(content, days_ahead)
            events.extend(policy_events)
            
            return events
            
        except Exception as e:
            logger.error(f"RBI press releases scraping error: {e}")
            return None
    
    async def _scrape_statistics_schedule(self, days_ahead: int) -> Optional[List[EconomicEvent]]:
        """Scrape RBI statistics page for regular data releases."""
        try:
            result = await self.crawler.arun(
                url=self.rbi_stats_url,
                word_count_threshold=10
            )
            
            if result.status_code != 200:
                logger.error(f"Failed to fetch RBI statistics: {result.status_code}")
                return None
            
            content = result.markdown
            events = []
            
            # Extract weekly data releases (like forex reserves)
            weekly_events = self._extract_weekly_releases(content, days_ahead)
            events.extend(weekly_events)
            
            # Extract monthly data releases
            monthly_events = self._extract_monthly_releases(content, days_ahead)
            events.extend(monthly_events)
            
            return events
            
        except Exception as e:
            logger.error(f"RBI statistics scraping error: {e}")
            return None
    
    def _extract_mpc_events(self, content: str, days_ahead: int) -> List[EconomicEvent]:
        """Extract MPC meeting events from press releases."""
        events = []
        
        # Look for MPC meeting minutes and announcements
        mpc_pattern = r"Minutes of the Monetary Policy Committee Meeting.*?(\w+ \d{1,2}.*?202\d)"
        matches = re.findall(mpc_pattern, content, re.IGNORECASE)
        
        for match in matches:
            try:
                # Parse date from "August 4 to 6, 2025" format
                date_match = re.search(r"(\w+) (\d{1,2}).*?(\d{4})", match)
                if date_match:
                    month_name, day, year = date_match.groups()
                    
                    # Convert month name to number
                    month_map = {
                        'january': 1, 'february': 2, 'march': 3, 'april': 4,
                        'may': 5, 'june': 6, 'july': 7, 'august': 8,
                        'september': 9, 'october': 10, 'november': 11, 'december': 12
                    }
                    
                    month_num = month_map.get(month_name.lower())
                    if month_num:
                        event_date = datetime(int(year), month_num, int(day), 14, 30)  # 2:30 PM typical MPC time
                        
                        # Only include future events within range
                        if datetime.utcnow() < event_date <= datetime.utcnow() + timedelta(days=days_ahead):
                            event = EconomicEvent(
                                event_id=f"rbi_mpc_{event_date.date()}",
                                title="RBI Monetary Policy Committee Decision",
                                country="IN",
                                currency="INR",
                                release_date=event_date,
                                period=f"{month_name} {year}",
                                impact=EventImpact.HIGH,
                                affected_currencies=["INR", "USD"],
                                previous_value=None,  # Would need to parse from previous meetings
                                forecast_value=None,
                                actual_value=None,
                                status=EventStatus.SCHEDULED,
                                source="RBI",
                                category="monetary_policy",
                                frequency="bi-monthly",
                                description="Reserve Bank of India Monetary Policy Committee rate decision"
                            )
                            events.append(event)
                            
            except Exception as e:
                logger.debug(f"Error parsing MPC date {match}: {e}")
                continue
        
        return events
    
    def _extract_inflation_events(self, content: str, days_ahead: int) -> List[EconomicEvent]:
        """Extract inflation survey and CPI events."""
        events = []
        
        # Look for inflation expectations survey
        inflation_pattern = r"Inflation Expectations Survey.*?(\w+ 202\d)"
        matches = re.findall(inflation_pattern, content, re.IGNORECASE)
        
        for match in matches:
            try:
                # Parse month and year
                month_year_match = re.search(r"(\w+) (\d{4})", match)
                if month_year_match:
                    month_name, year = month_year_match.groups()
                    
                    # Estimate release date (typically mid-month)
                    month_map = {
                        'january': 1, 'february': 2, 'march': 3, 'april': 4,
                        'may': 5, 'june': 6, 'july': 7, 'august': 8,
                        'september': 9, 'october': 10, 'november': 11, 'december': 12
                    }
                    
                    month_num = month_map.get(month_name.lower())
                    if month_num:
                        event_date = datetime(int(year), month_num, 15, 11, 30)  # Mid-month release
                        
                        if datetime.utcnow() < event_date <= datetime.utcnow() + timedelta(days=days_ahead):
                            event = EconomicEvent(
                                event_id=f"rbi_inflation_{event_date.date()}",
                                title="RBI Inflation Expectations Survey",
                                country="IN",
                                currency="INR",
                                release_date=event_date,
                                period=f"{month_name} {year}",
                                impact=EventImpact.MEDIUM,
                                affected_currencies=["INR"],
                                previous_value=None,
                                forecast_value=None,
                                actual_value=None,
                                status=EventStatus.SCHEDULED,
                                source="RBI",
                                category="inflation",
                                frequency="quarterly",
                                description="Household inflation expectations survey results"
                            )
                            events.append(event)
                            
            except Exception as e:
                logger.debug(f"Error parsing inflation date {match}: {e}")
                continue
        
        return events
    
    def _extract_policy_events(self, content: str, days_ahead: int) -> List[EconomicEvent]:
        """Extract general policy announcements and bulletins."""
        events = []
        
        # Look for RBI Bulletin releases
        bulletin_pattern = r"RBI Bulletin.*?(\w+ 202\d)"
        matches = re.findall(bulletin_pattern, content, re.IGNORECASE)
        
        for match in matches:
            try:
                month_year_match = re.search(r"(\w+) (\d{4})", match)
                if month_year_match:
                    month_name, year = month_year_match.groups()
                    
                    month_map = {
                        'january': 1, 'february': 2, 'march': 3, 'april': 4,
                        'may': 5, 'june': 6, 'july': 7, 'august': 8,
                        'september': 9, 'october': 10, 'november': 11, 'december': 12
                    }
                    
                    month_num = month_map.get(month_name.lower())
                    if month_num:
                        # Bulletin typically released end of month
                        event_date = datetime(int(year), month_num, 28, 16, 0)
                        
                        if datetime.utcnow() < event_date <= datetime.utcnow() + timedelta(days=days_ahead):
                            event = EconomicEvent(
                                event_id=f"rbi_bulletin_{event_date.date()}",
                                title="RBI Monthly Bulletin",
                                country="IN",
                                currency="INR",
                                release_date=event_date,
                                period=f"{month_name} {year}",
                                impact=EventImpact.MEDIUM,
                                affected_currencies=["INR"],
                                previous_value=None,
                                forecast_value=None,
                                actual_value=None,
                                status=EventStatus.SCHEDULED,
                                source="RBI",
                                category="bulletin",
                                frequency="monthly",
                                description="RBI monthly economic bulletin with policy insights"
                            )
                            events.append(event)
                            
            except Exception as e:
                logger.debug(f"Error parsing bulletin date {match}: {e}")
                continue
        
        return events
    
    def _extract_weekly_releases(self, content: str, days_ahead: int) -> List[EconomicEvent]:
        """Extract weekly data releases from statistics page."""
        events = []
        
        # Look for weekly releases (forex reserves, money supply)
        # This would parse the statistics schedule page
        # For now, return empty - would need to implement parsing logic
        
        return events
    
    def _extract_monthly_releases(self, content: str, days_ahead: int) -> List[EconomicEvent]:
        """Extract monthly data releases from statistics page.""" 
        events = []
        
        # Look for monthly releases (sectoral credit, deposits, etc.)
        # This would parse the statistics schedule page
        # For now, return empty - would need to implement parsing logic
        
        return events