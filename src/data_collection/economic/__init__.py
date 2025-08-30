"""
Economic calendar and event data integration module.
"""

from .calendar_collector import (
    EconomicEvent,
    EconomicCalendar,
    EconomicCalendarCollector,
    get_upcoming_events,
    get_events_by_impact
)

from .fred_provider import FREDProvider
from .ecb_provider import ECBProvider

__all__ = [
    # Core Calendar
    "EconomicEvent",
    "EconomicCalendar", 
    "EconomicCalendarCollector",
    "get_upcoming_events",
    "get_events_by_impact",
    
    # Providers
    "FREDProvider",
    "ECBProvider"
]