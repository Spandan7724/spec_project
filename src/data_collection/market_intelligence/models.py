from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Dict


@dataclass
class EconomicEvent:
    """Standardized economic calendar event."""

    when_utc: datetime
    when_local: datetime
    timezone: str
    country: str
    currency: str
    event: str
    importance: str  # "high" | "medium" | "low"
    source: str
    source_url: str

    @property
    def proximity_minutes(self) -> int:
        now = datetime.now(timezone.utc)
        return int((self.when_utc - now).total_seconds() / 60)

    @property
    def is_imminent(self) -> bool:
        return 0 <= self.proximity_minutes <= 60

    @property
    def is_today(self) -> bool:
        return self.when_utc.date() == datetime.now(timezone.utc).date()


@dataclass
class NewsClassification:
    """Classification result for a single article."""

    article_id: str
    url: str
    source: str
    title: str
    published_utc: Optional[datetime]
    relevance: Dict[str, float]  # {"USD": 0.9, "EUR": 0.2}
    sentiment: Dict[str, float]  # {"USD": 0.4, "EUR": -0.1} range: -1 to +1
    quality_flags: Dict[str, bool]  # {"clickbait": false, "rumor_speculative": false, "non_econ": false}

