from __future__ import annotations

import asyncio
from typing import Any, Dict

from src.data_collection.market_intelligence.serper_client import SerperClient
from src.data_collection.market_intelligence.calendar_collector import CalendarCollector
from src.data_collection.market_intelligence.news_collector import NewsCollector
from src.data_collection.market_intelligence.aggregator import NewsAggregator
from src.data_collection.market_intelligence.extractors.news_classifier import NewsClassifier
from src.data_collection.market_intelligence.extractors.narrative_generator import NarrativeGenerator
from src.data_collection.market_intelligence.extractors.calendar_extractor import CalendarExtractor
from src.data_collection.market_intelligence.bias_calculator import (
    calculate_policy_bias,
    next_high_impact_event,
)
from src.llm.manager import LLMManager
from src.utils.logging import get_logger

logger = get_logger(__name__)


class MarketIntelligenceService:
    def __init__(self, serper_api_key: str | None = None, llm_manager: LLMManager | None = None):
        self.llm_manager = llm_manager or LLMManager()
        self.serper = SerperClient(api_key=serper_api_key) if serper_api_key else SerperClient()
        self.calendar = CalendarCollector(self.serper)
        self.news_collector = NewsCollector(self.serper)
        self.classifier = NewsClassifier(self.llm_manager)
        self.aggregator = NewsAggregator(self.news_collector, self.classifier)
        self.narrative = NarrativeGenerator(self.llm_manager)
        self.cal_extractor = CalendarExtractor(self.llm_manager)

    async def get_pair_intelligence(self, base: str, quote: str) -> Dict[str, Any]:
        # Get configuration for calendar sources
        from src.config import get_config, load_config
        try:
            cfg = get_config()
        except Exception:
            cfg = load_config()
        max_cal_sources = int(cfg.get("agents.market_intelligence.max_calendar_sources", 3))
        import os as _os
        _ovr = _os.getenv("MI_MAX_CAL_SOURCES")
        if _ovr is not None:
            try:
                max_cal_sources = int(_ovr)
            except ValueError:
                pass

        # Run news aggregation, base calendar, and quote calendar in parallel
        news_task = self.aggregator.get_pair_snapshot(base, quote)
        base_cal_task = self.calendar.collect_calendar_urls(base, num_results=max(1, max_cal_sources * 2))
        quote_cal_task = self.calendar.collect_calendar_urls(quote, num_results=max(1, max_cal_sources * 2))

        news_snapshot, base_cal_full, quote_cal_full = await asyncio.gather(
            news_task, base_cal_task, quote_cal_task
        )

        # Limit results
        base_cal = base_cal_full[:max_cal_sources]
        quote_cal = quote_cal_full[:max_cal_sources]

        # Run event extraction for both currencies in parallel
        base_events, quote_events = await asyncio.gather(
            self.cal_extractor.extract_events_batch(base_cal, base),
            self.cal_extractor.extract_events_batch(quote_cal, quote)
        )
        all_events = base_events + quote_events

        # Concise evidence pipeline logging for calendar path
        logger.info(
            "MI.Calendar: base_cal=%d quote_cal=%d base_events=%d quote_events=%d total=%d",
            len(base_cal), len(quote_cal), len(base_events), len(quote_events), len(all_events)
        )

        policy_bias = calculate_policy_bias(all_events)
        next_event = next_high_impact_event(all_events)

        # Prepare list for display (cap for size)
        events_list = [
            {
                "when_utc": e.when_utc.isoformat(),
                "currency": e.currency,
                "event": e.event,
                "importance": e.importance,
                "source_url": e.source_url,
                "proximity_minutes": e.proximity_minutes,
            }
            for e in all_events
        ]

        # Narrative
        narrative_text = await self.narrative.generate_narrative(news_snapshot.__dict__)

        # Build response
        return {
            "pair": news_snapshot.pair,
            "ts_utc": news_snapshot.ts_utc.isoformat(),
            "news": {
                "sent_base": news_snapshot.sent_base,
                "sent_quote": news_snapshot.sent_quote,
                "pair_bias": news_snapshot.pair_bias,
                "confidence": news_snapshot.confidence,
                "n_articles_used": news_snapshot.n_articles_used,
                "top_evidence": news_snapshot.top_evidence,
                "narrative": narrative_text,
            },
            "calendar": {
                "next_high_event": (
                    {
                        "when_utc": next_event.when_utc.isoformat(),
                        "currency": next_event.currency,
                        "event": next_event.event,
                        "source_url": next_event.source_url,
                        "proximity_minutes": next_event.proximity_minutes,
                        "is_imminent": next_event.is_imminent,
                    }
                    if next_event
                    else None
                ),
                "total_high_impact_events_7d": sum(
                    1
                    for e in all_events
                    if (e.importance or "").lower() == "high" and 0 <= e.proximity_minutes <= 7 * 24 * 60
                ),
                # For UI: show a small list of events used (cap at 10 for readability)
                "events_extracted": events_list[:10],
            },
            "policy_bias": policy_bias,
        }
