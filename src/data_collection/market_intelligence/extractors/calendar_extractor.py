from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import List

from src.data_collection.market_intelligence.models import EconomicEvent
from src.data_collection.market_intelligence.serper_client import SerperSearchResult
from src.config import get_config, load_config
from src.llm.agent_helpers import chat_with_model, get_recommended_model_for_task
from src.utils.decorators import retry, log_execution, timeout
from src.utils.logging import get_logger


logger = get_logger(__name__)


class CalendarExtractor:
    """Extract economic events from search results using an LLM."""

    # Expanded default mapping for common currencies (ISO 3166-1 alpha-2 or region code)
    DEFAULT_CURRENCY_TO_COUNTRY = {
        # Majors
        "USD": "US",
        "EUR": "EA",  # Euro Area
        "GBP": "GB",
        "JPY": "JP",
        "CHF": "CH",
        "CAD": "CA",
        "AUD": "AU",
        "NZD": "NZ",
        # Nordics + Europe
        "SEK": "SE",
        "NOK": "NO",
        "DKK": "DK",
        "PLN": "PL",
        "HUF": "HU",
        "CZK": "CZ",
        "RON": "RO",
        "TRY": "TR",
        "ILS": "IL",
        # EM + Asia
        "ZAR": "ZA",
        "CNY": "CN",
        "HKD": "HK",
        "SGD": "SG",
        "KRW": "KR",
        "INR": "IN",
        "THB": "TH",
        "IDR": "ID",
        "TWD": "TW",
        # Americas
        "MXN": "MX",
        "BRL": "BR",
        # Others often encountered
        "RUB": "RU",
    }

    def __init__(self, llm_manager):
        self.llm_manager = llm_manager
        # Load mapping from config if provided, else use defaults
        try:
            cfg = get_config()
        except Exception:
            cfg = load_config()
        mapping = cfg.get("market_intelligence.currency_regions")
        # Accept mapping of str->str or str->List[str]; store normalized as str
        norm_map = {}
        if isinstance(mapping, dict):
            for k, v in mapping.items():
                if isinstance(v, list) and v:
                    norm_map[str(k).upper()] = str(v[0])
                elif isinstance(v, str):
                    norm_map[str(k).upper()] = v
        self.currency_to_country = {**self.DEFAULT_CURRENCY_TO_COUNTRY, **norm_map}

    @retry(max_attempts=2, delay=1.0)
    @timeout(15.0)
    @log_execution(log_args=False, log_result=False)
    async def extract_events_from_snippet(self, result: SerperSearchResult, currency: str) -> List[EconomicEvent]:
        prompt = f"""Extract economic calendar events from this search result.

Title: {result.title}
Snippet: {result.snippet}
Currency: {currency}

Return a JSON array of events with fields:
- date: ISO format (YYYY-MM-DD)
- time: ISO time (HH:MM) or null if not specified
- event: Event name/description
- importance: "high", "medium", or "low"
"""

        messages = [
            {"role": "system", "content": "Extract economic events. Return ONLY a JSON array."},
            {"role": "user", "content": prompt},
        ]
        model = get_recommended_model_for_task("data_extraction")
        resp = await chat_with_model(messages, model, self.llm_manager)
        content = resp.content.strip()
        # Strip code fences if present
        if content.startswith("```"):
            content = content.strip("`")
            if content.lower().startswith("json"):
                content = content[4:].strip()

        try:
            arr = json.loads(content)
        except Exception:
            logger.warning("Calendar extraction JSON parse failed; returning empty list")
            return []

        events: List[EconomicEvent] = []
        for e in arr:
            date_str = e.get("date")
            time_str = e.get("time")
            if not date_str:
                continue
            if time_str:
                dt = datetime.fromisoformat(f"{date_str}T{time_str}:00").replace(tzinfo=timezone.utc)
            else:
                dt = datetime.fromisoformat(f"{date_str}T00:00:00").replace(tzinfo=timezone.utc)

            events.append(
                EconomicEvent(
                    when_utc=dt,
                    when_local=dt,  # TODO: add TZ conversion later
                    timezone="UTC",
                    country=self.currency_to_country.get(currency, "UNKNOWN"),
                    currency=currency,
                    event=e.get("event", ""),
                    importance=e.get("importance", "medium"),
                    source=result.title,
                    source_url=result.url,
                )
            )
        return events

    async def extract_events_batch(self, results: List[SerperSearchResult], currency: str) -> List[EconomicEvent]:
        # Process snippets concurrently with a small semaphore to keep latency bounded
        import asyncio
        sem = asyncio.Semaphore(8)  # Increased from 3

        async def _one(r: SerperSearchResult) -> List[EconomicEvent]:
            async with sem:
                try:
                    return await self.extract_events_from_snippet(r, currency)
                except Exception:
                    return []

        tasks = [asyncio.create_task(_one(r)) for r in results]
        batches = await asyncio.gather(*tasks, return_exceptions=False)
        out: List[EconomicEvent] = []
        for b in batches:
            out.extend(b)
        # Deduplicate by date + event
        seen = set()
        deduped: List[EconomicEvent] = []
        for ev in out:
            key = (ev.when_utc.date(), ev.event.lower())
            if key not in seen:
                seen.add(key)
                deduped.append(ev)
        out_sorted = sorted(deduped, key=lambda ev: ev.when_utc)
        logger.info(
            "MI.CalendarExtractor: currency=%s results=%d extracted=%d",
            currency,
            len(results),
            len(out_sorted),
        )
        return out_sorted
