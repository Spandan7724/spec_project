from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import asdict
from typing import Any, Dict, Optional, Tuple
from datetime import datetime, timezone
from dateutil import parser as dateparser
import math

from src.llm.manager import LLMManager
from src.llm.agent_helpers import chat_with_model_for_task
from src.supervisor.models import ExtractedParameters
from src.supervisor.prompts import get_system_prompt, get_user_prompt, get_tools_schema
from src.supervisor.validation import (
    CURRENCY_CODES,
    CURRENCY_NAMES,
    RISK_LEVELS,
    TIMEFRAME_LEVELS,
    URGENCY_LEVELS,
    normalize_currency_code,
    timeframe_to_days,
    validate_currency_pair,
)


logger = logging.getLogger(__name__)


class NLUExtractor:
    """LLM-first parameter extraction with robust fallback and validation."""

    def __init__(self, llm_manager: Optional[LLMManager] = None, use_llm: Optional[bool] = None):
        self.llm = llm_manager or LLMManager()
        # Allow forcing no-network mode in tests or demos
        if use_llm is None:
            # OFFLINE_DEMO=true will disable LLM usage automatically
            use_llm = str(os.getenv("OFFLINE_DEMO", "false")).strip().lower() in {"0", "false", "no"}
        self.use_llm = use_llm

    async def aextract(self, text: str) -> ExtractedParameters:
        """Async extraction to align with async LLM interface."""
        if self.use_llm:
            try:
                primary = await self._extract_with_llm(text)
                fallback = self._extract_with_regex(text)
                merged = self._merge_results(primary, fallback)
                return self._postprocess(merged)
            except Exception as e:
                logger.error(f"LLM extraction failed, falling back to regex: {e}")
        # Fallback if disabled or failed
        return self._postprocess(self._extract_with_regex(text))

    def extract(self, text: str) -> ExtractedParameters:
        """Synchronous helper that runs async path (for simple usage)."""
        import asyncio

        try:
            return asyncio.get_event_loop().run_until_complete(self.aextract(text))
        except RuntimeError:
            # In case called from an existing event loop (e.g., Jupyter), create a new loop
            return asyncio.run(self.aextract(text))

    async def _extract_with_llm(self, text: str) -> Dict[str, Any]:
        """Use tool/function-calling to get structured parameters.

        Uses {provider}_main for complex NLU with function calling.
        """
        messages = [
            {"role": "system", "content": get_system_prompt()},
            {"role": "user", "content": get_user_prompt(text)},
        ]
        tools = get_tools_schema()
        # Use provider's main model for complex NLU reasoning with function calling
        response = await chat_with_model_for_task(messages, "nlu", self.llm, tools=tools)

        # Prefer tool-calls if present
        if response.tool_calls:
            for call in response.tool_calls:
                if call.get("function", {}).get("name") == "set_parameters":
                    args_raw = call.get("function", {}).get("arguments", "{}")
                    try:
                        return json.loads(args_raw)
                    except Exception:
                        logger.warning("Tool arguments were not valid JSON; attempting to repair")
                        repaired = self._repair_json(args_raw)
                        return json.loads(repaired)

        # Fallback: try to parse content as JSON
        try:
            return json.loads(response.content)
        except Exception as e:
            logger.warning(f"Model returned non-JSON content; attempting repair: {e}")
            repaired = self._repair_json(response.content)
            return json.loads(repaired)

    def _postprocess(self, data: Dict[str, Any]) -> ExtractedParameters:
        """Normalize and validate extracted fields; derive missing values."""
        cp = data.get("currency_pair")
        base = data.get("base_currency")
        quote = data.get("quote_currency")
        amount = data.get("amount")
        risk = data.get("risk_tolerance")
        urgency = data.get("urgency")
        timeframe = data.get("timeframe")
        tf_days = data.get("timeframe_days")
        timeframe_mode = data.get("timeframe_mode")
        deadline_utc = data.get("deadline_utc")
        window_days = data.get("window_days")
        time_unit = data.get("time_unit")
        timeframe_hours = data.get("timeframe_hours")

        # Normalize currency codes
        base_n = normalize_currency_code(base)
        quote_n = normalize_currency_code(quote)

        # Rebuild currency_pair from normalized codes if possible
        if base_n and quote_n:
            cp = f"{base_n}/{quote_n}"

        # Coerce amount
        if isinstance(amount, str):
            amount = self._coerce_amount(amount)
        if amount is not None:
            try:
                amount = float(amount)
            except Exception:
                amount = None

        # Normalize category values
        if isinstance(risk, str):
            risk = risk.strip().lower()
            if risk not in RISK_LEVELS:
                risk = None
        if isinstance(urgency, str):
            urgency = urgency.strip().lower()
            if urgency not in URGENCY_LEVELS:
                urgency = None
        if isinstance(timeframe, str):
            timeframe = timeframe.strip().lower()
            if timeframe not in TIMEFRAME_LEVELS:
                timeframe = None

        # Derive timeframe_days if missing
        if tf_days is None and timeframe:
            tf_days = timeframe_to_days(timeframe)

        # Validate currency pair
        if base_n and quote_n:
            ok, _ = validate_currency_pair(base_n, quote_n)
            if not ok:
                base_n, quote_n, cp = None, None, None

        params = ExtractedParameters(
            currency_pair=cp,
            base_currency=base_n,
            quote_currency=quote_n,
            amount=amount,
            risk_tolerance=risk,
            urgency=urgency,
            timeframe=timeframe,
            timeframe_days=tf_days,
            timeframe_mode=timeframe_mode,
            deadline_utc=deadline_utc,
            window_days=window_days,
            time_unit=time_unit,
            timeframe_hours=timeframe_hours,
        )
        return params

    def _merge_results(self, primary: Dict[str, Any], fallback: Dict[str, Any]) -> Dict[str, Any]:
        """Merge two extraction dicts, preferring non-empty primary fields.

        This helps when LLM returns partial results (e.g., amount but missing quote_currency);
        we fill gaps from a lightweight regex pass.
        """
        merged: Dict[str, Any] = dict(primary or {})

        def pick(key: str):
            v = merged.get(key)
            if v is None or (isinstance(v, str) and not v.strip()):
                merged[key] = fallback.get(key)

        for k in [
            "currency_pair",
            "base_currency",
            "quote_currency",
            "amount",
            "risk_tolerance",
            "urgency",
            "timeframe",
            "timeframe_days",
        ]:
            pick(k)

        # If base + quote present but currency_pair missing, synthesize
        if (merged.get("currency_pair") in (None, "")) and merged.get("base_currency") and merged.get("quote_currency"):
            merged["currency_pair"] = f"{merged['base_currency']}/{merged['quote_currency']}"

        return merged

    def _extract_with_regex(self, text: str) -> Dict[str, Any]:
        """Lightweight fallback extraction without LLM dependency."""
        # Amount
        amount = self._coerce_amount(text)

        # Currency pair via codes like USD/EUR or USD to EUR
        base, quote = self._extract_currency_codes(text)
        cp = f"{base}/{quote}" if base and quote else None

        # Risk/urgency via simple keyword checks
        low = text.lower()
        risk = (
            "conservative"
            if any(k in low for k in ["conservative", "safe", "cautious", "low risk"]) else
            "aggressive"
            if any(k in low for k in ["aggressive", "risky", "high risk"]) else
            "moderate" if "moderate" in low or "balanced" in low else None
        )
        urgency = (
            "urgent"
            if any(k in low for k in ["urgent", "asap", "immediately", "today", "now"]) else
            "flexible" if any(k in low for k in ["whenever", "no rush", "take time", "flexible"]) else
            "normal" if any(k in low for k in ["soon", "few days"]) else None
        )

        # Determine timeframe first using robust patterns
        tf_days: Optional[int] = None
        timeframe_mode: Optional[str] = None
        deadline_utc: Optional[str] = None
        window_days: Optional[Dict[str, int]] = None
        time_unit: Optional[str] = None
        timeframe_hours: Optional[int] = None
        # Explicit numeric days/weeks
        # Ranges like "in 3-5 days" or "in 2 to 4 weeks"
        m = re.search(r"\b(\d+)\s*(?:-|to|–)\s*(\d+)\s*day(s)?\b", low)
        if m:
            s, e = int(m.group(1)), int(m.group(2))
            window_days = {"start": min(s, e), "end": max(s, e)}
            tf_days = int(round((window_days["start"] + window_days["end"]) / 2))
            timeframe_mode = "duration"
            time_unit = "days"
        if tf_days is None:
            m = re.search(r"\b(\d+)\s*(?:-|to|–)\s*(\d+)\s*week(s)?\b", low)
            if m:
                s, e = int(m.group(1)) * 7, int(m.group(2)) * 7
                window_days = {"start": min(s, e), "end": max(s, e)}
                tf_days = int(round((window_days["start"] + window_days["end"]) / 2))
                timeframe_mode = "duration"
                time_unit = "days"

        # Hours like "in 12 hours"
        if tf_days is None:
            m = re.search(r"\b(\d+)\s*hour(s)?\b", low)
            if m:
                timeframe_hours = int(m.group(1))
                tf_days = 0
                timeframe_mode = "duration"
                time_unit = "hours"

        # Explicit numeric days/weeks
        if tf_days is None:
            m = re.search(r"\b(\d+)\s*day(s)?\b", low)
            if m:
                try:
                    tf_days = max(0, int(m.group(1)))
                    timeframe_mode = "duration"
                    time_unit = "days"
                except Exception:
                    tf_days = None
        if tf_days is None:
            m = re.search(r"\b(\d+)\s*week(s)?\b", low)
            if m:
                try:
                    tf_days = max(0, int(m.group(1)) * 7)
                    timeframe_mode = "duration"
                    time_unit = "days"
                except Exception:
                    tf_days = None
        # Natural phrases
        if tf_days is None and re.search(r"\btomorrow\b|\bin\s*24\s*hours?\b", low):
            tf_days = 1
            timeframe_mode = "duration"
        if tf_days is None and re.search(r"\bthis\s*week\b|\bnext\s*week\b", low):
            tf_days = 7
            timeframe_mode = "duration"
        if tf_days is None and re.search(r"\bnext\s*month\b", low):
            tf_days = 30
            timeframe_mode = "duration"

        # Absolute deadline like "by 2025-11-15" or "by Nov 15"
        if deadline_utc is None:
            m = re.search(r"\b(?:by|before|on)\s+([\w\-/:, ]{3,})\b", text, re.IGNORECASE)
            if m:
                raw_date = m.group(1).strip()
                try:
                    # Parse using dateutil; assume local tz if naive
                    dt = dateparser.parse(raw_date, fuzzy=True)
                    if dt is not None:
                        if dt.tzinfo is None:
                            local_tz = datetime.now().astimezone().tzinfo
                            dt = dt.replace(tzinfo=local_tz)
                        dt_utc = dt.astimezone(timezone.utc)
                        deadline_utc = dt_utc.isoformat()
                        # Derive days remaining (ceil)
                        now_utc = datetime.now(timezone.utc)
                        delta_days = (dt_utc - now_utc).total_seconds() / 86400.0
                        tf_days = max(0, int(math.ceil(delta_days)))
                        timeframe_mode = "deadline"
                except Exception:
                    pass

        # Categorical timeframe only if we did not find other forms
        timeframe = None
        if tf_days is None:
            if re.search(r"\b(immediate|today|now|asap)\b", low):
                timeframe = "immediate"
                timeframe_mode = "immediate"
            elif re.search(r"\b1\s*day(s)?\b|\btomorrow\b|\bin\s*24\s*hours?\b", low):
                timeframe = "1_day"
                timeframe_mode = "duration"
            elif re.search(r"\b(1\s*week|7\s*days|this\s*week|next\s*week)\b", low):
                timeframe = "1_week"
                timeframe_mode = "duration"
            elif re.search(r"\b(1\s*month|30\s*days|next\s*month)\b", low):
                timeframe = "1_month"
                timeframe_mode = "duration"

        return {
            "currency_pair": cp,
            "base_currency": base,
            "quote_currency": quote,
            "amount": amount,
            "risk_tolerance": risk,
            "urgency": urgency,
            "timeframe": timeframe,
            "timeframe_days": tf_days,
            "timeframe_mode": timeframe_mode,
            "deadline_utc": deadline_utc,
            "window_days": window_days,
            "time_unit": time_unit,
            "timeframe_hours": timeframe_hours,
        }

    def _extract_currency_codes(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        # Pattern: USD/EUR or USD-EUR or USDEUR (case-insensitive)
        m = re.search(r"\b([A-Za-z]{3})\s*[/\-]?\s*([A-Za-z]{3})\b", text, re.IGNORECASE)
        if m:
            b, q = m.groups()
            b_n, q_n = normalize_currency_code(b.upper()), normalize_currency_code(q.upper())
            if b_n and q_n:
                return b_n, q_n

        # Pattern: USD to EUR (case-insensitive)
        m = re.search(r"\b([A-Za-z]{3})\s+(?:to|into|→)\s+([A-Za-z]{3})\b", text, re.IGNORECASE)
        if m:
            b, q = m.groups()
            b_n, q_n = normalize_currency_code(b.upper()), normalize_currency_code(q.upper())
            if b_n and q_n:
                return b_n, q_n

        # Pattern: names like dollars to euros
        low = text.lower()
        found = []
        for name, code in CURRENCY_NAMES.items():
            if name in low:
                found.append((low.index(name), code))
        if len(found) >= 2:
            found.sort(key=lambda x: x[0])
            # pick the first two occurrences with distinct codes
            picked = []
            seen_codes = set()
            for _, code in found:
                if code not in seen_codes:
                    picked.append(code)
                    seen_codes.add(code)
                if len(picked) == 2:
                    break
            if len(picked) == 2:
                return picked[0], picked[1]

        return None, None

    def _coerce_amount(self, text: str) -> Optional[float]:
        """Extract the first positive numeric amount from text."""
        # Numbers with optional commas and decimals
        for match in re.findall(r"\b([\d,]+(?:\.\d+)?)\b", text):
            try:
                amt = float(match.replace(",", ""))
                if amt > 0:
                    return amt
            except Exception:
                continue
        return None

    def _repair_json(self, raw: str) -> str:
        """Attempt to repair common JSON formatting issues."""
        # Trim code fences if present
        raw = raw.strip()
        if raw.startswith("```") and raw.endswith("```"):
            raw = raw.strip("`\n")
        # Ensure it looks like a JSON object
        if not raw.startswith("{"):
            # Heuristic: find first "{" and last "}"
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end != -1 and end > start:
                raw = raw[start : end + 1]
        return raw
