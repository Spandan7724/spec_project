from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import asdict
from typing import Any, Dict, Optional, Tuple

from src.llm.manager import LLMManager
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
        """Use tool/function-calling to get structured parameters."""
        messages = [
            {"role": "system", "content": get_system_prompt()},
            {"role": "user", "content": get_user_prompt(text)},
        ]
        tools = get_tools_schema()
        response = await self.llm.chat(messages, tools=tools)

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

        # Risk/urgency/timeframe via simple keyword checks
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
        timeframe = (
            "immediate"
            if any(k in low for k in ["immediate", "today", "now", "asap"]) else
            "1_day" if any(k in low for k in ["tomorrow", "1 day", "24 hours"]) else
            "1_week" if any(k in low for k in ["week", "7 days", "this week"]) else
            "1_month" if any(k in low for k in ["month", "30 days", "next month"]) else None
        )

        # Flexible natural timeframe like "in 10 days" or "2 weeks"
        tf_days = timeframe_to_days(timeframe) if timeframe else None
        if tf_days is None:
            m = re.search(r"\b(\d+)\s*(day|days|d)\b", low)
            if m:
                try:
                    tf_days = max(0, int(m.group(1)))
                except Exception:
                    tf_days = None
        if tf_days is None:
            m = re.search(r"\b(\d+)\s*(week|weeks|w)\b", low)
            if m:
                try:
                    tf_days = max(0, int(m.group(1)) * 7)
                except Exception:
                    tf_days = None

        return {
            "currency_pair": cp,
            "base_currency": base,
            "quote_currency": quote,
            "amount": amount,
            "risk_tolerance": risk,
            "urgency": urgency,
            "timeframe": timeframe,
            "timeframe_days": tf_days,
        }

    def _extract_currency_codes(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        # Pattern: USD/EUR or USD-EUR or USDEUR
        m = re.search(r"\b([A-Z]{3})\s*[/\-]?\s*([A-Z]{3})\b", text)
        if m:
            b, q = m.groups()
            b_n, q_n = normalize_currency_code(b), normalize_currency_code(q)
            if b_n and q_n:
                return b_n, q_n

        # Pattern: USD to EUR
        m = re.search(r"\b([A-Z]{3})\s+(?:to|into|→)\s+([A-Z]{3})\b", text, re.IGNORECASE)
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
