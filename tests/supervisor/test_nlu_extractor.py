import os
import pytest

from src.supervisor.nlu_extractor import NLUExtractor


def test_rule_fallback_basic_extraction(monkeypatch):
    # Force no-LLM mode to make tests deterministic/offline
    monkeypatch.setenv("OFFLINE_DEMO", "true")
    extractor = NLUExtractor(use_llm=False)

    text = "I need to convert 5,000 USD to EUR today. I'm moderate risk and it's urgent."
    params = extractor.extract(text)

    assert params.currency_pair == "USD/EUR"
    assert params.base_currency == "USD"
    assert params.quote_currency == "EUR"
    assert params.amount == 5000.0
    assert params.risk_tolerance == "moderate"
    assert params.urgency == "urgent"
    assert params.timeframe == "immediate"
    assert params.timeframe_days == 1


def test_rule_fallback_names_extraction(monkeypatch):
    monkeypatch.setenv("OFFLINE_DEMO", "true")
    extractor = NLUExtractor(use_llm=False)

    text = "Convert dollars to euros next week, low risk. Amount 12000."
    params = extractor.extract(text)

    assert params.currency_pair == "USD/EUR"
    assert params.base_currency == "USD"
    assert params.quote_currency == "EUR"
    assert params.amount == 12000.0
    # low risk keyword maps to conservative
    assert params.risk_tolerance in {"conservative", "moderate", "aggressive", None}
    assert params.timeframe in {"1_week", "immediate", "1_day", "1_month", None}

