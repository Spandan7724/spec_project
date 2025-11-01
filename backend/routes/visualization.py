from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Depends

from ..dependencies import get_analysis_repository
from src.data_collection.market_data.snapshot import _load_historical
from src.data_collection.market_data.indicators import (
    calculate_indicators,
    _rsi,
    _macd,
    _atr,
)


router = APIRouter()


def _get_completed_record(correlation_id: str, analysis_repo):
    """Helper to get completed analysis record or raise 404."""
    record = analysis_repo.get_by_correlation_id(correlation_id)
    if not record:
        raise HTTPException(status_code=404, detail="Analysis not found")
    if record.status != "completed":
        raise HTTPException(status_code=400, detail="Analysis not completed yet")
    return record


@router.get("/confidence/{correlation_id}")
def confidence_breakdown(correlation_id: str, analysis_repo=Depends(get_analysis_repository)) -> Dict[str, Any]:
    """Get confidence breakdown (overall + components)."""
    record = _get_completed_record(correlation_id, analysis_repo)
    rec = record.recommendation or {}
    overall = rec.get("confidence")
    comps = rec.get("component_confidences") or {}
    return {
        "overall": overall,
        "components": comps,
        "correlation_id": correlation_id,
    }


@router.get("/risk-breakdown/{correlation_id}")
def risk_breakdown(correlation_id: str, analysis_repo=Depends(get_analysis_repository)) -> Dict[str, Any]:
    """Get detailed risk breakdown for visualization."""
    record = _get_completed_record(correlation_id, analysis_repo)
    rec = record.recommendation or {}
    risk_summary = rec.get("risk_summary") or {}

    return {
        "correlation_id": correlation_id,
        "risk_level": risk_summary.get("risk_level"),
        "event_risk": risk_summary.get("event_risk"),
        "volatility_risk": risk_summary.get("volatility_risk"),
        "liquidity_risk": risk_summary.get("liquidity_risk"),
        "market_regime": risk_summary.get("market_regime"),
        "details": risk_summary,
    }


@router.get("/cost-breakdown/{correlation_id}")
def cost_breakdown(correlation_id: str, analysis_repo=Depends(get_analysis_repository)) -> Dict[str, Any]:
    """Get cost breakdown for visualization (fees, spreads, etc)."""
    record = _get_completed_record(correlation_id, analysis_repo)
    rec = record.recommendation or {}
    cost_estimate = rec.get("cost_estimate") or {}

    return {
        "correlation_id": correlation_id,
        "total_cost_bps": cost_estimate.get("total_cost_bps"),
        "total_cost_absolute": cost_estimate.get("total_cost_absolute"),
        "spread_cost_bps": cost_estimate.get("spread_cost_bps"),
        "fee_bps": cost_estimate.get("fee_bps"),
        "slippage_bps": cost_estimate.get("slippage_bps"),
        "cost_percentage": cost_estimate.get("cost_percentage"),
        "breakdown": cost_estimate,
    }


@router.get("/timeline-data/{correlation_id}")
def timeline_data(correlation_id: str, analysis_repo=Depends(get_analysis_repository)) -> Dict[str, Any]:
    """Get timeline data for visualization."""
    record = _get_completed_record(correlation_id, analysis_repo)
    rec = record.recommendation or {}

    action = rec.get("action")
    timeline = rec.get("timeline")
    staged_plan = rec.get("staged_plan")
    expected_outcome = rec.get("expected_outcome")

    # Build timeline points
    timeline_points: List[Dict[str, Any]] = []

    if action == "staged_conversion" and staged_plan:
        tranches = staged_plan.get("tranches", [])
        for i, tranche in enumerate(tranches):
            timeline_points.append({
                "index": i + 1,
                "day": tranche.get("execute_on_day"),
                "amount": tranche.get("amount"),
                "percentage": tranche.get("percentage"),
                "note": tranche.get("note", ""),
            })
    else:
        # Single point for immediate or wait
        timeline_points.append({
            "index": 1,
            "day": 0 if action == "convert_now" else record.timeframe_days or 1,
            "amount": record.amount,
            "percentage": 100.0,
            "note": timeline or "",
        })

    return {
        "correlation_id": correlation_id,
        "action": action,
        "timeline": timeline,
        "timeline_points": timeline_points,
        "expected_outcome": expected_outcome,
    }


@router.get("/prediction-chart/{correlation_id}")
def prediction_chart(correlation_id: str, analysis_repo=Depends(get_analysis_repository)) -> Dict[str, Any]:
    """Get prediction data formatted for charting (quantiles, mean, etc)."""
    record = _get_completed_record(correlation_id, analysis_repo)
    prediction = record.prediction or {}

    predictions = prediction.get("predictions", {})
    latest_close = prediction.get("latest_close", 0.0)

    # Format for chart
    chart_data = []
    for horizon_key, pred_data in predictions.items():
        if isinstance(pred_data, dict):
            horizon = int(horizon_key) if str(horizon_key).isdigit() else horizon_key
            mean_change = pred_data.get("mean_change_pct", 0.0)
            quantiles = pred_data.get("quantiles", {})

            chart_data.append({
                "horizon": horizon,
                "mean_rate": latest_close * (1 + mean_change / 100) if latest_close > 0 else 0,
                "mean_change_pct": mean_change,
                "p10": quantiles.get("p10"),
                "p25": quantiles.get("p25"),
                "p50": quantiles.get("p50"),
                "p75": quantiles.get("p75"),
                "p90": quantiles.get("p90"),
                "direction_probability": pred_data.get("direction_probability"),
            })

    # Sort by horizon
    chart_data.sort(key=lambda x: x["horizon"] if isinstance(x["horizon"], (int, float)) else 0)

    return {
        "correlation_id": correlation_id,
        "currency_pair": record.currency_pair,
        "latest_close": latest_close,
        "chart_data": chart_data,
        "confidence": prediction.get("confidence"),
    }


@router.get("/evidence/{correlation_id}")
def evidence(correlation_id: str, analysis_repo=Depends(get_analysis_repository)) -> Dict[str, Any]:
    """Get supporting evidence (news, events, market data) formatted for UI."""
    record = _get_completed_record(correlation_id, analysis_repo)

    intelligence = record.intelligence or {}
    market_data = record.market_data or {}

    # Extract news
    news_data = intelligence.get("news", {})
    news_articles = news_data.get("top_evidence", [])
    news_narrative = news_data.get("narrative", "")

    # Extract calendar events
    calendar_data = intelligence.get("calendar", {})
    events = calendar_data.get("events_extracted", [])
    next_high_event = calendar_data.get("next_high_event")
    # If no extracted events but we have a next high impact event, surface it
    if (not events) and next_high_event:
        events = [next_high_event]

    # Market data summary
    market_summary = {
        "current_rate": market_data.get("current_rate"),
        "bid": market_data.get("bid"),
        "ask": market_data.get("ask"),
        "spread_bps": market_data.get("spread_bps"),
        "regime": market_data.get("regime"),
        "volatility": market_data.get("volatility"),
    }

    return {
        "correlation_id": correlation_id,
        "news": {
            "articles": news_articles,
            "narrative": news_narrative,
            "sentiment_base": news_data.get("sent_base"),
            "sentiment_quote": news_data.get("sent_quote"),
            "pair_bias": news_data.get("pair_bias"),
        },
        "calendar": {
            "upcoming_events": events,
            "next_high_impact": next_high_event,
            "total_high_impact_7d": calendar_data.get("total_high_impact_events_7d"),
        },
        "market": market_summary,
        "policy_bias": intelligence.get("policy_bias"),
    }


@router.get("/historical-prices/{correlation_id}")
async def historical_prices(
    correlation_id: str, 
    days: int = 90,
    analysis_repo=Depends(get_analysis_repository)
) -> Dict[str, Any]:
    """Get historical OHLC price data with technical indicators.
    
    Args:
        correlation_id: Analysis correlation ID
        days: Number of days of historical data (default: 90, min: 30, max: 365)
    """
    record = _get_completed_record(correlation_id, analysis_repo)

    try:
        # Fetch historical data using yfinance
        base = record.base_currency
        quote = record.quote_currency
        
        # Validate and clamp days parameter
        days = max(30, min(365, days))

        df = await _load_historical(base, quote, days)

        if df is None or df.empty:
            return {
                "correlation_id": correlation_id,
                "currency_pair": record.currency_pair,
                "data": [],
                "error": "No historical data available"
            }

        # Calculate indicators
        indicators = calculate_indicators(df)

        # Format for frontend - calculate SMA(5) manually since it's not in Indicators
        sma_5 = df['Close'].rolling(window=5).mean().iloc[-1] if len(df) >= 5 else None

        # Format for frontend
        chart_data = []
        for idx, row in df.iterrows():
            point = {
                "date": idx.isoformat(),
                "open": float(row.get("Open", 0)),
                "high": float(row.get("High", 0)),
                "low": float(row.get("Low", 0)),
                "close": float(row.get("Close", 0)),
                "volume": float(row.get("Volume", 0)),
            }
            chart_data.append(point)

        # Add indicators as separate series
        indicator_data = {
            "sma_5": float(sma_5) if sma_5 is not None and not pd.isna(sma_5) else None,
            "sma_20": indicators.sma_20,
            "sma_50": indicators.sma_50,
            "ema_12": indicators.ema_12,
            "ema_26": indicators.ema_26,
            "bb_upper": indicators.bb_upper,
            "bb_middle": indicators.bb_middle,
            "bb_lower": indicators.bb_lower,
        }

        return {
            "correlation_id": correlation_id,
            "currency_pair": record.currency_pair,
            "data": chart_data,
            "indicators": indicator_data,
        }
    except Exception as e:
        return {
            "correlation_id": correlation_id,
            "currency_pair": record.currency_pair,
            "data": [],
            "error": str(e)
        }


@router.get("/technical-indicators/{correlation_id}")
async def technical_indicators(correlation_id: str, analysis_repo=Depends(get_analysis_repository)) -> Dict[str, Any]:
    """Get technical indicators (RSI, MACD, ATR, volatility)."""
    record = _get_completed_record(correlation_id, analysis_repo)

    try:
        from src.data_collection.market_data.snapshot import _load_historical
        from src.data_collection.market_data.indicators import calculate_indicators

        base = record.base_currency
        quote = record.quote_currency
        days = 90

        df = await _load_historical(base, quote, days)

        if df is None or df.empty:
            return {
                "correlation_id": correlation_id,
                "data": {},
                "error": "No historical data available"
            }

        df = df.sort_index()
        indicators = calculate_indicators(df)

        close = df['Close']
        volume = df['Volume'] if 'Volume' in df.columns else None

        # Prepare indicator series (limit to last 250 points for payload size)
        limit = 250

        rsi_series = _rsi(close, 14).dropna().iloc[-limit:]
        macd_series, macd_signal_series, macd_hist_series = _macd(close)
        macd_series = macd_series.dropna().iloc[-limit:]
        macd_signal_series = macd_signal_series.dropna().iloc[-limit:]
        macd_hist_series = macd_hist_series.dropna().iloc[-limit:]

        atr_series = _atr(df, 14).dropna().iloc[-limit:]

        vol_20 = close.pct_change().rolling(window=20).std().dropna().iloc[-limit:]
        vol_30 = close.pct_change().rolling(window=30).std().dropna().iloc[-limit:]

        chart_data = []
        for idx in df.index[-limit:]:
            item = {
                "date": idx.isoformat(),
                "close": float(df.loc[idx, "Close"]),
            }
            if volume is not None:
                item["volume"] = float(volume.loc[idx]) if not pd.isna(volume.loc[idx]) else None
            chart_data.append(item)

        rsi_points = [
            {"date": idx.isoformat(), "value": float(val)}
            for idx, val in rsi_series.items()
            if not pd.isna(val)
        ]

        macd_points = []
        for idx in macd_series.index:
            macd_val = macd_series.loc[idx]
            signal_val = macd_signal_series.loc[idx] if idx in macd_signal_series.index else np.nan
            hist_val = macd_hist_series.loc[idx] if idx in macd_hist_series.index else np.nan
            if pd.isna(macd_val) and pd.isna(signal_val) and pd.isna(hist_val):
                continue
            macd_points.append({
                "date": idx.isoformat(),
                "macd": float(macd_val) if not pd.isna(macd_val) else None,
                "macd_signal": float(signal_val) if not pd.isna(signal_val) else None,
                "macd_histogram": float(hist_val) if not pd.isna(hist_val) else None,
            })

        atr_points = [
            {"date": idx.isoformat(), "atr": float(val)}
            for idx, val in atr_series.items()
            if not pd.isna(val)
        ]

        vol_points = []
        for idx in vol_20.index.union(vol_30.index):
            val20 = vol_20.loc[idx] if idx in vol_20.index else np.nan
            val30 = vol_30.loc[idx] if idx in vol_30.index else np.nan
            if pd.isna(val20) and pd.isna(val30):
                continue
            vol_points.append({
                "date": idx.isoformat(),
                "volatility_20d": float(val20) if not pd.isna(val20) else None,
                "volatility_30d": float(val30) if not pd.isna(val30) else None,
            })

        return {
            "correlation_id": correlation_id,
            "currency_pair": record.currency_pair,
            "data": chart_data,
            "indicators": {
                "latest": {
                    "rsi": indicators.rsi_14,
                    "macd": indicators.macd,
                    "macd_signal": indicators.macd_signal,
                    "macd_histogram": indicators.macd_histogram,
                    "atr": indicators.atr_14,
                    "volatility_30d": indicators.realized_vol_30d,
                },
                "series": {
                    "rsi": rsi_points,
                    "macd": macd_points,
                    "atr": atr_points,
                    "volatility": vol_points,
                },
            },
        }
    except Exception as e:
        return {
            "correlation_id": correlation_id,
            "data": {},
            "error": str(e)
        }


@router.get("/sentiment-timeline/{correlation_id}")
def sentiment_timeline(correlation_id: str, analysis_repo=Depends(get_analysis_repository)) -> Dict[str, Any]:
    """Get news sentiment analysis over time."""
    record = _get_completed_record(correlation_id, analysis_repo)

    intelligence = record.intelligence or {}
    news_data = intelligence.get("news", {})

    # Current sentiment scores
    def _to_float(value: Any, default: float = 0.0) -> float:
        try:
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return float(value)
            if isinstance(value, str) and value.strip():
                return float(value.strip())
        except (ValueError, TypeError):
            pass
        return default

    sent_base = _to_float(news_data.get("sent_base"), 0.0)
    sent_quote = _to_float(news_data.get("sent_quote"), 0.0)
    pair_bias_raw = news_data.get("pair_bias")
    pair_bias = _to_float(pair_bias_raw, None) if pair_bias_raw is not None else None

    # Get articles with sentiment
    articles = news_data.get("top_evidence", [])

    # Build timeline from articles
    timeline_data = []
    for article in articles:
        if isinstance(article, dict):
            pub_date = article.get("published_utc") or article.get("date")
            sentiment = article.get("sentiment", {})
            base_value = _to_float(sentiment.get(record.base_currency), 0.0) if isinstance(sentiment, dict) else 0.0
            quote_value = _to_float(sentiment.get(record.quote_currency), 0.0) if isinstance(sentiment, dict) else 0.0

            if pub_date:
                timeline_data.append({
                    "date": pub_date,
                    "sentiment_base": base_value,
                    "sentiment_quote": quote_value,
                    "title": article.get("title", ""),
                    "source": article.get("source", ""),
                })

    # Sort by date
    timeline_data.sort(key=lambda x: x["date"] if x["date"] else "")

    return {
        "correlation_id": correlation_id,
        "currency_pair": record.currency_pair,
        "current_sentiment": {
            "base_currency": record.base_currency,
            "quote_currency": record.quote_currency,
            "sentiment_base": sent_base,
            "sentiment_quote": sent_quote,
            "pair_bias": pair_bias if pair_bias is not None else pair_bias_raw,
            "narrative": news_data.get("narrative", ""),
        },
        "timeline": timeline_data,
        "n_articles": len(articles),
    }


@router.get("/events-timeline/{correlation_id}")
def events_timeline(correlation_id: str, analysis_repo=Depends(get_analysis_repository)) -> Dict[str, Any]:
    """Get economic events timeline with impact ratings."""
    record = _get_completed_record(correlation_id, analysis_repo)

    intelligence = record.intelligence or {}
    calendar_data = intelligence.get("calendar", {})

    events = calendar_data.get("events_extracted", [])
    next_high_event = calendar_data.get("next_high_event")

    # Normalize events for UI
    formatted_events = []
    for event in events:
        if isinstance(event, dict):
            formatted_events.append({
                "date": event.get("when_utc") or event.get("date"),
                "title": event.get("event") or event.get("title", "Economic Event"),
                "country": event.get("country", ""),
                "currency": event.get("currency", ""),
                "importance": event.get("importance", "medium"),
                "impact": event.get("importance", "medium"),  # Alias for importance
                "description": event.get("note") or event.get("description", ""),
                "source": event.get("source", ""),
            })

    # Add next high impact event if not in list
    if next_high_event and isinstance(next_high_event, dict):
        event_exists = any(
            e.get("title") == next_high_event.get("event")
            for e in formatted_events
        )
        if not event_exists:
            formatted_events.append({
                "date": next_high_event.get("when_utc"),
                "title": next_high_event.get("event", "High Impact Event"),
                "country": next_high_event.get("country", ""),
                "currency": next_high_event.get("currency", ""),
                "importance": "high",
                "impact": "high",
                "description": next_high_event.get("note", ""),
                "source": next_high_event.get("source", ""),
            })

    # Sort by date
    formatted_events.sort(key=lambda x: x["date"] if x["date"] else "")

    return {
        "correlation_id": correlation_id,
        "currency_pair": record.currency_pair,
        "events": formatted_events,
        "total_high_impact_7d": calendar_data.get("total_high_impact_events_7d", 0),
        "next_high_impact": next_high_event,
    }


@router.get("/shap-explanations/{correlation_id}")
def shap_explanations(correlation_id: str, analysis_repo=Depends(get_analysis_repository)) -> Dict[str, Any]:
    """Get SHAP explanations and feature importance for predictions."""
    record = _get_completed_record(correlation_id, analysis_repo)

    prediction = record.prediction or {}
    explanations = prediction.get("explanations") or {}

    if not explanations:
        return {
            "correlation_id": correlation_id,
            "currency_pair": record.currency_pair,
            "feature_importance": [],
            "waterfall_plot": None,
            "has_waterfall": False,
            "error": "SHAP explanations not available for this analysis",
        }

    selected = prediction.get("selected_prediction") or {}
    target_horizon = str(selected.get("horizon_key")) if selected.get("horizon_key") is not None else None

    daily_expl = explanations.get("daily") or {}
    intraday_expl = explanations.get("intraday") or {}

    target = None
    if target_horizon and target_horizon in daily_expl:
        target = daily_expl[target_horizon]
    elif daily_expl:
        target = next(iter(daily_expl.values()))
    elif intraday_expl:
        target = next(iter(intraday_expl.values()))

    if not target:
        return {
            "correlation_id": correlation_id,
            "currency_pair": record.currency_pair,
            "feature_importance": [],
            "waterfall_plot": None,
            "has_waterfall": False,
            "error": "No SHAP explanation payload available",
        }

    top_features = target.get("top_features") or {}
    importance_data = [
        {"feature": feature, "importance": float(score)}
        for feature, score in top_features.items()
    ]
    importance_data.sort(key=lambda item: abs(item["importance"]), reverse=True)

    waterfall_plot = target.get("shap_waterfall_base64")
    waterfall_data = target.get("shap_waterfall_data")

    return {
        "correlation_id": correlation_id,
        "currency_pair": record.currency_pair,
        "feature_importance": importance_data,
        "waterfall_plot": waterfall_plot,
        "has_waterfall": waterfall_plot is not None or bool(waterfall_data),
        "waterfall_data": waterfall_data,
    }


@router.get("/prediction-quantiles/{correlation_id}")
def prediction_quantiles(correlation_id: str, analysis_repo=Depends(get_analysis_repository)) -> Dict[str, Any]:
    """Get enhanced prediction data with full quantile information for fan charts."""
    record = _get_completed_record(correlation_id, analysis_repo)
    prediction = record.prediction or {}

    predictions = prediction.get("predictions", {})
    latest_close = prediction.get("latest_close", 0.0)

    # Format for fan chart with full quantile data
    chart_data = []
    for horizon_key, pred_data in predictions.items():
        if isinstance(pred_data, dict):
            horizon = int(horizon_key) if str(horizon_key).isdigit() else horizon_key
            mean_change = pred_data.get("mean_change_pct", 0.0) or 0.0
            quantiles = pred_data.get("quantiles") or {}

            # Calculate rates from percentage changes
            base_rate = latest_close if latest_close > 0 else 1.0

            def _quantile_or_mean(key: str) -> float:
                value = quantiles.get(key)
                return mean_change if value is None else value

            chart_data.append({
                "horizon": horizon,
                "horizon_label": f"{horizon}d" if isinstance(horizon, int) else str(horizon),
                "mean_rate": base_rate * (1 + mean_change / 100),
                "mean_change_pct": mean_change,
                "p10_rate": base_rate * (1 + _quantile_or_mean("p10") / 100),
                "p25_rate": base_rate * (1 + _quantile_or_mean("p25") / 100),
                "p50_rate": base_rate * (1 + _quantile_or_mean("p50") / 100),
                "p75_rate": base_rate * (1 + _quantile_or_mean("p75") / 100),
                "p90_rate": base_rate * (1 + _quantile_or_mean("p90") / 100),
                "p10": quantiles.get("p10"),
                "p25": quantiles.get("p25"),
                "p50": quantiles.get("p50"),
                "p75": quantiles.get("p75"),
                "p90": quantiles.get("p90"),
                "direction_probability": pred_data.get("direction_prob"),
                "confidence": pred_data.get("confidence"),
            })

    # Sort by horizon
    chart_data.sort(key=lambda x: x["horizon"] if isinstance(x["horizon"], (int, float)) else 0)

    return {
        "correlation_id": correlation_id,
        "currency_pair": record.currency_pair,
        "latest_close": latest_close,
        "predictions": chart_data,
        "overall_confidence": prediction.get("confidence"),
    }


@router.get("/market-regime/{correlation_id}")
async def market_regime(correlation_id: str, analysis_repo=Depends(get_analysis_repository)) -> Dict[str, Any]:
    """Get market regime classification over time."""
    record = _get_completed_record(correlation_id, analysis_repo)

    try:
        from src.data_collection.market_data.snapshot import _load_historical
        from src.data_collection.market_data.regime import classify_regime
        from src.data_collection.market_data.indicators import calculate_indicators

        base = record.base_currency
        quote = record.quote_currency
        days = 90

        df = await _load_historical(base, quote, days)

        if df is None or df.empty:
            # Return current regime only
            market_data = record.market_data or {}
            return {
                "correlation_id": correlation_id,
                "currency_pair": record.currency_pair,
                "current_regime": market_data.get("regime", "unknown"),
                "regime_history": [],
            }

        # Calculate regime for each point (using rolling window)
        regime_data = []
        for idx in df.index[-30:]:  # Last 30 days
            window_df = df.loc[:idx].tail(30)  # 30-day lookback
            if len(window_df) >= 20:
                indicators = calculate_indicators(window_df)
                latest_price = float(df.loc[idx, "Close"])
                regime_obj = classify_regime(latest_price, indicators)

                # Convert regime object to simple string
                if regime_obj.trend_direction == "up":
                    regime_str = "trending_up"
                elif regime_obj.trend_direction == "down":
                    regime_str = "trending_down"
                elif regime_obj.trend_direction == "sideways":
                    regime_str = "ranging"
                else:
                    regime_str = "unknown"

                regime_data.append({
                    "date": idx.isoformat(),
                    "regime": regime_str,
                    "close": latest_price,
                })

        # Current regime from market data
        market_data = record.market_data or {}
        current_regime = market_data.get("regime", "unknown")

        return {
            "correlation_id": correlation_id,
            "currency_pair": record.currency_pair,
            "current_regime": current_regime,
            "regime_history": regime_data,
        }
    except Exception as e:
        # Fallback: just return current regime
        market_data = record.market_data or {}
        return {
            "correlation_id": correlation_id,
            "currency_pair": record.currency_pair,
            "current_regime": market_data.get("regime", "unknown"),
            "regime_history": [],
            "error": str(e)
        }
