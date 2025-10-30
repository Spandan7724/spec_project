from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, Depends

from backend.dependencies import get_analysis_repository


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
