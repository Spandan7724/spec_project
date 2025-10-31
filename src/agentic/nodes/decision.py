"""Decision Engine LangGraph node implementation (Phase 3.4)."""
from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict

from src.agentic.state import AgentState
from src.decision.config import DecisionConfig
from src.decision.decision_maker import DecisionMaker
from src.decision.models import DecisionRequest
from src.utils.logging import get_logger


logger = get_logger(__name__)


def _timeframe_to_days(tf: str) -> int:
    m = (tf or "1_day").lower()
    if m == "immediate":
        return 0
    if m == "1_day":
        return 1
    if m == "1_week":
        return 7
    if m == "1_month":
        return 30
    # default fallback
    return 7


def decision_node(state: AgentState) -> Dict[str, Any]:
    """Build DecisionRequest from state, invoke DecisionMaker, and update state."""
    correlation_id = state.get("correlation_id", "unknown")
    try:
        # Extract runtime params
        timeframe = state.get("timeframe") or "1_day"
        tf_days = int(state.get("timeframe_days") or _timeframe_to_days(timeframe))
        amount = float(state.get("amount") or 0.0)

        # Build DecisionRequest
        req = DecisionRequest(
            amount=amount,
            risk_tolerance=state.get("risk_tolerance", "moderate"),
            urgency=state.get("urgency", "normal"),
            timeframe=timeframe,
            timeframe_days=tf_days,
            currency_pair=state.get("currency_pair"),
            market=state.get("market_snapshot") or {},
            intelligence=state.get("intelligence_report") or None,
            prediction=state.get("price_forecast") or None,
            spread_bps=None,
            fee_bps=None,
            components_available={
                "market": bool(state.get("market_snapshot")),
                "intelligence": bool(state.get("intelligence_report")),
                "prediction": bool(state.get("price_forecast")),
            },
            warnings=list(state.get("warnings") or []),
        )

        maker = DecisionMaker(DecisionConfig.from_yaml())
        resp = maker.make_decision(req)

        # Convert to dict for state
        recommendation = {
            "action": resp.action,
            "confidence": resp.confidence,
            "timeline": resp.timeline,
            "staged_plan": (
                {
                    "num_tranches": resp.staged_plan.num_tranches,
                    "spacing_days": resp.staged_plan.spacing_days,
                    "total_extra_cost_bps": resp.staged_plan.total_extra_cost_bps,
                    "tranches": [
                        {
                            "tranche_number": t.tranche_number,
                            "percentage": t.percentage,
                            "execute_day": t.execute_day,
                            "rationale": t.rationale,
                        }
                        for t in (resp.staged_plan.tranches if resp.staged_plan else [])
                    ],
                    "benefit": resp.staged_plan.benefit,
                }
                if resp.staged_plan
                else None
            ),
            "expected_outcome": {
                "expected_rate": resp.expected_outcome.expected_rate,
                "range_low": resp.expected_outcome.range_low,
                "range_high": resp.expected_outcome.range_high,
                "expected_improvement_bps": resp.expected_outcome.expected_improvement_bps,
            },
            "risk_summary": {
                "risk_level": resp.risk_summary.risk_level,
                "realized_vol_30d": resp.risk_summary.realized_vol_30d,
                "var_95": resp.risk_summary.var_95,
                "event_risk": resp.risk_summary.event_risk,
                "event_details": resp.risk_summary.event_details,
            },
            "cost_estimate": {
                "spread_bps": resp.cost_estimate.spread_bps,
                "fee_bps": resp.cost_estimate.fee_bps,
                "total_bps": resp.cost_estimate.total_bps,
                "staged_multiplier": resp.cost_estimate.staged_multiplier,
            },
            "rationale": resp.rationale,
            "warnings": resp.warnings,
            "utility_scores": resp.utility_scores,
            "component_confidences": resp.component_confidences,
            "timestamp": resp.timestamp.isoformat(),
        }

        # Attach meta with horizon used
        used_key = None
        pf = state.get("price_forecast") or {}
        preds = (pf.get("predictions") or {})
        if str(tf_days) in preds:
            used_key = str(tf_days)
        elif preds:
            try:
                used_key = str(sorted(int(k) for k in preds.keys())[0])
            except Exception:
                used_key = next(iter(preds.keys()))

        recommendation["meta"] = {
            "prediction_horizon_days": tf_days,
            "used_prediction_horizon_key": used_key,
        }

        # Build evidence block for GUI parsing
        ms = state.get("market_snapshot") or {}
        providers = list({p.get("source", "unknown") for p in (ms.get("provider_breakdown") or [])})
        quality_notes = (ms.get("quality") or {}).get("notes") or []
        mid_rate = ms.get("mid_rate")
        bid = ms.get("bid")
        ask = ms.get("ask")
        spread = ms.get("spread")
        rate_ts = ms.get("rate_timestamp")

        mi = state.get("intelligence_report") or {}
        news_citations = (mi.get("news") or {}).get("top_evidence") or []
        # Get all calendar events (including past ones for debugging)
        calendar_all = (mi.get("calendar") or {}).get("events_extracted") or []
        # Debug logging
        logger.info(
            "[%s] Decision.Evidence: news=%d calendar_total=%d",
            correlation_id,
            len(news_citations),
            len(calendar_all)
        )
        if calendar_all:
            # Log first event for debugging
            logger.info("[%s] First calendar event: %s", correlation_id, calendar_all[0])
        # Show all events temporarily to debug why they're empty
        calendar_sources = calendar_all[:10]  # Just cap at 10 for UI

        # Model evidence
        explanations = ((pf.get("explanations") or {}).get("daily") or {})
        tf_key = used_key if used_key in explanations else (next(iter(explanations.keys())) if explanations else None)
        top_features = (explanations.get(tf_key) or {}).get("top_features") if tf_key else None
        model_ev = {
            "horizon_key": tf_key,
            "horizon_days": int(tf_key) if tf_key and str(tf_key).isdigit() else None,
            "top_features": top_features or {},
            "model_id": pf.get("model_id"),
            "model_confidence": pf.get("confidence"),
        }

        # Market extras: indicators, regime, and sample provider quotes
        indicators = ms.get("indicators") or {}
        regime = ms.get("regime") or {}
        provider_quotes = []
        for pr in (ms.get("provider_breakdown") or [])[:5]:
            provider_quotes.append(
                {
                    "source": pr.get("source"),
                    "rate": pr.get("rate"),
                    "timestamp": pr.get("timestamp"),
                }
            )

        # Intelligence summary
        intel_summary = {
            "pair_bias": (mi.get("news") or {}).get("pair_bias"),
            "news_confidence": (mi.get("news") or {}).get("confidence"),
            "n_articles_used": (mi.get("news") or {}).get("n_articles_used"),
            "narrative": (mi.get("news") or {}).get("narrative"),
            "policy_bias": mi.get("policy_bias"),
            "next_high_event": (mi.get("calendar") or {}).get("next_high_event"),
            "total_high_impact_events_7d": (mi.get("calendar") or {}).get("total_high_impact_events_7d"),
        }

        # Prediction summary
        pred_summary = None
        if preds:
            used = preds.get(used_key) or {}
            pred_summary = {
                "horizon_key": used_key,
                "mean_change_pct": used.get("mean_change_pct"),
                "quantiles": used.get("quantiles"),
                "direction_prob": used.get("direction_prob"),
                "latest_close": pf.get("latest_close"),
            }
        predictions_all = {k: (v or {}).get("mean_change_pct") for k, v in (preds or {}).items()} if preds else {}

        recommendation["evidence"] = {
            "market": {
                "providers": providers,
                "quality_notes": quality_notes,
                "dispersion_bps": (ms.get("quality") or {}).get("dispersion_bps"),
                "mid_rate": mid_rate,
                "bid": bid,
                "ask": ask,
                "spread": spread,
                "rate_timestamp": rate_ts,
                "indicators": indicators,
                "regime": regime,
                "provider_quotes": provider_quotes,
            },
            "news": news_citations[:5],
            "calendar": calendar_sources[:10],
            "intelligence": intel_summary,
            "model": model_ev,
            "prediction": pred_summary,
            "predictions_all": predictions_all,
            "utility_scores": resp.utility_scores,
        }

        return {
            "decision_status": "success",
            "decision_error": None,
            "recommendation": recommendation,
            "processing_stage": "decision_complete",
        }

    except Exception as e:
        logger.error(f"Decision node failed [{correlation_id}]: {e}")
        # Minimal conservative fallback
        return {
            "decision_status": "error",
            "decision_error": str(e),
            "recommendation": {
                "action": "wait",
                "confidence": 0.2,
                "timeline": "Wait for more data",
                "rationale": ["Error occurred; returning conservative default"],
                "warnings": ["Decision engine error"],
            },
            "processing_stage": "decision_error",
        }
