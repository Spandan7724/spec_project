from __future__ import annotations

from typing import Dict, Any
import logging
import os

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse

from backend.dependencies import get_orchestrator, get_analysis_repository
from backend.models.requests import AnalysisRequest
from backend.models.responses import AnalysisStartResponse, AnalysisStatus
from src.supervisor.models import ExtractedParameters
from src.supervisor.nlu_extractor import NLUExtractor
from src.supervisor.validation import timeframe_to_days


router = APIRouter()


@router.post("/start", response_model=AnalysisStartResponse)
def start_analysis(
    request: AnalysisRequest,
    background: BackgroundTasks,
    orchestrator=Depends(get_orchestrator),
    analysis_repo=Depends(get_analysis_repository),
):
    # Normalize timeframe (prefer canonical, then free-text, then legacy categorical)
    params = ExtractedParameters(
        currency_pair=request.currency_pair,
        base_currency=request.base_currency,
        quote_currency=request.quote_currency,
        amount=request.amount,
        risk_tolerance=request.risk_tolerance,
        urgency=request.urgency,
        timeframe=request.timeframe,
        timeframe_days=request.timeframe_days,
    )

    # Fill base/quote/currency_pair if not consistent
    if not params.currency_pair and params.base_currency and params.quote_currency:
        params.currency_pair = f"{params.base_currency}/{params.quote_currency}"
    if not params.base_currency and params.currency_pair:
        try:
            base, quote = params.currency_pair.split("/")
            params.base_currency, params.quote_currency = base, quote
        except Exception:
            pass

    # Flexible timeframe
    if request.timeframe_days is not None or request.deadline_utc or request.window_days:
        # Canonical provided; trust
        params.timeframe_days = request.timeframe_days
        params.timeframe_mode = request.timeframe_mode
        params.deadline_utc = request.deadline_utc
        params.window_days = request.window_days
        params.time_unit = request.time_unit
        params.timeframe_hours = request.timeframe_hours
    elif request.timeframe_text:
        # Reuse NLU for normalization (no network)
        extractor = NLUExtractor(use_llm=False)
        tfp = extractor.extract(request.timeframe_text)
        params.timeframe = params.timeframe or tfp.timeframe
        params.timeframe_days = params.timeframe_days if params.timeframe_days is not None else tfp.timeframe_days
        params.timeframe_mode = tfp.timeframe_mode
        params.deadline_utc = tfp.deadline_utc
        params.window_days = tfp.window_days
        params.time_unit = tfp.time_unit
        params.timeframe_hours = tfp.timeframe_hours
    else:
        # Legacy categorical → derive days if missing
        if params.timeframe and params.timeframe_days is None:
            params.timeframe_days = timeframe_to_days(params.timeframe)

    # Create database record
    try:
        analysis_repo.create(
            correlation_id=request.correlation_id,
            currency_pair=params.currency_pair or "",
            base_currency=params.base_currency or "",
            quote_currency=params.quote_currency or "",
            amount=params.amount or 0.0,
            risk_tolerance=params.risk_tolerance or "moderate",
            urgency=params.urgency or "normal",
            timeframe_days=params.timeframe_days,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create analysis record: {str(e)}")

    # Background task
    background.add_task(_run_analysis_task, analysis_repo, orchestrator, params, request.correlation_id)
    return AnalysisStartResponse(correlation_id=request.correlation_id, status="pending")


def _run_analysis_task(analysis_repo, orchestrator, params: ExtractedParameters, correlation_id: str):
    try:
        analysis_repo.update_status(
            correlation_id=correlation_id,
            status="processing",
            progress=25,
            message="Analyzing market conditions...",
        )

        # Run orchestrator (async in sync wrapper using its own design)
        import asyncio

        async def _go():
            return await orchestrator.run_analysis(params, correlation_id)

        try:
            result = asyncio.run(_go())
        except RuntimeError:
            # If loop already running, create a new loop in a thread
            import threading

            container = {"result": None, "error": None}

            def _runner():
                try:
                    container["result"] = asyncio.run(_go())
                except Exception as e:  # noqa: BLE001
                    container["error"] = e

            t = threading.Thread(target=_runner, daemon=True)
            t.start()
            t.join()
            if container["error"]:
                raise container["error"]
            result = container["result"]

        # Extract components from orchestrator state for visualization
        recommendation = result if isinstance(result, dict) else {}

        # --- Evidence pipeline summary (concise) ---
        ev = (recommendation or {}).get("evidence") or {}
        market = ev.get("market") or {}
        news_list = ev.get("news") or []
        events_list = ev.get("calendar") or []
        offline_demo = os.getenv("OFFLINE_DEMO", "false").strip().lower() in {"1", "true", "yes", "on"}

        logging.info(
            "EvidencePipeline: corr=%s market_present=%s mid_rate=%s news=%d events=%d offline_demo=%s",
            correlation_id,
            bool(market),
            market.get("mid_rate"),
            len(news_list) if isinstance(news_list, list) else 0,
            len(events_list) if isinstance(events_list, list) else 0,
            offline_demo,
        )

        # Map evidence payloads from recommendation into DB fields expected by visualization routes
        ev = (recommendation or {}).get("evidence") or {}

        # Market data mapping
        md_src = ev.get("market") or {}
        market_data = None
        if md_src:
            indicators = md_src.get("indicators") or {}
            market_data = {
                "current_rate": md_src.get("mid_rate"),
                "bid": md_src.get("bid"),
                "ask": md_src.get("ask"),
                # spread_bps unknown here; keep None to avoid misleading values
                "spread_bps": None,
                "regime": md_src.get("regime"),
                "volatility": indicators.get("realized_vol_30d"),
            }

        # Intelligence mapping (shape expected by /api/viz/evidence)
        intel_summary = ev.get("intelligence") or {}
        news_list = ev.get("news") or []
        events_list = ev.get("calendar") or []
        intelligence = {
            "news": {
                "top_evidence": news_list,
                "narrative": intel_summary.get("narrative"),
                "sent_base": intel_summary.get("sent_base"),
                "sent_quote": intel_summary.get("sent_quote"),
                "pair_bias": intel_summary.get("pair_bias"),
                "confidence": intel_summary.get("news_confidence"),
                "n_articles_used": intel_summary.get("n_articles_used"),
            },
            "calendar": {
                "events_extracted": events_list,
                "next_high_event": intel_summary.get("next_high_event"),
                "total_high_impact_events_7d": intel_summary.get("total_high_impact_events_7d"),
            },
            "policy_bias": intel_summary.get("policy_bias"),
        }

        # Prediction mapping (minimal, enough for chart endpoint)
        pred_summary = ev.get("prediction") or {}
        preds_all = ev.get("predictions_all") or {}
        predictions = {
            str(k): {"mean_change_pct": v, "quantiles": {}}
            for k, v in preds_all.items()
        }
        prediction = {
            "latest_close": pred_summary.get("latest_close"),
            "predictions": predictions,
        }

        # Update with full result
        analysis_repo.update_result(
            correlation_id=correlation_id,
            recommendation=recommendation,
            market_data=market_data,
            intelligence=intelligence,
            prediction=prediction,
        )

    except Exception as e:  # noqa: BLE001
        analysis_repo.update_error(correlation_id=correlation_id, error_message=str(e))


@router.get("/status/{correlation_id}", response_model=AnalysisStatus)
def get_status(correlation_id: str, analysis_repo=Depends(get_analysis_repository)):
    record = analysis_repo.get_by_correlation_id(correlation_id)
    if not record:
        raise HTTPException(status_code=404, detail="Analysis not found")

    return AnalysisStatus(
        status=record.status,
        progress=record.progress,
        message=record.message,
    )


@router.get("/result/{correlation_id}")
def get_result(correlation_id: str, analysis_repo=Depends(get_analysis_repository)):
    """Get complete analysis result with all supporting data."""
    record = analysis_repo.get_by_correlation_id(correlation_id)
    if not record:
        raise HTTPException(status_code=404, detail="Analysis not found")

    if record.status != "completed":
        raise HTTPException(status_code=400, detail="Analysis not completed")

    recommendation = record.recommendation or {}

    # Build comprehensive response
    return {
        "status": "success",
        "correlation_id": correlation_id,
        # Core recommendation
        "action": recommendation.get("action"),
        "confidence": recommendation.get("confidence"),
        "timeline": recommendation.get("timeline"),
        "rationale": recommendation.get("rationale", []),
        "warnings": recommendation.get("warnings", []),
        # Detailed breakdowns
        "staged_plan": recommendation.get("staged_plan"),
        "expected_outcome": recommendation.get("expected_outcome"),
        "risk_summary": recommendation.get("risk_summary"),
        "cost_estimate": recommendation.get("cost_estimate"),
        "utility_scores": recommendation.get("utility_scores"),
        "component_confidences": recommendation.get("component_confidences"),
        # Supporting evidence (for webapp visualization)
        "evidence": {
            "market_data": record.market_data,
            "intelligence": record.intelligence,
            "prediction": record.prediction,
        },
        # Metadata
        "metadata": recommendation.get("metadata"),
        "created_at": record.created_at.isoformat() if record.created_at else None,
        "updated_at": record.updated_at.isoformat() if record.updated_at else None,
    }


@router.get("/stream/{correlation_id}")
async def stream_status(correlation_id: str, analysis_repo=Depends(get_analysis_repository)):
    """Server-Sent Events (SSE) stream for analysis progress.

    Emits events of type 'status' with JSON payload: {status, progress, message}.
    Ends when the analysis reaches completed or error state, or if not found.
    """
    import json as _json
    import asyncio as _asyncio

    async def event_generator():
        last_payload = None
        while True:
            record = analysis_repo.get_by_correlation_id(correlation_id)
            if not record:
                yield f"event: error\ndata: {_json.dumps({'error': 'not_found'})}\n\n"
                break

            payload = {
                "status": record.status,
                "progress": record.progress,
                "message": record.message,
            }
            data = _json.dumps(payload)
            if data != last_payload:
                yield f"event: status\ndata: {data}\n\n"
                last_payload = data

            if record.status in {"completed", "error"}:
                break

            await _asyncio.sleep(0.5)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.get("/list")
def list_analyses(
    status: str = Query(None, description="Filter by status (pending|processing|completed|error)"),
    currency_pair: str = Query(None, description="Filter by currency pair"),
    limit: int = Query(100, ge=1, le=500, description="Max results to return"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    analysis_repo=Depends(get_analysis_repository),
):
    """List analysis records with filters and pagination."""
    records = analysis_repo.list_all(
        status=status,
        currency_pair=currency_pair,
        limit=limit,
        offset=offset,
    )

    return {
        "total": len(records),
        "limit": limit,
        "offset": offset,
        "analyses": [
            {
                "correlation_id": r.correlation_id,
                "currency_pair": r.currency_pair,
                "status": r.status,
                "progress": r.progress,
                "created_at": r.created_at.isoformat() if r.created_at else None,
                "action": r.recommendation.get("action") if r.recommendation else None,
                "confidence": r.recommendation.get("confidence") if r.recommendation else None,
            }
            for r in records
        ],
    }


@router.delete("/{correlation_id}")
def delete_analysis(correlation_id: str, analysis_repo=Depends(get_analysis_repository)):
    """Delete a specific analysis record."""
    success = analysis_repo.delete_by_correlation_id(correlation_id)
    if not success:
        raise HTTPException(status_code=404, detail="Analysis not found")

    return {"status": "deleted", "correlation_id": correlation_id}


@router.post("/cleanup")
def cleanup_expired(analysis_repo=Depends(get_analysis_repository)):
    """Manually trigger cleanup of expired analysis records."""
    count = analysis_repo.delete_expired()
    return {"status": "ok", "deleted_count": count}
