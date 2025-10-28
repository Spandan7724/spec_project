from __future__ import annotations

from typing import Dict, Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.responses import StreamingResponse

from backend.dependencies import get_orchestrator
from backend.models.requests import AnalysisRequest
from backend.models.responses import AnalysisStartResponse, AnalysisStatus
from src.supervisor.models import ExtractedParameters
from src.supervisor.nlu_extractor import NLUExtractor
from src.supervisor.validation import timeframe_to_days


router = APIRouter()

# In-memory store (simple)
analysis_results: Dict[str, Dict[str, Any]] = {}


@router.post("/start", response_model=AnalysisStartResponse)
def start_analysis(request: AnalysisRequest, background: BackgroundTasks, orchestrator=Depends(get_orchestrator)):
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

    # Store initial status
    analysis_results[request.correlation_id] = {
        "status": "pending",
        "progress": 0,
        "message": "Analysis queued",
    }

    # Background task
    background.add_task(_run_analysis_task, orchestrator, params, request.correlation_id)
    return AnalysisStartResponse(correlation_id=request.correlation_id, status="pending")


def _run_analysis_task(orchestrator, params: ExtractedParameters, correlation_id: str):
    try:
        analysis_results[correlation_id] = {
            "status": "processing",
            "progress": 25,
            "message": "Analyzing market conditions...",
        }
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

        analysis_results[correlation_id] = {
            "status": "completed",
            "progress": 100,
            "message": "Analysis complete",
            "recommendation": result,
        }
    except Exception as e:  # noqa: BLE001
        analysis_results[correlation_id] = {
            "status": "error",
            "progress": 0,
            "message": str(e),
        }


@router.get("/status/{correlation_id}", response_model=AnalysisStatus)
def get_status(correlation_id: str):
    if correlation_id not in analysis_results:
        raise HTTPException(status_code=404, detail="Analysis not found")
    st = analysis_results[correlation_id]
    return AnalysisStatus(status=st["status"], progress=st["progress"], message=st["message"])


@router.get("/result/{correlation_id}")
def get_result(correlation_id: str):
    if correlation_id not in analysis_results:
        raise HTTPException(status_code=404, detail="Analysis not found")
    st = analysis_results[correlation_id]
    if st.get("status") != "completed":
        raise HTTPException(status_code=400, detail="Analysis not completed")
    return st.get("recommendation")


@router.get("/stream/{correlation_id}")
async def stream_status(correlation_id: str):
    """Server-Sent Events (SSE) stream for analysis progress.

    Emits events of type 'status' with JSON payload: {status, progress, message}.
    Ends when the analysis reaches completed or error state, or if not found.
    """
    import json as _json
    import asyncio as _asyncio

    async def event_generator():
        last_payload = None
        while True:
            st = analysis_results.get(correlation_id)
            if not st:
                yield f"event: error\ndata: {_json.dumps({'error': 'not_found'})}\n\n"
                break
            payload = {"status": st.get("status"), "progress": st.get("progress"), "message": st.get("message")}
            data = _json.dumps(payload)
            if data != last_payload:
                yield f"event: status\ndata: {data}\n\n"
                last_payload = data
            if st.get("status") in {"completed", "error"}:
                break
            await _asyncio.sleep(0.5)

    return StreamingResponse(event_generator(), media_type="text/event-stream")
