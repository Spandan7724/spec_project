from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from .analysis import analysis_results


router = APIRouter()


@router.get("/confidence/{correlation_id}")
def confidence_breakdown(correlation_id: str) -> Dict[str, Any]:
    st = analysis_results.get(correlation_id)
    if not st or st.get("status") != "completed":
        raise HTTPException(status_code=404, detail="Analysis not completed or not found")
    rec = st.get("recommendation") or {}
    overall = rec.get("confidence")
    comps = rec.get("component_confidences") or {}
    return {"overall": overall, "components": comps}

