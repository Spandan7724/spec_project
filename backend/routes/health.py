from __future__ import annotations

from fastapi import APIRouter

from src.health import get_health_status


router = APIRouter()


@router.get("/health")
async def health():
    status = await get_health_status()
    # health structure is already a dict with status, details
    return status

