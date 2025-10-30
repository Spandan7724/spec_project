from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse

from backend.routes import conversation, analysis, visualization, health, models
from src.database.connection import create_tables


app = FastAPI(
    title="Currency Assistant API",
    description="AI-powered currency conversion recommendations",
    version="0.1.0",
)


@app.on_event("startup")
async def startup_event():
    """Initialize database tables on startup."""
    try:
        from src.config import load_config
        load_config()
        # Import models to register with Base metadata
        from backend.database import models as backend_models  # noqa: F401
        create_tables()
        print("✓ Database tables created/verified")
    except Exception as e:
        print(f"⚠ Database initialization warning: {e}")


# CORS (broad for dev; tighten in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Routers
app.include_router(conversation.router, prefix="/api/conversation", tags=["conversation"])
app.include_router(analysis.router, prefix="/api/analysis", tags=["analysis"])
app.include_router(visualization.router, prefix="/api/viz", tags=["visualization"])
app.include_router(models.router, prefix="/api/models", tags=["models"])
app.include_router(health.router, tags=["health"])


@app.get("/", response_class=HTMLResponse)
def index():
    """Serve the main web interface."""
    html_path = Path(__file__).parent.parent / "ui" / "web" / "index.html"
    return FileResponse(html_path)

