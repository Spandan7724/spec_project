# Web UI Implementation Plan

## Overview

Build a modern web interface for the Currency Assistant using a FastAPI backend. This document outlines an HTMX + Jinja2 server-rendered option. For the SPA approach with Next.js + TypeScript, follow `plans/impli/phase_5/phase-5-3.plan.md`. In either case, the backend lives under `src/ui/web` and exposes the same API endpoints.

## Technology Stack

### Backend

- **FastAPI**: REST API and WebSocket support
- **Jinja2**: Server-side templating
- **Pydantic**: Request/response validation

### Frontend  

- **HTMX**: Dynamic updates without JavaScript framework
- **Chart.js** or **ApexCharts**: Interactive charts
- **TailwindCSS** or **Bootstrap 5**: Styling
- **Alpine.js**: Minimal client-side interactivity

### Alternative (More Complex)

- **React/Vue** + FastAPI: Full SPA experience
- **Recommendation**: Start with HTMX, upgrade if needed

## Visualizations Priority

### MVP Visualizations

1. **Confidence Gauge** - Overall recommendation confidence
2. **Current Rate Display** - Big number with trend
3. **Action Card** - Convert Now / Staged / Wait with icon
4. **Timeline Visual** - When to execute
5. **Staging Plan** - Tranche timeline if staged

### Phase 2 Visualizations

6. **Technical Indicators Dashboard** - RSI, MACD gauges
7. **Economic Calendar** - Timeline of upcoming events
8. **Historical Price Chart** - 7-30 day candlestick/line
9. **Prediction Forecast** - Multi-horizon with confidence bands
10. **Risk Assessment** - Visual risk gauge

### Phase 3 Visualizations

11. **Component Confidence Breakdown** - Bar chart
12. **Cost Comparison** - Staged vs immediate
13. **Sentiment Indicators** - News sentiment visualization

## Architecture

```
src/ui/web/
├── __init__.py
├── main.py                       # FastAPI app
├── routes/
│   ├── __init__.py
│   ├── conversation.py           # Conversation endpoints
│   ├── analysis.py               # Analysis endpoints
│   └── visualization.py          # Data for charts
├── models/
│   ├── __init__.py
│   ├── requests.py               # API request models
│   └── responses.py              # API response models
├── templates/
│   ├── base.html                 # Base template
│   ├── index.html                # Home/conversation page
│   ├── recommendation.html       # Results page
│   └── components/
│       ├── conversation.html     # Chat component
│       ├── recommendation.html   # Recommendation component
│       └── charts.html           # Chart components
└── static/
    ├── css/
    │   └── styles.css
    ├── js/
    │   ├── charts.js             # Chart initialization
    │   └── app.js                # Main JS
    └── images/

tests/api/
├── test_routes.py
└── test_integration.py
```

## Implementation

### 1. FastAPI Application

**File: `src/ui/web/main.py`**

```python
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import os

from .routes import conversation, analysis, visualization

# Create FastAPI app
app = FastAPI(
    title="Currency Assistant API",
    description="AI-powered currency conversion recommendations",
    version="0.1.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="src/ui/web/static"), name="static")

# Templates
templates = Jinja2Templates(directory="src/ui/web/templates")

# Include routers
app.include_router(conversation.router, prefix="/api/conversation", tags=["conversation"])
app.include_router(analysis.router, prefix="/api/analysis", tags=["analysis"])
app.include_router(visualization.router, prefix="/api/viz", tags=["visualization"])

@app.get("/")
async def home(request: Request):
    """Home page with conversation interface"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy"}
```

### 2. Conversation API Routes

**File: `src/ui/web/routes/conversation.py`**

```python
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any

from src.supervisor.conversation_manager import ConversationManager
from src.supervisor.models import SupervisorRequest

router = APIRouter()
conversation_manager = ConversationManager()

class ConversationInput(BaseModel):
    user_input: str
    session_id: Optional[str] = None

class ConversationOutput(BaseModel):
    session_id: str
    state: str
    message: str
    requires_input: bool
    parameters: Optional[Dict[str, Any]] = None

@router.post("/message", response_model=ConversationOutput)
async def process_message(input: ConversationInput):
    """Process user message in conversation"""
    
    try:
        request = SupervisorRequest(
            user_input=input.user_input,
            session_id=input.session_id
        )
        
        response = conversation_manager.process_input(request)
        
        return ConversationOutput(
            session_id=response.session_id,
            state=response.state.value,
            message=response.message,
            requires_input=response.requires_input,
            parameters=response.parameters.__dict__ if response.parameters else None
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/reset/{session_id}")
async def reset_conversation(session_id: str):
    """Reset conversation session"""
    # Implementation to clear session
    return {"status": "reset", "session_id": session_id}
```

### 3. Analysis API Routes

**File: `src/ui/web/routes/analysis.py`**

```python
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, Dict, Any
import asyncio

from src.supervisor.agent_orchestrator import AgentOrchestrator
from src.supervisor.models import ExtractedParameters

router = APIRouter()
orchestrator = AgentOrchestrator()

# Store results temporarily (use Redis in production)
analysis_results = {}

class AnalysisRequest(BaseModel):
    session_id: str
    correlation_id: str
    currency_pair: str
    base_currency: str
    quote_currency: str
    amount: float
    risk_tolerance: str
    urgency: str
    timeframe: str
    timeframe_days: int

class AnalysisStatus(BaseModel):
    status: str  # pending | processing | completed | error
    progress: int  # 0-100
    message: str

@router.post("/start")
async def start_analysis(request: AnalysisRequest, background_tasks: BackgroundTasks):
    """Start analysis in background"""
    
    # Convert to ExtractedParameters
    params = ExtractedParameters(
        currency_pair=request.currency_pair,
        base_currency=request.base_currency,
        quote_currency=request.quote_currency,
        amount=request.amount,
        risk_tolerance=request.risk_tolerance,
        urgency=request.urgency,
        timeframe=request.timeframe,
        timeframe_days=request.timeframe_days
    )
    
    # Store initial status
    analysis_results[request.correlation_id] = {
        "status": "pending",
        "progress": 0,
        "message": "Analysis queued"
    }
    
    # Run in background
    background_tasks.add_task(
        run_analysis_background,
        params,
        request.correlation_id
    )
    
    return {
        "correlation_id": request.correlation_id,
        "status": "pending"
    }

async def run_analysis_background(params: ExtractedParameters, correlation_id: str):
    """Run analysis in background"""
    
    try:
        # Update status
        analysis_results[correlation_id] = {
            "status": "processing",
            "progress": 25,
            "message": "Analyzing market conditions..."
        }
        
        # Run orchestrator
        recommendation = await orchestrator.run_analysis(params, correlation_id)
        
        # Store result
        analysis_results[correlation_id] = {
            "status": "completed",
            "progress": 100,
            "message": "Analysis complete",
            "recommendation": recommendation
        }
    
    except Exception as e:
        analysis_results[correlation_id] = {
            "status": "error",
            "progress": 0,
            "message": str(e)
        }

@router.get("/status/{correlation_id}", response_model=AnalysisStatus)
async def get_analysis_status(correlation_id: str):
    """Get analysis status"""
    
    if correlation_id not in analysis_results:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    result = analysis_results[correlation_id]
    
    return AnalysisStatus(
        status=result["status"],
        progress=result["progress"],
        message=result["message"]
    )

@router.get("/result/{correlation_id}")
async def get_analysis_result(correlation_id: str):
    """Get analysis result"""
    
    if correlation_id not in analysis_results:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    result = analysis_results[correlation_id]
    
    if result["status"] != "completed":
        raise HTTPException(status_code=400, detail="Analysis not completed")
    
    return result["recommendation"]
```

### 4. Visualization Data Routes

**File: `src/ui/web/routes/visualization.py`**

```python
from fastapi import APIRouter, HTTPException
from typing import Dict, Any

router = APIRouter()

@router.get("/confidence/{correlation_id}")
async def get_confidence_data(correlation_id: str) -> Dict[str, Any]:
    """Get confidence breakdown data for chart"""
    
    # Mock data - replace with actual from analysis
    return {
        "overall": 0.75,
        "components": {
            "market": 0.80,
            "intelligence": 0.65,
            "prediction": 0.70
        }
    }

@router.get("/technical/{currency_pair}")
async def get_technical_indicators(currency_pair: str) -> Dict[str, Any]:
    """Get technical indicators for dashboard"""
    
    # Mock data - replace with actual market data
    return {
        "rsi": 65.5,
        "macd": 0.0012,
        "macd_signal": 0.0008,
        "sma_20": 1.0845,
        "sma_50": 1.0820,
        "current_price": 1.0850
    }

@router.get("/calendar/{currency}")
async def get_calendar_events(currency: str) -> Dict[str, Any]:
    """Get upcoming calendar events"""
    
    # Mock data - replace with actual calendar
    return {
        "events": [
            {
                "date": "2025-10-30",
                "time": "12:15",
                "event": "ECB Interest Rate Decision",
                "importance": "high",
                "currency": "EUR"
            }
        ]
    }

@router.get("/history/{currency_pair}")
async def get_price_history(currency_pair: str, days: int = 30) -> Dict[str, Any]:
    """Get historical price data for chart"""
    
    # Mock data - replace with actual from yfinance
    return {
        "dates": ["2025-10-01", "2025-10-02", "..."],
        "prices": [1.0800, 1.0820, 1.0850],
        "volumes": [1000000, 1100000, 1050000]
    }
```

### 5. HTML Templates

**File: `src/ui/web/templates/base.html`**

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}Currency Assistant{% endblock %}</title>
    
    <!-- TailwindCSS CDN (use build process in production) -->
    <script src="https://cdn.tailwindcss.com"></script>
    
    <!-- HTMX -->
    <script src="https://unpkg.com/htmx.org@1.9.10"></script>
    
    <!-- Alpine.js -->
    <script defer src="https://cdn.jsdelivr.net/npm/alpinejs@3.x.x/dist/cdn.min.js"></script>
    
    <!-- Chart.js -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    
    <!-- Custom CSS -->
    <link rel="stylesheet" href="{{ url_for('static', path='/css/styles.css') }}">
    
    {% block extra_head %}{% endblock %}
</head>
<body class="bg-gray-50">
    <!-- Header -->
    <header class="bg-blue-600 text-white shadow-lg">
        <div class="container mx-auto px-4 py-4">
            <h1 class="text-2xl font-bold">💱 Currency Assistant</h1>
            <p class="text-blue-100 text-sm">AI-powered conversion recommendations</p>
        </div>
    </header>
    
    <!-- Main Content -->
    <main class="container mx-auto px-4 py-8">
        {% block content %}{% endblock %}
    </main>
    
    <!-- Footer -->
    <footer class="bg-gray-800 text-gray-300 mt-12">
        <div class="container mx-auto px-4 py-6 text-center">
            <p>Currency Assistant v0.1.0 | Powered by AI</p>
        </div>
    </footer>
    
    <!-- Custom JS -->
    <script src="{{ url_for('static', path='/js/app.js') }}"></script>
    {% block extra_scripts %}{% endblock %}
</body>
</html>
```

**File: `src/ui/web/templates/index.html`**

```html
{% extends "base.html" %}

{% block content %}
<div class="max-w-4xl mx-auto">
    <!-- Conversation Interface -->
    <div class="bg-white rounded-lg shadow-lg p-6 mb-6">
        <h2 class="text-xl font-bold mb-4">Start Your Conversion Analysis</h2>
        
        <!-- Chat Container -->
        <div id="chat-container" class="space-y-4 mb-4 h-96 overflow-y-auto border border-gray-200 rounded p-4">
            <!-- Messages will be added here -->
            <div class="text-gray-500 text-center">
                <p>Welcome! Tell me about your currency conversion needs.</p>
                <p class="text-sm mt-2">Example: "I need to convert 5000 USD to EUR"</p>
            </div>
        </div>
        
        <!-- Input Form -->
        <form hx-post="/api/conversation/message" 
              hx-target="#chat-container" 
              hx-swap="beforeend"
              class="flex gap-2">
            <input type="text" 
                   name="user_input" 
                   id="user-input"
                   class="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                   placeholder="Type your message..."
                   required>
            <button type="submit" 
                    class="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition">
                Send
            </button>
        </form>
    </div>
    
    <!-- Analysis Progress (hidden initially) -->
    <div id="analysis-progress" class="hidden bg-white rounded-lg shadow-lg p-6">
        <h3 class="text-lg font-bold mb-4">Analyzing...</h3>
        <div class="w-full bg-gray-200 rounded-full h-4">
            <div id="progress-bar" class="bg-blue-600 h-4 rounded-full transition-all" style="width: 0%"></div>
        </div>
        <p id="progress-message" class="text-sm text-gray-600 mt-2">Initializing...</p>
    </div>
    
    <!-- Recommendation Display (hidden initially) -->
    <div id="recommendation-container" class="hidden">
        <!-- Recommendation will be loaded here -->
    </div>
</div>
{% endblock %}

{% block extra_scripts %}
<script src="{{ url_for('static', path='/js/app.js') }}"></script>
{% endblock %}
```

## Visualization Examples

### Confidence Gauge (Chart.js)

```javascript
// static/js/charts.js
function createConfidenceGauge(confidence) {
    const ctx = document.getElementById('confidenceChart').getContext('2d');
    
    new Chart(ctx, {
        type: 'doughnut',
        data: {
            datasets: [{
                data: [confidence * 100, (1 - confidence) * 100],
                backgroundColor: [
                    confidence > 0.7 ? '#10b981' : confidence > 0.4 ? '#f59e0b' : '#ef4444',
                    '#e5e7eb'
                ],
                borderWidth: 0
            }]
        },
        options: {
            circumference: 180,
            rotation: 270,
            cutout: '75%',
            plugins: {
                legend: { display: false },
                tooltip: { enabled: false }
            }
        }
    });
}
```

### Technical Indicators Dashboard

```javascript
function createRSIGauge(rsi) {
    const ctx = document.getElementById('rsiChart').getContext('2d');
    
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['RSI'],
            datasets: [{
                data: [rsi],
                backgroundColor: rsi > 70 ? '#ef4444' : rsi < 30 ? '#10b981' : '#3b82f6'
            }]
        },
        options: {
            indexAxis: 'y',
            scales: {
                x: { min: 0, max: 100 }
            }
        }
    });
}
```

## Running the Web UI

### Development

```bash
# Install dependencies
pip install fastapi uvicorn jinja2 python-multipart

# Or with uv
uv pip install fastapi uvicorn jinja2 python-multipart

# Run server
uvicorn src.ui.web.main:app --reload --port 8000

# Access at http://localhost:8000
```

### Production

```bash
# With multiple workers
uvicorn src.ui.web.main:app --host 0.0.0.0 --port 8000 --workers 4
```

## Implementation Todos

- [ ] Create FastAPI application in `src/ui/web/main.py`
- [ ] Implement conversation routes in `src/ui/web/routes/conversation.py`
- [ ] Implement analysis routes in `src/ui/web/routes/analysis.py`
- [ ] Implement visualization routes in `src/ui/web/routes/visualization.py`
- [ ] Create base HTML template in `src/ui/web/templates/base.html`
- [ ] Create conversation interface in `src/ui/web/templates/index.html`
- [ ] Create recommendation display template
- [ ] Implement Chart.js visualizations in `static/js/charts.js`
- [ ] Add HTMX interaction handlers in `static/js/app.js`
- [ ] Create CSS styles in `static/css/styles.css`
- [ ] Add FastAPI dependencies to `pyproject.toml`
- [ ] Write API integration tests
- [ ] Add WebSocket support for real-time updates (Phase 2)
- [ ] Implement user authentication (Phase 2)
- [ ] Add data persistence/caching layer (Phase 2)
