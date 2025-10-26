from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime


@dataclass
class PredictionRequest:
    """Request for price prediction."""

    currency_pair: str
    horizons: List[int] = field(default_factory=lambda: [1, 7, 30])
    include_quantiles: bool = True
    include_direction_probabilities: bool = True
    max_age_hours: int = 1
    features_mode: str = "price_only"  # price_only | price_plus_intel
    correlation_id: Optional[str] = None
    # Optional intraday horizons (in hours) for hybrid routing with LSTM
    intraday_horizons_hours: List[int] = field(default_factory=list)
    # Optional backend preference override: "lightgbm" | "lstm" | "hybrid"
    backend_preference: Optional[str] = None
    # Include explanations/evidence in response
    include_explanations: bool = False


@dataclass
class HorizonPrediction:
    """Prediction for a single horizon."""

    horizon: int
    mean_change_pct: float  # Predicted % change
    quantiles: Optional[Dict[str, float]] = None  # {p10, p50, p90}
    direction_probability: Optional[float] = None  # P(up)


@dataclass
class PredictionQuality:
    """Quality metadata for predictions."""

    model_confidence: float  # 0-1
    calibrated: bool
    validation_metrics: Dict[str, float]
    notes: List[str] = field(default_factory=list)


@dataclass
class PredictionResponse:
    """Complete prediction response."""

    status: str  # success | partial | error
    confidence: float  # Overall confidence 0-1
    processing_time_ms: float
    # Data
    currency_pair: str
    horizons: List[int]
    predictions: Dict[int, HorizonPrediction]  # horizon -> prediction
    latest_close: float
    features_used: List[str]
    quality: PredictionQuality

    # Metadata
    model_id: str
    warnings: List[str] = field(default_factory=list)
    cached: bool = False
    timestamp: datetime = field(default_factory=datetime.now)
    # Optional explanations/evidence and model metadata for UI
    explanations: Optional[Dict[str, Any]] = None
    model_info: Optional[Dict[str, Any]] = None


@dataclass
class ModelMetadata:
    """Model registry metadata."""

    model_id: str
    model_type: str  # lightgbm | lstm
    currency_pair: str
    trained_at: datetime
    version: str

    # Validation metrics
    validation_metrics: Dict[str, float]
    min_samples: int
    calibration_ok: bool

    # Feature info
    features_used: List[str]
    horizons: List[int]

    # Storage
    model_path: str
    scaler_path: Optional[str] = None
