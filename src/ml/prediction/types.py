"""
Type definitions for prediction system
"""

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class MLPredictionRequest:
    """Request for ML prediction"""
    currency_pair: str
    horizons: List[int] = None  # [1, 7, 30] days
    include_confidence: bool = True
    include_direction_prob: bool = True
    max_age_hours: int = 1  # Use cached if available and fresh


@dataclass
class MLPredictionResponse:
    """Response from ML prediction"""
    currency_pair: str
    timestamp: str
    model_id: str
    model_confidence: float
    predictions: Dict[str, Dict[str, float]]  # {horizon: {mean, p10, p50, p90}}
    direction_probabilities: Dict[str, float]  # {horizon: probability_up}
    features_used: int
    processing_time_ms: float
    cached: bool = False