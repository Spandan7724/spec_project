"""
ML Price Prediction System for Currency Assistant

LSTM-based price prediction with multi-horizon forecasts and confidence intervals.
"""

from .prediction.predictor import MLPredictor
from .models.lstm_model import LSTMModel
from .config import MLConfig

__all__ = ['MLPredictor', 'LSTMModel', 'MLConfig']