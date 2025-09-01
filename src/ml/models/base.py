"""
Base model interface for all ML models
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass

@dataclass
class PredictionResult:
    """Result from model prediction"""
    predictions: np.ndarray  # Mean predictions for each horizon
    confidence_intervals: Dict[str, np.ndarray]  # p10, p50, p90 for each horizon
    direction_probabilities: np.ndarray  # Probability of up/down for each horizon
    model_confidence: float  # Overall model confidence
    timestamp: str
    model_version: str


class BaseModel(ABC, nn.Module):
    """Abstract base class for all prediction models"""
    
    def __init__(self, config: Dict[str, Any]):
        super(BaseModel, self).__init__()
        self.config = config
        self.model_version = "1.0"
        self.is_trained = False
        
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model"""
        pass
    
    @abstractmethod
    def predict(self, 
                features: np.ndarray, 
                horizons: List[int] = None) -> PredictionResult:
        """Make predictions with confidence intervals"""
        pass
    
    @abstractmethod
    def fit(self, 
            X_train: np.ndarray, 
            y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None) -> Dict[str, List[float]]:
        """Train the model"""
        pass
    
    @abstractmethod
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate the model"""
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "model_type": self.__class__.__name__,
            "model_version": self.model_version,
            "is_trained": self.is_trained,
            "config": self.config,
            "parameter_count": sum(p.numel() for p in self.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad)
        }
    
    def save_model(self, filepath: str):
        """Save model state"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'model_version': self.model_version,
            'is_trained': self.is_trained
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load model state"""
        checkpoint = torch.load(filepath, map_location='cpu')
        self.load_state_dict(checkpoint['model_state_dict'])
        self.config = checkpoint.get('config', self.config)
        self.model_version = checkpoint.get('model_version', self.model_version)
        self.is_trained = checkpoint.get('is_trained', False)
    
    def _calculate_confidence_intervals(self, 
                                      predictions: np.ndarray, 
                                      uncertainty: np.ndarray,
                                      confidence_levels: List[float] = None) -> Dict[str, np.ndarray]:
        """Calculate confidence intervals from predictions and uncertainty estimates"""
        if confidence_levels is None:
            confidence_levels = [0.1, 0.5, 0.9]
        
        confidence_intervals = {}
        
        for level in confidence_levels:
            if level == 0.5:
                # Median is just the prediction
                confidence_intervals[f'p{int(level*100)}'] = predictions
            else:
                # Calculate percentiles based on uncertainty
                z_score = torch.distributions.Normal(0, 1).icdf(torch.tensor(level)).item()
                if level < 0.5:
                    confidence_intervals[f'p{int(level*100)}'] = predictions - abs(z_score) * uncertainty
                else:
                    confidence_intervals[f'p{int(level*100)}'] = predictions + abs(z_score) * uncertainty
        
        return confidence_intervals
    
    def _calculate_direction_probabilities(self, 
                                         current_prices: np.ndarray,
                                         predictions: np.ndarray) -> np.ndarray:
        """Calculate probability of price going up vs down"""
        # Simple approach: sigmoid of relative change
        relative_change = (predictions - current_prices) / current_prices
        up_probabilities = torch.sigmoid(torch.tensor(relative_change * 10)).numpy()
        return up_probabilities