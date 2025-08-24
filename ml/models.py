"""
LSTM models for foreign exchange rate forecasting.

Implements time-series prediction models with uncertainty quantification
for currency exchange rate forecasting.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for LSTM forecasting model."""
    sequence_length: int = 168  # 7 days * 24 hours (weekly patterns)
    prediction_horizon: int = 24  # Predict 24 hours ahead
    input_features: int = 10  # Number of input features
    hidden_size: int = 128  # LSTM hidden size
    num_layers: int = 3  # Number of LSTM layers
    dropout: float = 0.2  # Dropout rate
    attention_heads: int = 8  # Multi-head attention
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100


class LSTMForecaster(nn.Module):
    """
    LSTM-based forecasting model with attention mechanism.
    
    Features:
    - Multi-layer LSTM for sequence modeling
    - Multi-head attention for important pattern focus
    - Uncertainty quantification (mean + variance prediction)
    - Dropout for regularization
    """
    
    def __init__(self, config: ModelConfig):
        super(LSTMForecaster, self).__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=config.input_features,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0,
            batch_first=True,
            bidirectional=False
        )
        
        # Multi-head attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.attention_heads,
            dropout=config.dropout,
            batch_first=True
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        
        # Output layers
        self.dropout = nn.Dropout(config.dropout)
        
        # Separate heads for mean and variance prediction
        self.mean_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size // 2, config.prediction_horizon)
        )
        
        self.variance_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(), 
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size // 2, config.prediction_horizon),
            nn.Softplus()  # Ensure positive variance
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights using Xavier/Glorot initialization."""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_features)
            
        Returns:
            Tuple of (predicted_mean, predicted_variance) tensors
            Both have shape (batch_size, prediction_horizon)
        """
        batch_size = x.size(0)
        
        # Initialize LSTM hidden states
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        # Apply attention mechanism
        attn_out, attention_weights = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Residual connection and layer normalization
        lstm_out = self.layer_norm(lstm_out + attn_out)
        
        # Use the last timestep output
        last_output = lstm_out[:, -1, :]  # Shape: (batch_size, hidden_size)
        
        # Apply dropout
        features = self.dropout(last_output)
        
        # Predict mean and variance
        predicted_mean = self.mean_head(features)
        predicted_variance = self.variance_head(features)
        
        return predicted_mean, predicted_variance
    
    def predict_with_uncertainty(self, x: torch.Tensor, num_samples: int = 100) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Make predictions with uncertainty quantification using Monte Carlo dropout.
        
        Args:
            x: Input tensor
            num_samples: Number of MC samples for uncertainty estimation
            
        Returns:
            Tuple of (mean_prediction, epistemic_uncertainty, aleatoric_uncertainty)
        """
        self.train()  # Enable dropout for MC sampling
        
        predictions = []
        variances = []
        
        with torch.no_grad():
            for _ in range(num_samples):
                pred_mean, pred_var = self.forward(x)
                predictions.append(pred_mean)
                variances.append(pred_var)
        
        predictions = torch.stack(predictions)  # (num_samples, batch_size, prediction_horizon)
        variances = torch.stack(variances)
        
        # Calculate uncertainties
        mean_prediction = predictions.mean(dim=0)
        epistemic_uncertainty = predictions.var(dim=0)  # Model uncertainty
        aleatoric_uncertainty = variances.mean(dim=0)   # Data uncertainty
        
        self.eval()  # Return to eval mode
        
        return mean_prediction, epistemic_uncertainty, aleatoric_uncertainty


class EarlyStopping:
    """Early stopping utility to prevent overfitting."""
    
    def __init__(self, patience: int = 10, min_delta: float = 1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


def gaussian_nll_loss(y_true: torch.Tensor, y_pred_mean: torch.Tensor, y_pred_var: torch.Tensor) -> torch.Tensor:
    """
    Gaussian negative log-likelihood loss for uncertainty-aware training.
    
    Args:
        y_true: Ground truth values
        y_pred_mean: Predicted means
        y_pred_var: Predicted variances
        
    Returns:
        NLL loss tensor
    """
    # Add small epsilon to prevent log(0)
    epsilon = 1e-8
    y_pred_var = y_pred_var + epsilon
    
    # Compute negative log-likelihood
    nll = 0.5 * (torch.log(y_pred_var) + (y_true - y_pred_mean)**2 / y_pred_var)
    return nll.mean()


def mape_loss(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    Mean Absolute Percentage Error loss.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        
    Returns:
        MAPE loss tensor
    """
    epsilon = 1e-8
    return torch.mean(torch.abs((y_true - y_pred) / (y_true + epsilon))) * 100


class ModelEvaluator:
    """Utility class for model evaluation metrics."""
    
    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            
        Returns:
            Dictionary of metrics
        """
        # Handle multi-dimensional arrays by flattening if needed
        if y_true.ndim > 1:
            y_true = y_true.flatten()
        if y_pred.ndim > 1:
            y_pred = y_pred.flatten()
        
        # Basic error metrics
        errors = y_true - y_pred
        absolute_errors = np.abs(errors)
        squared_errors = errors ** 2
        
        mse = np.mean(squared_errors)
        mae = np.mean(absolute_errors)
        rmse = np.sqrt(mse)
        
        # Percentage-based metrics
        mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100
        
        # Mean Absolute Scaled Error (MASE) - scaled by naive forecast
        naive_forecast_mae = np.mean(np.abs(np.diff(y_true)))
        mase = mae / (naive_forecast_mae + 1e-8)
        
        # Directional accuracy (did we predict the right direction?)
        if len(y_true) > 1:
            y_true_diff = np.diff(y_true)
            y_pred_diff = np.diff(y_pred)
            directional_accuracy = np.mean(np.sign(y_true_diff) == np.sign(y_pred_diff)) * 100
        else:
            directional_accuracy = 0.0
        
        # R-squared
        ss_res = np.sum(squared_errors)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))
        
        # Additional metrics
        median_ae = np.median(absolute_errors)
        max_error = np.max(absolute_errors)
        std_error = np.std(errors)
        
        # Symmetric metrics
        smape = 200 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true) + 1e-8))
        
        return {
            'mse': float(mse),
            'mae': float(mae),
            'rmse': float(rmse),
            'mape': float(mape),
            'smape': float(smape),
            'mase': float(mase),
            'r2': float(r2),
            'directional_accuracy': float(directional_accuracy),
            'median_absolute_error': float(median_ae),
            'max_error': float(max_error),
            'std_error': float(std_error),
            'mean_error': float(np.mean(errors))
        }