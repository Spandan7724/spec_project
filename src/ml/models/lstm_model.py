"""
LSTM model for currency price prediction
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from ..models.base import BaseModel, PredictionResult

logger = logging.getLogger(__name__)


class LSTMModel(BaseModel):
    """
    LSTM-based model for multi-horizon currency price prediction
    """
    
    def __init__(self, config: Dict[str, Any]):
        super(LSTMModel, self).__init__(config)
        
        # Model architecture parameters
        self.input_size = config.get('input_size', 30)
        self.hidden_size = config.get('hidden_size', 128)
        self.num_layers = config.get('num_layers', 2)
        self.dropout = config.get('dropout', 0.2)
        self.bidirectional = config.get('bidirectional', False)
        
        # Prediction parameters
        self.prediction_horizons = config.get('prediction_horizons', [1, 7, 30])
        self.output_size = len(self.prediction_horizons)
        
        # Build model
        self._build_model()
        
        # Training parameters
        self.learning_rate = config.get('learning_rate', 0.001)
        self.device = config.get('device', 'cpu')
        self.to(self.device)
        
        # Training state
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        self.training_history = {'loss': [], 'val_loss': []}
        
        logger.info(f"Initialized LSTM model: {self.get_model_info()}")
    
    def _build_model(self):
        """Build the LSTM architecture"""
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout if self.num_layers > 1 else 0,
            bidirectional=self.bidirectional,
            batch_first=True
        )
        
        # Calculate LSTM output size
        lstm_output_size = self.hidden_size * (2 if self.bidirectional else 1)
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(lstm_output_size, lstm_output_size // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(lstm_output_size // 2, lstm_output_size // 4),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(lstm_output_size // 4, self.output_size)
        )
        
        # Additional head for uncertainty estimation
        self.uncertainty_head = nn.Sequential(
            nn.Linear(lstm_output_size, lstm_output_size // 4),
            nn.ReLU(),
            nn.Linear(lstm_output_size // 4, self.output_size),
            nn.Softplus()  # Ensure positive uncertainty values
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the model
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
            predictions: Tensor of shape (batch_size, num_horizons)
            uncertainty: Tensor of shape (batch_size, num_horizons)
        """
        batch_size = x.size(0)
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use the last output of the sequence
        last_output = lstm_out[:, -1, :]
        
        # Predictions
        predictions = self.fc_layers(last_output)
        
        # Uncertainty estimates
        uncertainty = self.uncertainty_head(last_output)
        
        return predictions, uncertainty
    
    def fit(self, 
            X_train: np.ndarray, 
            y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None,
            epochs: int = None,
            batch_size: int = None,
            patience: int = None,
            verbose: bool = True) -> Dict[str, List[float]]:
        """
        Train the LSTM model
        """
        epochs = epochs or self.config.get('epochs', 100)
        batch_size = batch_size or self.config.get('batch_size', 32)
        patience = patience or self.config.get('patience', 15)
        
        logger.info(f"Starting training: {epochs} epochs, batch_size {batch_size}")
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)
        
        if X_val is not None:
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val).to(self.device)
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        self.train()
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            # Mini-batch training
            for i in range(0, len(X_train_tensor), batch_size):
                batch_X = X_train_tensor[i:i+batch_size]
                batch_y = y_train_tensor[i:i+batch_size]
                
                # Forward pass
                self.optimizer.zero_grad()
                predictions, uncertainty = self.forward(batch_X)
                
                # Calculate loss (MSE for predictions + regularization for uncertainty)
                pred_loss = self.criterion(predictions, batch_y)
                uncertainty_loss = torch.mean(uncertainty)  # Encourage reasonable uncertainty
                loss = pred_loss + 0.1 * uncertainty_loss
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            epoch_loss /= num_batches
            self.training_history['loss'].append(epoch_loss)
            
            # Validation
            val_loss = None
            if X_val is not None:
                val_loss = self._validate(X_val_tensor, y_val_tensor)
                self.training_history['val_loss'].append(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
            
            if verbose and (epoch + 1) % 10 == 0:
                val_str = f", Val Loss: {val_loss:.6f}" if val_loss else ""
                logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.6f}{val_str}")
        
        self.is_trained = True
        logger.info("Training completed")
        
        return self.training_history
    
    def _validate(self, X_val: torch.Tensor, y_val: torch.Tensor) -> float:
        """Validate the model on validation set"""
        self.eval()
        
        with torch.no_grad():
            val_predictions, val_uncertainty = self.forward(X_val)
            val_loss = self.criterion(val_predictions, y_val).item()
        
        self.train()
        return val_loss
    
    def predict(self,
                features: np.ndarray,
                horizons: List[int] = None) -> PredictionResult:
        """
        Make predictions with confidence intervals
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        horizons = horizons or self.prediction_horizons
        
        # Convert to tensor
        if features.ndim == 2:
            # Add batch dimension
            features = features.reshape(1, *features.shape)
        
        X_tensor = torch.FloatTensor(features).to(self.device)
        
        self.eval()
        with torch.no_grad():
            predictions, uncertainty = self.forward(X_tensor)
            predictions = predictions.cpu().numpy()[0]
            uncertainty = uncertainty.cpu().numpy()[0]

        # Calculate confidence intervals
        confidence_intervals = self._calculate_confidence_intervals(
            predictions, uncertainty
        )

        # Predicted returns -> direction probabilities via logistic transform
        direction_probs = 1.0 / (1.0 + np.exp(-predictions))

        # Overall model confidence (inverse of average uncertainty)
        model_confidence = float(1.0 / (1.0 + np.mean(uncertainty)))

        return PredictionResult(
            predictions=predictions,
            confidence_intervals=confidence_intervals,
            direction_probabilities=direction_probs,
            model_confidence=model_confidence,
            timestamp=datetime.now().isoformat(),
            model_version=self.model_version
        )
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate the model on test data"""
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        X_tensor = torch.FloatTensor(X_test).to(self.device)
        y_tensor = torch.FloatTensor(y_test).to(self.device)
        
        self.eval()
        with torch.no_grad():
            predictions, uncertainty = self.forward(X_tensor)
            
            # Calculate metrics
            mse = self.criterion(predictions, y_tensor).item()
            mae = torch.mean(torch.abs(predictions - y_tensor)).item()
            
            # RMSE
            rmse = np.sqrt(mse)
            
            # R²
            ss_res = torch.sum((y_tensor - predictions) ** 2).item()
            ss_tot = torch.sum((y_tensor - torch.mean(y_tensor)) ** 2).item()
            r2 = 1 - (ss_res / (ss_tot + 1e-8))
            
            # Directional accuracy (for each horizon)
            predictions_np = predictions.cpu().numpy()
            y_test_np = y_tensor.cpu().numpy()
            
            directional_accuracies = []
            for i in range(predictions_np.shape[1]):
                pred_direction = predictions_np[:, i] > 0
                actual_direction = y_test_np[:, i] > 0
                accuracy = np.mean(pred_direction == actual_direction)
                directional_accuracies.append(accuracy)
            
            avg_directional_accuracy = np.mean(directional_accuracies)
        
        metrics = {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'directional_accuracy': avg_directional_accuracy,
            'mean_uncertainty': float(torch.mean(uncertainty).cpu().item())
        }
        
        # Add per-horizon metrics
        for i, horizon in enumerate(self.prediction_horizons):
            metrics[f'directional_accuracy_{horizon}d'] = directional_accuracies[i]
        
        logger.info(f"Model evaluation: {metrics}")
        
        return metrics
    
    def get_feature_importance(self, 
                              X_sample: np.ndarray, 
                              method: str = 'gradient') -> np.ndarray:
        """
        Calculate feature importance using gradient-based method
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before calculating feature importance")
        
        if method != 'gradient':
            raise ValueError("Only 'gradient' method is currently supported")
        
        X_tensor = torch.FloatTensor(X_sample).to(self.device)
        X_tensor.requires_grad_(True)
        
        # Ensure model is in training mode for gradient computation
        # CUDA RNN backward pass requires training mode
        self.train()
        
        # Use torch.enable_grad() context to allow gradients
        with torch.enable_grad():
            predictions, _ = self.forward(X_tensor)
            
            # Calculate gradients
            prediction_sum = torch.sum(predictions)
            prediction_sum.backward()
        
        # Get importance as absolute gradient values
        importance = torch.abs(X_tensor.grad).mean(dim=(0, 1)).cpu().numpy()
        
        return importance
    
    def save_model(self, filepath: str):
        """Save model state"""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'model_version': self.model_version,
            'is_trained': self.is_trained,
            'training_history': self.training_history,
            'prediction_horizons': self.prediction_horizons
        }
        
        torch.save(checkpoint, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model state"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.load_state_dict(checkpoint['model_state_dict'])
        
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.config = checkpoint.get('config', self.config)
        self.model_version = checkpoint.get('model_version', self.model_version)
        self.is_trained = checkpoint.get('is_trained', False)
        self.training_history = checkpoint.get('training_history', {'loss': [], 'val_loss': []})
        self.prediction_horizons = checkpoint.get('prediction_horizons', [1, 7, 30])
        
        logger.info(f"Model loaded from {filepath}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get detailed model information"""
        base_info = super().get_model_info()
        
        lstm_info = {
            'architecture': {
                'input_size': self.input_size,
                'hidden_size': self.hidden_size,
                'num_layers': self.num_layers,
                'dropout': self.dropout,
                'bidirectional': self.bidirectional,
                'output_size': self.output_size
            },
            'prediction_horizons': self.prediction_horizons,
            'device': str(self.device),
            'training_history_length': len(self.training_history['loss'])
        }
        
        base_info.update(lstm_info)
        return base_info
