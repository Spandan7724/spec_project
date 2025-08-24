"""
Model training pipeline with walk-forward validation for FX forecasting.

Implements time-series appropriate validation, model training with early stopping,
and performance evaluation for LSTM forecasting models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import pickle
import os
from pathlib import Path

from .models import LSTMForecaster, ModelConfig, EarlyStopping, gaussian_nll_loss, mape_loss, ModelEvaluator
from .features import FeatureEngineering, SequenceGenerator, prepare_training_data, FeatureConfig

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    # Model parameters
    model_config: ModelConfig = field(default_factory=ModelConfig)
    feature_config: FeatureConfig = field(default_factory=FeatureConfig)
    
    # Training parameters
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    gradient_clip_norm: float = 1.0
    
    # Validation parameters
    validation_split: float = 0.2
    walk_forward_steps: int = 24  # Retrain every 24 hours
    min_train_size: int = 300  # Minimum samples for training (reduced for testing)
    
    # Early stopping
    patience: int = 15
    min_delta: float = 1e-6
    
    # Training loop
    max_epochs: int = 20  # Reduced for testing
    batch_size: int = 16  # Reduced for testing
    num_workers: int = 2
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Saving
    save_dir: str = "models"
    model_name: str = "fx_forecaster"


@dataclass
class TrainingResult:
    """Results from model training."""
    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    train_metrics: Dict[str, float] = field(default_factory=dict)
    val_metrics: Dict[str, float] = field(default_factory=dict)
    best_epoch: int = 0
    training_time: float = 0.0
    converged: bool = False


class ModelTrainer:
    """
    Trainer for LSTM forecasting models with time-series validation.
    
    Features:
    - Walk-forward validation for time series data
    - Early stopping and model checkpointing
    - Comprehensive metrics evaluation
    - Uncertainty-aware training with Gaussian NLL loss
    """
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.model: Optional[LSTMForecaster] = None
        self.feature_engineer: Optional[FeatureEngineering] = None
        self.sequence_generator: Optional[SequenceGenerator] = None
        
        # Ensure save directory exists
        Path(config.save_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized ModelTrainer on device: {self.device}")
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for training using feature engineering pipeline.
        
        Args:
            df: Raw DataFrame with FX rate data
            
        Returns:
            Tuple of (X, y) training arrays
        """
        logger.info("Preparing training data...")
        
        X, y, feature_engineer, sequence_generator = prepare_training_data(
            df=df,
            target_column='rate',
            feature_config=self.config.feature_config,
            sequence_length=self.config.model_config.sequence_length,
            prediction_horizon=self.config.model_config.prediction_horizon
        )
        
        # Store for later use in prediction
        self.feature_engineer = feature_engineer
        self.sequence_generator = sequence_generator
        
        # Update model config with actual input features
        self.config.model_config.input_features = X.shape[2]
        
        logger.info(f"Data preparation complete. Sequences: {X.shape[0]}, Features: {X.shape[2]}")
        
        return X, y
    
    def create_data_loaders(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> Tuple[DataLoader, Optional[DataLoader]]:
        """Create PyTorch data loaders."""
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train)
        
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True if self.config.device == "cuda" else False
        )
        
        val_loader = None
        if X_val is not None and y_val is not None:
            X_val_tensor = torch.FloatTensor(X_val)
            y_val_tensor = torch.FloatTensor(y_val)
            
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
                pin_memory=True if self.config.device == "cuda" else False
            )
        
        return train_loader, val_loader
    
    def train_epoch(
        self, 
        model: LSTMForecaster,
        train_loader: DataLoader,
        optimizer: optim.Optimizer,
        criterion: nn.Module
    ) -> float:
        """Train model for one epoch."""
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)
            
            optimizer.zero_grad()
            
            # Forward pass
            pred_mean, pred_var = model(batch_X)
            
            # Calculate loss (combination of MSE and NLL)
            mse_loss = nn.MSELoss()(pred_mean, batch_y)
            nll_loss = gaussian_nll_loss(batch_y, pred_mean, pred_var)
            loss = 0.7 * mse_loss + 0.3 * nll_loss
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.gradient_clip_norm)
            
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def validate_epoch(
        self,
        model: LSTMForecaster,
        val_loader: DataLoader,
        criterion: nn.Module
    ) -> Tuple[float, Dict[str, float]]:
        """Validate model for one epoch."""
        model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                pred_mean, pred_var = model(batch_X)
                
                # Calculate loss
                mse_loss = nn.MSELoss()(pred_mean, batch_y)
                nll_loss = gaussian_nll_loss(batch_y, pred_mean, pred_var)
                loss = 0.7 * mse_loss + 0.3 * nll_loss
                
                total_loss += loss.item()
                
                # Store for metrics calculation
                all_predictions.append(pred_mean.cpu().numpy())
                all_targets.append(batch_y.cpu().numpy())
        
        # Calculate comprehensive metrics
        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        
        # For metrics, use only the first prediction step (24h ahead -> 1h ahead)
        metrics = ModelEvaluator.calculate_metrics(
            targets[:, 0], predictions[:, 0]
        )
        
        avg_loss = total_loss / len(val_loader)
        return avg_loss, metrics
    
    def train_model(self, X: np.ndarray, y: np.ndarray) -> TrainingResult:
        """
        Train LSTM model with validation split.
        
        Args:
            X: Input sequences of shape (n_samples, sequence_length, n_features)
            y: Target sequences of shape (n_samples, prediction_horizon)
            
        Returns:
            TrainingResult with training history and metrics
        """
        start_time = datetime.now()
        result = TrainingResult()
        
        # Split data for validation
        split_idx = int(len(X) * (1 - self.config.validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        logger.info(f"Train samples: {len(X_train)}, Validation samples: {len(X_val)}")
        
        # Create data loaders
        train_loader, val_loader = self.create_data_loaders(X_train, y_train, X_val, y_val)
        
        # Initialize model
        self.model = LSTMForecaster(self.config.model_config).to(self.device)
        
        # Optimizer and scheduler
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Early stopping
        early_stopping = EarlyStopping(
            patience=self.config.patience,
            min_delta=self.config.min_delta
        )
        
        # Criterion (will be calculated in training loop)
        criterion = nn.MSELoss()
        
        # Training loop
        best_val_loss = float('inf')
        
        logger.info("Starting training...")
        
        for epoch in range(self.config.max_epochs):
            # Train
            train_loss = self.train_epoch(self.model, train_loader, optimizer, criterion)
            result.train_losses.append(train_loss)
            
            # Validate
            val_loss, val_metrics = self.validate_epoch(self.model, val_loader, criterion)
            result.val_losses.append(val_loss)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Log progress
            if epoch % 10 == 0 or epoch == self.config.max_epochs - 1:
                logger.info(
                    f"Epoch {epoch:3d}: Train Loss: {train_loss:.6f}, "
                    f"Val Loss: {val_loss:.6f}, Val MAPE: {val_metrics['mape']:.2f}%"
                )
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                result.best_epoch = epoch
                result.val_metrics = val_metrics
                self._save_checkpoint(self.model, epoch, val_loss, val_metrics)
            
            # Early stopping
            if early_stopping(val_loss):
                logger.info(f"Early stopping at epoch {epoch}")
                result.converged = True
                break
        
        # Training complete
        end_time = datetime.now()
        result.training_time = (end_time - start_time).total_seconds()
        
        # Load best model
        self._load_best_checkpoint()
        
        logger.info(f"Training completed in {result.training_time:.2f}s")
        logger.info(f"Best validation loss: {best_val_loss:.6f} at epoch {result.best_epoch}")
        
        return result
    
    def walk_forward_validation(
        self, 
        df: pd.DataFrame,
        currency_pair: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform walk-forward validation on historical data.
        
        This simulates real trading conditions by training on past data
        and testing on future data, progressively moving forward in time.
        """
        logger.info(f"Starting walk-forward validation for {currency_pair}")
        
        # Filter data for currency pair and date range
        df_pair = df[df['currency_pair'] == currency_pair].copy()
        
        if start_date:
            df_pair = df_pair[df_pair['timestamp'] >= start_date]
        if end_date:
            df_pair = df_pair[df_pair['timestamp'] <= end_date]
        
        df_pair = df_pair.sort_values('timestamp').reset_index(drop=True)
        
        # Prepare initial data
        X, y = self.prepare_data(df_pair)
        
        # Walk-forward validation parameters
        min_train_samples = max(self.config.min_train_size, self.config.model_config.sequence_length * 2)
        step_size = self.config.walk_forward_steps
        
        results = {
            'predictions': [],
            'actuals': [],
            'timestamps': [],
            'metrics_history': [],
            'training_history': []
        }
        
        current_start = min_train_samples
        
        while current_start + step_size < len(X):
            current_end = current_start + step_size
            
            logger.info(f"Training on samples 0:{current_start}, testing on {current_start}:{current_end}")
            
            # Split data
            X_train, y_train = X[:current_start], y[:current_start]
            X_test, y_test = X[current_start:current_end], y[current_start:current_end]
            
            # Train model on this window
            training_result = self.train_model(X_train, y_train)
            results['training_history'].append(training_result)
            
            # Make predictions on test set
            self.model.eval()
            with torch.no_grad():
                X_test_tensor = torch.FloatTensor(X_test).to(self.device)
                pred_mean, pred_var = self.model(X_test_tensor)
                predictions = pred_mean.cpu().numpy()
            
            # Store results
            results['predictions'].extend(predictions[:, 0])  # First prediction step
            results['actuals'].extend(y_test[:, 0])
            
            # Calculate metrics for this window
            window_metrics = ModelEvaluator.calculate_metrics(y_test[:, 0], predictions[:, 0])
            results['metrics_history'].append(window_metrics)
            
            logger.info(f"Window MAPE: {window_metrics['mape']:.2f}%")
            
            # Move to next window
            current_start += step_size
        
        # Calculate overall metrics
        overall_metrics = ModelEvaluator.calculate_metrics(
            np.array(results['actuals']),
            np.array(results['predictions'])
        )
        
        results['overall_metrics'] = overall_metrics
        
        logger.info("Walk-forward validation complete")
        logger.info(f"Overall MAPE: {overall_metrics['mape']:.2f}%")
        logger.info(f"Overall Directional Accuracy: {overall_metrics['directional_accuracy']:.2f}%")
        
        return results
    
    def _save_checkpoint(self, model: LSTMForecaster, epoch: int, val_loss: float, metrics: Dict[str, float]):
        """Save model checkpoint."""
        checkpoint_path = Path(self.config.save_dir) / f"{self.config.model_name}_best.pth"
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'val_loss': val_loss,
            'metrics': metrics,
            'model_config': self.config.model_config,
            'feature_engineer': self.feature_engineer,
            'sequence_generator': self.sequence_generator
        }, checkpoint_path)
        
        logger.debug(f"Saved checkpoint at epoch {epoch}")
    
    def _load_best_checkpoint(self):
        """Load the best model checkpoint."""
        checkpoint_path = Path(self.config.save_dir) / f"{self.config.model_name}_best.pth"
        
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded best checkpoint from epoch {checkpoint['epoch']}")
        else:
            logger.warning("No checkpoint found, using current model state")
    
    def save_model(self, filepath: Optional[str] = None) -> str:
        """Save the complete trained model."""
        if filepath is None:
            filepath = Path(self.config.save_dir) / f"{self.config.model_name}_complete.pkl"
        
        model_data = {
            'model_state_dict': self.model.state_dict(),
            'model_config': self.config.model_config,
            'feature_engineer': self.feature_engineer,
            'sequence_generator': self.sequence_generator,
            'training_config': self.config
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Saved complete model to {filepath}")
        return str(filepath)
    
    def load_model(self, filepath: str) -> None:
        """Load a complete trained model."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Restore components
        self.config.model_config = model_data['model_config']
        self.feature_engineer = model_data['feature_engineer']
        self.sequence_generator = model_data['sequence_generator']
        
        # Recreate and load model
        self.model = LSTMForecaster(self.config.model_config).to(self.device)
        self.model.load_state_dict(model_data['model_state_dict'])
        self.model.eval()
        
        logger.info(f"Loaded complete model from {filepath}")


def train_fx_model(
    df: pd.DataFrame,
    currency_pair: str,
    config: Optional[TrainingConfig] = None
) -> Tuple[ModelTrainer, TrainingResult]:
    """
    Convenience function to train a model for a specific currency pair.
    
    Args:
        df: DataFrame with FX rate data
        currency_pair: Currency pair to train on (e.g., 'USD/EUR')
        config: Training configuration
        
    Returns:
        Tuple of (trained_trainer, training_result)
    """
    if config is None:
        config = TrainingConfig()
    
    # Filter data for currency pair
    df_pair = df[df['currency_pair'] == currency_pair].copy()
    df_pair = df_pair.sort_values('timestamp').reset_index(drop=True)
    
    logger.info(f"Training model for {currency_pair} with {len(df_pair)} samples")
    
    # Initialize trainer
    trainer = ModelTrainer(config)
    
    # Prepare data and train
    X, y = trainer.prepare_data(df_pair)
    result = trainer.train_model(X, y)
    
    return trainer, result