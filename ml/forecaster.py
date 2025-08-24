"""
FX Rate Forecasting Service with confidence intervals.

Provides easy-to-use interface for making exchange rate predictions
with uncertainty quantification for the decision engine.
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

from .models import LSTMForecaster, ModelConfig
from .features import FeatureEngineering, SequenceGenerator
from .trainer import ModelTrainer, TrainingConfig
from data.providers.base import FXRateData

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Result of FX rate prediction with uncertainty."""
    currency_pair: str
    current_rate: float
    predicted_rates: List[float]  # Predictions for each hour in horizon
    confidence_intervals: List[Tuple[float, float]]  # (lower, upper) bounds
    prediction_horizon_hours: int
    confidence_level: float = 0.95  # 95% confidence intervals
    model_confidence: float = 0.0  # Overall model confidence (0-1)
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
    
    @property
    def next_hour_prediction(self) -> float:
        """Get prediction for next hour (most reliable)."""
        return self.predicted_rates[0] if self.predicted_rates else self.current_rate
    
    @property
    def next_day_prediction(self) -> float:
        """Get prediction for 24 hours ahead."""
        if len(self.predicted_rates) >= 24:
            return self.predicted_rates[23]
        return self.predicted_rates[-1] if self.predicted_rates else self.current_rate
    
    @property
    def expected_change_24h(self) -> float:
        """Expected percentage change over next 24 hours."""
        if not self.predicted_rates or self.current_rate == 0:
            return 0.0
        
        future_rate = self.next_day_prediction
        return ((future_rate - self.current_rate) / self.current_rate) * 100
    
    @property
    def trend_direction(self) -> str:
        """Get trend direction: 'up', 'down', or 'stable'."""
        change = self.expected_change_24h
        if abs(change) < 0.1:  # Less than 0.1% change
            return 'stable'
        return 'up' if change > 0 else 'down'


class FXForecaster:
    """
    Main forecasting service for FX rates.
    
    Provides high-level interface for making predictions with confidence intervals,
    integrates with the data collection system, and manages model lifecycle.
    """
    
    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        self.models: Dict[str, ModelTrainer] = {}  # currency_pair -> trained model
        self.last_predictions: Dict[str, PredictionResult] = {}
        self.prediction_cache_ttl = 3600  # Cache predictions for 1 hour
        
        logger.info(f"Initialized FX Forecaster with model directory: {model_dir}")
    
    def load_model(self, currency_pair: str, model_path: Optional[str] = None) -> bool:
        """
        Load a trained model for a currency pair.
        
        Args:
            currency_pair: Currency pair (e.g., 'USD/EUR')
            model_path: Path to model file (optional, will use default if None)
            
        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            if model_path is None:
                model_path = f"{self.model_dir}/fx_forecaster_{currency_pair.replace('/', '_')}_complete.pkl"
            
            trainer = ModelTrainer(TrainingConfig())
            trainer.load_model(model_path)
            
            self.models[currency_pair] = trainer
            logger.info(f"Loaded model for {currency_pair}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model for {currency_pair}: {e}")
            return False
    
    def train_model(
        self, 
        currency_pair: str, 
        historical_data: List[FXRateData],
        config: Optional[TrainingConfig] = None
    ) -> bool:
        """
        Train a new model for a currency pair.
        
        Args:
            currency_pair: Currency pair to train for
            historical_data: Historical FX rate data
            config: Training configuration
            
        Returns:
            True if training successful, False otherwise
        """
        try:
            # Convert historical data to DataFrame
            df = self._fx_data_to_dataframe(historical_data, currency_pair)
            
            if len(df) < 500:  # Need sufficient data for testing (reduced from 1000)
                logger.warning(f"Insufficient data for {currency_pair}: {len(df)} samples")
                return False
            
            # Train model
            if config is None:
                config = TrainingConfig()
            
            trainer = ModelTrainer(config)
            X, y = trainer.prepare_data(df)
            result = trainer.train_model(X, y)
            
            # Save trained model
            model_path = f"{self.model_dir}/fx_forecaster_{currency_pair.replace('/', '_')}_complete.pkl"
            trainer.save_model(model_path)
            
            # Store in memory
            self.models[currency_pair] = trainer
            
            logger.info(f"Successfully trained model for {currency_pair}")
            logger.info(f"Final validation MAPE: {result.val_metrics.get('mape', 'N/A'):.2f}%")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to train model for {currency_pair}: {e}")
            return False
    
    def predict_next_24h(
        self, 
        currency_pair: str,
        current_data: List[FXRateData],
        confidence_level: float = 0.95
    ) -> Optional[PredictionResult]:
        """
        Predict exchange rates for the next 24 hours.
        
        Args:
            currency_pair: Currency pair to predict
            current_data: Recent FX rate data (at least 168 hours)
            confidence_level: Confidence level for intervals
            
        Returns:
            PredictionResult with predictions and confidence intervals
        """
        # Check if we have a trained model
        if currency_pair not in self.models:
            logger.error(f"No trained model available for {currency_pair}")
            return None
        
        # Check cache
        cache_key = currency_pair
        if cache_key in self.last_predictions:
            cached = self.last_predictions[cache_key]
            age_seconds = (datetime.utcnow() - cached.timestamp).total_seconds()
            if age_seconds < self.prediction_cache_ttl:
                logger.debug(f"Returning cached prediction for {currency_pair}")
                return cached
        
        try:
            trainer = self.models[currency_pair]
            
            # Convert current data to DataFrame
            df = self._fx_data_to_dataframe(current_data, currency_pair)
            
            if len(df) < trainer.config.model_config.sequence_length:
                logger.error(f"Insufficient recent data for {currency_pair}: {len(df)} samples needed")
                return None
            
            # Prepare features
            df_features = trainer.feature_engineer.prepare_features(df)
            feature_columns = trainer.feature_engineer.get_feature_columns(df_features)
            features_scaled = trainer.feature_engineer.transform_features(df_features)
            
            # Create sequence for prediction
            sequence = trainer.sequence_generator.create_single_sequence(features_scaled)
            
            # Make prediction with uncertainty
            trainer.model.eval()
            with torch.no_grad():
                sequence_tensor = torch.FloatTensor(sequence).to(trainer.device)
                
                # Get predictions with Monte Carlo uncertainty
                pred_mean, epistemic_unc, aleatoric_unc = trainer.model.predict_with_uncertainty(
                    sequence_tensor, num_samples=50
                )
                
                # Convert to numpy
                predictions = pred_mean.cpu().numpy().flatten()
                epistemic_uncertainty = epistemic_unc.cpu().numpy().flatten()
                aleatoric_uncertainty = aleatoric_unc.cpu().numpy().flatten()
            
            # Calculate confidence intervals
            total_uncertainty = np.sqrt(epistemic_uncertainty + aleatoric_uncertainty)
            z_score = 1.96 if confidence_level == 0.95 else 2.58  # 95% or 99%
            
            confidence_intervals = [
                (pred - z_score * unc, pred + z_score * unc)
                for pred, unc in zip(predictions, total_uncertainty)
            ]
            
            # Calculate overall model confidence
            avg_uncertainty = np.mean(total_uncertainty)
            model_confidence = max(0.0, min(1.0, 1.0 - avg_uncertainty))  # Normalize to [0,1]
            
            # Get current rate
            current_rate = current_data[-1].rate if current_data else df.iloc[-1]['rate']
            
            # Create result
            result = PredictionResult(
                currency_pair=currency_pair,
                current_rate=float(current_rate),
                predicted_rates=predictions.tolist(),
                confidence_intervals=confidence_intervals,
                prediction_horizon_hours=len(predictions),
                confidence_level=confidence_level,
                model_confidence=model_confidence,
                timestamp=datetime.utcnow()
            )
            
            # Cache result
            self.last_predictions[cache_key] = result
            
            logger.info(f"Generated prediction for {currency_pair}")
            logger.info(f"24h change: {result.expected_change_24h:.2f}%, confidence: {model_confidence:.2f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed for {currency_pair}: {e}")
            return None
    
    def predict_simple(self, currency_pair: str, current_data: List[FXRateData]) -> Dict[str, Any]:
        """
        Simple prediction interface for integration with decision engine.
        
        Returns:
            Dictionary with prediction data suitable for decision making
        """
        prediction = self.predict_next_24h(currency_pair, current_data)
        
        if not prediction:
            return {
                'success': False,
                'error': 'Prediction failed',
                'currency_pair': currency_pair
            }
        
        return {
            'success': True,
            'currency_pair': currency_pair,
            'current_rate': prediction.current_rate,
            'predicted_rate': prediction.next_day_prediction,
            'expected_change_percent': prediction.expected_change_24h,
            'trend_direction': prediction.trend_direction,
            'confidence': prediction.model_confidence,
            'confidence_interval': prediction.confidence_intervals[23] if len(prediction.confidence_intervals) > 23 else prediction.confidence_intervals[-1],
            'timestamp': prediction.timestamp
        }
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get status of all loaded models."""
        status = {
            'loaded_models': list(self.models.keys()),
            'total_models': len(self.models),
            'cache_entries': len(self.last_predictions),
            'model_details': {}
        }
        
        for currency_pair, trainer in self.models.items():
            status['model_details'][currency_pair] = {
                'sequence_length': trainer.config.model_config.sequence_length,
                'prediction_horizon': trainer.config.model_config.prediction_horizon,
                'input_features': trainer.config.model_config.input_features,
                'device': str(trainer.device)
            }
        
        return status
    
    def _fx_data_to_dataframe(self, fx_data: List[FXRateData], currency_pair: str) -> pd.DataFrame:
        """Convert FXRateData objects to DataFrame for processing."""
        data = []
        for rate_data in fx_data:
            data.append({
                'timestamp': rate_data.timestamp,
                'rate': float(rate_data.rate),
                'currency_pair': currency_pair,
                'provider': rate_data.provider,
                'bid': float(rate_data.bid) if rate_data.bid else None,
                'ask': float(rate_data.ask) if rate_data.ask else None,
                'volume': rate_data.volume
            })
        
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        return df
    
    def clear_cache(self) -> None:
        """Clear prediction cache."""
        self.last_predictions.clear()
        logger.info("Prediction cache cleared")
    
    def cleanup(self) -> None:
        """Clean up resources."""
        self.models.clear()
        self.clear_cache()
        logger.info("FX Forecaster cleanup completed")


# Convenience functions for easy usage
def create_mock_historical_data(currency_pair: str, days: int = 30) -> List[FXRateData]:
    """Create mock historical data for testing purposes."""
    from decimal import Decimal
    import random
    
    base_rates = {
        'USD/EUR': 0.865,
        'USD/GBP': 0.789,
        'EUR/GBP': 0.912,
        'USD/JPY': 150.0,
        'EUR/USD': 1.156
    }
    
    base_rate = base_rates.get(currency_pair, 1.0)
    data = []
    
    # Generate hourly data
    start_time = datetime.utcnow() - timedelta(days=days)
    
    for hour in range(days * 24):
        timestamp = start_time + timedelta(hours=hour)
        
        # Add some realistic variation
        daily_variation = 0.02 * np.sin(2 * np.pi * hour / (24 * 7))  # Weekly cycle
        random_walk = random.gauss(0, 0.001)  # Random walk
        hourly_pattern = 0.001 * np.sin(2 * np.pi * hour / 24)  # Daily pattern
        
        rate = base_rate * (1 + daily_variation + random_walk + hourly_pattern)
        
        data.append(FXRateData(
            currency_pair=currency_pair,
            rate=Decimal(str(rate)),
            bid=Decimal(str(rate * 0.9995)),
            ask=Decimal(str(rate * 1.0005)),
            timestamp=timestamp,
            provider="MockProvider"
        ))
    
    return data


def quick_train_and_predict(currency_pair: str = "USD/EUR") -> Optional[PredictionResult]:
    """
    Quick function to train a model and make a prediction for testing.
    
    This is useful for testing the complete ML pipeline.
    """
    logger.info(f"Quick training and prediction for {currency_pair}")
    
    # Create mock data (need at least 45 days for sufficient training data)
    historical_data = create_mock_historical_data(currency_pair, days=45)  # 45 days of hourly data
    
    # Initialize forecaster
    forecaster = FXForecaster()
    
    # Train model
    success = forecaster.train_model(currency_pair, historical_data)
    
    if not success:
        logger.error("Training failed")
        return None
    
    # Make prediction using recent data
    recent_data = historical_data[-200:]  # Last 200 hours for prediction
    
    prediction = forecaster.predict_next_24h(currency_pair, recent_data)
    
    return prediction