"""
Main prediction API for ML models
"""

import pandas as pd
import numpy as np
import logging
import time
from typing import Dict, List, Any
from datetime import datetime, timedelta

from ..config import MLConfig, load_ml_config
from ..models.lstm_model import LSTMModel
from ..utils.data_loader import Layer1DataLoader
from ..features.engineering import FeatureEngineer
from ..features.preprocessing import DataPreprocessor
from ..utils.model_storage import ModelStorage
from .confidence import ConfidenceCalculator
from .cache import PredictionCache
from .types import MLPredictionRequest, MLPredictionResponse

logger = logging.getLogger(__name__)


class MLPredictor:
    """
    Main ML prediction system with caching and confidence intervals
    """
    
    def __init__(self, config: MLConfig = None):
        self.config = config or load_ml_config()
        
        # Initialize components
        self.data_loader = Layer1DataLoader()
        self.feature_engineer = FeatureEngineer(self.config.features.__dict__)
        self.preprocessor = DataPreprocessor(self.config.model.__dict__)
        self.model_storage = ModelStorage(self.config.model_storage_path)
        self.confidence_calc = ConfidenceCalculator()
        self.cache = PredictionCache()
        
        # Cache for loaded models
        self.loaded_models: Dict[str, LSTMModel] = {}
        
        # Track auto-training to prevent excessive retraining
        self.auto_training_history: Dict[str, datetime] = {}
        
        logger.info("ML Predictor initialized")
    
    async def predict(self, request: MLPredictionRequest) -> MLPredictionResponse:
        """
        Make ML predictions for currency pair
        """
        start_time = time.time()
        
        # Set default horizons
        if request.horizons is None:
            request.horizons = self.config.model.prediction_horizons
        
        logger.info(f"Making prediction for {request.currency_pair}, horizons: {request.horizons}")
        
        # Check cache first
        if request.max_age_hours > 0:
            cached_result = self.cache.get_prediction(
                request.currency_pair, 
                request.horizons,
                max_age_hours=request.max_age_hours
            )
            
            if cached_result:
                logger.info(f"Using cached prediction for {request.currency_pair}")
                cached_result.cached = True
                return cached_result
        
        try:
            # Load or get model
            model = self._get_model(request.currency_pair)
            
            # Get latest features
            features_df = self._get_latest_features(request.currency_pair)
            
            # Prepare features for prediction
            X_pred = self.preprocessor.prepare_single_prediction(features_df)
            
            # Make prediction
            prediction_result = model.predict(X_pred, request.horizons)
            
            # Create response
            response = self._create_response(
                request, prediction_result, model, 
                features_df.shape[1], start_time
            )
            
            # Cache the result
            self.cache.store_prediction(request.currency_pair, response)
            
            return response
            
        except Exception as e:
            logger.error(f"Prediction failed for {request.currency_pair}: {e}")
            raise e
    
    def _get_model(self, currency_pair: str) -> LSTMModel:
        """Get or load model for currency pair"""
        # Check if model is already loaded
        model_key = f"lstm_{currency_pair}"
        
        if model_key in self.loaded_models:
            return self.loaded_models[model_key]
        
        # Load model from storage
        try:
            model = self.model_storage.load_model(
                currency_pair=currency_pair,
                model_type="lstm"
            )
            
            # Also load the associated preprocessor
            self._load_preprocessor_for_model(currency_pair, model_type="lstm")
            
            self.loaded_models[model_key] = model
            logger.info(f"Loaded LSTM model for {currency_pair}")
            return model
            
        except (ValueError, FileNotFoundError) as e:
            # No trained model available - auto-train one
            logger.warning(f"No trained model available for {currency_pair}: {e}")
            
            # Check if we've auto-trained recently to prevent excessive retraining
            last_auto_train = self.auto_training_history.get(currency_pair)
            if last_auto_train:
                hours_since_training = (datetime.now() - last_auto_train).total_seconds() / 3600
                if hours_since_training < 24:  # Don't auto-train more than once per day
                    logger.error(f"Auto-training for {currency_pair} was attempted {hours_since_training:.1f} hours ago. "
                                f"Waiting 24 hours before next attempt.")
                    raise ValueError(f"No trained model available for {currency_pair}. "
                                   f"Auto-training was recently attempted. Please wait or train manually.")
            
            logger.info(f"Auto-training new model for {currency_pair}...")
            
            try:
                # Record the training attempt
                self.auto_training_history[currency_pair] = datetime.now()
                
                # Auto-train using configuration values
                training_days = self.config.data.min_training_days
                logger.info(f"Using {training_days} days of training data from config")
                
                training_result = self.train_model(
                    currency_pair=currency_pair,
                    days=training_days,
                    save_model=True,
                    set_as_default=True
                )
                
                logger.info(f"Auto-training completed for {currency_pair}: {training_result['model_id']}")
                
                # Now load the newly trained model
                model = self.model_storage.load_model(
                    currency_pair=currency_pair,
                    model_type="lstm"
                )
                
                # Load the associated preprocessor
                self._load_preprocessor_for_model(currency_pair, model_type="lstm")
                
                self.loaded_models[model_key] = model
                logger.info(f"Successfully loaded auto-trained model for {currency_pair}")
                return model
                
            except Exception as train_error:
                logger.error(f"Auto-training failed for {currency_pair}: {train_error}")
                # Remove the failed attempt from history so it can be retried later
                if currency_pair in self.auto_training_history:
                    del self.auto_training_history[currency_pair]
                raise ValueError(f"No trained model available for {currency_pair} and auto-training failed: {train_error}")
    
    def _load_preprocessor_for_model(self, currency_pair: str, model_type: str = "lstm"):
        """Load the preprocessor associated with a model"""
        try:
            # Get the default model ID for this currency pair
            default_models = self.model_storage.registry['default_models'].get(currency_pair, {})
            model_id = default_models.get(model_type)
            
            if model_id is None:
                raise ValueError(f"No default {model_type} model found for {currency_pair}")
            
            # Load the preprocessor
            preprocessor_path = self.model_storage.storage_path / model_id / "preprocessor.pkl"
            
            if preprocessor_path.exists():
                self.preprocessor.load_preprocessor(str(preprocessor_path))
                logger.info(f"Loaded preprocessor for {currency_pair} model {model_id}")
            else:
                logger.warning(f"Preprocessor file not found: {preprocessor_path}")
                
        except Exception as e:
            logger.error(f"Failed to load preprocessor for {currency_pair}: {e}")
            raise
    
    def _get_latest_features(self, currency_pair: str) -> pd.DataFrame:
        """Get latest features for prediction"""
        # Load recent data (enough for sequence + feature calculation)
        days_needed = self.config.model.sequence_length + 30  # Buffer for indicators
        
        # Load data
        prices, indicators, economic_events = self.data_loader.get_combined_dataset(
            currency_pair=currency_pair,
            days=days_needed
        )
        
        # Engineer features
        features = self.feature_engineer.engineer_features(
            prices, indicators, economic_events
        )
        
        logger.info(f"Generated {features.shape[1]} features from {features.shape[0]} days of data")
        
        return features
    
    def _create_response(self, 
                        request: MLPredictionRequest,
                        prediction_result,
                        model: LSTMModel,
                        features_count: int,
                        start_time: float) -> MLPredictionResponse:
        """Create response from prediction result"""
        
        # Format predictions by horizon
        predictions = {}
        direction_probs = {}
        
        for i, horizon in enumerate(request.horizons):
            pred_value = float(prediction_result.predictions[i])
            confidence_intervals = prediction_result.confidence_intervals
            
            predictions[f"{horizon}d"] = {
                "mean": pred_value,
                "p10": float(confidence_intervals["p10"][i]),
                "p50": float(confidence_intervals["p50"][i]),
                "p90": float(confidence_intervals["p90"][i])
            }
            
            direction_probs[f"{horizon}d"] = float(prediction_result.direction_probabilities[i])
        
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        
        return MLPredictionResponse(
            currency_pair=request.currency_pair,
            timestamp=prediction_result.timestamp,
            model_id=model.get_model_info()['model_version'],
            model_confidence=prediction_result.model_confidence,
            predictions=predictions,
            direction_probabilities=direction_probs,
            features_used=features_count,
            processing_time_ms=processing_time
        )
    
    def train_model(self, 
                   currency_pair: str,
                   days: int = 365,
                   save_model: bool = True,
                   set_as_default: bool = True) -> Dict[str, Any]:
        """
        Train a new LSTM model for a currency pair
        """
        logger.info(f"Training new LSTM model for {currency_pair}")
        
        # Ensure we have enough training data (minimum based on sequence length)
        min_days_needed = max(days, self.config.model.sequence_length * 3, 200)
        
        # Load training data
        prices, indicators, economic_events = self.data_loader.get_combined_dataset(
            currency_pair=currency_pair,
            days=min_days_needed
        )
        
        # Engineer features
        features = self.feature_engineer.engineer_features(
            prices, indicators, economic_events
        )
        
        # Create targets
        targets = self.feature_engineer.get_target_variables(
            prices, self.config.model.prediction_horizons
        )
        
        # Prepare data for training
        data_splits = self.preprocessor.prepare_data(
            features, targets,
            test_size=self.config.model.train_test_split,
            validation_size=self.config.model.validation_split
        )
        
        # Initialize model
        model_config = self.config.model.__dict__.copy()
        model_config['input_size'] = features.shape[1]
        model_config['device'] = self.config.get_device()
        
        model = LSTMModel(model_config)
        
        # Train model
        training_history = model.fit(
            data_splits['X_train'], data_splits['y_train'],
            data_splits['X_val'], data_splits['y_val']
        )
        
        # Evaluate model
        metrics = model.evaluate(data_splits['X_test'], data_splits['y_test'])
        
        # Save model if requested
        model_id = None
        if save_model:
            # Generate model name based on versioning configuration
            if self.config.model_versioning == "timestamp":
                model_name = f"LSTM_{currency_pair}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            else:  # "overwrite"
                model_name = f"LSTM_{currency_pair}"
            
            model_id = self.model_storage.save_model(
                model=model,
                model_name=model_name,
                currency_pair=currency_pair,
                model_type="lstm",
                metadata={
                    'training_samples': len(data_splits['X_train']),
                    'features_count': features.shape[1],
                    'performance_metrics': metrics,
                    'training_history': training_history,
                    'data_days': days
                },
                set_as_default=set_as_default
            )
            
            # Save preprocessor
            preprocessor_path = self.model_storage.storage_path / model_id / "preprocessor.pkl"
            self.preprocessor.save_preprocessor(str(preprocessor_path))
        
        result = {
            'model_id': model_id,
            'currency_pair': currency_pair,
            'training_samples': len(data_splits['X_train']),
            'features_count': features.shape[1],
            'performance_metrics': metrics,
            'training_history': training_history
        }
        
        logger.info(f"Training completed for {currency_pair}: {metrics}")
        
        return result
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available trained models"""
        return self.model_storage.list_models()
    
    def get_model_performance(self, model_id: str = None, currency_pair: str = None) -> Dict[str, Any]:
        """Get performance metrics for a model"""
        if model_id is None and currency_pair is not None:
            # Get default model for currency pair
            models = self.model_storage.list_models(currency_pair=currency_pair, model_type="lstm")
            default_models = [m for m in models if m.get('is_default', False)]
            
            if not default_models:
                raise ValueError(f"No default model found for {currency_pair}")
            
            model_id = default_models[0]['model_id']
        
        if model_id is None:
            raise ValueError("Must specify either model_id or currency_pair")
        
        return self.model_storage.get_model_performance(model_id)
    
    def validate_prediction_quality(self, 
                                  currency_pair: str,
                                  days_back: int = 30) -> Dict[str, Any]:
        """
        Validate prediction quality by comparing historical predictions with actual outcomes
        """
        logger.info(f"Validating prediction quality for {currency_pair}")
        
        # Get historical data
        end_date = datetime.now() - timedelta(days=days_back)
        prices, _, _ = self.data_loader.get_combined_dataset(
            currency_pair=currency_pair,
            days=days_back + 60  # Extra for features
        )
        
        # Simulate historical predictions
        validation_results = []
        
        for i in range(days_back):
            date = end_date + timedelta(days=i)
            
            try:
                # Get features up to this date
                historical_data = prices[:date]
                if len(historical_data) < self.config.model.sequence_length:
                    continue
                
                # Make prediction (simplified - would need full feature pipeline)
                # This is a placeholder for the actual validation logic
                actual_returns = []
                predicted_returns = []
                
                for horizon in self.config.model.prediction_horizons:
                    future_date = date + timedelta(days=horizon)
                    if future_date in prices.index:
                        actual_return = prices.loc[future_date, 'close'] / prices.loc[date, 'close'] - 1
                        actual_returns.append(actual_return)
                        
                        # Placeholder prediction
                        predicted_returns.append(0.001)  # Would be actual prediction
                
                validation_results.append({
                    'date': date.isoformat(),
                    'actual_returns': actual_returns,
                    'predicted_returns': predicted_returns
                })
                
            except Exception as e:
                logger.warning(f"Validation failed for {date}: {e}")
                continue
        
        # Calculate validation metrics
        if validation_results:
            all_actual = []
            all_predicted = []
            
            for result in validation_results:
                all_actual.extend(result['actual_returns'])
                all_predicted.extend(result['predicted_returns'])
            
            # Calculate metrics
            if all_actual and all_predicted:
                actual_array = np.array(all_actual)
                predicted_array = np.array(all_predicted)
                
                mse = np.mean((actual_array - predicted_array) ** 2)
                mae = np.mean(np.abs(actual_array - predicted_array))
                
                # Directional accuracy
                actual_directions = actual_array > 0
                predicted_directions = predicted_array > 0
                directional_accuracy = np.mean(actual_directions == predicted_directions)
                
                metrics = {
                    'validation_samples': len(all_actual),
                    'mse': float(mse),
                    'mae': float(mae),
                    'directional_accuracy': float(directional_accuracy),
                    'validation_period_days': days_back
                }
            else:
                metrics = {'error': 'No valid validation samples'}
        else:
            metrics = {'error': 'No validation results generated'}
        
        return metrics
    
    def get_feature_importance(self, currency_pair: str) -> Dict[str, float]:
        """Get feature importance for a currency pair's model"""
        model = self._get_model(currency_pair)
        features_df = self._get_latest_features(currency_pair)
        
        # Get a sample for importance calculation
        X_sample = self.preprocessor.prepare_single_prediction(features_df)
        
        # Calculate importance
        importance_scores = model.get_feature_importance(X_sample)
        
        # Map to feature names
        feature_names = self.feature_engineer.feature_names
        if len(feature_names) != len(importance_scores):
            logger.warning("Feature names and importance scores length mismatch")
            feature_names = [f"feature_{i}" for i in range(len(importance_scores))]
        
        feature_importance = dict(zip(feature_names, importance_scores))
        
        # Sort by importance
        sorted_importance = dict(sorted(feature_importance.items(), 
                                      key=lambda x: x[1], reverse=True))
        
        return sorted_importance
    
    def cleanup_cache(self, max_age_hours: int = 24):
        """Clean up old cached predictions"""
        self.cache.cleanup_old_predictions(max_age_hours)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        return {
            'models_loaded': len(self.loaded_models),
            'available_models': len(self.model_storage.list_models()),
            'cache_size': len(self.cache.cache),
            'storage_stats': self.model_storage.get_storage_stats(),
            'config': {
                'sequence_length': self.config.model.sequence_length,
                'prediction_horizons': self.config.model.prediction_horizons,
                'device': self.config.get_device()
            }
        }