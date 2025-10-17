"""
Data preprocessing for ML pipeline
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import joblib

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Data preprocessing pipeline for ML models
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.scaler = None
        self.sequence_length = self.config.get('sequence_length', 60)
        self.scaling_method = self.config.get('feature_scaling', 'minmax')
        self.fitted = False
        
    def prepare_data(self, 
                    features: pd.DataFrame,
                    targets: pd.DataFrame,
                    test_size: float = 0.2,
                    validation_size: float = 0.2) -> Dict[str, np.ndarray]:
        """
        Prepare data for ML training with proper scaling and sequencing
        """
        logger.info("Preparing data for ML training")
        
        # Align features and targets
        common_index = features.index.intersection(targets.index)
        features = features.loc[common_index]
        targets = targets.loc[common_index]
        
        # Remove NaN values
        combined = pd.concat([features, targets], axis=1)
        combined = combined.dropna()
        
        features = combined[features.columns]
        targets = combined[targets.columns]
        
        logger.info(f"Data shape after cleaning: {features.shape}")
        
        # Scale features
        features_scaled = self._scale_features(features)
        
        # Create sequences
        X, y = self._create_sequences(features_scaled, targets.values)
        
        # Split data (time-aware)
        data_splits = self._split_data(X, y, test_size, validation_size)
        
        logger.info(f"Train: {data_splits['X_train'].shape}, "
                   f"Val: {data_splits['X_val'].shape}, "
                   f"Test: {data_splits['X_test'].shape}")
        
        return data_splits
    
    def _scale_features(self, features: pd.DataFrame) -> np.ndarray:
        """Scale features using configured method"""
        if self.scaler is None:
            if self.scaling_method == 'standard':
                self.scaler = StandardScaler()
            elif self.scaling_method == 'minmax':
                self.scaler = MinMaxScaler()
            elif self.scaling_method == 'robust':
                self.scaler = RobustScaler()
            else:
                raise ValueError(f"Unknown scaling method: {self.scaling_method}")
        
        if not self.fitted:
            features_scaled = self.scaler.fit_transform(features)
            
            # Save training feature names for consistency during prediction
            self.training_features = features.columns.tolist()
            
            self.fitted = True
            logger.info(f"Fitted {self.scaling_method} scaler on {len(self.training_features)} features")
        else:
            features_scaled = self.scaler.transform(features)
            logger.info("Applied existing scaler")
        
        return features_scaled
    
    def _create_sequences(self, 
                         features: np.ndarray, 
                         targets: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM input
        """
        logger.info(f"Creating sequences with length {self.sequence_length}")
        
        X, y = [], []
        
        for i in range(self.sequence_length, len(features)):
            # Features: sequence of past values
            X.append(features[i-self.sequence_length:i])
            # Targets: future values
            y.append(targets[i])
        
        X = np.array(X)
        y = np.array(y)
        
        logger.info(f"Created sequences - X: {X.shape}, y: {y.shape}")
        
        return X, y
    
    def _split_data(self, 
                   X: np.ndarray, 
                   y: np.ndarray,
                   test_size: float,
                   validation_size: float) -> Dict[str, np.ndarray]:
        """
        Split data maintaining temporal order
        """
        n_samples = len(X)
        
        # Calculate split indices
        test_start = int(n_samples * (1 - test_size))
        val_start = int(test_start * (1 - validation_size))
        
        # Time-aware splits (no shuffling)
        X_train = X[:val_start]
        y_train = y[:val_start]
        
        X_val = X[val_start:test_start]
        y_val = y[val_start:test_start]
        
        X_test = X[test_start:]
        y_test = y[test_start:]
        
        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test
        }
    
    def inverse_transform_predictions(self, 
                                    predictions: np.ndarray,
                                    feature_indices: List[int] = None) -> np.ndarray:
        """
        Inverse transform predictions if they were scaled
        """
        if self.scaler is None:
            return predictions
        
        # If predictions are for specific features (like close prices)
        if feature_indices is not None and hasattr(self.scaler, 'scale_'):
            # Create dummy array with original feature dimensions
            dummy = np.zeros((len(predictions), len(self.scaler.scale_)))
            
            # Place predictions in correct positions
            for i, idx in enumerate(feature_indices):
                if predictions.ndim == 1:
                    dummy[:, idx] = predictions
                else:
                    dummy[:, idx] = predictions[:, i]
            
            # Inverse transform
            inverse = self.scaler.inverse_transform(dummy)
            
            # Extract the relevant predictions
            if len(feature_indices) == 1:
                return inverse[:, feature_indices[0]]
            else:
                return inverse[:, feature_indices]
        
        return predictions
    
    def prepare_single_prediction(self, 
                                latest_features: pd.DataFrame) -> np.ndarray:
        """
        Prepare a single sequence for real-time prediction
        """
        if latest_features.shape[0] < self.sequence_length:
            raise ValueError(f"Need at least {self.sequence_length} samples for prediction")
        
        # Take the most recent sequence
        recent_features = latest_features.tail(self.sequence_length)
        
        # Ensure we have the same features as during training
        if hasattr(self, 'training_features') and self.training_features is not None:
            missing = set(self.training_features) - set(recent_features.columns)
            extra = set(recent_features.columns) - set(self.training_features)

            if missing:
                logger.warning(
                    "Incoming features missing %d trained columns: %s",
                    len(missing),
                    sorted(missing),
                )

            if extra:
                logger.warning(
                    "Incoming features contain %d unseen columns that will be dropped: %s",
                    len(extra),
                    sorted(extra),
                )

            recent_features = recent_features.reindex(columns=self.training_features, fill_value=0)
        
        # Scale features
        features_scaled = self.scaler.transform(recent_features)
        
        # Reshape for model input: (1, sequence_length, n_features)
        X = features_scaled.reshape(1, self.sequence_length, -1)
        
        return X
    
    def save_preprocessor(self, filepath: str):
        """Save the fitted preprocessor"""
        if not self.fitted:
            raise ValueError("Preprocessor must be fitted before saving")
        
        joblib.dump({
            'scaler': self.scaler,
            'config': self.config,
            'fitted': self.fitted,
            'training_features': getattr(self, 'training_features', None)
        }, filepath)
        
        logger.info(f"Saved preprocessor to {filepath}")
    
    def load_preprocessor(self, filepath: str):
        """Load a fitted preprocessor"""
        saved_data = joblib.load(filepath)
        
        self.scaler = saved_data['scaler']
        self.config.update(saved_data.get('config', {}))
        self.fitted = saved_data['fitted']
        self.training_features = saved_data.get('training_features', None)
        
        # Update sequence length from config
        self.sequence_length = self.config.get('sequence_length', 60)
        
        logger.info(f"Loaded preprocessor from {filepath}")
    
    def get_preprocessing_info(self) -> Dict[str, Any]:
        """Get information about the preprocessing setup"""
        info = {
            'fitted': self.fitted,
            'scaling_method': self.scaling_method,
            'sequence_length': self.sequence_length,
            'scaler_type': type(self.scaler).__name__ if self.scaler else None
        }
        
        if self.fitted and hasattr(self.scaler, 'scale_'):
            info.update({
                'n_features': len(self.scaler.scale_),
                'feature_scale_range': [float(self.scaler.scale_.min()), float(self.scaler.scale_.max())],
                'feature_mean': float(getattr(self.scaler, 'mean_', [0])[0]) if hasattr(self.scaler, 'mean_') else None
            })
        
        return info
    
    @staticmethod
    def validate_data_quality(features: pd.DataFrame, 
                            targets: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Validate data quality and return metrics
        """
        quality_report = {
            'features': {
                'shape': features.shape,
                'missing_values': features.isnull().sum().sum(),
                'missing_percentage': features.isnull().mean().mean() * 100,
                'infinite_values': np.isinf(features.select_dtypes(include=[np.number])).sum().sum(),
                'constant_columns': (features.nunique() <= 1).sum(),
                'memory_usage_mb': features.memory_usage(deep=True).sum() / 1024 / 1024
            }
        }
        
        if targets is not None:
            quality_report['targets'] = {
                'shape': targets.shape,
                'missing_values': targets.isnull().sum().sum(),
                'missing_percentage': targets.isnull().mean().mean() * 100,
                'infinite_values': np.isinf(targets.select_dtypes(include=[np.number])).sum().sum()
            }
        
        # Data recommendations
        recommendations = []
        if quality_report['features']['missing_percentage'] > 20:
            recommendations.append("High missing value percentage - consider data cleaning")
        if quality_report['features']['infinite_values'] > 0:
            recommendations.append("Infinite values detected - requires handling")
        if quality_report['features']['constant_columns'] > 0:
            recommendations.append("Constant columns detected - should be removed")
        
        quality_report['recommendations'] = recommendations
        
        return quality_report
