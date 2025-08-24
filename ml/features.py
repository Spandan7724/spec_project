"""
Feature engineering pipeline for FX rate forecasting.

Transforms raw exchange rate data into features suitable for LSTM training,
including technical indicators, time-based features, and lag variables.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler, RobustScaler
import logging

logger = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    """Configuration for feature engineering pipeline."""
    # Technical indicators (reduced periods for testing)
    rsi_period: int = 6
    ma_short_period: int = 6
    ma_long_period: int = 12
    bb_period: int = 10
    bb_std_dev: float = 2.0
    
    # Volatility features
    volatility_windows: List[int] = None
    
    # Lag features
    lag_periods: List[int] = None
    
    # Time features
    include_time_features: bool = True
    
    # Scaling
    scaler_type: str = "robust"  # "standard" or "robust"
    
    def __post_init__(self):
        if self.volatility_windows is None:
            self.volatility_windows = [6, 12, 24]  # 6h, 12h, 24h (reduced for testing)
        
        if self.lag_periods is None:
            self.lag_periods = [1, 2, 3, 6]  # Reduced lag periods for testing


class FeatureEngineering:
    """
    Feature engineering pipeline for FX time series data.
    
    Transforms raw exchange rate data into ML-ready features including:
    - Technical indicators (RSI, Moving Averages, Bollinger Bands)
    - Time-based features (hour, day of week, etc.)
    - Lag variables and returns
    - Volatility measures
    """
    
    def __init__(self, config: FeatureConfig):
        self.config = config
        self.scaler = None
        self.feature_names: List[str] = []
        self.is_fitted = False
    
    def create_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create technical indicators from OHLC data.
        
        Args:
            df: DataFrame with columns ['rate', 'timestamp']
            
        Returns:
            DataFrame with additional technical indicator columns
        """
        result = df.copy()
        
        # RSI (Relative Strength Index)
        result['rsi'] = self._calculate_rsi(result['rate'], self.config.rsi_period)
        
        # Moving averages
        result['ma_short'] = result['rate'].rolling(window=self.config.ma_short_period).mean()
        result['ma_long'] = result['rate'].rolling(window=self.config.ma_long_period).mean()
        result['ma_ratio'] = result['ma_short'] / result['ma_long']
        
        # Bollinger Bands
        bb_mean = result['rate'].rolling(window=self.config.bb_period).mean()
        bb_std = result['rate'].rolling(window=self.config.bb_period).std()
        result['bb_upper'] = bb_mean + (bb_std * self.config.bb_std_dev)
        result['bb_lower'] = bb_mean - (bb_std * self.config.bb_std_dev)
        result['bb_position'] = (result['rate'] - result['bb_lower']) / (result['bb_upper'] - result['bb_lower'])
        
        # Price momentum
        result['momentum_24h'] = result['rate'].pct_change(periods=24)
        result['momentum_12h'] = result['rate'].pct_change(periods=12)
        result['momentum_6h'] = result['rate'].pct_change(periods=6)
        
        return result
    
    def create_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create volatility-based features."""
        result = df.copy()
        
        # Returns for volatility calculation
        result['returns'] = result['rate'].pct_change()
        
        # Rolling volatilities
        for window in self.config.volatility_windows:
            col_name = f'volatility_{window}h'
            result[col_name] = result['returns'].rolling(window=window).std()
        
        # Volatility of volatility (second-order)
        result['vol_of_vol'] = result['volatility_24h'].rolling(window=24).std()
        
        # Realized vs implied volatility proxy
        result['vol_ratio'] = result['volatility_6h'] / (result['volatility_24h'] + 1e-8)
        
        return result
    
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based cyclical features."""
        if not self.config.include_time_features:
            return df
        
        result = df.copy()
        
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(result['timestamp']):
            result['timestamp'] = pd.to_datetime(result['timestamp'])
        
        # Hour of day (cyclical encoding)
        result['hour_sin'] = np.sin(2 * np.pi * result['timestamp'].dt.hour / 24)
        result['hour_cos'] = np.cos(2 * np.pi * result['timestamp'].dt.hour / 24)
        
        # Day of week (cyclical encoding)
        result['dow_sin'] = np.sin(2 * np.pi * result['timestamp'].dt.dayofweek / 7)
        result['dow_cos'] = np.cos(2 * np.pi * result['timestamp'].dt.dayofweek / 7)
        
        # Month of year (for seasonal patterns)
        result['month_sin'] = np.sin(2 * np.pi * result['timestamp'].dt.month / 12)
        result['month_cos'] = np.cos(2 * np.pi * result['timestamp'].dt.month / 12)
        
        # Market session indicators
        result['is_market_hours'] = self._is_market_hours(result['timestamp'])
        result['is_weekend'] = (result['timestamp'].dt.dayofweek >= 5).astype(int)
        
        return result
    
    def create_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create lagged features and returns."""
        result = df.copy()
        
        # Lagged rates
        for lag in self.config.lag_periods:
            result[f'rate_lag_{lag}'] = result['rate'].shift(lag)
        
        # Lagged returns
        returns = result['rate'].pct_change()
        for lag in self.config.lag_periods:
            result[f'returns_lag_{lag}'] = returns.shift(lag)
        
        # Lagged technical indicators
        if 'rsi' in result.columns:
            for lag in [1, 6, 12]:
                result[f'rsi_lag_{lag}'] = result['rsi'].shift(lag)
        
        return result
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Complete feature engineering pipeline.
        
        Args:
            df: Raw DataFrame with 'rate' and 'timestamp' columns
            
        Returns:
            DataFrame with engineered features
        """
        logger.info("Starting feature engineering pipeline")
        
        # Ensure data is sorted by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Create all feature types
        df = self.create_technical_indicators(df)
        df = self.create_volatility_features(df)
        df = self.create_time_features(df)
        df = self.create_lag_features(df)
        
        # Remove rows with NaN values (from indicators and lags)
        initial_rows = len(df)
        df = df.dropna()
        final_rows = len(df)
        
        logger.info(f"Feature engineering complete: {initial_rows} -> {final_rows} rows")
        
        return df
    
    def fit_scaler(self, df: pd.DataFrame, feature_columns: List[str]) -> None:
        """Fit the scaler on training data."""
        if self.config.scaler_type == "standard":
            self.scaler = StandardScaler()
        else:
            self.scaler = RobustScaler()
        
        self.scaler.fit(df[feature_columns])
        self.feature_names = feature_columns
        self.is_fitted = True
        
        logger.info(f"Fitted {self.config.scaler_type} scaler on {len(feature_columns)} features")
    
    def transform_features(self, df: pd.DataFrame) -> np.ndarray:
        """Transform features using fitted scaler."""
        if not self.is_fitted:
            raise ValueError("Scaler not fitted. Call fit_scaler() first.")
        
        return self.scaler.transform(df[self.feature_names])
    
    def fit_transform_features(self, df: pd.DataFrame, feature_columns: List[str]) -> np.ndarray:
        """Fit scaler and transform features in one step."""
        self.fit_scaler(df, feature_columns)
        return self.transform_features(df)
    
    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Get list of feature columns (excluding target and metadata)."""
        exclude_cols = ['timestamp', 'rate', 'currency_pair', 'provider', 'bid', 'ask', 'volume']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        return feature_cols
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _is_market_hours(self, timestamps: pd.Series) -> pd.Series:
        """
        Determine if timestamp falls within major market hours.
        Simplified: Monday-Friday, 8 AM - 6 PM UTC
        """
        is_weekday = timestamps.dt.dayofweek < 5
        is_business_hour = (timestamps.dt.hour >= 8) & (timestamps.dt.hour <= 18)
        return (is_weekday & is_business_hour).astype(int)


class SequenceGenerator:
    """
    Generate sequences for LSTM training from time series data.
    
    Creates input-output pairs with sliding window approach.
    """
    
    def __init__(self, sequence_length: int = 168, prediction_horizon: int = 24):
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
    
    def create_sequences(
        self, 
        features: np.ndarray, 
        targets: np.ndarray,
        stride: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create input-output sequences for LSTM training.
        
        Args:
            features: Feature matrix of shape (n_samples, n_features)
            targets: Target values of shape (n_samples,)
            stride: Step size for sliding window
            
        Returns:
            Tuple of (X, y) where:
            - X: Input sequences of shape (n_sequences, sequence_length, n_features)
            - y: Target sequences of shape (n_sequences, prediction_horizon)
        """
        n_samples, n_features = features.shape
        
        # Calculate number of sequences we can create
        n_sequences = (n_samples - self.sequence_length - self.prediction_horizon + 1) // stride
        
        X = np.zeros((n_sequences, self.sequence_length, n_features))
        y = np.zeros((n_sequences, self.prediction_horizon))
        
        for i in range(n_sequences):
            start_idx = i * stride
            end_idx = start_idx + self.sequence_length
            target_start = end_idx
            target_end = target_start + self.prediction_horizon
            
            X[i] = features[start_idx:end_idx]
            y[i] = targets[target_start:target_end]
        
        return X, y
    
    def create_single_sequence(self, features: np.ndarray) -> np.ndarray:
        """
        Create a single sequence for prediction (no target needed).
        
        Args:
            features: Feature matrix of shape (sequence_length, n_features)
            
        Returns:
            Input sequence of shape (1, sequence_length, n_features)
        """
        if len(features) < self.sequence_length:
            raise ValueError(f"Not enough data points. Need {self.sequence_length}, got {len(features)}")
        
        # Take the last sequence_length rows
        sequence = features[-self.sequence_length:]
        return sequence.reshape(1, self.sequence_length, -1)


def prepare_training_data(
    df: pd.DataFrame,
    target_column: str = 'rate',
    feature_config: Optional[FeatureConfig] = None,
    sequence_length: int = 168,
    prediction_horizon: int = 24
) -> Tuple[np.ndarray, np.ndarray, FeatureEngineering, SequenceGenerator]:
    """
    Complete data preparation pipeline for LSTM training.
    
    Args:
        df: Raw DataFrame with FX data
        target_column: Name of target column
        feature_config: Feature engineering configuration
        sequence_length: Length of input sequences
        prediction_horizon: Number of time steps to predict
        
    Returns:
        Tuple of (X, y, feature_engineer, sequence_generator)
    """
    if feature_config is None:
        feature_config = FeatureConfig()
    
    # Feature engineering
    feature_engineer = FeatureEngineering(feature_config)
    df_features = feature_engineer.prepare_features(df)
    
    # Get feature columns and fit scaler
    feature_columns = feature_engineer.get_feature_columns(df_features)
    features_scaled = feature_engineer.fit_transform_features(df_features, feature_columns)
    
    # Extract targets
    targets = df_features[target_column].values
    
    # Generate sequences
    sequence_generator = SequenceGenerator(sequence_length, prediction_horizon)
    X, y = sequence_generator.create_sequences(features_scaled, targets)
    
    logger.info(f"Generated {len(X)} training sequences")
    logger.info(f"Input shape: {X.shape}, Target shape: {y.shape}")
    
    return X, y, feature_engineer, sequence_generator