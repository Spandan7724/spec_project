"""
Configuration management for ML prediction system
"""

import os
from dataclasses import dataclass
from typing import List
from pathlib import Path


@dataclass
class ModelConfig:
    """Configuration for model architecture"""
    # LSTM Architecture
    input_size: int = 30  # Number of features
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    bidirectional: bool = False
    
    # Training
    batch_size: int = 32
    learning_rate: float = 0.001
    epochs: int = 100
    patience: int = 15  # Early stopping
    
    # Data
    sequence_length: int = 60  # 60-day lookback
    prediction_horizons: List[int] = None  # [1, 7, 30] days
    train_test_split: float = 0.8
    validation_split: float = 0.2
    
    def __post_init__(self):
        if self.prediction_horizons is None:
            self.prediction_horizons = [1, 7, 30]


@dataclass
class FeatureConfig:
    """Configuration for feature engineering"""
    # Price features
    use_ohlcv: bool = True
    use_returns: bool = True
    use_log_returns: bool = True
    
    # Technical indicators
    use_moving_averages: bool = True
    ma_periods: List[int] = None
    use_bollinger_bands: bool = True
    use_rsi: bool = True
    use_macd: bool = True
    use_volatility: bool = True
    
    # Economic features
    use_economic_events: bool = True
    use_calendar_effects: bool = True  # Day of week, month, etc.
    
    # Scaling
    feature_scaling: str = "minmax"  # "minmax", "standard", "robust"
    
    def __post_init__(self):
        if self.ma_periods is None:
            self.ma_periods = [20, 50, 200]


@dataclass
class PredictionConfig:
    """Configuration for prediction system"""
    # Confidence intervals
    confidence_levels: List[float] = None  # [0.1, 0.5, 0.9] for p10, p50, p90
    
    # Prediction settings
    min_confidence_threshold: float = 0.3
    max_prediction_age_hours: int = 24
    
    # Caching
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600  # 1 hour
    
    # Performance
    max_inference_time_seconds: float = 2.0
    
    def __post_init__(self):
        if self.confidence_levels is None:
            self.confidence_levels = [0.1, 0.5, 0.9]


@dataclass
class DataConfig:
    """Configuration for data collection and training"""
    # Training data amount
    min_training_days: int = 200
    feature_lookback_days: int = 300
    
    # Data collection settings
    max_data_age_days: int = 1  # How old data can be before refresh


@dataclass
class BacktestConfig:
    """Configuration for backtesting"""
    # Validation
    walk_forward_steps: int = 30  # Days to step forward
    min_train_samples: int = 1000
    
    # Metrics
    calculate_sharpe: bool = True
    calculate_sortino: bool = True
    calculate_max_drawdown: bool = True
    
    # Currency pairs to test
    test_pairs: List[str] = None
    
    def __post_init__(self):
        if self.test_pairs is None:
            self.test_pairs = ["USD/EUR", "USD/GBP", "EUR/GBP"]


@dataclass 
class MLConfig:
    """Main ML configuration"""
    model: ModelConfig = None
    features: FeatureConfig = None
    prediction: PredictionConfig = None
    backtest: BacktestConfig = None
    data: DataConfig = None
    
    # Storage
    model_storage_path: str = "models/"
    data_storage_path: str = "data/"
    
    # Model versioning strategy
    model_versioning: str = "timestamp"  # "timestamp" or "overwrite"
    # timestamp: Create new folder with timestamp (lstm_USDEUR_20250831_184518)
    # overwrite: Use same folder name (lstm_USDEUR) and overwrite existing model
    
    # Device
    device: str = "auto"  # "auto", "cpu", "cuda"
    
    # Logging
    log_level: str = "INFO"
    
    def __post_init__(self):
        if self.model is None:
            self.model = ModelConfig()
        if self.features is None:
            self.features = FeatureConfig()
        if self.prediction is None:
            self.prediction = PredictionConfig()
        if self.backtest is None:
            self.backtest = BacktestConfig()
        if self.data is None:
            self.data = DataConfig()
    
    @classmethod
    def get_default(cls) -> 'MLConfig':
        """Get default configuration"""
        return cls()
    
    def get_device(self) -> str:
        """Get the device to use for computation"""
        if self.device == "auto":
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device
    
    def create_directories(self):
        """Create necessary directories"""
        Path(self.model_storage_path).mkdir(parents=True, exist_ok=True)
        Path(self.data_storage_path).mkdir(parents=True, exist_ok=True)


def load_ml_config(config_path: str = None) -> MLConfig:
    """Load ML configuration from YAML file or defaults"""
    if config_path and os.path.exists(config_path):
        try:
            import yaml
            with open(config_path, 'r') as f:
                yaml_config = yaml.safe_load(f)
            
            # Create config from YAML
            config = MLConfig()
            
            # Update model config
            if 'model' in yaml_config:
                model_data = yaml_config['model']
                config.model = ModelConfig(
                    hidden_size=model_data.get('hidden_size', 128),
                    num_layers=model_data.get('num_layers', 2),
                    dropout=model_data.get('dropout', 0.2),
                    bidirectional=model_data.get('bidirectional', False),
                    batch_size=model_data.get('batch_size', 32),
                    learning_rate=model_data.get('learning_rate', 0.001),
                    epochs=model_data.get('epochs', 100),
                    patience=model_data.get('patience', 15),
                    sequence_length=model_data.get('sequence_length', 60),
                    prediction_horizons=model_data.get('prediction_horizons', [1, 7, 30]),
                    train_test_split=model_data.get('train_test_split', 0.8),
                    validation_split=model_data.get('validation_split', 0.2)
                )
            
            # Update storage config
            if 'storage' in yaml_config:
                storage_data = yaml_config['storage']
                config.model_storage_path = storage_data.get('model_storage_path', 'models/')
                config.data_storage_path = storage_data.get('data_storage_path', 'data/')
                config.model_versioning = storage_data.get('model_versioning', 'timestamp')
            
            # Update feature config
            if 'features' in yaml_config:
                features_data = yaml_config['features']
                config.features = FeatureConfig(
                    use_ohlcv=features_data.get('use_ohlcv', True),
                    use_returns=features_data.get('use_returns', True),
                    use_log_returns=features_data.get('use_log_returns', True),
                    use_moving_averages=features_data.get('use_moving_averages', True),
                    ma_periods=features_data.get('ma_periods', [20, 50, 200]),
                    use_bollinger_bands=features_data.get('use_bollinger_bands', True),
                    use_rsi=features_data.get('use_rsi', True),
                    use_macd=features_data.get('use_macd', True),
                    use_volatility=features_data.get('use_volatility', True),
                    use_economic_events=features_data.get('use_economic_events', True),
                    use_calendar_effects=features_data.get('use_calendar_effects', True),
                    feature_scaling=features_data.get('feature_scaling', 'minmax')
                )
            
            # Update data config
            if 'data' in yaml_config:
                data_config = yaml_config['data']
                config.data = DataConfig(
                    min_training_days=data_config.get('min_training_days', 200),
                    feature_lookback_days=data_config.get('feature_lookback_days', 300),
                    max_data_age_days=data_config.get('max_data_age_days', 1)
                )
            
            # Update backtest config
            if 'backtest' in yaml_config:
                backtest_data = yaml_config['backtest']
                config.backtest = BacktestConfig(
                    walk_forward_steps=backtest_data.get('walk_forward_steps', 30),
                    min_train_samples=backtest_data.get('min_train_samples', 1000),
                    calculate_sharpe=backtest_data.get('calculate_sharpe', True),
                    calculate_sortino=backtest_data.get('calculate_sortino', True),
                    calculate_max_drawdown=backtest_data.get('calculate_max_drawdown', True),
                    test_pairs=backtest_data.get('test_pairs', ["USD/EUR", "USD/GBP", "EUR/GBP"])
                )
            
            # Update system config
            if 'system' in yaml_config:
                system_data = yaml_config['system']
                config.device = system_data.get('device', 'auto')
                config.log_level = system_data.get('log_level', 'INFO')
            
            config.create_directories()
            return config
            
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to load config from {config_path}: {e}. Using defaults.")
    
    # Try loading from default location
    default_paths = ['ml_config.yaml', 'config/ml_config.yaml', 'src/ml/config.yaml']
    for path in default_paths:
        if os.path.exists(path):
            return load_ml_config(path)
    
    # Use default config
    config = MLConfig.get_default()
    config.create_directories()
    return config