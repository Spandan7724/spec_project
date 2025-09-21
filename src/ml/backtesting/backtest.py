"""
ML model backtesting framework
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass

from ..config import MLConfig
from ..models.lstm_model import LSTMModel
from ..features.engineering import FeatureEngineer
from ..features.preprocessing import DataPreprocessor
from ..utils.data_loader import Layer1DataLoader
from .metrics import PerformanceMetrics

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Results from backtesting"""
    currency_pair: str
    test_period: Tuple[str, str]  # start_date, end_date
    total_predictions: int
    performance_metrics: Dict[str, float]
    prediction_history: List[Dict[str, Any]]
    model_info: Dict[str, Any]


class MLBacktester:
    """
    Comprehensive backtesting framework for ML models
    """
    
    def __init__(self, config: MLConfig = None):
        if config is None:
            from ..config import load_ml_config
            config = load_ml_config()
        self.config = config
        self.data_loader = Layer1DataLoader()
        self.feature_engineer = FeatureEngineer()
        self.preprocessor = DataPreprocessor()
        self.metrics_calculator = PerformanceMetrics()
        
    def run_walkforward_backtest(self,
                                currency_pair: str,
                                start_date: datetime,
                                end_date: datetime,
                                initial_train_days: int = 365,
                                retraining_frequency: int = 30,
                                prediction_horizons: List[int] = None) -> BacktestResult:
        """
        Run walk-forward backtesting with periodic model retraining
        
        Args:
            currency_pair: Currency pair to test
            start_date: Start date for backtesting
            end_date: End date for backtesting
            initial_train_days: Days of data for initial training
            retraining_frequency: Days between model retraining
            prediction_horizons: Horizons to test (uses config default if None)
            
        Returns:
            BacktestResult with comprehensive results
        """
        if prediction_horizons is None:
            prediction_horizons = self.config.model.prediction_horizons
        
        logger.info(f"Starting walk-forward backtest for {currency_pair}")
        logger.info(f"Period: {start_date.date()} to {end_date.date()}")
        
        # Load full dataset
        total_days = (end_date - start_date).days + initial_train_days + max(prediction_horizons)
        prices, indicators, economic_events = self.data_loader.get_combined_dataset(
            currency_pair=currency_pair,
            days=total_days
        )
        
        # Engineer features and targets
        features = self.feature_engineer.engineer_features(
            prices, indicators, economic_events
        )
        targets = self.feature_engineer.get_target_variables(
            prices, prediction_horizons
        )
        
        # Initialize tracking
        prediction_history = []
        model_retraining_dates = []
        
        # Current date pointer
        current_date = start_date
        last_retrain_date = start_date - timedelta(days=retraining_frequency)
        current_model = None
        
        while current_date <= end_date:
            try:
                # Check if we need to retrain
                days_since_retrain = (current_date - last_retrain_date).days
                
                if current_model is None or days_since_retrain >= retraining_frequency:
                    logger.info(f"Retraining model at {current_date.date()}")
                    
                    # Get training data up to current date
                    train_end_date = current_date - timedelta(days=1)
                    train_start_date = train_end_date - timedelta(days=initial_train_days)
                    
                    # Ensure we have enough data
                    if train_start_date not in features.index:
                        current_date += timedelta(days=1)
                        continue
                    
                    # Train new model
                    current_model = self._train_backtest_model(
                        features.loc[train_start_date:train_end_date],
                        targets.loc[train_start_date:train_end_date],
                        prediction_horizons
                    )
                    
                    last_retrain_date = current_date
                    model_retraining_dates.append(current_date)
                
                # Make prediction for current date
                if current_date in features.index:
                    prediction_result = self._make_backtest_prediction(
                        current_model, features, current_date, prediction_horizons
                    )
                    
                    # Get actual outcomes
                    actual_outcomes = self._get_actual_outcomes(
                        prices, current_date, prediction_horizons
                    )
                    
                    if actual_outcomes:
                        # Store prediction with outcomes
                        prediction_record = {
                            'date': current_date.isoformat(),
                            'predictions': prediction_result['predictions'],
                            'actuals': actual_outcomes,
                            'confidence_intervals': prediction_result.get('confidence_intervals', {}),
                            'model_confidence': prediction_result.get('model_confidence', 0.0)
                        }
                        
                        prediction_history.append(prediction_record)
                
                current_date += timedelta(days=1)
                
            except Exception as e:
                logger.warning(f"Backtest failed for {current_date.date()}: {e}")
                current_date += timedelta(days=1)
                continue
        
        # Calculate performance metrics
        performance_metrics = self._calculate_backtest_metrics(
            prediction_history, prediction_horizons
        )
        
        logger.info(f"Backtest completed: {len(prediction_history)} predictions made")
        
        return BacktestResult(
            currency_pair=currency_pair,
            test_period=(start_date.isoformat(), end_date.isoformat()),
            total_predictions=len(prediction_history),
            performance_metrics=performance_metrics,
            prediction_history=prediction_history,
            model_info=current_model.get_model_info() if current_model else {}
        )
    
    def _train_backtest_model(self,
                             features: pd.DataFrame,
                             targets: pd.DataFrame,
                             prediction_horizons: List[int]) -> LSTMModel:
        """Train model for backtesting"""
        # Prepare data
        data_splits = self.preprocessor.prepare_data(
            features, targets, test_size=0.2, validation_size=0.2
        )
        
        # Initialize model
        model_config = self.config.model.__dict__.copy()
        model_config['input_size'] = features.shape[1]
        model_config['prediction_horizons'] = prediction_horizons
        model_config['device'] = 'cpu'  # Use CPU for backtesting
        
        model = LSTMModel(model_config)
        
        # Train model (reduced epochs for backtesting speed)
        training_config = {
            'epochs': min(50, self.config.model.epochs),
            'batch_size': self.config.model.batch_size,
            'patience': 10,
            'verbose': False
        }
        
        model.fit(
            data_splits['X_train'], data_splits['y_train'],
            data_splits['X_val'], data_splits['y_val'],
            **training_config
        )
        
        return model
    
    def _make_backtest_prediction(self,
                                 model: LSTMModel,
                                 features: pd.DataFrame,
                                 prediction_date: datetime,
                                 prediction_horizons: List[int]) -> Dict[str, Any]:
        """Make prediction for backtesting"""
        # Get sequence ending at prediction_date
        sequence_length = self.config.model.sequence_length
        
        # Find the index for prediction_date
        try:
            date_idx = features.index.get_loc(prediction_date)
            
            if date_idx < sequence_length:
                raise ValueError(f"Not enough historical data for prediction on {prediction_date}")
            
            # Get feature sequence
            feature_sequence = features.iloc[date_idx-sequence_length:date_idx]
            
            # Prepare for prediction
            X_pred = self.preprocessor.prepare_single_prediction(feature_sequence)
            
            # Make prediction
            prediction_result = model.predict(X_pred, prediction_horizons)
            
            return {
                'predictions': prediction_result.predictions.tolist(),
                'confidence_intervals': {
                    k: v.tolist() if isinstance(v, np.ndarray) else v
                    for k, v in prediction_result.confidence_intervals.items()
                },
                'direction_probabilities': prediction_result.direction_probabilities.tolist(),
                'model_confidence': prediction_result.model_confidence
            }
            
        except Exception as e:
            logger.error(f"Prediction failed for {prediction_date}: {e}")
            return {
                'predictions': [0.0] * len(prediction_horizons),
                'confidence_intervals': {},
                'direction_probabilities': [0.5] * len(prediction_horizons),
                'model_confidence': 0.0
            }
    
    def _get_actual_outcomes(self,
                           prices: pd.DataFrame,
                           prediction_date: datetime,
                           prediction_horizons: List[int]) -> Optional[List[float]]:
        """Get actual outcomes for prediction validation"""
        try:
            base_price = prices.loc[prediction_date, 'close']
            actual_outcomes = []
            
            for horizon in prediction_horizons:
                future_date = prediction_date + timedelta(days=horizon)
                
                # Find the closest available date
                available_dates = prices.index[prices.index >= future_date]
                
                if len(available_dates) > 0:
                    closest_date = available_dates[0]
                    future_price = prices.loc[closest_date, 'close']
                    
                    # Calculate return
                    actual_return = (future_price / base_price) - 1
                    actual_outcomes.append(actual_return)
                else:
                    # Future data not available
                    return None
            
            return actual_outcomes
            
        except Exception as e:
            logger.warning(f"Could not get actual outcomes for {prediction_date}: {e}")
            return None
    
    def _calculate_backtest_metrics(self,
                                  prediction_history: List[Dict[str, Any]],
                                  prediction_horizons: List[int]) -> Dict[str, float]:
        """Calculate comprehensive backtest performance metrics"""
        if not prediction_history:
            return {}
        
        metrics = {}
        
        # Organize predictions and actuals by horizon
        horizon_data = {}
        for horizon in prediction_horizons:
            horizon_data[horizon] = {
                'predictions': [],
                'actuals': [],
                'dates': []
            }
        
        for record in prediction_history:
            predictions = record.get('predictions', [])
            actuals = record.get('actuals', [])
            
            for i, horizon in enumerate(prediction_horizons):
                if i < len(predictions) and i < len(actuals):
                    horizon_data[horizon]['predictions'].append(predictions[i])
                    horizon_data[horizon]['actuals'].append(actuals[i])
                    horizon_data[horizon]['dates'].append(record['date'])
        
        # Calculate metrics for each horizon
        for horizon in prediction_horizons:
            data = horizon_data[horizon]
            
            if not data['predictions']:
                continue
            
            predictions = np.array(data['predictions'])
            actuals = np.array(data['actuals'])
            
            # Basic metrics
            horizon_metrics = self.metrics_calculator.calculate_regression_metrics(
                actuals, predictions
            )
            
            # Directional accuracy
            pred_direction = predictions > 0
            actual_direction = actuals > 0
            directional_accuracy = np.mean(pred_direction == actual_direction)
            horizon_metrics['directional_accuracy'] = directional_accuracy
            
            # Add horizon prefix to metric names
            for key, value in horizon_metrics.items():
                metrics[f'{horizon}d_{key}'] = value
        
        # Overall metrics (average across horizons)
        if prediction_horizons:
            overall_metrics = {}
            metric_keys = ['mse', 'mae', 'rmse', 'r2', 'directional_accuracy']
            
            for metric_key in metric_keys:
                horizon_values = []
                for horizon in prediction_horizons:
                    metric_name = f'{horizon}d_{metric_key}'
                    if metric_name in metrics:
                        horizon_values.append(metrics[metric_name])
                
                if horizon_values:
                    overall_metrics[f'avg_{metric_key}'] = np.mean(horizon_values)
                    overall_metrics[f'std_{metric_key}'] = np.std(horizon_values)
            
            metrics.update(overall_metrics)
        
        # Time-based metrics
        total_predictions = len(prediction_history)
        if total_predictions > 0:
            first_date = datetime.fromisoformat(prediction_history[0]['date'])
            last_date = datetime.fromisoformat(prediction_history[-1]['date'])
            test_days = (last_date - first_date).days
            
            metrics.update({
                'total_predictions': total_predictions,
                'test_period_days': test_days,
                'predictions_per_day': total_predictions / max(test_days, 1),
                'data_coverage': total_predictions / max(test_days, 1)
            })
        
        return metrics
    
    def compare_models(self,
                      currency_pair: str,
                      model_configs: List[Dict[str, Any]],
                      test_start_date: datetime,
                      test_end_date: datetime,
                      model_names: List[str] = None) -> Dict[str, BacktestResult]:
        """
        Compare multiple model configurations on the same dataset
        """
        if model_names is None:
            model_names = [f"Model_{i+1}" for i in range(len(model_configs))]
        
        logger.info(f"Comparing {len(model_configs)} models for {currency_pair}")
        
        results = {}
        
        for i, (model_config, model_name) in enumerate(zip(model_configs, model_names)):
            logger.info(f"Testing {model_name} ({i+1}/{len(model_configs)})")
            
            try:
                # Update config
                original_config = self.config.model.__dict__.copy()
                self.config.model.__dict__.update(model_config)
                
                # Run backtest
                result = self.run_walkforward_backtest(
                    currency_pair=currency_pair,
                    start_date=test_start_date,
                    end_date=test_end_date
                )
                
                results[model_name] = result
                
                # Restore original config
                self.config.model.__dict__.update(original_config)
                
            except Exception as e:
                logger.error(f"Model comparison failed for {model_name}: {e}")
                results[model_name] = None
        
        return results
    
    def generate_backtest_report(self, 
                               results: Dict[str, BacktestResult],
                               output_path: str = None) -> Dict[str, Any]:
        """
        Generate comprehensive backtest report
        """
        report = {
            'summary': {
                'total_models_tested': len(results),
                'successful_tests': sum(1 for r in results.values() if r is not None),
                'test_period': None,
                'currency_pairs': set()
            },
            'model_comparison': {},
            'best_models': {},
            'detailed_results': {}
        }
        
        # Extract summary information
        valid_results = {k: v for k, v in results.items() if v is not None}
        
        if valid_results:
            # Test period (assuming all tests use same period)
            first_result = next(iter(valid_results.values()))
            report['summary']['test_period'] = first_result.test_period
            
            # Currency pairs
            for result in valid_results.values():
                report['summary']['currency_pairs'].add(result.currency_pair)
            
            report['summary']['currency_pairs'] = list(report['summary']['currency_pairs'])
        
        # Model comparison
        comparison_metrics = ['avg_directional_accuracy', 'avg_rmse', 'avg_mae', 'total_predictions']
        
        for model_name, result in valid_results.items():
            if result and result.performance_metrics:
                model_summary = {}
                
                for metric in comparison_metrics:
                    if metric in result.performance_metrics:
                        model_summary[metric] = result.performance_metrics[metric]
                
                report['model_comparison'][model_name] = model_summary
        
        # Find best models
        if report['model_comparison']:
            # Best by directional accuracy
            best_accuracy = max(
                report['model_comparison'].items(),
                key=lambda x: x[1].get('avg_directional_accuracy', 0),
                default=(None, {})
            )
            
            if best_accuracy[0]:
                report['best_models']['highest_accuracy'] = {
                    'model': best_accuracy[0],
                    'accuracy': best_accuracy[1].get('avg_directional_accuracy', 0)
                }
            
            # Best by RMSE (lowest)
            models_with_rmse = {k: v for k, v in report['model_comparison'].items() 
                               if 'avg_rmse' in v}
            
            if models_with_rmse:
                best_rmse = min(
                    models_with_rmse.items(),
                    key=lambda x: x[1]['avg_rmse']
                )
                
                report['best_models']['lowest_rmse'] = {
                    'model': best_rmse[0],
                    'rmse': best_rmse[1]['avg_rmse']
                }
        
        # Store detailed results
        report['detailed_results'] = {
            k: {
                'currency_pair': v.currency_pair,
                'test_period': v.test_period,
                'total_predictions': v.total_predictions,
                'performance_metrics': v.performance_metrics
            } for k, v in valid_results.items()
        }
        
        # Save report if path provided
        if output_path:
            import json
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Backtest report saved to {output_path}")
        
        return report