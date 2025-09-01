"""
Performance metrics calculation for ML models
"""

import numpy as np
from typing import Dict, List, Any
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class PerformanceMetrics:
    """
    Comprehensive performance metrics calculator for ML models
    """
    
    def __init__(self):
        pass
    
    def calculate_regression_metrics(self, 
                                   y_true: np.ndarray, 
                                   y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate regression performance metrics
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary of metrics
        """
        # Remove NaN values
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        
        if len(y_true) == 0:
            return {}
        
        # Basic metrics
        mse = np.mean((y_true - y_pred) ** 2)
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(mse)
        
        # R-squared
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))
        
        # Mean Absolute Percentage Error
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        
        # Median Absolute Error
        med_ae = np.median(np.abs(y_true - y_pred))
        
        # Max Error
        max_error = np.max(np.abs(y_true - y_pred))
        
        return {
            'mse': float(mse),
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(r2),
            'mape': float(mape),
            'median_ae': float(med_ae),
            'max_error': float(max_error)
        }
    
    def calculate_classification_metrics(self, 
                                       y_true: np.ndarray, 
                                       y_pred: np.ndarray,
                                       threshold: float = 0.5) -> Dict[str, float]:
        """
        Calculate classification metrics for directional predictions
        
        Args:
            y_true: True binary labels (0/1 or boolean)
            y_pred: Predicted probabilities or binary predictions
            threshold: Threshold for converting probabilities to binary
            
        Returns:
            Dictionary of classification metrics
        """
        # Convert probabilities to binary predictions if needed
        if y_pred.dtype == float and np.all((y_pred >= 0) & (y_pred <= 1)):
            y_pred_binary = (y_pred >= threshold).astype(int)
        else:
            y_pred_binary = y_pred.astype(int)
        
        y_true_binary = y_true.astype(int)
        
        # Remove NaN values
        mask = ~(np.isnan(y_true_binary) | np.isnan(y_pred_binary))
        y_true_binary = y_true_binary[mask]
        y_pred_binary = y_pred_binary[mask]
        
        if len(y_true_binary) == 0:
            return {}
        
        # Confusion matrix elements
        tp = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
        tn = np.sum((y_true_binary == 0) & (y_pred_binary == 0))
        fp = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
        fn = np.sum((y_true_binary == 1) & (y_pred_binary == 0))
        
        # Basic metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # F1 Score
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Balanced accuracy
        balanced_accuracy = (recall + specificity) / 2
        
        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'specificity': float(specificity),
            'f1_score': float(f1),
            'balanced_accuracy': float(balanced_accuracy),
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn)
        }
    
    def calculate_financial_metrics(self, 
                                  returns: np.ndarray,
                                  predictions: np.ndarray = None,
                                  benchmark_returns: np.ndarray = None) -> Dict[str, float]:
        """
        Calculate financial performance metrics
        
        Args:
            returns: Actual returns
            predictions: Predicted returns (optional)
            benchmark_returns: Benchmark returns for comparison (optional)
            
        Returns:
            Dictionary of financial metrics
        """
        returns = returns[~np.isnan(returns)]
        
        if len(returns) == 0:
            return {}
        
        # Basic return statistics
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        # Sharpe Ratio (assuming risk-free rate = 0 for simplicity)
        sharpe_ratio = mean_return / std_return if std_return > 0 else 0
        
        # Sortino Ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 0
        sortino_ratio = mean_return / downside_std if downside_std > 0 else 0
        
        # Maximum Drawdown
        cumulative_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        # Calmar Ratio
        calmar_ratio = mean_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # VaR and CVaR (5% level)
        var_5 = np.percentile(returns, 5)
        cvar_5 = np.mean(returns[returns <= var_5])
        
        # Skewness and Kurtosis
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns)
        
        metrics = {
            'mean_return': float(mean_return),
            'std_return': float(std_return),
            'sharpe_ratio': float(sharpe_ratio),
            'sortino_ratio': float(sortino_ratio),
            'max_drawdown': float(max_drawdown),
            'calmar_ratio': float(calmar_ratio),
            'var_5': float(var_5),
            'cvar_5': float(cvar_5),
            'skewness': float(skewness),
            'kurtosis': float(kurtosis)
        }
        
        # Information Ratio (if benchmark provided)
        if benchmark_returns is not None:
            benchmark_returns = benchmark_returns[~np.isnan(benchmark_returns)]
            if len(benchmark_returns) == len(returns):
                excess_returns = returns - benchmark_returns
                tracking_error = np.std(excess_returns)
                information_ratio = np.mean(excess_returns) / tracking_error if tracking_error > 0 else 0
                metrics['information_ratio'] = float(information_ratio)
                metrics['tracking_error'] = float(tracking_error)
        
        # Hit Ratio (if predictions provided)
        if predictions is not None:
            predictions = predictions[~np.isnan(predictions)]
            if len(predictions) == len(returns):
                # Directional accuracy
                correct_direction = np.sign(returns) == np.sign(predictions)
                hit_ratio = np.mean(correct_direction)
                metrics['hit_ratio'] = float(hit_ratio)
                
                # Prediction correlation
                correlation = np.corrcoef(returns, predictions)[0, 1]
                metrics['prediction_correlation'] = float(correlation) if not np.isnan(correlation) else 0.0
        
        return metrics
    
    def calculate_confidence_metrics(self,
                                   y_true: np.ndarray,
                                   predictions: np.ndarray,
                                   confidence_intervals: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Calculate metrics for confidence interval quality
        
        Args:
            y_true: True values
            predictions: Point predictions
            confidence_intervals: Dictionary with confidence interval arrays
            
        Returns:
            Confidence interval quality metrics
        """
        metrics = {}
        
        # Remove NaN values
        mask = ~np.isnan(y_true)
        y_true = y_true[mask]
        predictions = predictions[mask] if predictions is not None else None
        
        if len(y_true) == 0:
            return metrics
        
        # Analyze each confidence interval
        for level_name, interval_values in confidence_intervals.items():
            if len(interval_values) != len(y_true):
                continue
            
            interval_values = interval_values[mask]
            
            # Skip median (p50) for coverage analysis
            if level_name == 'p50':
                continue
            
            # Extract confidence level
            try:
                level_pct = int(level_name[1:])  # e.g., 'p90' -> 90
                expected_coverage = level_pct / 100.0
            except:
                continue
            
            # Calculate coverage
            if level_pct < 50:
                # Lower bound - actual should be >= bound
                actual_coverage = np.mean(y_true >= interval_values)
            else:
                # Upper bound - actual should be <= bound
                actual_coverage = np.mean(y_true <= interval_values)
            
            # Coverage metrics
            coverage_error = abs(actual_coverage - expected_coverage)
            metrics[f'{level_name}_coverage'] = float(actual_coverage)
            metrics[f'{level_name}_coverage_error'] = float(coverage_error)
        
        # Interval width analysis
        if 'p10' in confidence_intervals and 'p90' in confidence_intervals:
            p10_vals = confidence_intervals['p10'][mask]
            p90_vals = confidence_intervals['p90'][mask]
            
            interval_widths = p90_vals - p10_vals
            metrics['mean_interval_width'] = float(np.mean(interval_widths))
            metrics['median_interval_width'] = float(np.median(interval_widths))
            metrics['std_interval_width'] = float(np.std(interval_widths))
            
            # Normalized interval width (relative to prediction magnitude)
            if predictions is not None:
                predictions_filtered = predictions[mask]
                non_zero_mask = np.abs(predictions_filtered) > 1e-8
                
                if np.sum(non_zero_mask) > 0:
                    normalized_widths = interval_widths[non_zero_mask] / np.abs(predictions_filtered[non_zero_mask])
                    metrics['mean_normalized_width'] = float(np.mean(normalized_widths))
        
        # Prediction interval coverage probability (PICP)
        if 'p10' in confidence_intervals and 'p90' in confidence_intervals:
            p10_vals = confidence_intervals['p10'][mask]
            p90_vals = confidence_intervals['p90'][mask]
            
            within_interval = (y_true >= p10_vals) & (y_true <= p90_vals)
            picp = np.mean(within_interval)
            metrics['picp_80'] = float(picp)  # 80% prediction interval
        
        return metrics
    
    def calculate_stability_metrics(self,
                                  prediction_history: List[Dict[str, Any]],
                                  window_size: int = 30) -> Dict[str, float]:
        """
        Calculate model stability metrics over time
        
        Args:
            prediction_history: List of prediction records with dates
            window_size: Rolling window size for stability calculation
            
        Returns:
            Stability metrics
        """
        if len(prediction_history) < window_size:
            return {}
        
        # Extract time series of predictions and actuals
        dates = []
        predictions_by_horizon = {}
        actuals_by_horizon = {}
        
        for record in prediction_history:
            dates.append(record['date'])
            
            preds = record.get('predictions', [])
            acts = record.get('actuals', [])
            
            for i, (pred, actual) in enumerate(zip(preds, acts)):
                horizon_key = f'horizon_{i}'
                
                if horizon_key not in predictions_by_horizon:
                    predictions_by_horizon[horizon_key] = []
                    actuals_by_horizon[horizon_key] = []
                
                predictions_by_horizon[horizon_key].append(pred)
                actuals_by_horizon[horizon_key].append(actual)
        
        stability_metrics = {}
        
        # Calculate rolling performance metrics
        for horizon_key in predictions_by_horizon.keys():
            preds = np.array(predictions_by_horizon[horizon_key])
            acts = np.array(actuals_by_horizon[horizon_key])
            
            if len(preds) < window_size:
                continue
            
            # Rolling MAE
            rolling_mae = []
            rolling_accuracy = []
            
            for i in range(window_size, len(preds) + 1):
                window_preds = preds[i-window_size:i]
                window_acts = acts[i-window_size:i]
                
                mae = np.mean(np.abs(window_acts - window_preds))
                rolling_mae.append(mae)
                
                # Directional accuracy
                direction_acc = np.mean(np.sign(window_acts) == np.sign(window_preds))
                rolling_accuracy.append(direction_acc)
            
            # Stability metrics for this horizon
            if rolling_mae:
                mae_stability = np.std(rolling_mae) / np.mean(rolling_mae)
                accuracy_stability = np.std(rolling_accuracy)
                
                stability_metrics[f'{horizon_key}_mae_stability'] = float(mae_stability)
                stability_metrics[f'{horizon_key}_accuracy_stability'] = float(accuracy_stability)
                stability_metrics[f'{horizon_key}_mean_rolling_mae'] = float(np.mean(rolling_mae))
                stability_metrics[f'{horizon_key}_mean_rolling_accuracy'] = float(np.mean(rolling_accuracy))
        
        # Overall stability (average across horizons)
        if stability_metrics:
            mae_stabilities = [v for k, v in stability_metrics.items() if k.endswith('_mae_stability')]
            acc_stabilities = [v for k, v in stability_metrics.items() if k.endswith('_accuracy_stability')]
            
            if mae_stabilities:
                stability_metrics['overall_mae_stability'] = float(np.mean(mae_stabilities))
            if acc_stabilities:
                stability_metrics['overall_accuracy_stability'] = float(np.mean(acc_stabilities))
        
        return stability_metrics
    
    def generate_performance_report(self,
                                  y_true: np.ndarray,
                                  y_pred: np.ndarray,
                                  confidence_intervals: Dict[str, np.ndarray] = None,
                                  returns: np.ndarray = None,
                                  prediction_history: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate comprehensive performance report
        
        Args:
            y_true: True values
            y_pred: Predicted values
            confidence_intervals: Confidence intervals (optional)
            returns: Financial returns (optional)
            prediction_history: Historical predictions for stability analysis (optional)
            
        Returns:
            Comprehensive performance report
        """
        report = {
            'regression_metrics': {},
            'classification_metrics': {},
            'financial_metrics': {},
            'confidence_metrics': {},
            'stability_metrics': {},
            'summary': {}
        }
        
        # Regression metrics
        report['regression_metrics'] = self.calculate_regression_metrics(y_true, y_pred)
        
        # Classification metrics (directional predictions)
        y_true_direction = y_true > 0
        y_pred_direction = y_pred > 0
        report['classification_metrics'] = self.calculate_classification_metrics(
            y_true_direction, y_pred_direction
        )
        
        # Financial metrics
        if returns is not None:
            report['financial_metrics'] = self.calculate_financial_metrics(
                returns, y_pred
            )
        
        # Confidence metrics
        if confidence_intervals is not None:
            report['confidence_metrics'] = self.calculate_confidence_metrics(
                y_true, y_pred, confidence_intervals
            )
        
        # Stability metrics
        if prediction_history is not None:
            report['stability_metrics'] = self.calculate_stability_metrics(
                prediction_history
            )
        
        # Summary metrics
        summary = {}
        
        # Key regression metrics
        if 'rmse' in report['regression_metrics']:
            summary['rmse'] = report['regression_metrics']['rmse']
        if 'mae' in report['regression_metrics']:
            summary['mae'] = report['regression_metrics']['mae']
        if 'r2' in report['regression_metrics']:
            summary['r2'] = report['regression_metrics']['r2']
        
        # Key classification metrics
        if 'accuracy' in report['classification_metrics']:
            summary['directional_accuracy'] = report['classification_metrics']['accuracy']
        
        # Key financial metrics
        if 'sharpe_ratio' in report['financial_metrics']:
            summary['sharpe_ratio'] = report['financial_metrics']['sharpe_ratio']
        if 'max_drawdown' in report['financial_metrics']:
            summary['max_drawdown'] = report['financial_metrics']['max_drawdown']
        
        # Overall score (weighted combination)
        score_components = []
        if 'directional_accuracy' in summary:
            score_components.append(summary['directional_accuracy'] * 0.4)
        if 'r2' in summary and summary['r2'] > 0:
            score_components.append(min(summary['r2'], 1.0) * 0.3)
        if 'sharpe_ratio' in summary:
            normalized_sharpe = max(0, min(summary['sharpe_ratio'], 2.0)) / 2.0
            score_components.append(normalized_sharpe * 0.3)
        
        if score_components:
            summary['overall_score'] = sum(score_components)
        
        report['summary'] = summary
        
        return report