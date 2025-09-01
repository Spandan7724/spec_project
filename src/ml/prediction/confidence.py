"""
Confidence interval calculation for predictions
"""

import numpy as np
import scipy.stats as stats
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)


class ConfidenceCalculator:
    """
    Advanced confidence interval calculation for ML predictions
    """
    
    def __init__(self):
        self.calibration_data = {}  # Store historical prediction vs actual for calibration
        
    def calculate_prediction_intervals(self,
                                     predictions: np.ndarray,
                                     model_uncertainty: np.ndarray,
                                     confidence_levels: List[float] = None,
                                     method: str = 'gaussian') -> Dict[str, np.ndarray]:
        """
        Calculate prediction intervals using various methods
        
        Args:
            predictions: Point predictions
            model_uncertainty: Model uncertainty estimates
            confidence_levels: Confidence levels (e.g., [0.1, 0.5, 0.9])
            method: Method for interval calculation ('gaussian', 'bootstrap', 'quantile')
            
        Returns:
            Dictionary with confidence intervals
        """
        if confidence_levels is None:
            confidence_levels = [0.1, 0.5, 0.9]
        
        intervals = {}
        
        if method == 'gaussian':
            intervals = self._gaussian_intervals(predictions, model_uncertainty, confidence_levels)
        elif method == 'bootstrap':
            intervals = self._bootstrap_intervals(predictions, model_uncertainty, confidence_levels)
        elif method == 'quantile':
            intervals = self._quantile_intervals(predictions, model_uncertainty, confidence_levels)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return intervals
    
    def _gaussian_intervals(self,
                           predictions: np.ndarray,
                           uncertainty: np.ndarray,
                           confidence_levels: List[float]) -> Dict[str, np.ndarray]:
        """Calculate intervals assuming Gaussian distribution"""
        intervals = {}
        
        for level in confidence_levels:
            if level == 0.5:
                # Median is the prediction itself
                intervals[f'p{int(level*100)}'] = predictions
            else:
                # Calculate z-score for the confidence level
                z_score = stats.norm.ppf(level)
                
                if level < 0.5:
                    intervals[f'p{int(level*100)}'] = predictions + z_score * uncertainty
                else:
                    intervals[f'p{int(level*100)}'] = predictions + z_score * uncertainty
        
        return intervals
    
    def _bootstrap_intervals(self,
                            predictions: np.ndarray,
                            uncertainty: np.ndarray,
                            confidence_levels: List[float],
                            n_bootstrap: int = 1000) -> Dict[str, np.ndarray]:
        """Calculate intervals using bootstrap method"""
        intervals = {}
        
        # Generate bootstrap samples
        n_samples = len(predictions)
        bootstrap_predictions = []
        
        for _ in range(n_bootstrap):
            # Add noise based on uncertainty
            noise = np.random.normal(0, uncertainty)
            bootstrap_pred = predictions + noise
            bootstrap_predictions.append(bootstrap_pred)
        
        bootstrap_predictions = np.array(bootstrap_predictions)
        
        # Calculate percentiles
        for level in confidence_levels:
            percentile = level * 100
            intervals[f'p{int(percentile)}'] = np.percentile(bootstrap_predictions, percentile, axis=0)
        
        return intervals
    
    def _quantile_intervals(self,
                           predictions: np.ndarray,
                           uncertainty: np.ndarray,
                           confidence_levels: List[float]) -> Dict[str, np.ndarray]:
        """Calculate intervals using quantile regression approach"""
        # Simplified quantile approach - assumes uncertainty represents quantile width
        intervals = {}
        
        for level in confidence_levels:
            if level == 0.5:
                intervals[f'p{int(level*100)}'] = predictions
            elif level < 0.5:
                # Lower quantile
                width_factor = (0.5 - level) * 2  # Scale factor
                intervals[f'p{int(level*100)}'] = predictions - width_factor * uncertainty
            else:
                # Upper quantile
                width_factor = (level - 0.5) * 2  # Scale factor
                intervals[f'p{int(level*100)}'] = predictions + width_factor * uncertainty
        
        return intervals
    
    def calibrate_confidence(self,
                           historical_predictions: List[np.ndarray],
                           historical_actuals: List[np.ndarray],
                           currency_pair: str) -> Dict[str, float]:
        """
        Calibrate confidence intervals using historical data
        
        Args:
            historical_predictions: List of prediction arrays
            historical_actuals: List of actual value arrays
            currency_pair: Currency pair identifier
            
        Returns:
            Calibration metrics
        """
        if len(historical_predictions) != len(historical_actuals):
            raise ValueError("Predictions and actuals must have same length")
        
        # Combine all predictions and actuals
        all_predictions = np.concatenate(historical_predictions)
        all_actuals = np.concatenate(historical_actuals)
        
        # Calculate prediction errors
        errors = all_actuals - all_predictions
        
        # Calculate calibration metrics
        calibration_metrics = {
            'mean_error': float(np.mean(errors)),
            'std_error': float(np.std(errors)),
            'mae': float(np.mean(np.abs(errors))),
            'rmse': float(np.sqrt(np.mean(errors**2)))
        }
        
        # Coverage analysis for different confidence levels
        confidence_levels = [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]
        coverage_analysis = {}
        
        for level in confidence_levels:
            # Calculate theoretical interval
            z_score = stats.norm.ppf(level)
            std_err = calibration_metrics['std_error']
            
            if level < 0.5:
                lower_bound = all_predictions + z_score * std_err
                coverage = np.mean(all_actuals >= lower_bound)
            else:
                upper_bound = all_predictions + z_score * std_err
                coverage = np.mean(all_actuals <= upper_bound)
            
            coverage_analysis[f'coverage_p{int(level*100)}'] = float(coverage)
        
        calibration_metrics.update(coverage_analysis)
        
        # Store calibration data
        self.calibration_data[currency_pair] = {
            'metrics': calibration_metrics,
            'error_distribution': errors
        }
        
        logger.info(f"Calibrated confidence intervals for {currency_pair}")
        
        return calibration_metrics
    
    def get_calibrated_intervals(self,
                                predictions: np.ndarray,
                                model_uncertainty: np.ndarray,
                                currency_pair: str,
                                confidence_levels: List[float] = None) -> Dict[str, np.ndarray]:
        """
        Get calibrated confidence intervals using historical calibration data
        """
        if currency_pair not in self.calibration_data:
            logger.warning(f"No calibration data for {currency_pair}, using uncalibrated intervals")
            return self.calculate_prediction_intervals(predictions, model_uncertainty, confidence_levels)
        
        if confidence_levels is None:
            confidence_levels = [0.1, 0.5, 0.9]
        
        calibration = self.calibration_data[currency_pair]
        calibrated_std = calibration['metrics']['std_error']
        
        intervals = {}
        
        for level in confidence_levels:
            if level == 0.5:
                # Apply bias correction
                bias_correction = calibration['metrics']['mean_error']
                intervals[f'p{int(level*100)}'] = predictions - bias_correction
            else:
                # Use calibrated standard deviation
                z_score = stats.norm.ppf(level)
                intervals[f'p{int(level*100)}'] = predictions + z_score * calibrated_std
        
        return intervals
    
    def calculate_directional_confidence(self,
                                       predictions: np.ndarray,
                                       uncertainty: np.ndarray) -> np.ndarray:
        """
        Calculate confidence in directional predictions (up/down)
        """
        # Convert predictions to probabilities using sigmoid
        # Higher uncertainty -> lower confidence in direction
        
        # Normalize predictions by uncertainty
        normalized_predictions = predictions / (uncertainty + 1e-8)
        
        # Convert to probabilities using sigmoid
        probabilities = 1 / (1 + np.exp(-normalized_predictions))
        
        # Convert to directional confidence (0.5 = no confidence, 1.0 = high confidence)
        directional_confidence = np.abs(probabilities - 0.5) * 2
        
        return directional_confidence
    
    def evaluate_prediction_quality(self,
                                  predictions: np.ndarray,
                                  actuals: np.ndarray,
                                  confidence_intervals: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Evaluate the quality of predictions and confidence intervals
        """
        metrics = {}
        
        # Basic prediction metrics
        errors = actuals - predictions
        metrics['mae'] = float(np.mean(np.abs(errors)))
        metrics['mse'] = float(np.mean(errors**2))
        metrics['rmse'] = float(np.sqrt(metrics['mse']))
        
        # Directional accuracy
        pred_direction = predictions > 0
        actual_direction = actuals > 0
        metrics['directional_accuracy'] = float(np.mean(pred_direction == actual_direction))
        
        # Interval coverage analysis
        for key, interval_values in confidence_intervals.items():
            if key == 'p50':
                continue  # Skip median
            
            level = int(key[1:]) / 100.0
            
            if level < 0.5:
                # Lower bound coverage
                coverage = np.mean(actuals >= interval_values)
                expected_coverage = level
            else:
                # Upper bound coverage
                coverage = np.mean(actuals <= interval_values)
                expected_coverage = level
            
            metrics[f'{key}_coverage'] = float(coverage)
            metrics[f'{key}_coverage_error'] = float(abs(coverage - expected_coverage))
        
        # Interval width analysis
        if 'p10' in confidence_intervals and 'p90' in confidence_intervals:
            interval_width = confidence_intervals['p90'] - confidence_intervals['p10']
            metrics['mean_interval_width'] = float(np.mean(interval_width))
            metrics['median_interval_width'] = float(np.median(interval_width))
        
        return metrics
    
    def suggest_recalibration(self, 
                            evaluation_metrics: Dict[str, float],
                            threshold_coverage_error: float = 0.1) -> Dict[str, Any]:
        """
        Suggest recalibration based on evaluation metrics
        """
        suggestions = {
            'needs_recalibration': False,
            'issues': [],
            'recommendations': []
        }
        
        # Check coverage errors
        for key, value in evaluation_metrics.items():
            if key.endswith('_coverage_error'):
                if value > threshold_coverage_error:
                    suggestions['needs_recalibration'] = True
                    level = key.replace('_coverage_error', '')
                    suggestions['issues'].append(f"Poor coverage for {level}: {value:.3f} error")
        
        # Check directional accuracy
        if evaluation_metrics.get('directional_accuracy', 0) < 0.5:
            suggestions['issues'].append("Poor directional accuracy")
            suggestions['recommendations'].append("Consider model retraining")
        
        # Check prediction accuracy
        if evaluation_metrics.get('mae', float('inf')) > 0.05:  # 5% error threshold
            suggestions['issues'].append("High prediction error")
            suggestions['recommendations'].append("Model may need more training data")
        
        if suggestions['needs_recalibration']:
            suggestions['recommendations'].append("Run confidence calibration with more historical data")
        
        return suggestions