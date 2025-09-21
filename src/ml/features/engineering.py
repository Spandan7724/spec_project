"""
Feature engineering for price prediction
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Feature engineering pipeline for currency price prediction
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.feature_names = []
        
    def engineer_features(self, 
                         prices: pd.DataFrame,
                         indicators: pd.DataFrame = None,
                         economic_events: pd.DataFrame = None) -> pd.DataFrame:
        """
        Create comprehensive feature set for ML model
        """
        logger.info("Starting feature engineering")
        
        features = pd.DataFrame(index=prices.index)
        
        # Price-based features
        features = self._add_price_features(features, prices)
        
        # Technical indicator features
        if indicators is not None and not indicators.empty:
            features = self._add_indicator_features(features, indicators)
        
        # Economic event features
        if economic_events is not None:
            features = self._add_economic_features(features, economic_events)
        
        # Time-based features
        features = self._add_time_features(features)
        
        # Interaction features
        features = self._add_interaction_features(features)
        
        # Clean up and finalize
        features = self._finalize_features(features)
        
        logger.info(f"Feature engineering complete: {len(features.columns)} features")
        self.feature_names = list(features.columns)
        
        return features
    
    def _add_price_features(self, features: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
        """Add price-based features"""
        logger.debug("Adding price features")
        
        # Raw prices (normalized)
        features['close_norm'] = prices['close'] / prices['close'].iloc[0]
        features['open_norm'] = prices['open'] / prices['close'].iloc[0]
        features['high_norm'] = prices['high'] / prices['close'].iloc[0]  
        features['low_norm'] = prices['low'] / prices['close'].iloc[0]
        
        # Returns
        features['returns'] = prices['returns']
        features['log_returns'] = prices['log_returns']
        
        # Return lags
        for lag in [1, 2, 3, 5, 10]:
            features[f'returns_lag_{lag}'] = prices['returns'].shift(lag)
            features[f'log_returns_lag_{lag}'] = prices['log_returns'].shift(lag)
        
        # Rolling statistics
        for window in [5, 10, 20]:
            features[f'returns_mean_{window}d'] = prices['returns'].rolling(window).mean()
            features[f'returns_std_{window}d'] = prices['returns'].rolling(window).std()
            features[f'returns_skew_{window}d'] = prices['returns'].rolling(window).skew()
            features[f'returns_kurt_{window}d'] = prices['returns'].rolling(window).kurt()
        
        # Price ranges and ratios
        features['hl_ratio'] = (prices['high'] - prices['low']) / prices['close']
        features['oc_ratio'] = (prices['open'] - prices['close']) / prices['close']
        
        # Volume features (if available)
        if 'volume' in prices.columns and prices['volume'].sum() > 0:
            features['volume_norm'] = prices['volume'] / prices['volume'].rolling(20).mean()
            features['price_volume'] = prices['close'] * prices['volume']
            features['vwap'] = (features['price_volume'].rolling(20).sum() / 
                              prices['volume'].rolling(20).sum())
        
        return features
    
    def _add_indicator_features(self, features: pd.DataFrame, indicators: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicator features"""
        logger.debug("Adding technical indicator features")
        
        # Moving averages
        if 'sma_20' in indicators.columns:
            features['sma_20'] = indicators['sma_20']
            features['price_sma20_ratio'] = features['close_norm'] / (indicators['sma_20'] / indicators['sma_20'].iloc[0])
        
        if 'sma_50' in indicators.columns:
            features['sma_50'] = indicators['sma_50'] / indicators['sma_50'].iloc[0]
            features['sma20_sma50_ratio'] = features.get('sma_20', 1) / features['sma_50']
        
        if 'sma_200' in indicators.columns:
            features['sma_200'] = indicators['sma_200'] / indicators['sma_200'].iloc[0]
            features['sma50_sma200_ratio'] = features.get('sma_50', 1) / features['sma_200']
        
        # Bollinger Bands
        if all(col in indicators.columns for col in ['bb_upper', 'bb_middle', 'bb_lower']):
            bb_width = indicators['bb_upper'] - indicators['bb_lower']
            bb_middle = indicators['bb_middle']
            
            # Align indicators with features index using forward fill
            bb_middle_aligned = indicators['bb_middle'].reindex(features.index, method='ffill')
            bb_width_aligned = bb_width.reindex(features.index, method='ffill')
            bb_upper_aligned = indicators['bb_upper'].reindex(features.index, method='ffill')
            bb_lower_aligned = indicators['bb_lower'].reindex(features.index, method='ffill')
            
            current_price = features['close_norm'] * features['close_norm'].iloc[0]
            
            features['bb_position'] = ((current_price - bb_middle_aligned) / 
                                     (bb_width_aligned + 1e-8))
            features['bb_width_norm'] = bb_width_aligned / bb_middle_aligned
        
        # RSI
        if 'rsi' in indicators.columns:
            rsi_aligned = indicators['rsi'].reindex(features.index, method='ffill')
            features['rsi'] = rsi_aligned / 100.0  # Normalize to 0-1
            features['rsi_overbought'] = (rsi_aligned > 70).astype(int)
            features['rsi_oversold'] = (rsi_aligned < 30).astype(int)
            
            # RSI changes
            features['rsi_change'] = rsi_aligned.diff()
            features['rsi_momentum'] = rsi_aligned.diff().rolling(5).mean()
        
        # MACD
        if all(col in indicators.columns for col in ['macd', 'macd_signal']):
            macd_aligned = indicators['macd'].reindex(features.index, method='ffill')
            macd_signal_aligned = indicators['macd_signal'].reindex(features.index, method='ffill')
            macd_hist_aligned = indicators.get('macd_histogram', macd_aligned - macd_signal_aligned)
            if 'macd_histogram' in indicators.columns:
                macd_hist_aligned = indicators['macd_histogram'].reindex(features.index, method='ffill')
            
            features['macd'] = macd_aligned
            features['macd_signal'] = macd_signal_aligned
            features['macd_histogram'] = macd_hist_aligned
            
            # MACD signals
            features['macd_bullish'] = (macd_aligned > macd_signal_aligned).astype(int)
            features['macd_cross_up'] = ((macd_aligned > macd_signal_aligned) & 
                                       (macd_aligned.shift(1) <= macd_signal_aligned.shift(1))).astype(int)
        
        # Volatility
        if 'volatility' in indicators.columns:
            vol_aligned = indicators['volatility'].reindex(features.index, method='ffill')
            features['volatility'] = vol_aligned
            
            # Handle NaN values in volatility before binning
            if not vol_aligned.isna().all():
                vol_clean = vol_aligned.dropna()
                if len(vol_clean) > 0:
                    # Create bins based on non-NaN data
                    vol_regime = pd.cut(vol_aligned, bins=3, labels=[0, 1, 2])
                    # Fill NaN values with 1 (medium regime)
                    features['volatility_regime'] = vol_regime.fillna(1).astype(int)
                else:
                    features['volatility_regime'] = 1  # Default to medium regime
            else:
                features['volatility_regime'] = 1  # Default to medium regime
        
        return features
    
    def _add_economic_features(self, features: pd.DataFrame, economic_events: pd.DataFrame) -> pd.DataFrame:
        """Add economic event features"""
        logger.debug("Adding economic features")
        
        # Get unique indicators and currencies
        indicators_list = economic_events['indicator'].unique()
        currencies = economic_events['currency'].unique()
        
        # Create features for each indicator-currency combination
        for indicator in indicators_list:
            for currency in currencies:
                mask = (economic_events['indicator'] == indicator) & (economic_events['currency'] == currency)
                indicator_data = economic_events[mask].copy()
                
                if len(indicator_data) == 0:
                    continue

                feature_name = f'{indicator.lower()}_{currency.lower()}'

                value_series = None
                for column in ("actual", "forecast", "previous"):
                    if column in indicator_data.columns:
                        series = pd.to_numeric(indicator_data[column], errors='coerce')
                        if series.dropna().empty:
                            continue
                        if hasattr(series.index, "tz") and series.index.tz is not None:
                            series = series.tz_convert(None)
                        value_series = series
                        break

                if value_series is None:
                    logger.debug("Skipping economic feature %s (no usable values)", feature_name)
                    continue

                indicator_index = value_series.index
                if hasattr(features.index, "tz") and features.index.tz is not None:
                    target_index = features.index.tz_convert(None)
                else:
                    target_index = features.index

                indicator_series = value_series.reindex(target_index, method='ffill')

                if indicator_series.dropna().nunique() <= 1:
                    logger.debug("Skipping economic feature %s (constant series)", feature_name)
                    continue

                features[feature_name] = indicator_series
                features[f'{feature_name}_change'] = indicator_series.diff()
                features[f'{feature_name}_pct_change'] = indicator_series.pct_change()

                last_event_dates = indicator_data.index
                days_since_event: List[int] = []

                for date in features.index:
                    recent_events = last_event_dates[last_event_dates <= date]
                    if len(recent_events) > 0:
                        days_since = (date - recent_events[-1]).days
                    else:
                        days_since = 999
                    days_since_event.append(days_since)

                features[f'{feature_name}_days_since'] = days_since_event
        
        # Economic regime features
        if 'gdp_usd' in features.columns and 'cpi_usd' in features.columns:
            # Simple economic regime classification
            gdp_threshold = features['gdp_usd'].quantile(0.5)
            cpi_threshold = features['cpi_usd'].quantile(0.7)  # Higher threshold for inflation concern
            
            features['economic_regime'] = 0  # Base case
            features.loc[(features['gdp_usd'] > gdp_threshold) & (features['cpi_usd'] < cpi_threshold), 'economic_regime'] = 1  # Good growth, low inflation
            features.loc[(features['gdp_usd'] > gdp_threshold) & (features['cpi_usd'] >= cpi_threshold), 'economic_regime'] = 2  # Good growth, high inflation
            features.loc[(features['gdp_usd'] <= gdp_threshold) & (features['cpi_usd'] >= cpi_threshold), 'economic_regime'] = 3  # Low growth, high inflation
        
        return features
    
    def _add_time_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        logger.debug("Adding time features")
        
        # Day of week (0=Monday, 6=Sunday)
        features['day_of_week'] = features.index.dayofweek
        
        # Month of year
        features['month'] = features.index.month
        
        # Quarter
        features['quarter'] = features.index.quarter
        
        # Is month end/start
        features['is_month_end'] = features.index.is_month_end.astype(int)
        features['is_month_start'] = features.index.is_month_start.astype(int)
        
        # Is quarter end/start
        features['is_quarter_end'] = features.index.is_quarter_end.astype(int)
        features['is_quarter_start'] = features.index.is_quarter_start.astype(int)
        
        # Cyclical encoding for time features
        features['day_of_week_sin'] = np.sin(2 * np.pi * features['day_of_week'] / 7)
        features['day_of_week_cos'] = np.cos(2 * np.pi * features['day_of_week'] / 7)
        
        features['month_sin'] = np.sin(2 * np.pi * features['month'] / 12)
        features['month_cos'] = np.cos(2 * np.pi * features['month'] / 12)
        
        return features
    
    def _add_interaction_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Add interaction features between existing features"""
        logger.debug("Adding interaction features")
        
        # Volatility-Return interactions
        if 'volatility' in features.columns and 'returns' in features.columns:
            features['vol_return_interaction'] = features['volatility'] * features['returns']
        
        # RSI-Return interactions
        if 'rsi' in features.columns and 'returns' in features.columns:
            features['rsi_return_interaction'] = features['rsi'] * features['returns']
        
        # Economic-Technical interactions
        if 'gdp_usd' in features.columns and 'sma_20' in features.columns:
            features['gdp_tech_interaction'] = (features['gdp_usd'].ffill() * 
                                              features['sma_20'])
        
        return features
    
    def _finalize_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Clean up and finalize features"""
        logger.debug("Finalizing features")
        
        # Remove features with too many NaN values
        nan_threshold = 0.5  # Remove if more than 50% NaN
        features = features.loc[:, features.isnull().mean() < nan_threshold]
        
        # Forward fill remaining NaN values
        features = features.ffill()
        
        # Backward fill any remaining NaN at the beginning
        features = features.bfill()
        
        # Replace any infinite values
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(0)
        
        # Remove constant columns (with better diagnostic info)
        constant_cols = []
        for col in features.columns:
            unique_vals = features[col].nunique()
            if unique_vals <= 1:
                sample_values = features[col].dropna().head(3).tolist()
                logger.debug(
                    "Dropping constant feature %s (sample=%s)",
                    col,
                    sample_values,
                )
                constant_cols.append(col)
        
        if constant_cols:
            logger.warning(f"Removing {len(constant_cols)} constant columns: {constant_cols}")
            features = features.drop(columns=constant_cols)
            
            # If too many columns are constant, warn about data quality
            if len(constant_cols) > len(features.columns) * 0.5:
                logger.error(f"Data quality issue: {len(constant_cols)} constant columns removed out of {len(features.columns) + len(constant_cols)} total. Check data source and time range.")
        
        # Final validation
        if features.empty:
            raise ValueError("No valid features remaining after preprocessing")
        
        logger.info(f"Feature engineering completed: {len(features.columns)} features, {len(features)} samples")
        
        return features
    
    def get_target_variables(self, 
                           prices: pd.DataFrame, 
                           horizons: List[int] = None) -> pd.DataFrame:
        """
        Create target variables for multiple prediction horizons
        """
        if horizons is None:
            horizons = [1, 7, 30]
        
        targets = pd.DataFrame(index=prices.index)
        
        for horizon in horizons:
            # Future returns (main prediction target for LSTM)
            targets[f'return_{horizon}d'] = prices['close'].shift(-horizon).pct_change(fill_method=None)
            
            # Note: Only using returns as the main prediction target
            # Direction and price levels can be derived from returns
        
        return targets
    
    def get_feature_importance_names(self) -> Dict[str, List[str]]:
        """Get feature names grouped by category for importance analysis"""
        categories = {
            'price': [name for name in self.feature_names if any(x in name for x in ['close', 'open', 'high', 'low', 'returns', 'hl_ratio'])],
            'technical': [name for name in self.feature_names if any(x in name for x in ['sma', 'rsi', 'macd', 'bb_', 'volatility'])],
            'economic': [name for name in self.feature_names if any(x in name for x in ['gdp', 'cpi', 'unemployment', 'interest_rate'])],
            'temporal': [name for name in self.feature_names if any(x in name for x in ['day_of', 'month', 'quarter', 'is_'])],
            'interaction': [name for name in self.feature_names if 'interaction' in name]
        }
        
        return categories
