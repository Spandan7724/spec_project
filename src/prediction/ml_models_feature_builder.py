"""
Feature Builder compatible with ml_models advanced feature engineering.
This module creates features that match exactly what the ml_models were trained on.
"""

import pandas as pd
import numpy as np
from ta.trend import MACD, EMAIndicator, SMAIndicator, ADXIndicator
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator
from ta.volatility import BollingerBands, AverageTrueRange
import logging

logger = logging.getLogger(__name__)


class MLModelsFeatureBuilder:
    """
    Feature builder that creates the exact features used in ml_models training.
    Compatible with the 174 features from train_forex_models.py
    """

    def __init__(self):
        self.feature_columns = []

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create advanced features matching ml_models training.

        Args:
            df: DataFrame with columns: Date, Open, High, Low, Close

        Returns:
            DataFrame with all engineered features
        """
        try:
            data = df.copy()

            # Ensure Date is datetime
            if 'Date' in data.columns:
                data['Date'] = pd.to_datetime(data['Date'])

            # Sort by date
            data = data.sort_values('Date').reset_index(drop=True)

            logger.info(f"Building features for {len(data)} rows")

            # ============================================================================
            # Basic price features
            # ============================================================================
            data['Returns'] = data['Close'].pct_change()
            data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))
            data['HL_Pct'] = (data['High'] - data['Low']) / data['Close']
            data['OC_Pct'] = (data['Close'] - data['Open']) / data['Open']

            # ============================================================================
            # Price momentum and changes
            # ============================================================================
            for period in [1, 2, 3, 5, 7, 14, 21, 30]:
                data[f'Price_Change_{period}d'] = data['Close'].pct_change(period)
                data[f'High_Change_{period}d'] = data['High'].pct_change(period)
                data[f'Low_Change_{period}d'] = data['Low'].pct_change(period)

            # ============================================================================
            # Moving averages
            # ============================================================================
            for window in [5, 10, 20, 50, 100, 200]:
                data[f'SMA_{window}'] = SMAIndicator(close=data['Close'], window=window).sma_indicator()
                data[f'EMA_{window}'] = EMAIndicator(close=data['Close'], window=window).ema_indicator()
                data[f'Close_to_SMA_{window}'] = (data['Close'] - data[f'SMA_{window}']) / data[f'SMA_{window}']

            # MA crossovers
            data['SMA_5_20_cross'] = data['SMA_5'] - data['SMA_20']
            data['SMA_10_50_cross'] = data['SMA_10'] - data['SMA_50']
            data['SMA_50_200_cross'] = data['SMA_50'] - data['SMA_200']

            # ============================================================================
            # RSI for multiple periods
            # ============================================================================
            for period in [7, 14, 21, 28]:
                data[f'RSI_{period}'] = RSIIndicator(close=data['Close'], window=period).rsi()

            # ============================================================================
            # MACD
            # ============================================================================
            macd = MACD(close=data['Close'])
            data['MACD'] = macd.macd()
            data['MACD_Signal'] = macd.macd_signal()
            data['MACD_Diff'] = macd.macd_diff()

            # ============================================================================
            # Bollinger Bands
            # ============================================================================
            for window in [20, 50]:
                bb = BollingerBands(close=data['Close'], window=window, window_dev=2)
                data[f'BB_High_{window}'] = bb.bollinger_hband()
                data[f'BB_Low_{window}'] = bb.bollinger_lband()
                data[f'BB_Mid_{window}'] = bb.bollinger_mavg()
                data[f'BB_Width_{window}'] = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()
                data[f'BB_Position_{window}'] = (data['Close'] - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband())

            # ============================================================================
            # ATR (Average True Range)
            # ============================================================================
            for period in [7, 14, 21]:
                data[f'ATR_{period}'] = AverageTrueRange(high=data['High'], low=data['Low'], close=data['Close'], window=period).average_true_range()

            # ============================================================================
            # ADX (Average Directional Index)
            # ============================================================================
            adx = ADXIndicator(high=data['High'], low=data['Low'], close=data['Close'], window=14)
            data['ADX'] = adx.adx()
            data['ADX_Pos'] = adx.adx_pos()
            data['ADX_Neg'] = adx.adx_neg()

            # ============================================================================
            # Stochastic Oscillator
            # ============================================================================
            stoch = StochasticOscillator(high=data['High'], low=data['Low'], close=data['Close'])
            data['Stoch_K'] = stoch.stoch()
            data['Stoch_D'] = stoch.stoch_signal()

            # ============================================================================
            # Williams %R
            # ============================================================================
            data['Williams_R'] = WilliamsRIndicator(high=data['High'], low=data['Low'], close=data['Close']).williams_r()

            # ============================================================================
            # Volatility measures
            # ============================================================================
            for window in [5, 10, 20, 30]:
                data[f'Volatility_{window}'] = data['Returns'].rolling(window=window).std()
                data[f'Volatility_HL_{window}'] = (data['High'] - data['Low']).rolling(window=window).std()

            # ============================================================================
            # Statistical features
            # ============================================================================
            for window in [5, 10, 20, 30, 50, 60, 100]:
                data[f'Rolling_Mean_{window}'] = data['Close'].rolling(window=window).mean()
                data[f'Rolling_Std_{window}'] = data['Close'].rolling(window=window).std()
                data[f'Rolling_Min_{window}'] = data['Close'].rolling(window=window).min()
                data[f'Rolling_Max_{window}'] = data['Close'].rolling(window=window).max()
                data[f'Rolling_Median_{window}'] = data['Close'].rolling(window=window).median()
                data[f'Rolling_Skew_{window}'] = data['Returns'].rolling(window=window).skew()
                data[f'Rolling_Kurt_{window}'] = data['Returns'].rolling(window=window).kurt()

            # ============================================================================
            # Distance from extremes
            # ============================================================================
            for window in [20, 50, 100]:
                data[f'Distance_from_High_{window}'] = (data[f'Rolling_Max_{window}'] - data['Close']) / data['Close']
                data[f'Distance_from_Low_{window}'] = (data['Close'] - data[f'Rolling_Min_{window}']) / data['Close']

            # ============================================================================
            # Lag features
            # ============================================================================
            for lag in [1, 2, 3, 5, 7, 14, 21, 30]:
                data[f'Close_Lag_{lag}'] = data['Close'].shift(lag)
                data[f'Returns_Lag_{lag}'] = data['Returns'].shift(lag)
                data[f'Volume_Proxy_Lag_{lag}'] = data['HL_Pct'].shift(lag)  # Using HL% as volume proxy

            # ============================================================================
            # Time-based features
            # ============================================================================
            if 'Date' in data.columns:
                data['Day_of_Week'] = data['Date'].dt.dayofweek
                data['Day_of_Month'] = data['Date'].dt.day
                data['Month'] = data['Date'].dt.month
                data['Quarter'] = data['Date'].dt.quarter
                data['Year'] = data['Date'].dt.year

                # Cyclical encoding for time features
                data['Day_of_Week_Sin'] = np.sin(2 * np.pi * data['Day_of_Week'] / 7)
                data['Day_of_Week_Cos'] = np.cos(2 * np.pi * data['Day_of_Week'] / 7)
                data['Month_Sin'] = np.sin(2 * np.pi * data['Month'] / 12)
                data['Month_Cos'] = np.cos(2 * np.pi * data['Month'] / 12)

            # ============================================================================
            # Trend indicators
            # ============================================================================
            for window in [10, 20, 50]:
                data[f'Trend_{window}'] = data['Close'].rolling(window=window).apply(
                    lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == window else np.nan,
                    raw=True
                )

            # Drop rows with NaN (from rolling windows and lags)
            # Keep track of which features we created
            exclude_cols = ['Date', 'Close', 'Open', 'High', 'Low']
            self.feature_columns = [col for col in data.columns if col not in exclude_cols]

            logger.info(f"Created {len(self.feature_columns)} features")

            return data

        except Exception as e:
            logger.error(f"Error building features: {e}")
            raise

    def get_feature_columns(self):
        """Return list of feature column names"""
        return self.feature_columns

    def prepare_for_prediction(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for prediction by building features and selecting only feature columns.

        Args:
            df: Raw OHLC data

        Returns:
            DataFrame with only feature columns, ready for model input
        """
        # Build all features
        df_with_features = self.build_features(df)

        # Get the last row (most recent data)
        if len(df_with_features) == 0:
            raise ValueError("No data after feature engineering")

        # Select only the feature columns (exclude Date, OHLC)
        exclude_cols = ['Date', 'Close', 'Open', 'High', 'Low', 'Target_1d', 'Target_3d', 'Target_7d']
        feature_cols = [col for col in df_with_features.columns if col not in exclude_cols]

        # Take the last row and return only features
        latest_features = df_with_features[feature_cols].iloc[[-1]].copy()

        # Drop any remaining NaN values (fill with 0 as fallback)
        latest_features = latest_features.fillna(0)

        return latest_features
