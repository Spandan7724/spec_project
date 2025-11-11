#!/usr/bin/env python3
"""
Generate Figure 4: Before and After Preprocessing Comparison
Shows raw forex data vs preprocessed/normalized data with technical indicators
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.prediction.data_loader import HistoricalDataLoader
from src.prediction.feature_builder import FeatureBuilder
from src.prediction.config import PredictionConfig


async def fetch_data(base: str = "USD", quote: str = "EUR", days: int = 30):
    """Fetch historical forex data"""
    loader = HistoricalDataLoader()
    df = await loader.fetch_historical_data(
        base=base,
        quote=quote,
        days=days,
        interval="1d"
    )
    return df


def preprocess_data(df: pd.DataFrame):
    """Apply preprocessing and feature engineering"""
    # Load config to get technical indicators
    config = PredictionConfig.from_yaml()

    # Build features using the existing feature builder
    feature_builder = FeatureBuilder(indicators=config.technical_indicators)

    # Build features (this adds all technical indicators)
    features_df = feature_builder.build_features(
        df=df.copy(),
        mode="price_only",
        market_intel=None
    )

    return features_df


def create_figure_4(raw_df: pd.DataFrame, processed_df: pd.DataFrame, output_path: Path):
    """
    Create two-panel figure showing before/after preprocessing

    Args:
        raw_df: Raw OHLC data from Yahoo Finance
        processed_df: Preprocessed data with technical indicators
        output_path: Where to save the figure
    """
    # Select a 1-week window for visualization from the processed data
    # (which has already dropped NaN values from indicator calculation)
    if len(processed_df) < 7:
        # If we have less than 7 days after preprocessing, use all available
        processed_window = processed_df
        # Match the window in raw data
        start_date = processed_df.index[0]
        end_date = processed_df.index[-1]
        raw_window = raw_df[(raw_df.index >= start_date) & (raw_df.index <= end_date)]
    else:
        # Use the most recent 7 data points from processed data
        processed_window = processed_df.iloc[-7:]
        # Match the window in raw data
        start_date = processed_window.index[0]
        end_date = processed_window.index[-1]
        raw_window = raw_df[(raw_df.index >= start_date) & (raw_df.index <= end_date)]

    # Create figure with 2 panels
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle('Fig. 4: Forex Data Before and After Preprocessing',
                 fontsize=14, fontweight='bold', y=0.995)

    # ==================== Panel (a): Raw Data ====================
    ax1.plot(raw_window.index, raw_window['Close'],
             color='#2E86AB', linewidth=2, label='Close Price')
    ax1.fill_between(raw_window.index,
                      raw_window['Low'],
                      raw_window['High'],
                      alpha=0.2, color='#2E86AB', label='High-Low Range')

    ax1.set_title('(a) Raw Exchange-Rate Data (Before Preprocessing)',
                  fontsize=12, fontweight='bold', pad=10)
    ax1.set_xlabel('Date', fontsize=10)
    ax1.set_ylabel('USD/EUR Exchange Rate', fontsize=10)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax1.tick_params(axis='x', rotation=45)

    # Add annotation showing raw characteristics
    raw_mean = raw_window['Close'].mean()
    raw_std = raw_window['Close'].std()
    raw_missing = raw_window['Close'].isna().sum()

    textstr = f'Mean: {raw_mean:.4f}\nStd Dev: {raw_std:.4f}\nMissing: {raw_missing}'
    ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes,
             fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # ==================== Panel (b): Preprocessed Data ====================
    # Normalize the close price for visualization
    processed_window_copy = processed_window.copy()

    # Check if we have data
    if len(processed_window_copy) == 0:
        print("Warning: No data available after preprocessing for visualization")
        return None

    scaler = StandardScaler()

    # Normalize Close price
    close_normalized = scaler.fit_transform(
        processed_window_copy[['Close']].values
    ).flatten()

    # Plot normalized close price
    ax2.plot(processed_window_copy.index, close_normalized,
             color='#A23B72', linewidth=2, label='Normalized Close', zorder=3)

    # Add technical indicators if they exist
    if 'sma_20' in processed_window_copy.columns:
        sma_20_norm = scaler.fit_transform(
            processed_window_copy[['sma_20']].dropna().values
        ).flatten()
        ax2.plot(processed_window_copy.dropna().index[-len(sma_20_norm):],
                 sma_20_norm,
                 color='#F18F01', linewidth=1.5,
                 linestyle='--', label='SMA(20)', alpha=0.8)

    if 'ema_12' in processed_window_copy.columns:
        ema_12_norm = scaler.fit_transform(
            processed_window_copy[['ema_12']].dropna().values
        ).flatten()
        ax2.plot(processed_window_copy.dropna().index[-len(ema_12_norm):],
                 ema_12_norm,
                 color='#C73E1D', linewidth=1.5,
                 linestyle=':', label='EMA(12)', alpha=0.8)

    # Add Bollinger Bands if they exist
    if all(col in processed_window_copy.columns for col in ['bb_upper', 'bb_lower']):
        bb_data = processed_window_copy[['bb_upper', 'bb_middle', 'bb_lower']].dropna()
        bb_upper_norm = scaler.fit_transform(bb_data[['bb_upper']].values).flatten()
        bb_middle_norm = scaler.fit_transform(bb_data[['bb_middle']].values).flatten()
        bb_lower_norm = scaler.fit_transform(bb_data[['bb_lower']].values).flatten()

        ax2.fill_between(bb_data.index, bb_lower_norm, bb_upper_norm,
                         alpha=0.15, color='green', label='Bollinger Bands')
        ax2.plot(bb_data.index, bb_middle_norm,
                 color='green', linewidth=1, linestyle='-.', alpha=0.6)

    ax2.set_title('(b) Normalized and Feature-Enhanced Data (After Preprocessing)',
                  fontsize=12, fontweight='bold', pad=10)
    ax2.set_xlabel('Date', fontsize=10)
    ax2.set_ylabel('Normalized Value (Z-score)', fontsize=10)
    ax2.legend(loc='upper right', fontsize=9, ncol=2)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax2.tick_params(axis='x', rotation=45)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)

    # Add annotation showing preprocessing applied
    features_added = [col for col in processed_window_copy.columns
                      if col in ['sma_5', 'sma_20', 'ema_12', 'rsi_14',
                                 'macd', 'bb_upper', 'atr_14', 'volatility_20']]

    textstr = (f'Transformations Applied:\n'
               f'• Normalization (Z-score)\n'
               f'• Missing Values Handled\n'
               f'• {len(features_added)} Technical Indicators\n'
               f'  (SMA, EMA, RSI, MACD, BB, ATR)')
    ax2.text(0.02, 0.98, textstr, transform=ax2.transAxes,
             fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Figure saved to: {output_path}")

    return fig


async def main():
    """Main execution function"""
    print("=" * 60)
    print("Generating Figure 4: Before/After Preprocessing Comparison")
    print("=" * 60)

    # Configuration
    BASE = "USD"
    QUOTE = "EUR"
    DAYS = 90  # Fetch 90 days to ensure we have enough data after indicator calculation

    print(f"\n1. Fetching {DAYS} days of {BASE}/{QUOTE} data...")
    raw_df = await fetch_data(BASE, QUOTE, DAYS)
    print(f"   ✓ Fetched {len(raw_df)} data points")
    print(f"   Date range: {raw_df.index[0]} to {raw_df.index[-1]}")

    print(f"\n2. Applying preprocessing and feature engineering...")
    processed_df = preprocess_data(raw_df)
    print(f"   ✓ Generated {len(processed_df.columns)} features")
    print(f"   Features: {', '.join(processed_df.columns[:10])}...")

    print(f"\n3. Creating visualization...")
    output_path = project_root / "research_paper" / "figure_4_preprocessing_comparison.png"
    fig = create_figure_4(raw_df, processed_df, output_path)

    print(f"\n✓ Complete! Figure 4 generated successfully.")
    print(f"\nOutput location: {output_path}")
    print(f"Figure shows:")
    print(f"  • Panel (a): Raw USD/EUR exchange rates (1 week window)")
    print(f"  • Panel (b): Normalized data with technical indicators")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
