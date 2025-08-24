  1. Change Prediction Horizon (How Far Ahead to Predict)

  In the Model Configuration section (cell with ModelConfig), modify the prediction_horizon parameter:

  model_config = ModelConfig(
      sequence_length=168,  # 1 week of hourly data (7 * 24)
      prediction_horizon=72,  # CHANGE THIS: Predict 72 hours (3 days) ahead
      hidden_size=64,
      num_layers=2,
      dropout=0.2,
      attention_heads=8
  )

  Common prediction horizons:
  - prediction_horizon=12 → 12 hours ahead
  - prediction_horizon=24 → 1 day ahead (default)
  - prediction_horizon=48 → 2 days ahead
  - prediction_horizon=72 → 3 days ahead
  - prediction_horizon=168 → 1 week ahead

  2. Change Training Data Period (How Much Historical Data)

  In the Data Collection section, modify the DATA_PERIOD parameter:

  # Configuration for data collection
  CURRENCY_PAIRS = ['USD/EUR', 'USD/GBP', 'EUR/GBP']
  DATA_PERIOD = "2y"  # CHANGE THIS: Use 2 years of historical data
  DATA_INTERVAL = "1h"  # Hourly data

  Available data periods:
  - DATA_PERIOD = "6mo" → 6 months
  - DATA_PERIOD = "1y" → 1 year (default)
  - DATA_PERIOD = "2y" → 2 years
  - DATA_PERIOD = "5y" → 5 years
  - DATA_PERIOD = "10y" → 10 years
  - DATA_PERIOD = "max" → Maximum available data

  3. Example: Predict 1 Week Ahead Using 2 Years of Data

  # In the Model Configuration cell:
  model_config = ModelConfig(
      sequence_length=168,
      prediction_horizon=168,  # Predict 1 week (168 hours) ahead
      hidden_size=64,
      num_layers=2,
      dropout=0.2,
      attention_heads=8
  )

  # In the Data Collection cell:
  DATA_PERIOD = "2y"  # Use 2 years of training data