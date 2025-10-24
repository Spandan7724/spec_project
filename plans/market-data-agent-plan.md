# Market Data Agent — Data Contract, Schemas, and Validation

## Purpose
Aggregate live FX rates from multiple providers, compute core technical context from recent OHLC, and publish a normalized market snapshot for downstream agents (Decision, Prediction).

Providers in scope: ExchangeRate.host + yfinance (Fixer skipped).

---

## Data Contracts (Schemas)

The agent returns a standard envelope plus a `data` object. Schemas below are normative for producer/consumer code.

### 1) Envelope
```json
{
  "status": "success | partial | error",
  "confidence": 0.0,
  "processing_time_ms": 0,
  "warnings": ["..."],
  "metadata": {
    "version": "1.0",
    "data_sources": ["exchange_rate_host", "yfinance"]
  },
  "data": { /* LiveSnapshot */ }
}
```

### 2) ProviderRate
```json
{
  "type": "object",
  "required": ["source", "rate", "timestamp"],
  "properties": {
    "source": {"type": "string", "enum": ["exchange_rate_host", "yfinance"]},
    "rate": {"type": "number", "exclusiveMinimum": 0},
    "bid": {"type": ["number", "null"], "exclusiveMinimum": 0},
    "ask": {"type": ["number", "null"], "exclusiveMinimum": 0},
    "timestamp": {"type": "string", "format": "date-time"},
    "notes": {"type": "array", "items": {"type": "string"}}
  }
}
```

### 3) LiveSnapshot
```json
{
  "type": "object",
  "required": [
    "currency_pair", "rate_timestamp", "mid_rate",
    "provider_breakdown", "quality", "indicators", "regime"
  ],
  "properties": {
    "currency_pair": {"type": "string", "pattern": "^[A-Z]{3}/[A-Z]{3}$"},
    "rate_timestamp": {"type": "string", "format": "date-time"},
    "mid_rate": {"type": "number", "exclusiveMinimum": 0},
    "bid": {"type": ["number", "null"], "exclusiveMinimum": 0},
    "ask": {"type": ["number", "null"], "exclusiveMinimum": 0},
    "spread": {"type": ["number", "null"], "minimum": 0},

    "provider_breakdown": {
      "type": "array",
      "minItems": 1,
      "items": {"$ref": "#/definitions/ProviderRate"}
    },

    "quality": {
      "type": "object",
      "required": ["sources_success", "sources_total", "dispersion_bps", "fresh", "notes"],
      "properties": {
        "sources_success": {"type": "integer", "minimum": 0},
        "sources_total": {"type": "integer", "minimum": 1},
        "dispersion_bps": {"type": "number", "minimum": 0},
        "fresh": {"type": "boolean"},
        "notes": {"type": "array", "items": {"type": "string"}}
      }
    },

    "indicators": {"$ref": "#/definitions/Indicators"},
    "regime": {"$ref": "#/definitions/Regime"}
  },
  "definitions": {
    "ProviderRate": {},
    "Indicators": {},
    "Regime": {}
  }
}
```

### 4) Indicators
```json
{
  "type": "object",
  "required": [
    "sma_20", "sma_50", "ema_12", "ema_26", "rsi_14",
    "macd", "macd_signal", "macd_histogram",
    "bb_middle", "bb_upper", "bb_lower", "bb_position",
    "atr_14", "realized_vol_30d"
  ],
  "properties": {
    "sma_20": {"type": ["number", "null"]},
    "sma_50": {"type": ["number", "null"]},
    "ema_12": {"type": ["number", "null"]},
    "ema_26": {"type": ["number", "null"]},
    "rsi_14": {"type": ["number", "null"], "minimum": 0, "maximum": 100},
    "macd": {"type": ["number", "null"]},
    "macd_signal": {"type": ["number", "null"]},
    "macd_histogram": {"type": ["number", "null"]},
    "bb_middle": {"type": ["number", "null"]},
    "bb_upper": {"type": ["number", "null"]},
    "bb_lower": {"type": ["number", "null"]},
    "bb_position": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
    "atr_14": {"type": ["number", "null"]},
    "realized_vol_30d": {"type": ["number", "null"]}
  }
}
```

### 5) Regime
```json
{
  "type": "object",
  "required": ["trend_direction", "bias"],
  "properties": {
    "trend_direction": {"type": ["string", "null"], "enum": ["up", "down", "sideways", null]},
    "bias": {"type": ["string", "null"], "enum": ["bullish", "bearish", "neutral", null]}
  }
}
```

### 6) HistoricalBar (yfinance OHLC)
```json
{
  "type": "object",
  "required": ["ts_utc", "open", "high", "low", "close"],
  "properties": {
    "ts_utc": {"type": "string", "format": "date-time"},
    "open": {"type": "number"},
    "high": {"type": "number"},
    "low": {"type": "number"},
    "close": {"type": "number"},
    "volume": {"type": ["number", "null"]}
  }
}
```

---

## Example Output (LiveSnapshot `data`)
```json
{
  "currency_pair": "USD/EUR",
  "rate_timestamp": "2025-10-23T18:55:58Z",
  "mid_rate": 0.86085,
  "bid": 0.8606,
  "ask": 0.8611,
  "spread": 0.0005,
  "provider_breakdown": [
    {
      "source": "exchange_rate_host",
      "rate": 0.86084,
      "bid": null,
      "ask": null,
      "timestamp": "2025-10-23T18:55:05Z"
    },
    {
      "source": "yfinance",
      "rate": 0.86085,
      "bid": 0.8606,
      "ask": 0.8611,
      "timestamp": "2025-10-23T18:55:58Z"
    }
  ],
  "quality": {
    "sources_success": 2,
    "sources_total": 2,
    "dispersion_bps": 0.12,
    "fresh": true,
    "notes": []
  },
  "indicators": {
    "sma_20": 0.8599,
    "sma_50": 0.8562,
    "ema_12": 0.8602,
    "ema_26": 0.8589,
    "rsi_14": 58.3,
    "macd": 0.0011,
    "macd_signal": 0.0009,
    "macd_histogram": 0.0002,
    "bb_middle": 0.8599,
    "bb_upper": 0.8655,
    "bb_lower": 0.8543,
    "bb_position": 0.62,
    "atr_14": 0.0031,
    "realized_vol_30d": 0.07
  },
  "regime": {
    "trend_direction": "up",
    "bias": "bullish"
  }
}
```

---

## Validation Checklist

- Currency pair
  - Must match `^[A-Z]{3}/[A-Z]{3}$`; reject invalid or identical codes.
- Timestamps
  - All timestamps UTC ISO‑8601 (append `Z` or `+00:00`).
- Rates
  - `mid_rate > 0`; if `bid` and `ask` present: `0 < bid <= ask` and `spread = ask − bid`.
  - Provider rates must be positive; discard outliers > X bps from median (configurable).
- Consensus
  - `mid_rate` = median of provider rates (use (bid+ask)/2 when available).
  - `rate_timestamp` = max(provider timestamps).
- Quality
  - `sources_total = len(provider_breakdown)`; `sources_success >= 1`.
  - `dispersion_bps = ((max − min)/min) × 10000` over provider rates.
  - `fresh = true` only if latest timestamp and/or cache age within TTL.
- Indicators
  - Require minimum N bars (configurable) or set fields to null with a warning.
  - RSI in [0,100]; `bb_position` in [0,1].
- Regime
  - `trend_direction ∈ {up, down, sideways, null}`; `bias ∈ {bullish, bearish, neutral, null}`.
- Provider names
  - Use canonical strings: `exchange_rate_host`, `yfinance`.

---

## Configuration & Defaults

- ENV
  - `EXCHANGE_RATE_HOST_API_KEY` (optional)
  - `MARKET_CACHE_TTL_SECONDS` (e.g., 5)
  - `HISTORICAL_LOOKBACK_DAYS` (e.g., 90)
  - `INDICATOR_MIN_HISTORY_DAYS` (e.g., 50)
  - `OFFLINE_DEMO` (true/false)
- Runtime
  - Provider enable/disable; outlier threshold (bps); lookback override; correlation_id.

---

## Fallbacks & Offline

- If all live providers fail → serve cached snapshot (mark `fresh=false`, add warning) or return `partial/error` with whatever is available.
- If historical fetch fails → return live snapshot with `indicators = null` and a warning.
- Offline/demo mode → load a canned USD/EUR snapshot and OHLC slice from fixtures; still follow the same schema.

---

## File Layout (Suggested)

- `src/data_collection/providers/exchange_rate_host_client.py`
- `src/data_collection/providers/yfinance_client.py`
- `src/data_collection/market_data/aggregator.py`
- `src/data_collection/market_data/indicators.py`
- `src/data_collection/market_data/snapshot.py`
- `src/data_collection/market_data/cache.py` (optional)
- `tests/data_collection/market_data/*` (fixtures + tests)

