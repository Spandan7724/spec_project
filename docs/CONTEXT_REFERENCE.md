# Context Reference (Concise)

This document lists the fields included in the chat/answer context after a recommendation is generated. Fields come from the user parameters, agent state, and the decision output.

- Parameters
  - `currency_pair` (e.g., USD/EUR)
  - `amount` (float)
  - `risk_tolerance` (conservative|moderate|aggressive)
  - `urgency` (urgent|normal|flexible)
  - `timeframe` (immediate|1_day|1_week|1_month)
  - `timeframe_days` (int; derived or explicit, e.g., 10)
  - `timeframe_mode` (immediate|deadline|duration)
  - `deadline_utc` (ISO timestamp when provided, e.g., "2025-11-15T00:00:00Z")
  - `window_days` (range object for phrases like "3–5 days", e.g., `{start: 3, end: 5}`)
  - `time_unit` ("hours"|"days"; optional hint)
  - `timeframe_hours` (int; for sub-day horizons like "in 12 hours")

- Recommendation
  - `action` (convert_now|staged_conversion|wait)
  - `confidence` (0.0–1.0)
  - `timeline` (human text)
  - `rationale` (list of short reasons)
  - `risk_summary`
    - `risk_level` (low|moderate|high)
    - `realized_vol_30d` (pct)
    - `var_95` (pct)
    - `event_risk` + `event_details`
  - `cost_estimate`
    - `spread_bps`, `fee_bps`, `total_bps`, `staged_multiplier`
  - `expected_outcome`
    - `expected_rate`, `range_low`, `range_high`, `expected_improvement_bps`
  - `staged_plan` (when applicable)
    - `num_tranches`, `spacing_days`, `total_extra_cost_bps`, `tranches[] { tranche_number, percentage, execute_day, rationale }`, `benefit`
  - `utility_scores` (map of action → score)
  - `component_confidences` (map of component → 0.0–1.0)
  - `meta`
    - `prediction_horizon_days`, `used_prediction_horizon_key`

- Evidence
  - `market`
    - `providers` (list of source names)
    - `quality_notes` (list)
    - `dispersion_bps`
    - `mid_rate`, `bid`, `ask`, `spread`, `rate_timestamp`
    - `indicators` (technical)
      - `sma_20`, `sma_50`, `ema_12`, `ema_26`, `rsi_14`, `macd`, `macd_signal`, `macd_histogram`, `bb_middle`, `bb_upper`, `bb_lower`, `bb_position`, `atr_14`, `realized_vol_30d`
    - `regime` { `trend_direction`, `bias` }
    - `provider_quotes[]` (up to 5) { `source`, `rate`, `timestamp` }
  - `news` (top citations, up to 5)
    - items: { `source`, `title`, `url` }
  - `calendar` (future-facing subset, up to 10)
    - items: { `currency`, `event`, `importance`, `source_url`, `proximity_minutes` }
  - `intelligence` (summary)
    - `pair_bias`, `news_confidence`, `n_articles_used`, `narrative`
    - `policy_bias`
    - `next_high_event` { `when_utc`, `currency`, `event`, `source_url`, `proximity_minutes`, `is_imminent` }
    - `total_high_impact_events_7d`
  - `model`
    - `horizon_key`, `horizon_days`, `top_features` (feature → importance), `model_id`, `model_confidence`
  - `prediction`
    - `horizon_key`, `mean_change_pct`, `quantiles` (e.g., 0.05/0.95), `direction_prob`, `latest_close`
  - `predictions_all` (map horizon → `mean_change_pct`)
  - `utility_scores` (duplicated for convenience)

- System Metadata (not passed to LLM as-is)
  - `metadata` { `correlation_id`, `timestamp` } at the top level of the recommendation payload returned to the UI

Notes
- Counts and truncation
  - News: top 5 items
  - Calendar: up to 10 future events
  - Provider quotes: up to 5 entries
- Sources
  - Market fields from `market_snapshot`
  - Intelligence from `intelligence_report`
  - Prediction fields from `price_forecast`
  - Recommendation fields from the decision engine response
