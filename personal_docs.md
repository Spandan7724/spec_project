 run all tests including integration tests:
 pytest tests/ -v --tb=short
 pytest tests/ -q

 for specific ones 
 pytest tests/unit/test_graph.py -v
 to train the ml models
 currency-assistant train-model --pair USD/EUR -d 365 -h 1 -h 7 -h 30 -v 1.0
 
 
 Train and register LSTM (programmatic)
 from src.prediction.training import train_and_register_lstm
 await train_and_register_lstm("USD/EUR", days=120, interval="1h", horizons_hours=[1,4,24])
 
 
 LightGBM daily:
 currency-assistant train-model --pair USD/EUR -d 365 -h 1 -h 7 -h 30 -v 1.0
 LSTM intraday:
 currency-assistant train-lstm --pair USD/EUR -d 120 -i 1h -H 1 -H 4 -H 24 -v 1.0
 
 
 Options:
 --pair/-p (required): currency pair (e.g., USD/EUR)
 --days/-d (default 180): intraday history length
 --interval/-i (default "1h"): intraday interval
 --horizon-hours/-H (repeatable): intraday horizons in hours (e.g., -H 1 -H 4 -H 24)
 --version/-v (default "1.0")
 
 
 
 
 
 currency-assistant train-model --pair USD/EUR -d 365 -h 1 -h 7 -h 30
 currency-assistant train-lstm --pair USD/EUR -d 120 -i 1h -H 1 -H 4 -H 24
 
 
 
 
 
 ightGBM (train-model)
 
 New options
 --gbm-rounds INT: total boosting rounds (applies to mean, quantile, and direction models)
 --gbm-patience INT: early-stopping patience (rounds)
 --gbm-lr FLOAT: learning rate override
 --gbm-leaves INT: number of leaves override
 Changes
 src/prediction/backends/lightgbm_backend.py: train() now accepts num_boost_round and patience
 src/prediction/training.py: train_and_register_lightgbm(...) accepts these knobs and passes them through
 src/cli/main.py: wires CLI options to helper
 Example
 currency-assistant train-model --pair USD/EUR -d 365 -h 1 -h 7 -h 30 --gbm-rounds 600 --gbm-patience 30 --gbm-lr 0.03 --gbm-leaves 63
 LSTM (train-lstm)
 
 New options
 --lstm-epochs INT: training epochs (default 5)
 --lstm-hidden-dim INT: hidden size (default 64)
 --lstm-seq-len INT: sequence length (default 64)
 --lstm-lr FLOAT: learning rate (default 1e-3)
 Changes
 src/prediction/training.py: train_and_register_lstm(...) accepts and passes seq_len/hidden_dim and epochs/lr
 src/cli/main.py: wires CLI options to helper
 Example
 currency-assistant train-lstm --pair USD/EUR -d 120 -i 1h -H 1 -H 4 -H 24 --lstm-epochs 40 --lstm-hidden-dim 128 --lstm-seq-len 96 --lstm-lr 5e-4
 
 
 
 
 
 
 LightGBM (daily)
 Command: currency-assistant train-model
 Prompts:
 Currency pair (default USD/EUR)
 History (days) (default from config)
 Horizons (comma-separated; default from config)
 Optional advanced settings if you choose to tune:
 GBM rounds (default 120)
 Early-stopping patience (default 10)
 Learning rate (default 0.05)
 Num leaves (default 31)
 LSTM (intraday)
 Command: currency-assistant train-lstm
 Prompts:
 Currency pair (default USD/EUR)
 Intraday history (days) (default 180)
 Interval (default 1h)
 Intraday horizons (hours, comma-separated; default 1,4,24)
 Optional advanced settings if you choose to tune:
 Epochs (default 5)
 Hidden dim (default 64)
 Sequence length (default 64)
 Learning rate (default 1e-3)
 
 
 
 
 
  I split training into its own CLI module and exposed it both as a sub‑app and a separate entrypoint, while keeping your existing commands for backward compatibility.
 
 What I changed
 
 New training CLI
 src/cli/train.py
 Commands:
 model: interactive or flag‑driven LightGBM training
 lstm: interactive or flag‑driven LSTM training
 Prompts for missing values; optional advanced tuning prompts
 Main CLI mounts training as sub‑app
 src/cli/main.py
 app.add_typer(train_app, name="train")
 Your existing train‑model and train‑lstm commands remain (no breaking change)
 Separate entrypoint
 pyproject.toml
 Added script: currency-assistant-train = "src.cli.train:app"
 
 How to use
 
 Unified (sub‑app):
 currency-assistant train model
 currency-assistant train lstm
 Dedicated training entrypoint:
 currency-assistant-train model
 currency-assistant-train lstm
 Backward compatible (existing commands still work):
 currency-assistant train-model ...
 currency-assistant train-lstm ...
 
 
 
 
 
 ________________________________________________________________________________________________________________________________
 
 Here’s what the two model backends “find out,” and whether both are required.
 
 What the models estimate
 
 Expected % change (signal): For each requested horizon, they estimate the expected percentage move of the pair (e.g., +0.25% over 1 day, −0.10% over 4 hours).
 Uncertainty (quantiles): Bounds for the move (p10/p50/p90). Tighter bands → higher certainty; wider bands → higher risk.
 Direction probability: Probability the move is up vs down (P(up)). Useful for confidence gating and staging.
 Who computes what
 
 LightGBM (daily/weekly horizons: 1, 7, 30 days)
 Outputs: mean % change, quantiles, direction probability
 Best for medium-term timing and plan shaping (today vs this week vs this month)
 LSTM (intraday horizons: 1h, 4h, 24h)
 Outputs: mean % change, and (via MC‑dropout) quantiles and direction probability
 Best for near-term execution windows and finer timing
 How the system uses them
 
 Decision Engine inputs:
 Expected improvement (from mean % change)
 Confidence (from direction probability and calibration)
 Risk/uncertainty (from quantiles → spreads/stop widths, staging)
 Timeframe mapping:
 immediate → use intraday + short daily horizons
 1_day/1_week/1_month → favor daily/weekly horizons
 Hybrid routing:
 “hybrid” = GBM for daily, LSTM for intraday, merged into one forecast
 You can force “lightgbm” or “lstm” in config or per request
 Are both required?
 
 Not strictly. The system runs with any subset:
 If only GBM is available: you get daily/weekly signals (still useful for most user timeframes)
 If only LSTM is available: you get intraday signals (useful for immediate execution)
 If neither loads: fallback heuristics kick in (last resort)
 Recommended: Having both improves coverage (planning + fine execution) and generally yields better decisions.
 Practical guidance
 
 For users with immediate or near-term conversion needs: train/use LSTM (intraday horizons).
 For users planning over days to weeks: train/use LightGBM (daily/weekly).
 Default hybrid gives the best of both; the Decision Engine weighs them based on the user’s timeframe and risk.
 
 
 
 
 
 
 
 ________________________________________________________________________________________________________________________________
 
 
 GUI FOR PHASE 2 EXTRA CITATION THING 
 
 so like in the phase 1 script which I added I addeded liek citations thign which showerd which articles were being used so it woudl b eb better for the uiux for the gui can I do something like that for this phase
 
 
 Yes — add “prediction evidence” alongside forecasts so the UI can show why.
 
 What to include
 
 LightGBM (daily/weekly)
 Top features (gain-based) per horizon (e.g., top 5)
 Optional SHAP waterfall plot for the latest row (base64) for an expanded view
 LSTM (intraday)
 MC‑dropout summary: p10/p50/p90 (already computed) and direction_prob
 mc_samples count and note that dropout drives uncertainty
 Model metadata
 model_id, trained_at, horizons, validation metrics (directional_accuracy, quantile_coverage) from registry
 Where to attach
 
 Attach an explanations payload to the prediction output so the UI can render it:
 price_forecast.explanations:
 daily: {horizon -> {top_features, shap_waterfall_base64?}}
 intraday: {horizon -> {mc_samples, quantiles, direction_prob}}
 price_forecast.model_info: {model_id, trained_at, validation_metrics}
 How to control overhead
 
 Add a flag to enable explanations only when needed:
 Config: prediction.explain.enabled, prediction.explain.include_plots, prediction.explain.top_n
 Per‑request: PredictionRequest.include_explanations = True
 Minimal changes (high level)
 
 In PredictionRequest: add include_explanations: bool = False
 In MLPredictor.predict():
 If include_explanations:
 GBM: for each daily horizon in predictions
 top_features = backend_gbm.get_feature_importance(h, top_n)
 optional: explainer = PredictionExplainer(backend_gbm.models[h], backend_gbm.feature_names)
 shap_waterfall_base64 = explainer.generate_waterfall_plot(X_latest)
 LSTM: for each intraday horizon in predictions
 add {'mc_samples': backend_lstm.mc_samples}
 Attach explanations dict + model_info (from registry meta if loaded) to the response that the node passes to state.price_forecast
 In the node (src/agentic/nodes/prediction.py):
 Pass include_explanations from config or a query flag to PredictionRequest
 Copy pred_response.explanations into price_forecast.explanations for the UI
 UI rendering idea (compact)
 
 Per horizon row:
 Mean ± quantiles, P(up)
 Top features chips (GBM) with small bars
 Optional “View details” → SHAP waterfall (GBM) or MC‑dropout histogram (LSTM)
 Footer:
 Model: id, trained_at, accuracy/coverage
 
 
 
 For a GUI/Web UI, you can render top features as chips/bars and (optionally) the SHAP waterfall (base64) when explain_include_plots is true.
 