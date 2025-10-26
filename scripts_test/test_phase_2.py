#!/usr/bin/env python
"""Phase 2 end-to-end runner using the LangGraph workflow with Prediction node.

Usage:
  uv run python scripts_test/test_phase_2.py

Notes:
- If no trained models are found in the registry, predictions may fallback to heuristics.
- Train models via CLI:
    currency-assistant train-model --pair USD/EUR -d 365 -h 1 -h 7 -h 30
    currency-assistant train-lstm  --pair USD/EUR -d 120 -i 1h -H 1 -H 4 -H 24
"""

import os
import time
from src.agentic.graph import create_graph
from src.agentic.state import initialize_state
from src.prediction.config import PredictionConfig
from src.prediction.registry import ModelRegistry


def _has_models(pair: str) -> bool:
    cfg = PredictionConfig.from_yaml()
    reg = ModelRegistry(cfg.model_registry_path, cfg.model_storage_dir)
    gbm = reg.get_model(pair, "lightgbm")
    lstm = reg.get_model(pair, "lstm")
    return bool(gbm or lstm)


def main():
    # Ensure we run from repo root so relative paths (config.yaml, data/models) resolve
    here = os.path.abspath(os.path.dirname(__file__))
    repo_root = os.path.abspath(os.path.join(here, ".."))
    try:
        os.chdir(repo_root)
    except Exception:
        pass
    # Optional: ensure demo mode is OFF for real providers/LLMs (honors your env)
    offline = os.getenv("OFFLINE_DEMO", "false").strip().lower() in {"1", "true", "yes", "on"}
    if offline:
        print("[warn] OFFLINE_DEMO=true -> upstream nodes may return fallback data.")

    pair = os.getenv("PAIR", "USD/EUR")
    if not _has_models(pair):
        print("[warn] No trained models found in registry for", pair)
        print("       Predictions may use fallback heuristics.")
        print("       Train models via:\n"
              "         currency-assistant train-model --pair USD/EUR -d 365 -h 1 -h 7 -h 30\n"
              "         currency-assistant train-lstm  --pair USD/EUR -d 120 -i 1h -H 1 -H 4 -H 24")

    # Start timing
    start_time = time.time()

    # Create the graph
    g = create_graph()

    # Initialize state
    s = initialize_state(
        f"Convert 5000 {pair.split('/')[0]} to {pair.split('/')[1]}",
        base_currency=pair.split('/')[0],
        quote_currency=pair.split('/')[1],
        timeframe=os.getenv("TIMEFRAME", "1_day"),  # set to "immediate" to include intraday horizons
    )

    # Invoke the graph
    r = g.invoke(s)

    # Calculate execution time
    end_time = time.time()
    execution_time = end_time - start_time

    # Market Data output
    md_status = r.get("market_data_status", "unknown")
    print("market_status:", md_status)
    if md_status == "success" and r.get("market_snapshot"):
        print("mid_rate:", r["market_snapshot"].get("mid_rate"))
    else:
        print("market_error:", r.get("market_data_error"))

    # Market Intelligence output
    mi_status = r.get("intelligence_status", "unknown")
    print("intel_status:", mi_status)
    if mi_status == "success" and r.get("intelligence_report"):
        rep = r["intelligence_report"]
        print("policy_bias:", rep.get("policy_bias"))
        print("news_bias:", rep.get("news", {}).get("pair_bias"), "conf:", rep.get("news", {}).get("confidence"))
        print("next_event:", rep.get("calendar", {}).get("next_high_event"))
    else:
        print("intel_error:", r.get("intelligence_error"))

    # Prediction output
    pr_status = r.get("prediction_status", "unknown")
    print("prediction_status:", pr_status)
    pf = r.get("price_forecast") or {}
    if pf:
        print("model_id:", pf.get("model_id"))
        print("confidence:", pf.get("confidence"))
        print("cached:", pf.get("cached"))
        preds = pf.get("predictions") or {}
        if preds:
            print("\nPredictions:")
            for hz, p in sorted(preds.items(), key=lambda x: int(x[0])):
                print(f"  {hz}: mean={p.get('mean_change_pct'):+.3f}% ", end="")
                if p.get("direction_prob") is not None:
                    print(f"(P(up)={p.get('direction_prob'):.2f}) ", end="")
                q = p.get("quantiles") or {}
                if q:
                    print(f"q10={q.get('p10'):+.3f}% q50={q.get('p50'):+.3f}% q90={q.get('p90'):+.3f}%", end="")
                print()
        warns = pf.get("warnings") or []
        if warns:
            print("\nWarnings:")
            for w in warns:
                print(" -", w)
        # Optional explanations/evidence
        expl = pf.get("explanations") or {}
        if expl:
            daily = expl.get("daily") or {}
            intraday = expl.get("intraday") or {}
            if daily:
                print("\nExplanations (daily):")
                for hz, item in sorted(daily.items(), key=lambda x: int(x[0])):
                    tf = item.get("top_features") or {}
                    if tf:
                        top_line = ", ".join(f"{k}:{v:.1f}" for k, v in list(tf.items()))
                        print(f"  {hz}d top_features: {top_line}")
            if intraday:
                print("\nExplanations (intraday):")
                for hz, item in sorted(intraday.items(), key=lambda x: int(x[0])):
                    print(f"  {hz}h mc_samples: {item.get('mc_samples')}")
            mi = pf.get("model_info") or {}
            if mi:
                print("\nModel info:")
                for name, meta in mi.items():
                    print(f"  {name}: id={meta.get('model_id')} trained_at={meta.get('trained_at')}")
    else:
        print("prediction_error:", r.get("prediction_error"))

    # Execution time
    print(f"\n⏱️  Execution time: {execution_time:.2f} seconds")


if __name__ == "__main__":
    main()
