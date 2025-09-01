
  🏗️ System Overview:

  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
  │  Market Agent   │    │ Economic Agent  │    │ Risk Agent      │
  │                 │    │                 │    │                 │
  │ Uses:           │    │ Uses:           │    │ Uses:           │
  │ • ML Predictor  │    │ • FRED API      │    │ • Vol Calculator│
  │ • Tech Indicators│    │ • News Scraper  │    │ • VaR Models    │
  └─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
            │                      │                      │
            └──────────────────────┼──────────────────────┘
                                   │
                      ┌────────────▼────────────┐
                      │ Decision Coordinator    │
                      │                         │
                      │ Synthesizes all inputs  │
                      │ → Convert/Wait Decision │
                      └─────────────────────────┘

  🔧 Tool Communication Protocol:

  1. Tool Input Format:

  # What agents send TO tools
  tool_request = {
      "tool_name": "ml_predictor",
      "parameters": {
          "currency_pair": "USD/EUR",
          "horizons": [1, 7, 30],
          "confidence_level": 0.95
      },
      "context": {
          "agent_id": "market_analysis_agent",
          "request_id": "req_12345",
          "timestamp": "2025-09-01T10:30:00Z"
      }
  }

  2. Tool Output Format:

  # What tools send BACK to agents
  tool_response = {
      "status": "success",  # "success", "error", "timeout"
      "data": {
          "predictions": {
              "1d": {"mean": -0.0093, "p90": 0.0917, "direction_up": 0.42},
              "7d": {"mean": 0.0076, "p90": 0.0160, "direction_up": 0.68},
              "30d": {"mean": -0.0051, "p90": 0.0639, "direction_up": 0.35}
          },
          "confidence_score": 0.956,
          "model_version": "lstm_v1.2.3"
      },
      "metadata": {
          "processing_time_ms": 87,
          "features_used": 73,
          "data_freshness": "2025-09-01T10:25:00Z"
      },
      "error": None  # Contains error details if status != "success"
  }

  3. Specific Tool Communications:

  🧠 ML Prediction Tool:

  # Market Agent → ML Predictor
  {
      "tool": "ml_predictor",
      "input": {
          "currency_pair": "USD/EUR",
          "horizons": [1, 7, 30],
          "include_confidence": True
      }
  }

  # ML Predictor → Market Agent
  {
      "predictions": {
          "1d": {"return": -0.009, "probability_up": 0.42, "confidence": 0.89},
          "7d": {"return": 0.008, "probability_up": 0.68, "confidence": 0.73},
          "30d": {"return": -0.005, "probability_up": 0.35, "confidence": 0.45}
      },
      "signal": "BEARISH_SHORT_TERM",
      "reasoning": "LSTM model shows temporary USD weakness vs EUR"
  }

  📊 Technical Analysis Tool:

  # Market Agent → Technical Indicators
  {
      "tool": "technical_analysis",
      "input": {
          "currency_pair": "USD/EUR",
          "indicators": ["rsi", "macd", "bollinger_bands"],
          "timeframe": "1D"
      }
  }

  # Technical Indicators → Market Agent
  {
      "indicators": {
          "rsi": {"value": 67.3, "signal": "NEUTRAL", "overbought": False},
          "macd": {"signal": "BULLISH", "cross": "recent_golden_cross"},
          "bollinger": {"position": "middle", "squeeze": False, "signal": "NEUTRAL"}
      },
      "overall_signal": "MIXED_NEUTRAL",
      "strength": 0.6
  }

  🏛️ Economic Analysis Tool:

  # Economic Agent → FRED API
  {
      "tool": "economic_events",
      "input": {
          "currencies": ["USD", "EUR"],
          "upcoming_days": 7,
          "impact_level": "HIGH"
      }
  }

  # FRED API → Economic Agent
  {
      "events": [
          {
              "date": "2025-09-03",
              "event": "US_NFP_RELEASE",
              "impact": "HIGH",
              "expected": "180K",
              "previous": "175K",
              "currency_impact": {"USD": "POSITIVE_IF_BEAT"}
          }
      ],
      "overall_calendar_bias": "USD_POSITIVE",
      "risk_level": "ELEVATED"
  }

  ⚖️ Risk Assessment Tool:

  # Risk Agent → Volatility Calculator
  {
      "tool": "risk_calculator",
      "input": {
          "currency_pair": "USD/EUR",
          "amount": 10000,
          "timeframe": 7
      }
  }

  # Volatility Calculator → Risk Agent
  {
      "risk_metrics": {
          "var_95": -0.0234,  # 2.34% max loss at 95% confidence
          "max_drawdown": -0.0156,
          "volatility": 0.087,
          "risk_level": "MEDIUM"
      },
      "scenarios": {
          "best_case": 0.0445,
          "worst_case": -0.0412,
          "expected": 0.0023
      }
  }

  4. Inter-Agent Communication in LangGraph:

  🔄 LangGraph State Flow:

  # Shared state between agents
  langgraph_state = {
      "request": {
          "currency_pair": "USD/EUR",
          "amount": 10000,
          "user_risk_tolerance": "MODERATE"
      },
      "market_analysis": {
          "ml_prediction": {...},
          "technical_signals": {...},
          "market_regime": "RANGING"
      },
      "economic_analysis": {
          "upcoming_events": [...],
          "fundamental_bias": "USD_POSITIVE",
          "calendar_risk": "ELEVATED"
      },
      "risk_assessment": {
          "var_95": -0.0234,
          "risk_level": "MEDIUM",
          "scenarios": {...}
      },
      "final_decision": {
          "action": "WAIT",  # "CONVERT" or "WAIT"
          "confidence": 0.78,
          "reasoning": "High economic event risk this week",
          "timeline": "Wait until after NFP on Sept 3rd"
      }
  }

  🎯 Decision Flow Example:

  1. Market Agent calls ML Predictor → Gets bearish 7-day forecast
  2. Economic Agent calls FRED API → Finds high-impact NFP release
  3. Risk Agent calls Risk Calculator → Calculates 2.34% potential loss
  4. Decision Coordinator synthesizes all inputs:
  decision = {
      "action": "WAIT",
      "confidence": 0.78,
      "reasoning": {
          "market": "ML shows temporary USD weakness (-0.9%)",
          "economic": "Major NFP release in 2 days could cause 2%+ move",
          "risk": "Current volatility elevated, wait for clarity",
          "recommendation": "Wait until after Sept 3rd NFP release"
      },
      "timeline": "Re-evaluate on Sept 4th"
  }

  5. Error Handling & Fallbacks:

  # Tool failure response
  {
      "status": "error",
      "error": {
          "code": "ML_MODEL_UNAVAILABLE",
          "message": "LSTM model is retraining",
          "fallback_available": True
      },
      "fallback_data": {
          "technical_signal": "NEUTRAL",
          "confidence": 0.3,
          "source": "RSI_FALLBACK"
      }
  }