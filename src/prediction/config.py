from dataclasses import dataclass, field
from typing import List, Dict, Any
import os
import yaml
from pathlib import Path
from src.utils.paths import find_project_root, resolve_project_path


@dataclass
class PredictionConfig:
    """Prediction system configuration."""

    # General
    prediction_horizons: List[int] = field(default_factory=lambda: [1, 7, 30])
    cache_ttl_hours: int = 1

    # Backend
    predictor_backend: str = "hybrid"  # lightgbm | lstm | hybrid | advanced_ensemble

    # Advanced ensemble settings
    advanced_ensemble: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "ml_models_dir": "ml_models/models",
        "fallback_to_hybrid": True
    })

    # Features
    features_mode: str = "price_only"  # price_only | price_plus_intel
    technical_indicators: List[str] = field(
        default_factory=lambda: [
            "sma_5",
            "sma_20",
            "sma_50",
            "ema_12",
            "ema_26",
            "rsi_14",
            "macd",
            "macd_signal",
            "bb_upper",
            "bb_lower",
            "bb_middle",
            "atr_14",
            "volatility_20",
        ]
    )

    # Data
    min_history_days: int = 100
    max_history_days: int = 365

    # Quality gates
    min_samples_required: int = 500
    min_validation_coverage: float = 0.85  # 85% quantile coverage
    min_directional_accuracy: float = 0.52  # Better than coin flip

    # Fallback
    enable_fallback_heuristics: bool = True
    fallback_strength_pct: float = 0.15  # ±0.15% default signal

    # Model registry
    model_registry_path: str = "data/models/prediction_registry.json"
    model_storage_dir: str = "data/models/prediction/"

    # Explanations
    explain_enabled: bool = False
    explain_include_plots: bool = False
    explain_top_n: int = 5

    # Horizon routing preferences
    use_user_timeframe: bool = True
    default_timeframe: str = "1_day"
    include_intraday_for_1_day: bool = False
    intraday_horizons: List[int] = field(default_factory=lambda: [1, 4, 24])
    daily_horizons: List[int] = field(default_factory=lambda: [1, 7, 30])

    @classmethod
    def from_yaml(cls, config_path: str = "config.yaml"):
        """Load from YAML config file.

        Prefers root-level `prediction:`; falls back to `agents.prediction` if missing.
        """
        env_cfg = os.getenv("CURRENCY_ASSISTANT_CONFIG")
        cfg_path = Path(env_cfg).expanduser() if env_cfg else Path(config_path)
        if not cfg_path.exists():
            root = find_project_root()
            candidate = root / cfg_path.name
            if candidate.exists():
                cfg_path = candidate
        if cfg_path.exists():
            with open(cfg_path, "r") as f:
                full_config = yaml.safe_load(f) or {}
                pred = full_config.get("prediction")
                if pred is None:
                    pred = full_config.get("agents", {}).get("prediction", {})
                # Normalize model paths to absolute
                root = find_project_root()
                if "model_registry_path" in pred:
                    pred["model_registry_path"] = str(
                        resolve_project_path(pred["model_registry_path"], root)
                    )
                if "model_storage_dir" in pred:
                    pred["model_storage_dir"] = str(
                        resolve_project_path(pred["model_storage_dir"], root)
                    )
                # Optional routing keys may be absent; defaults apply otherwise
                for key, default in (
                    ("default_timeframe", "1_day"),
                    ("use_user_timeframe", True),
                    ("include_intraday_for_1_day", False),
                    ("intraday_horizons", [1, 4, 24]),
                    ("daily_horizons", [1, 7, 30]),
                ):
                    if key not in pred:
                        pred[key] = default
                return cls(**pred)
        return cls()
