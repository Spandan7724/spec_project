from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List
import os
import yaml

from src.utils.paths import find_project_root


@dataclass
class UtilityWeights:
    profit: float
    risk: float
    cost: float
    urgency: float


@dataclass
class RiskProfile:
    weights: UtilityWeights
    min_improvement_bps: float
    event_proximity_threshold_days: float
    volatility_penalty_multiplier: float


@dataclass
class DecisionThresholds:
    convert_now_min_utility: float
    staged_min_timeframe_days: int
    wait_event_proximity_days: float
    min_model_confidence: float
    max_prediction_age_hours: int


@dataclass
class StagingConfig:
    max_tranches: int
    min_spacing_days: int
    short_timeframe_tranches: int
    long_timeframe_tranches: int
    urgent_pattern: List[float]
    normal_pattern: List[float]
    flexible_pattern: List[float]


@dataclass
class CostConfig:
    default_spread_bps: float
    default_fee_bps: float
    staging_cost_multiplier: float


@dataclass
class DecisionConfig:
    risk_profiles: Dict[str, RiskProfile]
    thresholds: DecisionThresholds
    staging: StagingConfig
    costs: CostConfig
    # Heuristics policy
    heuristics_enabled: bool = False
    heuristics_trigger_policy: str = "strict"  # strict | relaxed

    @classmethod
    def from_yaml(cls, config_path: str = "config.yaml") -> "DecisionConfig":
        # Resolve config path similar to other components
        env_cfg = os.getenv("CURRENCY_ASSISTANT_CONFIG")
        cfg_path = Path(env_cfg).expanduser() if env_cfg else Path(config_path)
        if not cfg_path.exists():
            root = find_project_root()
            candidate = root / cfg_path.name
            if candidate.exists():
                cfg_path = candidate
        with open(cfg_path, "r") as f:
            data = yaml.safe_load(f) or {}
        d = data.get("decision") or {}

        # Defaults in case config is missing
        def_w = UtilityWeights(1.0, 1.0, 0.3, 0.4)
        def_profile = RiskProfile(def_w, 5.0, 1.5, 1.0)

        rp: Dict[str, RiskProfile] = {}
        for key in ("conservative", "moderate", "aggressive"):
            src = d.get("risk_profiles", {}).get(key, {})
            wsrc = src.get("weights", {})
            weights = UtilityWeights(
                profit=float(wsrc.get("profit", def_w.profit)),
                risk=float(wsrc.get("risk", def_w.risk)),
                cost=float(wsrc.get("cost", def_w.cost)),
                urgency=float(wsrc.get("urgency", def_w.urgency)),
            )
            rp[key] = RiskProfile(
                weights=weights,
                min_improvement_bps=float(src.get("min_improvement_bps", def_profile.min_improvement_bps)),
                event_proximity_threshold_days=float(
                    src.get("event_proximity_threshold_days", def_profile.event_proximity_threshold_days)
                ),
                volatility_penalty_multiplier=float(
                    src.get("volatility_penalty_multiplier", def_profile.volatility_penalty_multiplier)
                ),
            )

        th_src = d.get("thresholds", {})
        thresholds = DecisionThresholds(
            convert_now_min_utility=float(th_src.get("convert_now_min_utility", 0.3)),
            staged_min_timeframe_days=int(th_src.get("staged_min_timeframe_days", 3)),
            wait_event_proximity_days=float(th_src.get("wait_event_proximity_days", 1.5)),
            min_model_confidence=float(th_src.get("min_model_confidence", 0.4)),
            max_prediction_age_hours=int(th_src.get("max_prediction_age_hours", 6)),
        )

        st_src = d.get("staging", {})
        staging = StagingConfig(
            max_tranches=int(st_src.get("max_tranches", 3)),
            min_spacing_days=int(st_src.get("min_spacing_days", 1)),
            short_timeframe_tranches=int(st_src.get("short_timeframe_tranches", 2)),
            long_timeframe_tranches=int(st_src.get("long_timeframe_tranches", 3)),
            urgent_pattern=list(st_src.get("urgent_pattern", [0.6, 0.4])),
            normal_pattern=list(st_src.get("normal_pattern", [0.5, 0.5])),
            flexible_pattern=list(st_src.get("flexible_pattern", [0.33, 0.33, 0.34])),
        )

        c_src = d.get("costs", {})
        costs = CostConfig(
            default_spread_bps=float(c_src.get("default_spread_bps", 5.0)),
            default_fee_bps=float(c_src.get("default_fee_bps", 0.0)),
            staging_cost_multiplier=float(c_src.get("staging_cost_multiplier", 1.2)),
        )

        h_src = d.get("heuristics", {})
        heuristics_enabled = bool(h_src.get("enabled", False))
        heuristics_trigger_policy = str(h_src.get("trigger_policy", "strict"))

        return cls(
            risk_profiles=rp,
            thresholds=thresholds,
            staging=staging,
            costs=costs,
            heuristics_enabled=heuristics_enabled,
            heuristics_trigger_policy=heuristics_trigger_policy,
        )
