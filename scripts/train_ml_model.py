#!/usr/bin/env python3
"""Train or retrain the LSTM model for a given currency pair."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from ml.config import load_ml_config
from ml.prediction.predictor import MLPredictor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the LSTM currency prediction model",
    )
    parser.add_argument(
        "currency_pair",
        help="Currency pair to train on (e.g. USD/EUR)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=365,
        help="Number of days of history to load (default: 365)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="ml_config.yaml",
        help="Path to ML configuration file (default: ml_config.yaml)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not persist the trained model/preprocessor",
    )
    parser.add_argument(
        "--no-default",
        action="store_true",
        help="Train but do not mark the model as the default",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger("train_ml_model")

    logger.info("Loading ML configuration from %s", args.config)
    config = load_ml_config(args.config)

    predictor = MLPredictor(config)

    try:
        result = predictor.train_model(
            currency_pair=args.currency_pair,
            days=args.days,
            save_model=not args.no_save,
            set_as_default=not args.no_default,
        )
    except RuntimeError as exc:
        logger.error("Training failed: %s", exc)
        return 2

    logger.info("Training completed: model_id=%s", result.get("model_id"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
