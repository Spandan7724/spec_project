from __future__ import annotations

import asyncio
from typing import List, Optional

import typer

from src.prediction.config import PredictionConfig
from src.prediction.training import train_and_register_lightgbm, train_and_register_lstm
from src.cli.train import app as train_app


app = typer.Typer(add_completion=False, help="Currency Assistant CLI")
app.add_typer(train_app, name="train")


@app.command("train-model")
def train_model(
    pair: Optional[str] = typer.Option(None, "--pair", "-p", prompt=False, help="Currency pair, e.g., USD/EUR"),
    days: Optional[int] = typer.Option(None, "--days", "-d", prompt=False, help="History in days"),
    horizons: Optional[List[int]] = typer.Option(
        None, "--horizons", "-h", help="Horizons in days (repeat option), e.g., -h 1 -h 7 -h 30"
    ),
    version: str = typer.Option("1.0", "--version", "-v", help="Model version label"),
    # GBM knobs
    gbm_rounds: Optional[int] = typer.Option(None, "--gbm-rounds", help="Boost rounds (applies to all GBM models)"),
    gbm_patience: Optional[int] = typer.Option(None, "--gbm-patience", help="Early stopping patience (rounds)"),
    gbm_learning_rate: Optional[float] = typer.Option(None, "--gbm-lr", help="Learning rate"),
    gbm_num_leaves: Optional[int] = typer.Option(None, "--gbm-leaves", help="Num leaves"),
):
    """Train a LightGBM model on historical data and register it in the local registry."""

    cfg = PredictionConfig.from_yaml()

    # Interactive prompts for missing values
    if not pair:
        pair = typer.prompt("Currency pair (e.g., USD/EUR)", default="USD/EUR")
    if days is None:
        days = int(typer.prompt("History (days)", default=cfg.max_history_days))
    if horizons is None or len(horizons) == 0:
        default_hz = ",".join(str(h) for h in (cfg.prediction_horizons or [1, 7, 30]))
        hz_str = typer.prompt("Horizons (days, comma-separated)", default=default_hz)
        try:
            horizons = [int(x.strip()) for x in hz_str.split(",") if x.strip()]
        except Exception:
            typer.secho("Invalid horizons; using defaults", fg=typer.colors.YELLOW)
            horizons = cfg.prediction_horizons or [1, 7, 30]

    # Optional advanced tuning
    if gbm_rounds is None and gbm_patience is None and gbm_learning_rate is None and gbm_num_leaves is None:
        if typer.confirm("Adjust advanced LightGBM settings?", default=False):
            gbm_rounds = int(typer.prompt("GBM rounds", default=120))
            gbm_patience = int(typer.prompt("GBM early-stopping patience", default=10))
            gbm_learning_rate = float(typer.prompt("GBM learning rate", default=0.05))
            gbm_num_leaves = int(typer.prompt("GBM num_leaves", default=31))

    typer.echo(
        f"Training LightGBM for {pair} (days={days}, horizons={horizons})"
    )

    try:
        meta = asyncio.run(
            train_and_register_lightgbm(
                pair,
                config=cfg,
                days=days,
                horizons=horizons,
                version=version,
                gbm_rounds=gbm_rounds,
                gbm_patience=gbm_patience,
                gbm_learning_rate=gbm_learning_rate,
                gbm_num_leaves=gbm_num_leaves,
            )
        )
        typer.echo(
            "\n✅ Model registered: "
            f"id={meta.model_id}, pair={meta.currency_pair}, version={meta.version}"
        )
        typer.echo(
            f"Registry: {cfg.model_registry_path}\nStorage: {cfg.model_storage_dir}"
        )
    except Exception as e:
        typer.secho(f"Failed to train/register: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)


@app.command("train-lstm")
def train_lstm(
    pair: Optional[str] = typer.Option(None, "--pair", "-p", prompt=False, help="Currency pair, e.g., USD/EUR"),
    days: Optional[int] = typer.Option(None, "--days", "-d", prompt=False, help="History (intraday) in days"),
    interval: Optional[str] = typer.Option(None, "--interval", "-i", prompt=False, help="Intraday interval (e.g., 1h)"),
    horizons_hours: Optional[List[int]] = typer.Option(
        None,
        "--horizon-hours",
        "-H",
        help="Intraday horizons in hours (repeat option), e.g., -H 1 -H 4 -H 24",
    ),
    version: str = typer.Option("1.0", "--version", "-v", help="Model version label"),
    # LSTM knobs
    lstm_epochs: int = typer.Option(5, "--lstm-epochs", help="Training epochs"),
    lstm_hidden_dim: int = typer.Option(64, "--lstm-hidden-dim", help="Hidden dimension"),
    lstm_seq_len: int = typer.Option(64, "--lstm-seq-len", help="Sequence length"),
    lstm_lr: float = typer.Option(1e-3, "--lstm-lr", help="Learning rate"),
):
    """Train an LSTM intraday model and register it in the local registry."""

    cfg = PredictionConfig.from_yaml()
    # Interactive prompts for missing values
    if not pair:
        pair = typer.prompt("Currency pair (e.g., USD/EUR)", default="USD/EUR")
    if days is None:
        days = int(typer.prompt("History (intraday days)", default=180))
    if not interval:
        interval = typer.prompt("Intraday interval", default="1h")
    if horizons_hours is None or len(horizons_hours) == 0:
        hz_str = typer.prompt("Intraday horizons (hours, comma-separated)", default="1,4,24")
        try:
            horizons_hours = [int(x.strip()) for x in hz_str.split(",") if x.strip()]
        except Exception:
            typer.secho("Invalid intraday horizons; using defaults", fg=typer.colors.YELLOW)
            horizons_hours = [1, 4, 24]

    # Optional advanced tuning
    if (
        lstm_epochs == 5 and lstm_hidden_dim == 64 and lstm_seq_len == 64 and abs(lstm_lr - 1e-3) < 1e-12
    ):
        if typer.confirm("Adjust advanced LSTM settings?", default=False):
            lstm_epochs = int(typer.prompt("LSTM epochs", default=5))
            lstm_hidden_dim = int(typer.prompt("LSTM hidden dim", default=64))
            lstm_seq_len = int(typer.prompt("LSTM sequence length", default=64))
            lstm_lr = float(typer.prompt("LSTM learning rate", default=1e-3))

    show_hz = horizons_hours or [1, 4, 24]
    typer.echo(
        f"Training LSTM for {pair} (days={days}, interval={interval}, horizons_hours={show_hz})"
    )

    try:
        meta = asyncio.run(
            train_and_register_lstm(
                pair,
                config=cfg,
                days=days or 180,
                interval=interval,
                horizons_hours=horizons_hours,
                version=version,
                lstm_epochs=lstm_epochs,
                lstm_hidden_dim=lstm_hidden_dim,
                lstm_seq_len=lstm_seq_len,
                lstm_lr=lstm_lr,
            )
        )
        typer.echo(
            "\n✅ LSTM model registered: "
            f"id={meta.model_id}, pair={meta.currency_pair}, version={meta.version}"
        )
        typer.echo(
            f"Registry: {cfg.model_registry_path}\nStorage: {cfg.model_storage_dir}"
        )
    except Exception as e:
        typer.secho(f"Failed to train/register LSTM: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

if __name__ == "__main__":
    app()
