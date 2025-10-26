from __future__ import annotations

from src.ui.tui.display import create_welcome_panel, create_parameter_table
from src.supervisor.models import ExtractedParameters


def test_create_welcome_panel_renderable():
    panel = create_welcome_panel()
    assert hasattr(panel, "render")  # basic sanity: Rich renderable


def test_parameter_table_basic():
    params = ExtractedParameters(
        currency_pair="USD/EUR",
        base_currency="USD",
        quote_currency="EUR",
        amount=5000,
        risk_tolerance="moderate",
        urgency="normal",
        timeframe="1_week",
        timeframe_days=7,
    )
    table = create_parameter_table(params)
    assert table.row_count >= 5

