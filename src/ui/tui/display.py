from __future__ import annotations

"""Rich display components for the TUI."""

from typing import Any, Dict, List, Optional

from rich import box
from rich.console import RenderableType, Group
from rich.panel import Panel
from rich.table import Table

from .config import THEME, BOX, WELCOME_TEXT
from .renderer import format_currency, format_percentage, format_confidence, get_color_for_confidence, format_action


def create_welcome_panel() -> Panel:
    return Panel(WELCOME_TEXT, title="Welcome", border_style=THEME.primary, box=getattr(box, BOX.welcome))


def create_parameter_table(params: Any) -> Table:
    table = Table(title="Parameters", box=box.SIMPLE, show_header=False)
    table.add_column("Field", style=f"{THEME.primary} bold", width=22)
    table.add_column("Value", style=THEME.neutral)

    cp = getattr(params, "currency_pair", None) or "—"
    amt = getattr(params, "amount", None)
    base = getattr(params, "base_currency", None) or ""
    risk = getattr(params, "risk_tolerance", None) or "—"
    urg = getattr(params, "urgency", None) or "—"
    # Timeframe: show categorical timeframe if set, otherwise use numeric days if available
    tf_val = getattr(params, "timeframe", None)
    tf_days_val = getattr(params, "timeframe_days", None)
    if tf_val:
        tf = str(tf_val)
    elif tf_days_val is not None:
        try:
            d = int(tf_days_val)
            tf = f"{d} day" if d == 1 else f"{d} days"
        except Exception:
            tf = str(tf_days_val)
    else:
        tf = "—"

    table.add_row("Currency Pair", cp)
    table.add_row("Amount", format_currency(amt, base))
    table.add_row("Risk Tolerance", str(risk).capitalize())
    table.add_row("Urgency", str(urg).capitalize())
    table.add_row("Timeframe", tf)
    return table


def _confidence_label(c: float) -> str:
    if c > 0.7:
        return "High"
    if c > 0.4:
        return "Moderate"
    return "Low"


def create_recommendation_panel(reco: Dict[str, Any]) -> Panel:
    # Header fields
    action = format_action(reco.get("action"))
    conf = reco.get("confidence")
    conf_color = get_color_for_confidence(conf)
    conf_label = _confidence_label(float(conf)) if isinstance(conf, (int, float)) else "—"
    timeline = reco.get("timeline") or "—"

    # Summary table
    table = Table(title="Recommendation Summary", box=box.ROUNDED, show_header=False)
    table.add_column("Field", style=f"{THEME.primary} bold", width=16)
    table.add_column("Value", style=THEME.neutral)
    table.add_row("Action", action)
    table.add_row("Confidence", f"[{conf_color}]{float(conf):.2f}[/] ({conf_label})" if isinstance(conf, (int, float)) else "—")
    table.add_row("Timeline", str(timeline))

    # Build body renderables
    renders: List[RenderableType] = [table]

    # Staged plan
    sp = reco.get("staged_plan")
    if sp and isinstance(sp.get("tranches"), list) and sp["tranches"]:
        renders.append(create_staged_plan_table(sp))

    # Rationale
    rationale = reco.get("rationale") or []
    if rationale:
        rat_table = Table(title="Rationale", box=box.SIMPLE, show_header=False)
        rat_table.add_column("#", width=3, style=THEME.primary)
        rat_table.add_column("Reason", style=THEME.neutral)
        for i, r in enumerate(rationale, 1):
            rat_table.add_row(str(i), str(r))
        renders.append(rat_table)

    # Risk summary
    rs = reco.get("risk_summary")
    if isinstance(rs, dict):
        renders.append(create_risk_summary_table(rs))

    # Cost estimate
    ce = reco.get("cost_estimate")
    if isinstance(ce, dict):
        cost_table = Table(title="Estimated Costs", box=box.SIMPLE, show_header=False)
        cost_table.add_column("Component", style=f"{THEME.primary} bold", width=18)
        cost_table.add_column("Value", style=THEME.neutral)
        total = ce.get("total_bps")
        spread = ce.get("spread_bps")
        fee = ce.get("fee_bps")
        if total is not None:
            cost_table.add_row("Total", f"{float(total):.1f} bps")
        if spread is not None:
            cost_table.add_row("Spread", f"{float(spread):.1f} bps")
        if fee is not None:
            cost_table.add_row("Fee", f"{float(fee):.1f} bps")
        renders.append(cost_table)

    # Warnings
    warnings = [w for w in (reco.get("warnings") or []) if w]
    if warnings:
        warn_table = Table(title="Warnings", box=box.SIMPLE, show_header=False)
        warn_table.add_column("", width=2)
        warn_table.add_column("Message", style=THEME.warning)
        for w in warnings:
            warn_table.add_row("⚠", str(w))
        renders.append(warn_table)

    # Compose panel: stack all renderables vertically
    body = Group(*renders)
    return Panel(body, title="Recommendation", border_style=THEME.primary, box=box.ROUNDED)


def create_staged_plan_table(plan: Dict[str, Any]) -> Table:
    table = Table(title="Staged Conversion Plan", box=box.SIMPLE)
    table.add_column("Tranche", style=THEME.primary)
    table.add_column("Percentage", justify="right", style=THEME.success)
    table.add_column("Execute Day", justify="center", style=THEME.warning)
    table.add_column("Rationale", style=THEME.neutral)

    for tr in plan.get("tranches", []) or []:
        num = tr.get("tranche_number") or tr.get("number") or "—"
        pct = tr.get("percentage")
        day = tr.get("execute_day")
        rationale = tr.get("rationale") or ""
        pct_str = f"{float(pct):.0f}%" if isinstance(pct, (int, float)) else "—"
        day_str = f"Day {int(day)}" if isinstance(day, (int, float)) else "—"
        table.add_row(f"#{num}", pct_str, day_str, rationale)
    return table


def create_risk_summary_table(risk: Dict[str, Any]) -> Table:
    table = Table(title="Risk Summary", box=box.SIMPLE, show_header=False)
    table.add_column("Metric", style=f"{THEME.primary} bold", width=22)
    table.add_column("Value", style=THEME.neutral)

    level = (risk.get("risk_level") or "").title()
    level_color = ("green" if level == "Low" else ("yellow" if level == "Moderate" else "red")) if level else "white"
    table.add_row("Risk Level", f"[{level_color}]{level}[/]")

    rv = risk.get("realized_vol_30d")
    var95 = risk.get("var_95")
    if isinstance(rv, (int, float)):
        table.add_row("30d Realized Vol", format_percentage(rv))
    if isinstance(var95, (int, float)):
        table.add_row("95% VaR", format_percentage(var95))
    if risk.get("event_risk"):
        table.add_row("Event Risk", str(risk.get("event_risk")).title())
    if risk.get("event_details"):
        table.add_row("Details", str(risk.get("event_details")))
    return table


def create_evidence_panel(evidence: Dict[str, Any]) -> Panel:
    """Render evidence & citations gathered by upstream agents and models."""
    renders: List[RenderableType] = []

    # Market providers & notes
    market = evidence.get("market") or {}
    providers = market.get("providers") or []
    notes = market.get("quality_notes") or []
    if providers or notes:
        mt = Table(title="Market Data Sources", box=box.SIMPLE, show_header=False)
        mt.add_column("Field", style=f"{THEME.primary} bold", width=20)
        mt.add_column("Value", style=THEME.neutral)
        if providers:
            mt.add_row("Providers", ", ".join(sorted(set(p for p in providers if p))))
        if notes:
            for n in notes:
                mt.add_row("Note", str(n))
        renders.append(mt)

    # News citations
    news = evidence.get("news") or []
    if news:
        nt = Table(title="News Citations", box=box.SIMPLE)
        nt.add_column("#", width=3, style=THEME.primary)
        nt.add_column("Source", style=THEME.neutral)
        nt.add_column("Title", style=THEME.neutral)
        nt.add_column("URL", style=THEME.neutral)
        for i, ev in enumerate(news[:5], 1):
            nt.add_row(str(i), str(ev.get("source", "")), str(ev.get("title", ""))[:80], str(ev.get("url", ""))[:80])
        renders.append(nt)

    # Calendar citations
    cal = evidence.get("calendar") or []
    if cal:
        ct = Table(title="Calendar Sources", box=box.SIMPLE)
        ct.add_column("Currency", style=THEME.primary)
        ct.add_column("Event", style=THEME.neutral)
        ct.add_column("Importance", style=THEME.neutral)
        ct.add_column("URL", style=THEME.neutral)
        for e in cal[:5]:
            ct.add_row(str(e.get("currency", "")), str(e.get("event", ""))[:40], str(e.get("importance", "")), str(e.get("source_url", ""))[:80])
        renders.append(ct)

    # Model evidence (top features)
    model = evidence.get("model") or {}
    top_features = model.get("top_features") or {}
    if top_features:
        ft = Table(title="Model Evidence (Top Features)", box=box.SIMPLE, show_header=False)
        ft.add_column("Feature", style=f"{THEME.primary} bold", width=30)
        ft.add_column("Importance", style=THEME.neutral)
        for k, v in list(top_features.items()):
            try:
                ft.add_row(str(k), f"{float(v):.1f}")
            except Exception:
                ft.add_row(str(k), str(v))
        renders.append(ft)

    if not renders:
        return Panel("No evidence available.", title="Evidence & Citations", border_style=THEME.primary, box=box.ROUNDED)

    return Panel(Group(*renders), title="Evidence & Citations", border_style=THEME.primary, box=box.ROUNDED)
