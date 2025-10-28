from __future__ import annotations

import asyncio
import threading
import time
import uuid
from typing import Any, Dict

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from rich import box

from src.supervisor.conversation_manager import ConversationManager
from src.supervisor.agent_orchestrator import AgentOrchestrator
from src.supervisor.answer_generator import AnswerGenerator
from src.supervisor.response_formatter import ResponseFormatter
from src.supervisor.models import SupervisorRequest, ConversationState, ExtractedParameters

from .display import (
    create_welcome_panel,
    create_parameter_table,
    create_recommendation_panel,
    create_evidence_panel,
)
from .input_handler import get_user_input, ask_yes_no


console = Console()


class CurrencyAssistantTUI:
    """Terminal User Interface for Currency Assistant."""

    def __init__(self) -> None:
        self.conversation_manager = ConversationManager()
        self.orchestrator = AgentOrchestrator()
        self.formatter = ResponseFormatter()
        self.session_id = str(uuid.uuid4())
        self.answer_generator = AnswerGenerator()

    def run(self) -> None:
        """Main entry point (sync)."""
        console.print(create_welcome_panel())
        console.print()

        while True:
            try:
                self._conversation_loop()

                another = ask_yes_no("Would you like to make another conversion analysis?", default=False)
                if not another:
                    console.print("\n[bold green]Thank you for using Currency Assistant! Goodbye![/]")
                    break

                self.session_id = str(uuid.uuid4())
                console.print("\n" + "=" * 60 + "\n")
            except KeyboardInterrupt:
                console.print("\n\n[yellow]Interrupted. Goodbye![/]")
                break
            except Exception as e:  # noqa: BLE001
                console.print(f"\n[bold red]Error: {e}[/]")
                console.print("[yellow]Please try again or contact support.[/]")

    def _conversation_loop(self) -> str:
        """Main conversation loop (sync)."""

        console.print("[bold]Let's start with your currency conversion request.[/]\n")
        user_input = get_user_input("You")

        while True:
            req = SupervisorRequest(user_input=user_input, session_id=self.session_id, correlation_id=str(uuid.uuid4()))
            resp = self.conversation_manager.process_input(req)

            # Show assistant message
            console.print(f"\n[bold magenta]Assistant[/]: {resp.message}\n")

            # Show parameters table; include current rate at confirmation step
            if resp.parameters:
                current_rate = None
                try:
                    if resp.state == ConversationState.CONFIRMING and resp.parameters.base_currency and resp.parameters.quote_currency:
                        current_rate = self._fetch_current_rate(resp.parameters.base_currency, resp.parameters.quote_currency)
                except Exception:
                    current_rate = None
                console.print(create_parameter_table(resp.parameters, current_rate=current_rate))

            if resp.requires_input:
                user_input = get_user_input("You")
                continue

            if resp.state == ConversationState.PROCESSING:
                # Drive agents and show progress
                reco = self._run_agents_sync(resp.parameters, req.correlation_id or str(uuid.uuid4()))
                self._display_recommendation(reco)
                # Enter post-recommendation chat loop
                return self._post_recommendation_chat(reco, resp.parameters)

            if resp.state in (ConversationState.COMPLETED, ConversationState.ERROR):
                # If completed without a recommendation, ask user to start new or exit
                return "new" if ask_yes_no("Start a new analysis?", default=False) else "exit"
        # Fallback: ask user to start new or exit
        return "new" if ask_yes_no("Start a new analysis?", default=False) else "exit"

    def _run_coro_blocking(self, coro):
        """Run an async coroutine safely from sync context, even if a loop is running.

        - If no event loop is running, use asyncio.run.
        - If an event loop is already running (e.g., in notebooks), run in a worker thread.
        """
        try:
            asyncio.get_running_loop()
            loop_running = True
        except RuntimeError:
            loop_running = False

        if not loop_running:
            return asyncio.run(coro)

        result: Dict[str, Any] = {"value": None, "error": None}

        def runner():
            try:
                result["value"] = asyncio.run(coro)
            except Exception as e:  # noqa: BLE001
                result["error"] = e

        t = threading.Thread(target=runner, daemon=True)
        t.start()
        t.join()
        if result["error"] is not None:
            raise result["error"]
        return result["value"]

    def _run_agents_sync(self, parameters: Any, correlation_id: str) -> Dict[str, Any]:
        """Run agent orchestration with progress feedback (sync)."""

        console.print()
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=None),
            TaskProgressColumn(style="cyan"),
            TimeElapsedColumn(style="cyan"),
            console=console,
        ) as progress:
            task = progress.add_task("[cyan]Analyzing market conditions...", total=100)
            # Stage 1 → 25%
            time.sleep(0.2)
            progress.advance(task, 25)
            progress.update(task, description="[cyan]Fetching economic calendar and news...")
            # Stage 2 → 50%
            time.sleep(0.2)
            progress.advance(task, 25)
            progress.update(task, description="[cyan]Generating price predictions...")
            # Stage 3 → 75%
            time.sleep(0.2)
            progress.advance(task, 25)
            progress.update(task, description="[cyan]Calculating optimal recommendation...")

            recommendation = self._run_coro_blocking(self.orchestrator.run_analysis(parameters, correlation_id))
            # Stage 4 → 100%
            progress.update(task, completed=100, description="[green]✓ Analysis complete!")

        console.print()
        return recommendation

    def _fetch_current_rate(self, base: str, quote: str) -> float | None:
        """Fetch a quick current mid rate for the pair (sync wrapper)."""
        async def _run():
            from src.config import get_config, load_config
            from src.data_collection.providers import get_provider
            from src.data_collection.market_data.snapshot import build_snapshot
            from src.cache import cache
            try:
                cfg = get_config()
            except Exception:
                cfg = load_config()
            provider_names = cfg.get("agents.market_data.providers", ["exchange_rate_host", "yfinance"]) or []
            providers = [get_provider(name) for name in provider_names]
            snap = await build_snapshot(base, quote, providers, cache=cache)
            return snap.mid_rate

        try:
            return self._run_coro_blocking(_run())
        except Exception:
            return None

    def _display_recommendation(self, recommendation: Dict[str, Any]) -> None:
        """Display recommendation in rich format and a text summary."""

        if recommendation.get("status") == "error":
            panel = Panel(
                f"[bold red]Error:[/] {recommendation.get('error', 'Unknown error')}",
                title="Error",
                border_style="red",
                box=box.HEAVY,
            )
            console.print(panel)
            warnings = recommendation.get("warnings") or []
            for w in warnings:
                console.print(f"[yellow]• {w}")
            return

        console.print(create_recommendation_panel(recommendation))
        console.print()

        # Show evidence & citations if available
        ev = recommendation.get("evidence") or {}
        if ev:
            console.print(create_evidence_panel(ev))
            console.print()

        # Also print a concise text block using ResponseFormatter
        try:
            text_block = self.formatter.format_recommendation(recommendation)
            console.print(Panel(text_block, title="Summary", border_style="cyan", box=box.ROUNDED))
        except Exception:
            # Non-blocking if formatting fails
            pass

    def _post_recommendation_chat(self, recommendation: Dict[str, Any], parameters: ExtractedParameters) -> str:
        """Lightweight, chatty follow-up loop after presenting a recommendation.

        Supports:
        - help / ?: show commands
        - why / rationale / explain: show rationale list
        - evidence / citations: show evidence panel
        - providers: list market data providers
        - calendar: list a few calendar sources
        - model: show model top features
        - change <param> [to <value>]: update a parameter and rerun analysis
        - rerun: rerun analysis with current parameters
        - new: start a new analysis
        - exit / quit: exit application
        """
        console.print("[dim]Type 'help' to see available commands. Type 'new' to start over, 'exit' to quit.[/]")
        current_reco = recommendation

        while True:
            user = get_user_input("You")
            low = user.strip().lower()

            if low in {"help", "?"}:
                console.print(
                    "\n[bold]Commands:[/]\n"
                    "- why | rationale | explain  → show rationale\n"
                    "- evidence | citations       → show evidence & sources\n"
                    "- providers                  → list market data providers\n"
                    "- calendar                   → list calendar sources\n"
                    "- model                      → show model top features\n"
                    "- scenarios                  → compare convert now vs wait 24h/48h\n"
                    "- change <param> [to <value>]→ update parameter (pair, amount, risk, urgency, timeframe)\n"
                    "- rerun                      → rerun analysis\n"
                    "- new                        → start a new analysis\n"
                    "- exit | quit                → exit\n"
                )
                continue

            if low in {"why", "rationale", "explain"}:
                rats = current_reco.get("rationale") or []
                if rats:
                    console.print("\n[bold cyan]Rationale:[/]")
                    for i, r in enumerate(rats, 1):
                        console.print(f"  {i}. {r}")
                else:
                    console.print("No rationale available.")
                continue

            if low in {"evidence", "citations"}:
                ev = current_reco.get("evidence") or {}
                if ev:
                    from .display import create_evidence_panel

                    console.print(create_evidence_panel(ev))
                else:
                    console.print("No evidence available.")
                continue

            if low == "providers":
                ev = current_reco.get("evidence") or {}
                market = ev.get("market") or {}
                providers = market.get("providers") or []
                if providers:
                    console.print("Providers: " + ", ".join(sorted(set(p for p in providers if p))))
                else:
                    console.print("No provider info available.")
                continue

            if low == "calendar":
                ev = current_reco.get("evidence") or {}
                cal = ev.get("calendar") or []
                if cal:
                    console.print("\n[bold]Calendar sources:[/]")
                    for e in cal[:5]:
                        console.print(f" - {e.get('currency','')}: {e.get('event','')} [{e.get('importance','')}] → {e.get('source_url','')}")
                else:
                    console.print("No calendar citations available.")
                continue

            if low == "model":
                ev = current_reco.get("evidence") or {}
                model = ev.get("model") or {}
                tf = model.get("top_features") or {}
                if tf:
                    console.print("\n[bold]Model top features:[/]")
                    for k, v in list(tf.items()):
                        try:
                            console.print(f" - {k}: {float(v):.1f}")
                        except Exception:
                            console.print(f" - {k}: {v}")
                else:
                    console.print("No model feature info available.")
                continue

            if low == "scenarios":
                try:
                    panel = self._build_scenarios_panel(current_reco)
                    console.print(panel)
                except Exception as e:  # noqa: BLE001
                    console.print(f"[red]Failed to build scenarios:[/] {e}")
                continue

            if low in {"new"}:
                return "new"
            if low in {"exit", "quit"}:
                return "exit"
            if low == "rerun":
                # Rerun with current parameters
                reco = self._run_agents_sync(parameters, str(uuid.uuid4()))
                self._display_recommendation(reco)
                current_reco = reco
                console.print("[dim]Type 'help' to see available commands. Type 'new' to start over, 'exit' to quit.[/]")
                continue

            # change <param> [to <value>]
            import re as _re

            m = _re.match(r"\s*change\s+(\w+)(?:\s+to\s+(.+))?\s*", user, _re.IGNORECASE)
            if m:
                param = m.group(1).lower()
                value = (m.group(2) or "").strip()
                # If no value provided, prompt
                if not value:
                    value = get_user_input(f"New {param}")
                try:
                    self._apply_parameter_change(parameters, param, value)
                    console.print("[green]Updated parameters.[/]")
                    console.print(create_parameter_table(parameters))
                    # Rerun automatically
                    reco = self._run_agents_sync(parameters, str(uuid.uuid4()))
                    self._display_recommendation(reco)
                    current_reco = reco
                except Exception as e:  # noqa: BLE001
                    console.print(f"[red]Failed to update parameter:[/] {e}")
                continue

            # Free-form question answering about current recommendation
            # Prefer LLM answer when available; fallback to heuristic answer
            answer = self._llm_answer_safe(user, current_reco, parameters)
            if not answer:
                answer = self._answer_question(user, current_reco, parameters)
            if answer:
                console.print(answer)
            else:
                console.print("I didn't catch that. Type 'help' for options, or 'new' / 'exit'.")

    def _llm_answer_safe(self, question: str, reco: Dict[str, Any], params: ExtractedParameters) -> str:
        try:
            return self._run_coro_blocking(self.answer_generator.agenerate_answer(question, reco, params)) or ""
        except Exception:
            return ""

    def _build_scenarios_panel(self, reco: Dict[str, Any]) -> Panel:
        from rich.table import Table
        from rich import box as _box
        ev = reco.get("evidence") or {}
        mkt = ev.get("market") or {}
        preds_all = ev.get("predictions_all") or {}
        current = mkt.get("mid_rate")
        if current is None:
            raise ValueError("current rate not available")

        def exp_rate(days: int) -> float | None:
            # Use matching horizon if present
            if str(days) in preds_all and isinstance(preds_all[str(days)], (int, float)):
                return float(current) * (1.0 + float(preds_all[str(days)]) / 100.0)
            # Approximate 48h using 7d scaled if available
            if days == 2 and ("7" in preds_all) and isinstance(preds_all["7"], (int, float)):
                daily = float(preds_all["7"]) / 7.0
                return float(current) * (1.0 + daily * 2.0 / 100.0)
            return None

        now_rate = float(current)
        d1 = exp_rate(1)
        d2 = exp_rate(2)

        tbl = Table(title="Scenarios (Expected Rate)", box=_box.ROUNDED, show_header=True)
        tbl.add_column("Scenario", style=f"{THEME.primary} bold", width=20)
        tbl.add_column("Expected Rate", style=THEME.neutral)
        tbl.add_column("Change (bps)", style=THEME.neutral)

        def add_row(name: str, rate: float | None):
            if rate is None:
                tbl.add_row(name, "—", "—")
            else:
                try:
                    chg_bps = (rate - now_rate) / now_rate * 10000.0
                    tbl.add_row(name, f"{rate:.6f}", f"{chg_bps:.1f}")
                except Exception:
                    tbl.add_row(name, str(rate), "—")

        add_row("Convert Now", now_rate)
        add_row("Wait 24h", d1)
        add_row("Wait 48h", d2)

        return Panel(tbl, title="Scenarios", border_style=THEME.primary, box=box.ROUNDED)

    def _answer_question(self, question: str, reco: Dict[str, Any], params: ExtractedParameters) -> str:
        """Heuristic Q&A over the current recommendation and evidence."""
        low = question.strip().lower()
        lines = []

        # Confidence
        if any(k in low for k in ["confidence", "confident", "sure"]):
            c = reco.get("confidence")
            if isinstance(c, (int, float)):
                level = "High" if c > 0.7 else ("Moderate" if c > 0.4 else "Low")
                lines.append(f"Confidence is {c:.2f} ({level}).")
            cc = reco.get("component_confidences") or {}
            if cc:
                parts = ", ".join(f"{k}:{v:.2f}" for k, v in cc.items() if isinstance(v, (int, float)))
                if parts:
                    lines.append(f"Component confidences: {parts}.")

        # Risk
        if "risk" in low:
            rs = reco.get("risk_summary") or {}
            if rs:
                lvl = str(rs.get("risk_level", "")).title()
                rv = rs.get("realized_vol_30d")
                var95 = rs.get("var_95")
                evr = str(rs.get("event_risk", "")).title()
                lines.append(f"Risk level is {lvl}; event risk {evr}.")
                if isinstance(rv, (int, float)):
                    lines.append(f"30d realized volatility: {rv:.2f}%.")
                if isinstance(var95, (int, float)):
                    lines.append(f"95% VaR: {var95:.2f}%.")

        # Timeline / When
        if any(k in low for k in ["when", "timeline", "time", "execute"]):
            tl = reco.get("timeline")
            if tl:
                lines.append(f"Timeline: {tl}.")
            sp = reco.get("staged_plan") or {}
            tr = sp.get("tranches") or []
            if tr:
                schedule = ", ".join(f"{t.get('percentage','?'):.0f}% on day {t.get('execute_day','?')}" if isinstance(t.get('percentage'), (int,float)) else f"{t.get('percentage','?')}% on day {t.get('execute_day','?')}" for t in tr)
                lines.append(f"Staging schedule: {schedule}.")

        # Costs
        if any(k in low for k in ["cost", "fee", "spread", "bps"]):
            ce = reco.get("cost_estimate") or {}
            total = ce.get("total_bps")
            spread = ce.get("spread_bps")
            fee = ce.get("fee_bps")
            parts = []
            if isinstance(total, (int, float)):
                parts.append(f"Total {total:.1f} bps")
            if isinstance(spread, (int, float)):
                parts.append(f"spread {spread:.1f}")
            if isinstance(fee, (int, float)):
                parts.append(f"fee {fee:.1f}")
            if parts:
                lines.append("Estimated cost: " + ", ".join(parts) + ".")

        # Current rate / price / quote
        if any(k in low for k in ["rate", "price", "quote", "current"]):
            ev = reco.get("evidence") or {}
            market = ev.get("market") or {}
            mr = market.get("mid_rate")
            bid = market.get("bid")
            ask = market.get("ask")
            ts = market.get("rate_timestamp")
            if mr is not None:
                try:
                    lines.append(f"Current rate (mid): {float(mr):.6f}.")
                except Exception:
                    lines.append(f"Current rate (mid): {mr}.")
                if isinstance(bid, (int, float)) and isinstance(ask, (int, float)):
                    lines.append(f"Bid/Ask: {bid:.6f} / {ask:.6f}.")
                elif bid is not None or ask is not None:
                    lines.append(f"Bid/Ask: {bid} / {ask}.")
                if ts:
                    lines.append(f"As of: {ts}.")

        # Providers / sources
        if any(k in low for k in ["provider", "source", "data source"]):
            ev = reco.get("evidence") or {}
            market = ev.get("market") or {}
            providers = market.get("providers") or []
            if providers:
                names = ", ".join(sorted(set(p for p in providers if p)))
                lines.append(f"Market data providers: {names}.")

        # Calendar / events
        if any(k in low for k in ["event", "calendar"]):
            ev = reco.get("evidence") or {}
            cal = ev.get("calendar") or []
            if cal:
                examples = ", ".join(f"{e.get('currency','')}: {e.get('event','')}" for e in cal[:3])
                lines.append(f"Upcoming events (examples): {examples}.")

        # Technicals / regime
        if any(k in low for k in ["technical", "rsi", "macd", "trend", "bias", "regime"]):
            ev = reco.get("evidence") or {}
            mkt = ev.get("market") or {}
            tech = mkt.get("indicators") or {}
            reg = mkt.get("regime") or {}
            rsi = tech.get("rsi_14")
            macd = tech.get("macd")
            macd_sig = tech.get("macd_signal")
            trend = reg.get("trend_direction")
            bias = reg.get("bias")
            if isinstance(rsi, (int, float)):
                lines.append(f"RSI(14): {rsi:.1f}.")
            if isinstance(macd, (int, float)) and isinstance(macd_sig, (int, float)):
                lines.append(f"MACD/Signal: {macd:.6f} / {macd_sig:.6f}.")
            if trend:
                lines.append(f"Trend: {str(trend).title()}.")
            if bias:
                lines.append(f"Bias: {str(bias).title()}.")

        # Forecast / prediction
        if any(k in low for k in ["forecast", "prediction", "expected", "mean change", "quantile"]):
            ev = reco.get("evidence") or {}
            pred = ev.get("prediction") or {}
            if pred:
                mc = pred.get("mean_change_pct")
                if isinstance(mc, (int, float)):
                    lines.append(f"Forecast mean change: {mc:.2f}%.")
                q = pred.get("quantiles") or {}
                lo = q.get("0.05") or q.get(0.05)
                hi = q.get("0.95") or q.get(0.95)
                if isinstance(lo, (int, float)) and isinstance(hi, (int, float)):
                    lines.append(f"95% interval: {lo:.2f}% to {hi:.2f}%.")
            else:
                lines.append("No forecast summary available.")

        # Model / features / why
        if any(k in low for k in ["model", "feature", "shap", "why"]):
            ev = reco.get("evidence") or {}
            model = ev.get("model") or {}
            top = model.get("top_features") or {}
            if top:
                parts = ", ".join(f"{k}:{v:.1f}" if isinstance(v,(int,float)) else f"{k}:{v}" for k,v in list(top.items())[:5])
                lines.append(f"Top model features: {parts}.")

        # Staging / tranches
        if any(k in low for k in ["staged", "tranche", "split", "plan"]):
            sp = reco.get("staged_plan") or {}
            if sp:
                n = sp.get("num_tranches")
                spacing = sp.get("spacing_days")
                lines.append(f"Staged plan has {n} tranches; average spacing ~{spacing:.1f} days." if isinstance(spacing,(int,float)) else f"Staged plan has {n} tranches.")

        return "\n".join(l for l in lines if l)

    def _apply_parameter_change(self, parameters: ExtractedParameters, param: str, value: str) -> None:
        """Update ExtractedParameters in place based on a change command."""
        # Reuse the NLU extractor for robust parsing
        extractor = self.conversation_manager.extractor
        extracted = extractor.extract(value)

        alias = {"pair": "currency_pair", "risk": "risk_tolerance", "time": "timeframe"}
        target = alias.get(param, param)

        if target == "currency_pair":
            base = extracted.base_currency or parameters.base_currency
            quote = extracted.quote_currency or parameters.quote_currency
            if base and quote:
                parameters.base_currency = base
                parameters.quote_currency = quote
                parameters.currency_pair = f"{base}/{quote}"
            elif extracted.currency_pair:
                parameters.currency_pair = extracted.currency_pair
        elif target == "amount":
            amt = extracted.amount
            if amt is None:
                # fallback to simple coercion
                try:
                    amt = float(value.replace(",", "").strip())
                except Exception:
                    pass
            if amt is None or amt <= 0:
                raise ValueError("Please provide a positive amount (e.g., 5000)")
            parameters.amount = float(amt)
        elif target == "risk_tolerance":
            if not extracted.risk_tolerance:
                raise ValueError("Please specify risk as conservative/moderate/aggressive")
            parameters.risk_tolerance = extracted.risk_tolerance
        elif target == "urgency":
            if not extracted.urgency:
                raise ValueError("Please specify urgency as urgent/normal/flexible")
            parameters.urgency = extracted.urgency
        elif target == "timeframe":
            # Support categorical timeframe or numeric days/weeks
            if extracted.timeframe:
                parameters.timeframe = extracted.timeframe
                parameters.timeframe_days = extracted.timeframe_days
            elif extracted.timeframe_days is not None:
                parameters.timeframe = None
                parameters.timeframe_days = extracted.timeframe_days
            else:
                raise ValueError("Please specify timeframe like '1_week' or 'in 10 days'")
        elif target == "timeframe_days":
            try:
                d = int(value)
            except Exception:
                raise ValueError("Please provide numeric days (e.g., 10)")
            parameters.timeframe = None
            parameters.timeframe_days = max(0, d)
        else:
            raise ValueError(f"Unknown parameter '{param}'")


def main() -> None:
    app = CurrencyAssistantTUI()
    app.run()


if __name__ == "__main__":  # pragma: no cover
    main()
