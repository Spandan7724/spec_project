"""
Clean, professional display system for currency advisor CLI
"""

from typing import Dict, Any, List, Optional
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.tree import Tree
from rich.columns import Columns
from rich.status import Status
from rich.text import Text
from rich.rule import Rule
from rich.align import Align
from rich import box
from rich.padding import Padding


class DisplayManager:
    """Manages all CLI display operations using Rich with clean, professional styling"""

    def __init__(self):
        self.console = Console(width=100)  # Optimize for standard terminal width

    def show_recommendation(self, result: Any, chat_mode: bool = False) -> None:
        """
        Display a currency recommendation result with clean, professional formatting

        Args:
            result: AdvisorResult object
            chat_mode: If True, use chat-optimized formatting
        """

        # Create main recommendation panel
        self._show_main_recommendation(result, chat_mode)

        if not chat_mode:
            # Show detailed analysis in non-chat mode
            self._show_detailed_analysis(result)

    def _show_main_recommendation(self, result: Any, chat_mode: bool = False) -> None:
        """Display the main recommendation panel with clean layout"""

        # Color code based on action
        action_colors = {
            "convert_now": "green",
            "wait": "yellow",
            "staged_conversion": "blue",
            "error": "red",
            "unknown": "gray"
        }

        action_color = action_colors.get(result.action, "blue")
        action_display = result.action.replace("_", " ").title()

        # Create clean table for key metrics
        metrics_table = Table(show_header=False, box=None, padding=0)
        metrics_table.add_column("Metric", style="bold", width=15)
        metrics_table.add_column("Value", style=action_color, width=25)

        metrics_table.add_row("Action", action_display)
        metrics_table.add_row("Confidence", self._format_confidence(result.confidence))
        metrics_table.add_row("Timeline", result.timeline or "Not specified")

        # Build clean content
        content_lines = []

        # Add metrics manually as formatted text
        content_lines.append(f"[bold]Action:[/bold]     [{action_color}]{action_display}[/{action_color}]")
        content_lines.append(f"[bold]Confidence:[/bold] {self._format_confidence(result.confidence)}")
        content_lines.append(f"[bold]Timeline:[/bold]    {result.timeline or 'Not specified'}")

        # Add summary
        content_lines.append(f"\n[bold]{result.recommendation}[/bold]")

        # Add rationale if available
        if result.rationale:
            content_lines.append("\n[bold]Key Reasons:[/bold]")
            for i, reason in enumerate(result.rationale[:3], 1):
                content_lines.append(f"  {chr(96+i)}. {reason}")

        # Add warnings if any
        if result.warnings:
            content_lines.append("\n[bold yellow]Warnings:[/bold yellow]")
            for warning in result.warnings[:2]:
                content_lines.append(f"  • {warning}")

        # Join content
        clean_content = "\n".join(content_lines)

        # Create clean panel title
        panel_title = f"Currency Recommendation"
        if not chat_mode:
            panel_title += f" - {result.request_summary}"

        # Display the panel with clean styling
        panel = Panel(
            clean_content,
            title=panel_title,
            border_style=action_color,
            padding=(1, 2)
        )

        self.console.print(Padding(panel, (1, 0, 0, 0)))

        # Show processing info in chat mode only if requested
        if chat_mode and result.processing_time:
            self.console.print(f"[dim]Processed in {result.processing_time:.1f}s | ID: {result.correlation_id[:8]}...[/dim]")

    def _show_detailed_analysis(self, result: Any) -> None:
        """Show detailed analysis breakdown with clean, professional styling"""

        self.console.print(Rule("Analysis Details"))

        # Create cleaner columns for different analysis types
        analyses = []

        # Market Analysis
        market_content = self._format_market_analysis(result.market_analysis)
        analyses.append(Panel(
            market_content,
            title="Market Analysis",
            border_style="cyan",
            padding=(1, 1)
        ))

        # Economic Analysis
        economic_content = self._format_economic_analysis(result.economic_analysis)
        analyses.append(Panel(
            economic_content,
            title="Economic Calendar",
            border_style="green",
            padding=(1, 1)
        ))

        # Risk Assessment
        risk_content = self._format_risk_assessment(result.risk_assessment)
        analyses.append(Panel(
            risk_content,
            title="Risk Assessment",
            border_style="yellow",
            padding=(1, 1)
        ))

        # Display columns with proper spacing
        self.console.print(Columns(analyses, equal=True, expand=True, column_first=True))

        # Show additional details
        self._show_additional_details(result)

    def _format_market_analysis(self, market: Dict[str, Any]) -> str:
        """Format market analysis with clean, professional styling"""

        lines = []

        # Summary
        summary = market.get('summary', 'N/A')
        lines.append(f"Summary: {summary}")

        # Bias with clean indicator
        if market.get('bias'):
            bias_indicators = {
                "bullish": "[green]▲[/green] Bullish",
                "bearish": "[red]▼[/red] Bearish",
                "neutral": "[gray]▪[/gray] Neutral"
            }
            bias_indicator = bias_indicators.get(market['bias'], f"[dim]{market['bias']}[/dim]")
            lines.append(f"Bias: {bias_indicator}")

        # Other metrics
        if market.get('confidence'):
            lines.append(f"Confidence: {self._format_confidence(market['confidence'])}")

        if market.get('rate'):
            lines.append(f"Current Rate: {market['rate']:.4f}")

        if market.get('regime'):
            lines.append(f"Market Regime: {market['regime'].title()}")

        return "\n".join(lines)

    def _format_economic_analysis(self, economic: Dict[str, Any]) -> str:
        """Format economic analysis with clean styling"""

        lines = []

        # Summary
        summary = economic.get('summary', 'N/A')
        lines.append(f"Summary: {summary}")

        # Bias with clean indicator
        if economic.get('bias'):
            bias_indicators = {
                "supportive": "[green]✓[/green] Supportive",
                "risk_on": "[green]ON[/green] Risk On",
                "risk_off": "[red]OFF[/red] Risk Off",
                "neutral": "[gray]-[/gray] Neutral"
            }
            bias_indicator = bias_indicators.get(economic['bias'].replace('_', ' ').title(), f"[dim]{economic['bias']}[/dim]")
            lines.append(f"Overall Bias: {bias_indicator}")

        # Events count
        high_impact = economic.get('high_impact_events', 0)
        if high_impact > 0:
            event_color = "red" if high_impact >= 3 else "yellow" if high_impact >= 1 else "white"
            lines.append(f"High-Impact Events: [{event_color}]{high_impact}[/{event_color}]")

        return "\n".join(lines)

    def _format_risk_assessment(self, risk: Dict[str, Any]) -> str:
        """Format risk assessment with clean styling"""

        lines = []

        # Summary
        summary = risk.get('summary', 'N/A')
        lines.append(f"Summary: {summary}")

        # Risk level with clean indicator
        if risk.get('risk_level'):
            risk_colors = {
                "low": "[green]LOW[/green]",
                "moderate": "[yellow]MEDIUM[/yellow]",
                "high": "[red]HIGH[/red]"
            }
            risk_color = risk_colors.get(risk['risk_level'], f"[dim]{risk['risk_level']}[/dim]")
            lines.append(f"Risk Level: {risk_color}")

        # Other metrics
        if risk.get('volatility'):
            lines.append(f"Volatility: {risk['volatility']:.2%}")

        if risk.get('var_95'):
            lines.append(f"95% VaR: {risk['var_95']:.2%}")

        return "\n".join(lines)

    def _show_additional_details(self, result: Any) -> None:
        """Show additional details with clean table formatting"""

        # Create clean details table
        details_table = Table(show_header=True, box=box.ROUNDED, show_lines=False)
        details_table.add_column("Metric", style="bold cyan", width=20)
        details_table.add_column("Value", style="white", width=40)

        details_table.add_row("Processing Time", f"{result.processing_time:.2f} seconds")
        details_table.add_row("Correlation ID", result.correlation_id[:8] + "...")
        details_table.add_row("Request", result.request_summary)

        self.console.print(Padding(details_table, (1, 0, 0, 0)))

        # Show errors if any
        if result.errors:
            self.console.print("\n[bold red]Errors Encountered:[/bold red]")
            for error in result.errors:
                self.console.print(f"  • [red]{error}[/red]")

    def show_system_status(self, status: Dict[str, Any], verbose: bool = False) -> None:
        """Display system status with clean professional styling"""

        # Overall health with clean indicators
        health_colors = {
            "healthy": "green",
            "degraded": "yellow",
            "partial": "blue",
            "unknown": "gray",
            "error": "red"
        }

        overall_health = status.get("overall_health", "unknown")
        health_color = health_colors.get(overall_health, "gray")
        health_indicator = {
            "healthy": "●",
            "degraded": "◆",
            "partial": "◐",
            "unknown": "○",
            "error": "✕"
        }.get(overall_health, "?")

        # Create status table
        status_table = Table(show_header=False, box=None, padding=0)
        status_table.add_column("Component", style="bold", width=20)
        status_table.add_column("Status", width=15)

        # Add component statuses
        components = status.get("components", {})
        for component_name, component_info in components.items():
            comp_status = component_info.get("status", "unknown")
            comp_color = health_colors.get(comp_status, "gray")

            # Clean component name
            display_name = component_name.replace('_', ' ').title()
            status_table.add_row(display_name, f"[{comp_color}]{comp_status}[/{comp_color}]")

        # Create main status panel
        main_content = f"[{health_color}]{health_indicator}[/{health_color}] System Status: {overall_health.upper()}"

        panel = Panel(
            Align.center(main_content),
            title=f"System Health (as of {status.get('timestamp', 'unknown').split('T')[0]})",
            border_style=health_color
        )

        self.console.print(panel)
        self.console.print()
        self.console.print(status_table)

        # Add verbose details if requested
        if verbose and components:
            for component_name, component_info in components.items():
                if component_info and any(key in component_info for key in ['healthy_providers', 'providers_count', 'error']):
                    comp_status = component_info.get("status", "unknown")
                    comp_color = health_colors.get(comp_status, "gray")

                    details_lines = []
                    if "healthy_providers" in component_info:
                        details_lines.append(f"Healthy: {', '.join(component_info['healthy_providers'])}")
                    if "total_providers" in component_info:
                        details_lines.append(f"Total: {', '.join(component_info['total_providers'])}")
                    if "providers_count" in component_info:
                        details_lines.append(f"Providers: {component_info['providers_count']}")
                    if "error" in component_info:
                        details_lines.append(f"[red]Error: {component_info['error']}[/red]")

                    if details_lines:
                        details_content = "\n".join(details_lines)
                        details_panel = Panel(
                            details_content,
                            title=f"{component_name.replace('_', ' ').title()} Details",
                            border_style=comp_color
                        )
                        self.console.print(Padding(details_panel, (0, 4)))

    def show_configuration(self, config: Dict[str, Any]) -> None:
        """Display configuration with clean table formatting"""

        config_table = Table(title="Configuration", box=box.ROUNDED, show_lines=True)
        config_table.add_column("Setting", style="bold cyan", width=25)
        config_table.add_column("Value", style="white", width=25)

        for key, value in config.items():
            # Hide sensitive values
            if any(sensitive in key.lower() for sensitive in ['key', 'token', 'secret', 'password']):
                display_value = "***" if value else "Not set"
            else:
                display_value = str(value) if value is not None else "Not set"

            config_table.add_row(key, display_value)

        self.console.print(config_table)

    def show_progress(self, steps: List[str]) -> Progress:
        """Create and return a progress display with clean styling"""

        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=self.console
        )

        return progress

    def show_error(self, message: str, title: str = "Error") -> None:
        """Display an error message with clean styling"""
        self.console.print(Panel(
            f"[red]{message}[/red]",
            title=title,
            border_style="red"
        ))

    def show_success(self, message: str, title: str = "Success") -> None:
        """Display a success message with clean styling"""
        self.console.print(Panel(
            f"[green]{message}[/green]",
            title=title,
            border_style="green"
        ))

    def show_warning(self, message: str, title: str = "Warning") -> None:
        """Display a warning message with clean styling"""
        self.console.print(Panel(
            f"[yellow]{message}[/yellow]",
            title=title,
            border_style="yellow"
        ))

    def _format_confidence(self, confidence: float) -> str:
        """Format confidence level with clean color coding"""

        if confidence >= 0.7:
            color = "green"
            level = "High"
        elif confidence >= 0.5:
            color = "yellow"
            level = "Medium"
        else:
            color = "red"
            level = "Low"

        return f"[{color}]{level} ({confidence:.0%})[/{color}]"

    def print_header(self, title: str, subtitle: str = "") -> None:
        """Print a clean styled header"""
        self.console.print(Panel(
            Text(title, style="bold blue"),
            subtitle=subtitle,
            border_style="blue"
        ))

    def print_rule(self, title: str = "") -> None:
        """Print a clean horizontal rule"""
        self.console.print(Rule(title))

    def clear(self) -> None:
        """Clear the console"""
        self.console.clear()