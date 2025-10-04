"""
Interactive chat session management for currency advisor CLI
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from src.cli.services import CurrencyAdvisorService, AdvisorResult
from src.cli.display import DisplayManager

logger = logging.getLogger(__name__)


PROMPT_STYLE = Style.from_dict({"": "bold cyan"})


class ChatSession:
    """Interactive chat session for currency advice"""

    def __init__(self, session_id: Optional[str] = None, debug: bool = False):
        self.session_id = session_id or str(uuid.uuid4())
        self.debug = debug
        self.console = Console()
        self.service = CurrencyAdvisorService(debug=debug)
        self.display = DisplayManager()

        # Chat history
        self.messages: List[Dict[str, Any]] = []
        self.started_at = datetime.now()

        # Prompt toolkit setup
        self.history_file = Path.home() / ".currency_assistant_history"
        self.prompt_session = PromptSession(
            history=FileHistory(str(self.history_file)),
            auto_suggest=AutoSuggestFromHistory(),
            completer=self._create_completer(),
            key_bindings=self._create_key_bindings(),
        )

        # Session file for persistence
        self.session_file = Path.home() / ".currency_assistant_sessions" / f"{self.session_id}.json"
        self.session_file.parent.mkdir(exist_ok=True)

        # Load existing session if provided
        if session_id and self.session_file.exists():
            self._load_session()

    def _create_completer(self) -> WordCompleter:
        """Create command completer for better UX"""
        commands = [
            "help", "status", "clear", "history", "exit", "quit",
            "What should I do with", "Should I convert", "Convert", "Exchange",
            "USD to EUR", "EUR to USD", "GBP to USD", "USD to JPY",
            "this week", "next week", "this month", "today", "now",
            "low risk", "moderate risk", "high risk",
        ]
        return WordCompleter(commands, ignore_case=True)

    def _create_key_bindings(self) -> KeyBindings:
        """Create custom key bindings"""
        kb = KeyBindings()

        @kb.add('c-c')
        def _(event):
            """Handle Ctrl+C gracefully"""
            event.app.exit(exception=KeyboardInterrupt)

        @kb.add('c-d')
        def _(event):
            """Handle Ctrl+D to exit"""
            event.app.exit(exception=EOFError)

        return kb

    async def start(self) -> None:
        """Start the interactive chat session"""

        self._show_welcome_message()

        try:
            while True:
                try:
                    # Get user input
                    user_input = await self._get_user_input()

                    if not user_input.strip():
                        continue

                    # Handle commands
                    if await self._handle_command(user_input):
                        continue

                    # Process the question
                    await self._process_message(user_input)

                except KeyboardInterrupt:
                    self.console.print("\n[yellow]👋 Use 'exit' or 'quit' to end the session[/yellow]")
                except EOFError:
                    break
                except Exception as e:
                    self.console.print(f"[red]❌ Error: {e}[/red]")
                    if self.debug:
                        import traceback
                        self.console.print(traceback.format_exc())

        finally:
            self._save_session()
            self._show_goodbye_message()

    async def _get_user_input(self) -> str:
        """Get user input with styled prompt"""

        # Create a nice prompt showing current context
        prompt_text = f"[{datetime.now().strftime('%H:%M')}] You: "

        user_input = await self.prompt_session.prompt_async(
            prompt_text,
            style=PROMPT_STYLE,
        )

        return user_input.strip()

    async def _handle_command(self, user_input: str) -> bool:
        """Handle special commands - returns True if command was handled"""

        command = user_input.lower().strip()

        if command in ['exit', 'quit', 'bye']:
            raise EOFError
        elif command == 'help':
            self._show_help()
            return True
        elif command == 'status':
            await self._show_status()
            return True
        elif command == 'clear':
            self._clear_screen()
            return True
        elif command == 'history':
            self._show_history()
            return True
        elif command.startswith('export'):
            await self._export_session(command)
            return True

        return False

    async def _process_message(self, user_input: str) -> None:
        """Process a user message and show the response"""

        # Add to message history
        self.messages.append({
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now().isoformat()
        })

        try:
            # Show typing indicator
            with self.console.status("[bold green]Analyzing your request...", spinner="dots"):
                result = await self.service.ask_question(user_input)

            # Display the result
            self.display.show_recommendation(result, chat_mode=True)

            # Add to message history
            self.messages.append({
                "role": "assistant",
                "content": result.to_dict(),
                "timestamp": datetime.now().isoformat()
            })

        except Exception as e:
            error_msg = f"Sorry, I encountered an error: {e}"
            self.console.print(f"[red]{error_msg}[/red]")

            self.messages.append({
                "role": "assistant",
                "content": {"error": str(e)},
                "timestamp": datetime.now().isoformat()
            })

        # Add spacing for readability
        self.console.print()

    def _show_welcome_message(self) -> None:
        """Show welcome message when session starts"""

        welcome_text = f"""
**Welcome to Currency Assistant!**

I'm your AI-powered currency conversion timing advisor. I can help you decide:
• **When** to convert currencies for optimal timing
• **What** market conditions to consider
• **How** to manage risk in your conversions

**Session ID**: `{self.session_id[:8]}...`

**Examples of questions you can ask:**
• "Should I convert $1000 USD to EUR this week?"
• "What's the outlook for GBP/USD over the next 30 days?"
• "I need to exchange 5000 JPY to EUR, what's your recommendation?"

Type **'help'** for commands or **'exit'** to quit.
        """

        self.console.print(Panel(
            Markdown(welcome_text),
            title="Chat Session Started",
            border_style="blue"
        ))

    def _show_help(self) -> None:
        """Show help information"""

        help_text = """
**Available Commands:**
• **help** - Show this help message
• **status** - Check system status and health
• **history** - Show conversation history
• **clear** - Clear the screen
• **export** - Export session to file
• **exit/quit** - End the session

**Example Questions:**
• "Convert $1000 USD to EUR this week"
• "Should I wait or convert GBP to USD now?"
• "What's the risk for EUR/JPY this month?"
• "I have 5000 CAD, convert to USD with low risk"
        """

        self.console.print(Panel(
            Markdown(help_text),
            title="Help",
            border_style="green"
        ))

    async def _show_status(self) -> None:
        """Show system status"""

        with self.console.status("[bold blue]Checking system status...", spinner="dots"):
            status = await self.service.get_system_status()

        self.display.show_system_status(status, verbose=False)

    def _clear_screen(self) -> None:
        """Clear the console screen"""
        self.console.clear()

    def _show_history(self) -> None:
        """Show conversation history"""

        if not self.messages:
            self.console.print("[yellow]No messages in this session yet[/yellow]")
            return

        self.console.print(Panel(
            f"**Session History** ({len(self.messages)} messages)",
            title="History",
            border_style="purple"
        ))

        for i, msg in enumerate(self.messages, 1):
            if msg["role"] == "user":
                self.console.print(f"[cyan]{i}. You:[/cyan] {msg['content']}")
            else:
                content = msg["content"]
                if isinstance(content, dict) and "recommendation" in content:
                    self.console.print(f"[green]{i}. Assistant:[/green] {content['recommendation']}")
                elif isinstance(content, dict) and "error" in content:
                    self.console.print(f"[red]{i}. Assistant:[/red] Error: {content['error']}")
                else:
                    self.console.print(f"[green]{i}. Assistant:[/green] {str(content)[:100]}...")

        self.console.print()

    async def _export_session(self, command: str) -> None:
        """Export session to file"""

        try:
            # Parse filename from command
            parts = command.split(maxsplit=1)
            filename = parts[1] if len(parts) > 1 else f"currency_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

            if not filename.endswith('.json'):
                filename += '.json'

            export_path = Path.cwd() / filename

            # Prepare export data
            export_data = {
                "session_id": self.session_id,
                "started_at": self.started_at.isoformat(),
                "exported_at": datetime.now().isoformat(),
                "messages": self.messages,
                "total_messages": len(self.messages)
            }

            # Write to file
            with open(export_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)

            self.console.print(f"[green]Session exported to {export_path}[/green]")

        except Exception as e:
            self.console.print(f"[red]Failed to export session: {e}[/red]")

    def _show_goodbye_message(self) -> None:
        """Show goodbye message when session ends"""

        session_duration = datetime.now() - self.started_at
        duration_str = str(session_duration).split('.')[0]  # Remove microseconds

        goodbye_text = f"""
**Session Ended!**

**Session Stats:**
• **Duration**: {duration_str}
• **Messages**: {len(self.messages)}
• **Session ID**: `{self.session_id[:8]}...`

Your session has been saved and can be resumed with:
```bash
currency-assistant chat --session {self.session_id}
```

Thank you for using Currency Assistant!
        """

        self.console.print(Panel(
            Markdown(goodbye_text),
            title="Goodbye!",
            border_style="blue"
        ))

    def _save_session(self) -> None:
        """Save session to file for persistence"""

        try:
            session_data = {
                "session_id": self.session_id,
                "started_at": self.started_at.isoformat(),
                "last_activity": datetime.now().isoformat(),
                "messages": self.messages
            }

            with open(self.session_file, 'w') as f:
                json.dump(session_data, f, indent=2, default=str)

        except Exception as e:
            logger.warning(f"Failed to save session: {e}")

    def _load_session(self) -> None:
        """Load existing session from file"""

        try:
            with open(self.session_file, 'r') as f:
                session_data = json.load(f)

            self.messages = session_data.get("messages", [])
            old_started_at = session_data.get("started_at")

            if old_started_at:
                self.started_at = datetime.fromisoformat(old_started_at)

            self.console.print(f"[green]Resumed session with {len(self.messages)} previous messages[/green]")

        except Exception as e:
            logger.warning(f"Failed to load session: {e}")
            self.console.print("[yellow]Could not fully restore previous session[/yellow]")
