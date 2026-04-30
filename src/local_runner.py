"""Local CLI runner for the PydanticAI supervisor."""

import asyncio
import getpass
import os

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text

from src.agent_graph import get_supervisor, run_supervisor_with_critic
from src.modeling import get_openai_api_key
from src.tracing import configure_adk_tracing

DEFAULT_BRAINTRUST_PROJECT = "pydantic-supervisor"


def _set_if_undefined(var: str) -> None:
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"Please provide your {var}: ")


async def _run_chat() -> None:
    load_dotenv()
    if not get_openai_api_key():
        _set_if_undefined("OPENAI_API_KEY")
    _set_if_undefined("TAVILY_API_KEY")

    if os.environ.get("BRAINTRUST_API_KEY"):
        configure_adk_tracing(
            api_key=os.environ.get("BRAINTRUST_API_KEY"),
            project_id=os.environ.get("BRAINTRUST_PROJECT_ID"),
            project_name=os.environ.get("BRAINTRUST_PROJECT", DEFAULT_BRAINTRUST_PROJECT),
        )

    console = Console()
    supervisor = get_supervisor()

    welcome_text = Text("Pydantic Supervisor Chat", style="bold cyan")
    welcome_panel = Panel(
        welcome_text,
        subtitle="Type 'quit' or 'q' to exit",
        border_style="cyan",
    )
    console.print(welcome_panel)
    console.print()

    while True:
        user_input = Prompt.ask("[bold green]You[/bold green]", console=console)

        if user_input.lower() in {"q", "quit", "exit"}:
            console.print("\n[bold yellow]Goodbye![/bold yellow]")
            break

        if not user_input.strip():
            continue

        with console.status("[bold blue]Processing...", spinner="dots"):
            run_result = await run_supervisor_with_critic(
                supervisor=supervisor,
                query=user_input,
                app_name="pydantic-supervisor-local",
            )

        final_output = run_result.get("final_output", "")
        console.print(
            Panel(
                str(final_output) if final_output else "(No response generated)",
                title="Assistant",
                border_style="blue",
            )
        )
        console.print()


def main() -> None:
    asyncio.run(_run_chat())


if __name__ == "__main__":
    main()
