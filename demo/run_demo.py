"""Demo script — run a full Video Agent job end-to-end from the CLI.

Usage:
    python demo/run_demo.py --prompt "A lone astronaut discovers a garden on Mars"
    python demo/run_demo.py --prompt "..." --provider mock
    python demo/run_demo.py --prompt "..." --server http://localhost:8000
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text

# Add src to path for local runs without install
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

app = typer.Typer(help="Video Agent demo — one prompt → 40-second story")
console = Console()


@app.command()
def main(
    prompt: str = typer.Option(
        "A lone astronaut discovers a hidden garden on the surface of Mars at sunset.",
        "--prompt", "-p",
        help="Your story idea",
    ),
    provider: str = typer.Option("mock", "--provider", help="'mock' or 'higgsfield'"),
    server: str = typer.Option("", "--server", help="Remote server URL. Leave empty for local embedded run."),
    timeout: int = typer.Option(300, "--timeout", help="Max seconds to wait for completion"),
):
    """Run a full Video Agent job and display results."""
    asyncio.run(_run(prompt, provider, server, timeout))


async def _run(prompt: str, provider: str, server: str, timeout: int):
    console.print()
    console.print(Panel.fit(
        "[bold cyan]🎬 Video Agent[/bold cyan]\n"
        "[dim]One prompt → continuous 40-second story[/dim]\n\n"
        f"[yellow]Prompt:[/yellow] {prompt}\n"
        f"[yellow]Provider:[/yellow] {provider}",
        border_style="cyan",
    ))
    console.print()

    if server:
        await _run_remote(prompt, provider, server, timeout)
    else:
        await _run_local(prompt, provider, timeout)


async def _run_local(prompt: str, provider: str, timeout: int):
    """Run directly via the agent graph (no HTTP server required)."""
    import uuid

    # Import here so the script works from any directory
    from video_agent.config import get_settings
    from video_agent.agent.state import initial_state, JobStatus
    from video_agent.agent.graph import get_graph
    from video_agent.observability.langfuse_client import create_trace

    settings = get_settings()

    # Override provider
    import video_agent.agent.nodes.generator as gen_mod
    if provider == "mock":
        from video_agent.providers.mock import MockVideoProvider
        gen_mod._get_provider = lambda: MockVideoProvider(simulate_latency=True)

    job_id = str(uuid.uuid4())[:8]
    trace_id = create_trace(job_id, prompt)
    state = initial_state(job_id=job_id, user_prompt=prompt, trace_id=trace_id)
    graph = get_graph()
    config = {"configurable": {"thread_id": job_id}}

    console.print(f"[dim]Job ID: {job_id}[/dim]")
    console.print()

    node_times: dict[str, float] = {}
    final_state: dict = {}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Running agent...", total=None)

        async for event in graph.astream(state, config=config):
            for node_name, node_state in event.items():
                progress.update(task, description=f"[cyan]✓ {node_name}")
                if isinstance(node_state, dict):
                    final_state.update(node_state)

    _display_results(final_state, job_id)


async def _run_remote(prompt: str, provider: str, server: str, timeout: int):
    """Submit job to a running server and poll for results."""
    import httpx

    async with httpx.AsyncClient(base_url=server, timeout=30) as client:
        resp = await client.post(
            "/api/v1/jobs",
            json={"prompt": prompt, "provider": provider},
        )
        resp.raise_for_status()
        job_id = resp.json()["job_id"]
        console.print(f"[dim]Job ID: {job_id}[/dim]")
        console.print()

        terminal = {"success", "partial", "failed", "failed_no_progress", "escalated"}
        start = time.time()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("[cyan]Running agent...", total=None)

            while time.time() - start < timeout:
                await asyncio.sleep(3)
                poll = await client.get(f"/api/v1/jobs/{job_id}")
                data = poll.json()
                status = data["status"]
                progress.update(task, description=f"[cyan]{status} — {len(data.get('shots', []))} shots")

                if status in terminal:
                    break

        _display_results(data, job_id)


def _display_results(state: dict, job_id: str):
    status = state.get("status", "unknown")
    colour = "green" if status == "success" else "yellow" if status == "partial" else "red"
    console.print(Panel.fit(
        f"[bold {colour}]Status: {status.upper()}[/bold {colour}]",
        border_style=colour,
    ))
    console.print()

    # Story Plan
    sp = state.get("story_plan")
    if sp:
        console.print(f"[bold]📖 Story Plan:[/bold] {sp.get('title')} ({sp.get('genre')})")
        for beat in sp.get("beats", []):
            console.print(f"  Beat {beat['index']+1} [{beat['label']}]: {beat['action']}")
        console.print()

    # Shots table
    shots = state.get("shots", [])
    if shots:
        table = Table(title="🎬 Shots", border_style="cyan")
        table.add_column("Shot", style="bold")
        table.add_column("Status")
        table.add_column("QC Score")
        table.add_column("Attempts")
        table.add_column("Latency")
        table.add_column("URL")

        for s in sorted(shots, key=lambda x: x.get("shot_index", 0)):
            sc = s.get("qc_score", 0)
            sc_color = "green" if sc >= 0.75 else "yellow" if sc >= 0.5 else "red"
            table.add_row(
                str(s.get("shot_index", 0) + 1),
                s.get("status", "?"),
                f"[{sc_color}]{sc:.2f}[/{sc_color}]",
                str(s.get("qc_attempts", 0)),
                f"{s.get('latency_seconds', 0):.1f}s",
                (s.get("clip_url") or "")[:50],
            )
        console.print(table)
        console.print()

    # Artifacts
    artifacts = state.get("artifacts")
    if artifacts:
        console.print("[bold]📦 Artifacts:[/bold]")
        console.print(f"  Stitched MP4: {artifacts.get('stitched_mp4_url', '')}")
        console.print(f"  StoryPlan JSON: {artifacts.get('story_plan_url', '')}")
        console.print(f"  Bible JSON: {artifacts.get('continuity_bible_url', '')}")
        console.print()

    # Budget
    budget = state.get("budget")
    if budget:
        console.print(
            f"[dim]💰 Cost: ${budget.get('cost_usd', 0):.4f} | "
            f"Tokens: {budget.get('tokens_used', 0)} | "
            f"Elapsed: {budget.get('elapsed_seconds', 0):.1f}s | "
            f"Iterations: {budget.get('iterations', 0)}[/dim]"
        )
    console.print()
    console.print(f"[bold green]✅ Done![/bold green] Job: {job_id}")


if __name__ == "__main__":
    app()
