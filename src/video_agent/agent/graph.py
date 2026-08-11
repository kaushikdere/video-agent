"""LangGraph StateGraph — observe → think → act → evaluate → repeat | terminate."""
from __future__ import annotations

import structlog
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from video_agent.agent.state import AgentState, JobStatus
from video_agent.agent.nodes.planner import plan_story_node
from video_agent.agent.nodes.bible import lock_bible_node
from video_agent.agent.nodes.generator import generate_shot_node
from video_agent.agent.nodes.qc import qc_shot_node
from video_agent.agent.nodes.assembler import assemble_node
from video_agent.config import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()


# ─── Routing functions ────────────────────────────────────────────────────────

def after_generate(state: AgentState) -> str:
    """Route to QC after generation."""
    return "qc_shot"


def after_qc(state: AgentState) -> str:
    """
    After QC:
    - If shot failed and repair budget allows → back to generator (repair)
    - If shot failed and no budget → assemble with what we have
    - If shot passed and more shots to go → back to generator (next shot)
    - If shot passed and all shots done → assemble
    """
    current = state["current_shot_index"]
    shots = state["shots"]
    budget = state["budget"]
    repair_count = state["repair_count"]

    # Check global budget first
    if (
        budget["cost_usd"] >= settings.max_job_budget_usd
        or budget["iterations"] >= settings.max_job_iterations
        or budget["elapsed_seconds"] >= settings.max_job_wall_clock_seconds
    ):
        logger.warning("budget_exhausted", job_id=state["job_id"], budget=budget)
        return "assemble"

    # Look at the last shot result
    if shots and shots[-1]["shot_index"] == current:
        last_shot = shots[-1]
        if last_shot["status"] == "failed":
            if repair_count < settings.max_qc_repair_attempts:
                return "generate_shot"  # repair
            else:
                # Failed after max repairs — still try to assemble partial
                return "assemble"

    # Advance to next shot or assemble
    if current < settings.shots_per_story - 1:
        return "generate_shot"
    return "assemble"


def after_assemble(state: AgentState) -> str:
    return END


def check_failed(state: AgentState) -> str:
    """Terminal check — abort if status already failed."""
    if state["status"] in (
        JobStatus.FAILED,
        JobStatus.FAILED_NO_PROGRESS,
        JobStatus.ESCALATED,
    ):
        return "assemble"
    return "lock_bible"


# ─── Graph construction ───────────────────────────────────────────────────────

def build_graph(checkpointer=None) -> "CompiledGraph":  # type: ignore[name-defined]
    """Build and compile the Video Agent StateGraph."""
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("plan_story", plan_story_node)
    graph.add_node("lock_bible", lock_bible_node)
    graph.add_node("generate_shot", generate_shot_node)
    graph.add_node("qc_shot", qc_shot_node)
    graph.add_node("assemble", assemble_node)

    # Entry point
    graph.set_entry_point("plan_story")

    # plan_story → lock_bible (or abort)
    graph.add_conditional_edges(
        "plan_story",
        check_failed,
        {"lock_bible": "lock_bible", "assemble": "assemble"},
    )

    # lock_bible → generate_shot
    graph.add_edge("lock_bible", "generate_shot")

    # generate_shot → qc_shot
    graph.add_edge("generate_shot", "qc_shot")

    # qc_shot → conditional routing
    graph.add_conditional_edges(
        "qc_shot",
        after_qc,
        {
            "generate_shot": "generate_shot",
            "assemble": "assemble",
        },
    )

    # assemble → END
    graph.add_edge("assemble", END)

    checkpointer = checkpointer or MemorySaver()
    return graph.compile(checkpointer=checkpointer)


# Singleton compiled graph
_graph = None


def get_graph() -> "CompiledGraph":  # type: ignore[name-defined]
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph
