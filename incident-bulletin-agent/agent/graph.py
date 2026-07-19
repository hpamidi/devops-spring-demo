"""LangGraph StateGraph definition for the Incident Bulletin Agent."""
from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from agent.nodes import (
    fetch_template,
    generate_bulletin,
    handle_error,
    search_historical,
    send_to_bcu,
    validate_incident,
    validate_output,
)
from agent.state import AgentState


def _route_on_error(state: AgentState) -> str:
    """Route to handle_error if the previous node set an error, else continue."""
    return "handle_error" if state.get("error") else "continue"


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    # ── Register nodes ────────────────────────────────────────────────────────
    graph.add_node("validate_incident", validate_incident)
    graph.add_node("fetch_template", fetch_template)
    graph.add_node("search_historical", search_historical)
    graph.add_node("generate_bulletin", generate_bulletin)
    graph.add_node("validate_output", validate_output)
    graph.add_node("send_to_bcu", send_to_bcu)
    graph.add_node("handle_error", handle_error)

    # ── Entry ─────────────────────────────────────────────────────────────────
    graph.add_edge(START, "validate_incident")

    # ── Conditional edges after each node ─────────────────────────────────────
    graph.add_conditional_edges(
        "validate_incident",
        _route_on_error,
        {"continue": "fetch_template", "handle_error": "handle_error"},
    )
    graph.add_conditional_edges(
        "fetch_template",
        _route_on_error,
        {"continue": "search_historical", "handle_error": "handle_error"},
    )
    graph.add_conditional_edges(
        "search_historical",
        _route_on_error,
        {"continue": "generate_bulletin", "handle_error": "handle_error"},
    )
    graph.add_conditional_edges(
        "generate_bulletin",
        _route_on_error,
        {"continue": "validate_output", "handle_error": "handle_error"},
    )
    graph.add_conditional_edges(
        "validate_output",
        _route_on_error,
        {"continue": "send_to_bcu", "handle_error": "handle_error"},
    )
    graph.add_conditional_edges(
        "send_to_bcu",
        _route_on_error,
        {"continue": END, "handle_error": "handle_error"},
    )

    # ── Error terminal ────────────────────────────────────────────────────────
    graph.add_edge("handle_error", END)

    return graph


def compile_graph():
    """Return a compiled, runnable LangGraph application."""
    return build_graph().compile()


# Module-level compiled graph (singleton)
bulletin_agent = compile_graph()


def run_agent(incident: dict[str, Any], run_id: str | None = None) -> dict[str, Any]:
    """
    Execute the bulletin agent for a single incident.
    Returns the final AgentState.
    """
    import uuid
    from agent.monitoring import get_logger

    logger = get_logger(__name__)
    run_id = run_id or str(uuid.uuid4())

    initial_state: AgentState = {
        "incident": incident,
        "messages": [],
        "template": None,
        "historical_context": None,
        "bulletin_draft": None,
        "error": None,
        "error_node": None,
        "retry_count": 0,
        "bcu_response": None,
        "run_id": run_id,
        "trace_url": None,
    }

    logger.info("Agent run started", extra={"run_id": run_id, "incident_id": incident.get("id")})
    result = bulletin_agent.invoke(initial_state, config={"run_id": run_id})
    logger.info(
        "Agent run completed",
        extra={
            "run_id": run_id,
            "incident_id": incident.get("id"),
            "success": result.get("error") is None,
        },
    )
    return result
