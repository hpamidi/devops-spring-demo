"""LangGraph agent state definition."""
from __future__ import annotations

from typing import Annotated, Any, Optional
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class IncidentInput(TypedDict):
    id: str
    title: str
    description: str
    severity: str          # CRITICAL | HIGH | MEDIUM | LOW
    category: str          # network_outage | security_incident | performance_degradation | other
    timestamp: str         # ISO-8601


class BulletinDraft(TypedDict):
    bulletin_id: str
    incident_id: str
    category: str
    subject: str
    body: str
    severity: str
    template_name: str
    generated_at: str


class AgentState(TypedDict):
    # Input
    incident: IncidentInput

    # Conversation messages (accumulated via add_messages reducer)
    messages: Annotated[list[BaseMessage], add_messages]

    # Intermediate state
    template: Optional[dict[str, Any]]
    historical_context: Optional[list[dict[str, Any]]]
    bulletin_draft: Optional[BulletinDraft]

    # Flow control
    error: Optional[str]
    error_node: Optional[str]
    retry_count: int
    bcu_response: Optional[dict[str, Any]]

    # Metadata
    run_id: str
    trace_url: Optional[str]
