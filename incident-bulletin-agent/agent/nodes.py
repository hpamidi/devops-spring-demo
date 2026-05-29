"""LangGraph node implementations for the Incident Bulletin Agent."""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from agent.guardrails import validate_bulletin_output, validate_incident_input
from agent.monitoring import (
    AZURE_SEARCH_REQUESTS,
    BCU_REQUESTS,
    BULLETINS_GENERATED,
    TOKEN_USAGE,
    get_logger,
    node_span,
)
from agent.state import AgentState
from services.azure_search import AzureSearchClient
from services.bcu_client import BCUClient
from services.template_store import TemplateStore

logger = get_logger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _mark_error(state: AgentState, node: str, message: str) -> dict[str, Any]:
    logger.error("Node error", extra={"node": node, "error": message})
    return {"error": message, "error_node": node}


def _llm_factory():
    """Build LLM client from settings (lazy import to keep startup fast)."""
    from config import get_settings
    settings = get_settings()

    if settings.llm_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=settings.llm_model,
            anthropic_api_key=settings.anthropic_api_key,
            max_tokens=2048,
        )
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=settings.llm_model,
        openai_api_key=settings.openai_api_key,
        max_tokens=2048,
    )


# ── Node: validate_incident ───────────────────────────────────────────────────

def validate_incident(state: AgentState) -> dict[str, Any]:
    with node_span("validate_incident"):
        try:
            validated = validate_incident_input(state["incident"])
            logger.info(
                "Incident validated",
                extra={
                    "incident_id": validated.id,
                    "severity": validated.severity,
                    "category": validated.category,
                },
            )
            return {
                "incident": validated.model_dump(),
                "error": None,
                "error_node": None,
            }
        except Exception as exc:
            return _mark_error(state, "validate_incident", str(exc))


# ── Node: fetch_template ──────────────────────────────────────────────────────

def fetch_template(state: AgentState) -> dict[str, Any]:
    with node_span("fetch_template"):
        try:
            category = state["incident"]["category"]
            store = TemplateStore()
            template = store.get_template(category)
            logger.info("Template fetched", extra={"template": template.get("name"), "category": category})
            return {"template": template, "error": None}
        except Exception as exc:
            return _mark_error(state, "fetch_template", str(exc))


# ── Node: search_historical ───────────────────────────────────────────────────

def search_historical(state: AgentState) -> dict[str, Any]:
    with node_span("search_historical"):
        try:
            incident = state["incident"]
            query = f"{incident['category']} {incident['title']} {incident['description'][:200]}"
            client = AzureSearchClient()
            results = client.search(query=query, top=5)
            AZURE_SEARCH_REQUESTS.labels(status="success").inc()
            logger.info("Historical search complete", extra={"results_count": len(results)})
            return {"historical_context": results, "error": None}
        except Exception as exc:
            AZURE_SEARCH_REQUESTS.labels(status="error").inc()
            # Non-fatal: continue with empty context
            logger.warning("Azure Search failed, continuing without history", extra={"error": str(exc)})
            return {"historical_context": [], "error": None}


# ── Node: generate_bulletin ───────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are an expert IT communications specialist. Your job is to produce
professional, clear, and actionable customer-facing incident bulletins.

Guidelines:
- Use formal but accessible language
- Include all key facts: what happened, impact, current status, next steps
- Fill every placeholder in the template with appropriate content
- Do NOT invent facts; base content strictly on the incident details and historical context
- Return ONLY valid JSON matching the bulletin schema, no extra text
"""


def generate_bulletin(state: AgentState) -> dict[str, Any]:
    with node_span("generate_bulletin"):
        try:
            incident = state["incident"]
            template = state.get("template") or {}
            history = state.get("historical_context") or []

            human_content = f"""
## Incident Details
{json.dumps(incident, indent=2)}

## Bulletin Template
{json.dumps(template, indent=2)}

## Historical Similar Bulletins (for tone/style reference)
{json.dumps(history[:3], indent=2)}

## Task
Fill in the template placeholders using the incident details and historical context.
Return a JSON object with these fields:
{{
  "bulletin_id": "<new UUID>",
  "incident_id": "{incident['id']}",
  "category": "{incident['category']}",
  "subject": "<bulletin email subject line>",
  "body": "<full bulletin body with ALL {{{{placeholders}}}} replaced>",
  "severity": "{incident['severity']}",
  "template_name": "{template.get('name', 'default')}",
  "generated_at": "<ISO-8601 timestamp>"
}}
"""
            llm = _llm_factory()
            messages = [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=human_content)]
            response = llm.invoke(messages)

            # Track token usage if available
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                usage = response.usage_metadata
                TOKEN_USAGE.labels(type="prompt").inc(usage.get("input_tokens", 0))
                TOKEN_USAGE.labels(type="completion").inc(usage.get("output_tokens", 0))

            raw = response.content.strip()
            # Strip markdown code fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            bulletin_data = json.loads(raw.strip())

            # Ensure bulletin_id and generated_at are set
            if not bulletin_data.get("bulletin_id"):
                bulletin_data["bulletin_id"] = str(uuid.uuid4())
            if not bulletin_data.get("generated_at"):
                bulletin_data["generated_at"] = datetime.now(timezone.utc).isoformat()

            BULLETINS_GENERATED.labels(
                category=incident["category"], severity=incident["severity"]
            ).inc()

            logger.info(
                "Bulletin generated",
                extra={"bulletin_id": bulletin_data["bulletin_id"], "incident_id": incident["id"]},
            )
            return {"bulletin_draft": bulletin_data, "error": None}

        except Exception as exc:
            return _mark_error(state, "generate_bulletin", str(exc))


# ── Node: validate_output ─────────────────────────────────────────────────────

def validate_output(state: AgentState) -> dict[str, Any]:
    with node_span("validate_output"):
        try:
            draft = state.get("bulletin_draft")
            if not draft:
                raise ValueError("No bulletin draft to validate")
            validated = validate_bulletin_output(draft)
            logger.info("Bulletin output validated", extra={"bulletin_id": validated.bulletin_id})
            return {"bulletin_draft": validated.model_dump(), "error": None}
        except Exception as exc:
            return _mark_error(state, "validate_output", str(exc))


# ── Node: send_to_bcu ─────────────────────────────────────────────────────────

def send_to_bcu(state: AgentState) -> dict[str, Any]:
    with node_span("send_to_bcu"):
        try:
            draft = state.get("bulletin_draft")
            if not draft:
                raise ValueError("No bulletin draft to send")
            client = BCUClient()
            response = client.send_bulletin(draft)
            BCU_REQUESTS.labels(status="success").inc()
            logger.info(
                "Bulletin sent to BCU",
                extra={"bulletin_id": draft["bulletin_id"], "bcu_response": response},
            )
            return {"bcu_response": response, "error": None}
        except Exception as exc:
            BCU_REQUESTS.labels(status="error").inc()
            return _mark_error(state, "send_to_bcu", str(exc))


# ── Node: handle_error ────────────────────────────────────────────────────────

def handle_error(state: AgentState) -> dict[str, Any]:
    with node_span("handle_error"):
        error = state.get("error", "Unknown error")
        error_node = state.get("error_node", "unknown")
        incident_id = state.get("incident", {}).get("id", "unknown")

        logger.error(
            "Incident bulletin pipeline failed",
            extra={
                "incident_id": incident_id,
                "failed_node": error_node,
                "error": error,
            },
        )
        # Dead-letter: log and surface the error for downstream alerting
        return {
            "bcu_response": {
                "status": "failed",
                "error": error,
                "failed_at_node": error_node,
                "incident_id": incident_id,
            }
        }
