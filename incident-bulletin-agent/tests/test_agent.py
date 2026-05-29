"""Integration tests for the LangGraph agent graph flow."""
import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

VALID_INCIDENT = {
    "id": "INC-002",
    "title": "Database Performance Degradation",
    "description": "Primary database cluster showing high query latency. P99 response times exceeding 5 seconds.",
    "severity": "HIGH",
    "category": "performance_degradation",
    "timestamp": "2026-05-29T11:00:00Z",
}

MOCK_BULLETIN_RESPONSE = {
    "bulletin_id": "BUL-test-001",
    "incident_id": "INC-002",
    "category": "performance_degradation",
    "subject": "[HIGH] Service Performance Degradation - Database - INC-002",
    "body": (
        "Dear Customer, We are experiencing performance degradation affecting the primary database cluster. "
        "Query latency has significantly increased. Our SRE team is actively investigating and applying mitigations. "
        "We expect restoration of normal performance by 13:00 UTC. Thank you for your patience."
    ),
    "severity": "HIGH",
    "template_name": "performance_degradation",
    "generated_at": "2026-05-29T11:05:00Z",
}


class TestAgentNodes:
    """Unit tests for individual graph nodes."""

    def test_validate_incident_node_success(self):
        from agent.nodes import validate_incident
        state = _make_state(VALID_INCIDENT)
        result = validate_incident(state)
        assert result["error"] is None
        assert result["incident"]["severity"] == "HIGH"

    def test_validate_incident_node_failure(self):
        from agent.nodes import validate_incident
        bad_incident = {**VALID_INCIDENT, "severity": "EXTREME"}
        state = _make_state(bad_incident)
        result = validate_incident(state)
        assert result["error"] is not None
        assert result["error_node"] == "validate_incident"

    def test_fetch_template_node(self):
        from agent.nodes import fetch_template
        state = _make_state(VALID_INCIDENT)
        result = fetch_template(state)
        assert result["error"] is None
        assert result["template"] is not None
        assert result["template"]["category"] == "performance_degradation"

    def test_fetch_template_unknown_category(self):
        from agent.nodes import fetch_template
        incident = {**VALID_INCIDENT, "category": "other"}
        state = _make_state(incident)
        result = fetch_template(state)
        assert result["error"] is None
        assert result["template"]["name"] == "default"

    @patch("services.azure_search.AzureSearchClient.search")
    def test_search_historical_node(self, mock_search):
        mock_search.return_value = [{"bulletin_id": "BUL-hist-1", "body": "Past bulletin"}]
        from agent.nodes import search_historical
        state = _make_state(VALID_INCIDENT)
        result = search_historical(state)
        assert result["error"] is None
        assert len(result["historical_context"]) == 1

    @patch("services.azure_search.AzureSearchClient.search")
    def test_search_historical_failure_is_non_fatal(self, mock_search):
        mock_search.side_effect = Exception("Azure unavailable")
        from agent.nodes import search_historical
        state = _make_state(VALID_INCIDENT)
        result = search_historical(state)
        # Should NOT set error — failure is graceful
        assert result["error"] is None
        assert result["historical_context"] == []

    def test_validate_output_node_success(self):
        from agent.nodes import validate_output
        state = _make_state(VALID_INCIDENT, bulletin_draft=MOCK_BULLETIN_RESPONSE)
        result = validate_output(state)
        assert result["error"] is None

    def test_validate_output_node_with_unfilled_placeholder(self):
        from agent.nodes import validate_output
        bad_draft = {**MOCK_BULLETIN_RESPONSE, "body": "Dear {{customer}}, see {{eta}} for details. Extra padding here."}
        state = _make_state(VALID_INCIDENT, bulletin_draft=bad_draft)
        result = validate_output(state)
        assert result["error"] is not None
        assert result["error_node"] == "validate_output"

    @patch("services.bcu_client.BCUClient.send_bulletin")
    def test_send_to_bcu_node_success(self, mock_send):
        mock_send.return_value = {"tracking_id": "TRK-001", "status": "accepted"}
        from agent.nodes import send_to_bcu
        state = _make_state(VALID_INCIDENT, bulletin_draft=MOCK_BULLETIN_RESPONSE)
        result = send_to_bcu(state)
        assert result["error"] is None
        assert result["bcu_response"]["tracking_id"] == "TRK-001"

    @patch("services.bcu_client.BCUClient.send_bulletin")
    def test_send_to_bcu_failure_sets_error(self, mock_send):
        mock_send.side_effect = Exception("BCU unreachable")
        from agent.nodes import send_to_bcu
        state = _make_state(VALID_INCIDENT, bulletin_draft=MOCK_BULLETIN_RESPONSE)
        result = send_to_bcu(state)
        assert result["error"] is not None
        assert result["error_node"] == "send_to_bcu"

    def test_handle_error_node(self):
        from agent.nodes import handle_error
        state = _make_state(VALID_INCIDENT)
        state["error"] = "Something went wrong"
        state["error_node"] = "generate_bulletin"
        result = handle_error(state)
        assert result["bcu_response"]["status"] == "failed"
        assert result["bcu_response"]["failed_at_node"] == "generate_bulletin"


class TestAgentGraph:
    """Full graph execution tests with mocked external services."""

    @patch("services.azure_search.AzureSearchClient.search")
    @patch("services.bcu_client.BCUClient.send_bulletin")
    @patch("agent.nodes._llm_factory")
    def test_full_happy_path(self, mock_llm_factory, mock_bcu_send, mock_search):
        mock_search.return_value = []
        mock_bcu_send.return_value = {"tracking_id": "TRK-happy", "status": "accepted"}

        # Mock LLM returning a valid bulletin JSON
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content=json.dumps(MOCK_BULLETIN_RESPONSE),
            usage_metadata=None,
        )
        mock_llm_factory.return_value = mock_llm

        from agent.graph import run_agent
        result = run_agent(VALID_INCIDENT)

        assert result["error"] is None
        assert result["bcu_response"]["tracking_id"] == "TRK-happy"
        assert result["bulletin_draft"]["bulletin_id"] == "BUL-test-001"

    def test_invalid_incident_routes_to_error(self):
        bad_incident = {**VALID_INCIDENT, "severity": "EXTREME"}
        from agent.graph import run_agent
        result = run_agent(bad_incident)
        assert result["error"] is not None
        assert result["bcu_response"]["status"] == "failed"

    @patch("services.azure_search.AzureSearchClient.search")
    @patch("services.bcu_client.BCUClient.send_bulletin")
    @patch("agent.nodes._llm_factory")
    def test_llm_failure_routes_to_error(self, mock_llm_factory, mock_bcu_send, mock_search):
        mock_search.return_value = []
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("LLM API error")
        mock_llm_factory.return_value = mock_llm

        from agent.graph import run_agent
        result = run_agent(VALID_INCIDENT)

        assert result["error"] is not None
        assert result["bcu_response"]["failed_at_node"] == "generate_bulletin"


class TestTemplateStore:
    def test_loads_network_outage_template(self):
        from services.template_store import TemplateStore
        store = TemplateStore()
        t = store.get_template("network_outage")
        assert t["name"] == "network_outage"
        assert "placeholders" in t

    def test_loads_security_template(self):
        from services.template_store import TemplateStore
        store = TemplateStore()
        t = store.get_template("security_incident")
        assert t["category"] == "security_incident"

    def test_fallback_to_default_for_unknown(self):
        from services.template_store import TemplateStore
        store = TemplateStore()
        t = store.get_template("unknown")
        assert t["name"] == "default"

    def test_list_categories(self):
        from services.template_store import TemplateStore
        store = TemplateStore()
        cats = store.list_categories()
        assert "network_outage" in cats
        assert "security_incident" in cats


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_state(incident: dict, bulletin_draft: dict | None = None) -> dict:
    return {
        "incident": incident,
        "messages": [],
        "template": None,
        "historical_context": None,
        "bulletin_draft": bulletin_draft,
        "error": None,
        "error_node": None,
        "retry_count": 0,
        "bcu_response": None,
        "run_id": "test-run-001",
        "trace_url": None,
    }
