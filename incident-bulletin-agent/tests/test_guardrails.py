"""Tests for input/output guardrails."""
import pytest

from agent.guardrails import (
    validate_incident_input,
    validate_bulletin_output,
    IncidentInputSchema,
    BulletinOutputSchema,
)
from pydantic import ValidationError


VALID_INCIDENT = {
    "id": "INC-001",
    "title": "East Coast Network Outage",
    "description": "Multiple routers in the east coast data center are unreachable causing packet loss.",
    "severity": "HIGH",
    "category": "network_outage",
    "timestamp": "2026-05-29T10:00:00Z",
}

VALID_BULLETIN = {
    "bulletin_id": "BUL-abc-123",
    "incident_id": "INC-001",
    "category": "network_outage",
    "subject": "[HIGH] Network Outage - East Coast - INC-001",
    "body": "Dear Customer, We are experiencing a network outage in the East Coast region. "
            "Our team is actively working to resolve this. We expect restoration by 14:00 UTC. "
            "Thank you for your patience.",
    "severity": "HIGH",
    "template_name": "network_outage",
    "generated_at": "2026-05-29T10:05:00Z",
}


class TestIncidentInputValidation:
    def test_valid_incident_passes(self):
        result = validate_incident_input(VALID_INCIDENT)
        assert result.id == "INC-001"
        assert result.severity == "HIGH"
        assert result.category == "network_outage"

    def test_severity_normalized_to_uppercase(self):
        data = {**VALID_INCIDENT, "severity": "high"}
        result = validate_incident_input(data)
        assert result.severity == "HIGH"

    def test_unknown_category_falls_back_to_other(self):
        data = {**VALID_INCIDENT, "category": "unknown_category"}
        result = validate_incident_input(data)
        assert result.category == "other"

    def test_invalid_severity_raises(self):
        data = {**VALID_INCIDENT, "severity": "EXTREME"}
        with pytest.raises(ValidationError):
            validate_incident_input(data)

    def test_empty_id_raises(self):
        data = {**VALID_INCIDENT, "id": ""}
        with pytest.raises(ValidationError):
            validate_incident_input(data)

    def test_short_title_raises(self):
        data = {**VALID_INCIDENT, "title": "AB"}
        with pytest.raises(ValidationError):
            validate_incident_input(data)

    def test_short_description_raises(self):
        data = {**VALID_INCIDENT, "description": "Too short"}
        with pytest.raises(ValidationError):
            validate_incident_input(data)

    def test_sql_injection_blocked(self):
        data = {**VALID_INCIDENT, "description": "DROP TABLE incidents; -- hack attempt here with enough chars"}
        with pytest.raises(ValidationError):
            validate_incident_input(data)

    def test_xss_blocked(self):
        data = {**VALID_INCIDENT, "title": "<script>alert('xss')</script> outage affecting services now"}
        with pytest.raises(ValidationError):
            validate_incident_input(data)


class TestBulletinOutputValidation:
    def test_valid_bulletin_passes(self):
        result = validate_bulletin_output(VALID_BULLETIN)
        assert result.bulletin_id == "BUL-abc-123"
        assert result.severity == "HIGH"

    def test_unfilled_placeholder_raises(self):
        data = {**VALID_BULLETIN, "body": "Dear {{customer}}, something happened {{eta}} ago."}
        with pytest.raises(ValidationError, match="unfilled placeholders"):
            validate_bulletin_output(data)

    def test_short_body_raises(self):
        data = {**VALID_BULLETIN, "body": "Too short"}
        with pytest.raises(ValidationError):
            validate_bulletin_output(data)

    def test_short_subject_raises(self):
        data = {**VALID_BULLETIN, "subject": "Hi"}
        with pytest.raises(ValidationError):
            validate_bulletin_output(data)
