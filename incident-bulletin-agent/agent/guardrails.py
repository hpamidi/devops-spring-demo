"""Input validation, content safety, and output guardrails."""
from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field, field_validator

from agent.monitoring import get_logger

logger = get_logger(__name__)

VALID_SEVERITIES = {"CRITICAL", "HIGH", "MEDIUM", "LOW"}
VALID_CATEGORIES = {"network_outage", "security_incident", "performance_degradation", "other"}

# Basic PII / sensitive pattern detection
_PII_PATTERNS = [
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),            # SSN
    re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),  # email
    re.compile(r"\b(?:password|passwd|secret|token|api[_-]?key)\s*[:=]\s*\S+", re.I),
]

# Profanity / harmful content (simplified word list - production should use a service)
_BLOCKED_PATTERNS = [
    re.compile(r"\b(drop\s+table|delete\s+from|insert\s+into)\b", re.I),  # SQL injection
    re.compile(r"<script[\s\S]*?>[\s\S]*?</script>", re.I),               # XSS
]


class IncidentInputSchema(BaseModel):
    id: str = Field(min_length=1, max_length=128)
    title: str = Field(min_length=3, max_length=256)
    description: str = Field(min_length=10, max_length=4096)
    severity: str
    category: str
    timestamp: str = Field(min_length=10)

    @field_validator("severity")
    @classmethod
    def validate_severity(cls, v: str) -> str:
        upper = v.upper()
        if upper not in VALID_SEVERITIES:
            raise ValueError(f"severity must be one of {VALID_SEVERITIES}, got '{v}'")
        return upper

    @field_validator("category")
    @classmethod
    def validate_category(cls, v: str) -> str:
        lower = v.lower()
        if lower not in VALID_CATEGORIES:
            logger.warning("Unknown category, falling back to 'other'", extra={"category": v})
            return "other"
        return lower

    @field_validator("title", "description")
    @classmethod
    def check_injection(cls, v: str) -> str:
        for pattern in _BLOCKED_PATTERNS:
            if pattern.search(v):
                raise ValueError("Input contains potentially harmful content")
        return v


class BulletinOutputSchema(BaseModel):
    bulletin_id: str = Field(min_length=1)
    incident_id: str = Field(min_length=1)
    category: str
    subject: str = Field(min_length=5, max_length=512)
    body: str = Field(min_length=50)
    severity: str
    template_name: str
    generated_at: str

    @field_validator("body")
    @classmethod
    def no_unfilled_placeholders(cls, v: str) -> str:
        remaining = re.findall(r"\{\{[^}]+\}\}", v)
        if remaining:
            raise ValueError(f"Bulletin contains unfilled placeholders: {remaining}")
        return v

    @field_validator("body")
    @classmethod
    def check_pii(cls, v: str) -> str:
        for pattern in _PII_PATTERNS:
            if pattern.search(v):
                logger.warning("Potential PII detected in bulletin body; review before sending")
                break
        return v


def validate_incident_input(data: dict[str, Any]) -> IncidentInputSchema:
    return IncidentInputSchema.model_validate(data)


def validate_bulletin_output(data: dict[str, Any]) -> BulletinOutputSchema:
    return BulletinOutputSchema.model_validate(data)
