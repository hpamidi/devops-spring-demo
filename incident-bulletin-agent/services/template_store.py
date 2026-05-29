"""Template management: load and serve bulletin templates by incident category."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent.monitoring import get_logger

logger = get_logger(__name__)

_CATEGORY_MAP = {
    "network_outage": "network_outage.json",
    "security_incident": "security_incident.json",
    "performance_degradation": "performance_degradation.json",
    "other": "default.json",
}


class TemplateStore:
    """Loads templates from the templates/ directory."""

    def __init__(self, templates_dir: Path | None = None) -> None:
        if templates_dir is None:
            from config import get_settings
            templates_dir = get_settings().templates_dir
        # Resolve relative to the package root (incident-bulletin-agent/)
        if not templates_dir.is_absolute():
            templates_dir = Path(__file__).parent.parent / templates_dir
        self._dir = templates_dir
        self._cache: dict[str, dict[str, Any]] = {}

    def get_template(self, category: str) -> dict[str, Any]:
        """Return the template dict for the given category, falling back to default."""
        filename = _CATEGORY_MAP.get(category.lower(), "default.json")
        if filename in self._cache:
            return self._cache[filename]

        path = self._dir / filename
        if not path.exists():
            logger.warning("Template file missing, using default", extra={"path": str(path)})
            path = self._dir / "default.json"

        with path.open() as fh:
            template = json.load(fh)

        self._cache[filename] = template
        logger.info("Template loaded", extra={"name": template.get("name"), "category": category})
        return template

    def list_categories(self) -> list[str]:
        return list(_CATEGORY_MAP.keys())
