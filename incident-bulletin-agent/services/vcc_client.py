"""VCC (incident source) client with retry logic."""
from __future__ import annotations

from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from agent.monitoring import get_logger

logger = get_logger(__name__)


class VCCClient:
    """Client for fetching incident details from VCC."""

    def __init__(self) -> None:
        from config import get_settings
        s = get_settings()
        self._base_url = s.vcc_base_url.rstrip("/")
        self._headers = {
            "Authorization": f"Bearer {s.vcc_api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        self._timeout = s.http_timeout

    @retry(
        retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
        wait=wait_exponential(multiplier=1, min=2, max=16),
        stop=stop_after_attempt(4),
        reraise=True,
    )
    def get_incident(self, incident_id: str) -> dict[str, Any]:
        """Fetch a single incident by ID from VCC."""
        url = f"{self._base_url}/incidents/{incident_id}"
        with httpx.Client(timeout=self._timeout) as client:
            logger.info("Fetching incident from VCC", extra={"incident_id": incident_id, "url": url})
            response = client.get(url, headers=self._headers)
            response.raise_for_status()
            data = response.json()
            logger.info("Incident fetched", extra={"incident_id": incident_id})
            return data

    @retry(
        retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
        wait=wait_exponential(multiplier=1, min=2, max=16),
        stop=stop_after_attempt(4),
        reraise=True,
    )
    def list_incidents(self, status: str = "open", limit: int = 20) -> list[dict[str, Any]]:
        """List open incidents from VCC."""
        url = f"{self._base_url}/incidents"
        params = {"status": status, "limit": limit}
        with httpx.Client(timeout=self._timeout) as client:
            response = client.get(url, headers=self._headers, params=params)
            response.raise_for_status()
            return response.json().get("incidents", [])
