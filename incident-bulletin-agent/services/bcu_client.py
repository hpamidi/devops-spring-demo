"""BCU (bulletin delivery) client with retry logic."""
from __future__ import annotations

from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from agent.monitoring import get_logger

logger = get_logger(__name__)


class BCUClient:
    """Client for delivering draft bulletins to BCU for distribution."""

    def __init__(self) -> None:
        from config import get_settings
        s = get_settings()
        self._base_url = s.bcu_base_url.rstrip("/")
        self._headers = {
            "Authorization": f"Bearer {s.bcu_api_key}",
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
    def send_bulletin(self, bulletin: dict[str, Any]) -> dict[str, Any]:
        """
        POST a draft bulletin to BCU.
        BCU returns a delivery receipt with tracking ID.
        """
        url = f"{self._base_url}/bulletins/draft"
        with httpx.Client(timeout=self._timeout) as client:
            logger.info(
                "Sending bulletin to BCU",
                extra={"bulletin_id": bulletin.get("bulletin_id"), "url": url},
            )
            response = client.post(url, json=bulletin, headers=self._headers)
            response.raise_for_status()
            receipt = response.json()
            logger.info(
                "BCU accepted bulletin",
                extra={"bulletin_id": bulletin.get("bulletin_id"), "tracking_id": receipt.get("tracking_id")},
            )
            return receipt

    @retry(
        retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
        wait=wait_exponential(multiplier=1, min=2, max=16),
        stop=stop_after_attempt(4),
        reraise=True,
    )
    def get_delivery_status(self, tracking_id: str) -> dict[str, Any]:
        """Check the delivery status of a previously submitted bulletin."""
        url = f"{self._base_url}/bulletins/{tracking_id}/status"
        with httpx.Client(timeout=self._timeout) as client:
            response = client.get(url, headers=self._headers)
            response.raise_for_status()
            return response.json()
