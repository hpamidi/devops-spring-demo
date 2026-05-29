"""Azure AI Search client for historical bulletin retrieval."""
from __future__ import annotations

from typing import Any

from tenacity import retry, retry_if_exception_type, retry_if_not_exception_type, stop_after_attempt, wait_exponential

from agent.monitoring import get_logger

logger = get_logger(__name__)


class AzureSearchClient:
    """Wrapper around Azure Cognitive Search for semantic bulletin retrieval."""

    def __init__(self) -> None:
        from config import get_settings
        s = get_settings()
        self._endpoint = s.azure_search_endpoint
        self._key = s.azure_search_key
        self._index = s.azure_search_index
        self._client = None

    def _get_client(self):
        if self._client is None:
            if not self._endpoint or not self._key:
                raise ValueError(
                    "AZURE_SEARCH_ENDPOINT and AZURE_SEARCH_KEY must be configured"
                )
            from azure.search.documents import SearchClient
            from azure.core.credentials import AzureKeyCredential
            self._client = SearchClient(
                endpoint=self._endpoint,
                index_name=self._index,
                credential=AzureKeyCredential(self._key),
            )
        return self._client

    @retry(
        retry=retry_if_not_exception_type((ValueError, TypeError)),
        wait=wait_exponential(multiplier=1, min=2, max=16),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def search(self, query: str, top: int = 5) -> list[dict[str, Any]]:
        """
        Full-text + semantic search over historical bulletins.
        Returns a list of matching bulletin documents.
        """
        client = self._get_client()
        logger.info("Searching Azure AI Search", extra={"query": query[:100], "top": top})

        results = client.search(
            search_text=query,
            top=top,
            select=["bulletin_id", "incident_id", "category", "subject", "body", "severity", "generated_at"],
            query_type="semantic",
            semantic_configuration_name="default",
        )
        docs = []
        for result in results:
            doc = {k: v for k, v in result.items() if not k.startswith("@")}
            doc["_score"] = result.get("@search.score", 0)
            docs.append(doc)

        logger.info("Azure Search returned results", extra={"count": len(docs)})
        return docs

    def index_bulletin(self, bulletin: dict[str, Any]) -> None:
        """Index a completed bulletin for future historical retrieval."""
        client = self._get_client()
        from azure.search.documents.models import IndexingResult
        results: list[IndexingResult] = client.upload_documents(documents=[bulletin])
        for r in results:
            if not r.succeeded:
                logger.warning("Failed to index bulletin", extra={"key": r.key, "error": r.error_message})
