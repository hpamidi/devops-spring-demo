"""LangChain tools exposed to the LLM for template lookup and historical search."""
from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import tool

from services.azure_search import AzureSearchClient
from services.template_store import TemplateStore


@tool
def lookup_template(category: str) -> str:
    """
    Look up the bulletin template for the given incident category.
    Returns the template JSON as a string, or the default template if category not found.
    """
    store = TemplateStore()
    template = store.get_template(category)
    return json.dumps(template, indent=2)


@tool
def search_historical_bulletins(query: str, top: int = 5) -> str:
    """
    Search historical incident bulletins using Azure AI Search.
    Returns a JSON array of the most relevant past bulletins for context.
    Args:
        query: Natural language search query (e.g. 'network outage east region')
        top: Number of results to return (default 5)
    """
    client = AzureSearchClient()
    results = client.search(query=query, top=top)
    return json.dumps(results, indent=2)


def get_tools() -> list[Any]:
    return [lookup_template, search_historical_bulletins]
