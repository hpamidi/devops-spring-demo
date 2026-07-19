"""Structured logging, Prometheus metrics, and observability setup."""
import logging
import sys
import time
from contextlib import contextmanager
from typing import Generator

from prometheus_client import Counter, Histogram, start_http_server
from pythonjsonlogger import json as jsonlogger

# ── Metrics ──────────────────────────────────────────────────────────────────

NODE_DURATION = Histogram(
    "bulletin_agent_node_duration_seconds",
    "Time spent in each LangGraph node",
    labelnames=["node_name"],
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
)

NODE_ERRORS = Counter(
    "bulletin_agent_node_errors_total",
    "Total errors per LangGraph node",
    labelnames=["node_name", "error_type"],
)

BULLETINS_GENERATED = Counter(
    "bulletin_agent_bulletins_generated_total",
    "Total bulletins successfully generated",
    labelnames=["category", "severity"],
)

BCU_REQUESTS = Counter(
    "bulletin_agent_bcu_requests_total",
    "Total BCU delivery requests",
    labelnames=["status"],
)

AZURE_SEARCH_REQUESTS = Counter(
    "bulletin_agent_azure_search_requests_total",
    "Total Azure AI Search queries",
    labelnames=["status"],
)

TOKEN_USAGE = Counter(
    "bulletin_agent_llm_tokens_total",
    "Total LLM tokens consumed",
    labelnames=["type"],  # prompt | completion
)


# ── Logging ───────────────────────────────────────────────────────────────────

def configure_logging(level: str = "INFO") -> logging.Logger:
    """Configure JSON structured logging for the application."""
    logger = logging.getLogger("bulletin_agent")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = jsonlogger.JsonFormatter(
            fmt="%(asctime)s %(name)s %(levelname)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False

    return logger


def get_logger(name: str = "bulletin_agent") -> logging.Logger:
    return logging.getLogger(name)


# ── Prometheus server ─────────────────────────────────────────────────────────

def start_metrics_server(port: int = 8001) -> None:
    """Start Prometheus metrics HTTP server on a separate port."""
    try:
        start_http_server(port)
        get_logger().info("Prometheus metrics server started", extra={"port": port})
    except OSError:
        # Port already in use (e.g. duplicate startup in tests)
        pass


# ── Node timing context manager ───────────────────────────────────────────────

@contextmanager
def node_span(node_name: str) -> Generator[None, None, None]:
    """Record duration and errors for a LangGraph node execution."""
    logger = get_logger()
    start = time.perf_counter()
    logger.info("Node started", extra={"node": node_name})
    try:
        yield
        duration = time.perf_counter() - start
        NODE_DURATION.labels(node_name=node_name).observe(duration)
        logger.info("Node completed", extra={"node": node_name, "duration_s": round(duration, 3)})
    except Exception as exc:
        duration = time.perf_counter() - start
        NODE_DURATION.labels(node_name=node_name).observe(duration)
        NODE_ERRORS.labels(node_name=node_name, error_type=type(exc).__name__).inc()
        logger.error(
            "Node failed",
            extra={"node": node_name, "error": str(exc), "duration_s": round(duration, 3)},
        )
        raise
