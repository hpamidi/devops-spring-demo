"""
FastAPI entry point for the Incident Bulletin Agent.

Endpoints:
  POST /incidents/process   - Process a single incident (VCC webhook)
  GET  /health              - Liveness probe
  GET  /metrics             - Prometheus metrics (also on :8001 scrape port)
"""
from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from agent.graph import run_agent
from agent.monitoring import configure_logging, get_logger, start_metrics_server
from config import get_settings


# ── Startup ───────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)
    start_metrics_server(settings.prometheus_port)
    get_logger().info(
        "Incident Bulletin Agent started",
        extra={"env": settings.app_env, "metrics_port": settings.prometheus_port},
    )
    yield
    get_logger().info("Incident Bulletin Agent shutting down")


app = FastAPI(
    title="Incident Bulletin Agent",
    description="LangGraph agent that converts VCC incidents into BCU-ready bulletins",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Request / Response models ─────────────────────────────────────────────────

class IncidentRequest(BaseModel):
    id: str = Field(description="Unique incident identifier from VCC")
    title: str
    description: str
    severity: str = Field(description="CRITICAL | HIGH | MEDIUM | LOW")
    category: str = Field(description="network_outage | security_incident | performance_degradation | other")
    timestamp: str = Field(description="ISO-8601 detection timestamp")


class BulletinResponse(BaseModel):
    run_id: str
    bulletin_id: str | None
    incident_id: str
    status: str                  # success | failed
    bcu_tracking_id: str | None
    error: str | None


# ── Routes ────────────────────────────────────────────────────────────────────

@app.post("/incidents/process", response_model=BulletinResponse, status_code=202)
async def process_incident(incident: IncidentRequest) -> BulletinResponse:
    """
    Webhook endpoint for VCC incident events.
    Runs the full bulletin generation pipeline and delivers to BCU.
    """
    logger = get_logger()
    run_id = str(uuid.uuid4())
    logger.info(
        "Received incident",
        extra={"run_id": run_id, "incident_id": incident.id, "severity": incident.severity},
    )

    result = run_agent(incident.model_dump(), run_id=run_id)

    draft = result.get("bulletin_draft")
    bcu = result.get("bcu_response") or {}

    return BulletinResponse(
        run_id=run_id,
        bulletin_id=draft["bulletin_id"] if draft else None,
        incident_id=incident.id,
        status="failed" if result.get("error") else "success",
        bcu_tracking_id=bcu.get("tracking_id"),
        error=result.get("error"),
    )


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "incident-bulletin-agent"}


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics() -> PlainTextResponse:
    """Prometheus metrics scrape endpoint (also available on dedicated port)."""
    return PlainTextResponse(
        content=generate_latest().decode("utf-8"),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    get_logger().error("Unhandled exception", extra={"path": request.url.path, "error": str(exc)})
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


# ── Dev runner ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=settings.app_env == "development")
