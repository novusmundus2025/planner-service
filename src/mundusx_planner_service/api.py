from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

from . import __version__
from .models import PlanRequest
from .planner import plan_request

app = FastAPI(title="MundusX Planner Service", version=__version__)


class PlanRequestBody(BaseModel):
    request_id: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    model: str | None = None
    classification: dict[str, Any] = Field(default_factory=dict)
    available_capability_summary: dict[str, Any] = Field(default_factory=dict)
    policy: dict[str, Any] = Field(default_factory=dict)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@app.post("/v1/plan")
def create_plan(body: PlanRequestBody) -> dict[str, Any]:
    request = PlanRequest(
        request_id=body.request_id,
        prompt=body.prompt,
        model=body.model,
        classification=body.classification,
        available_capability_summary=body.available_capability_summary,
        policy=body.policy,
    )
    return plan_request(request).to_dict()
