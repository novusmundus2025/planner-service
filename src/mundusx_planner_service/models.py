from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class PlannerProvider(StrEnum):
    LANGGRAPH = "langgraph"
    DETERMINISTIC = "deterministic"


class PlannerStatus(StrEnum):
    PLANNED = "planned"
    DEGRADED = "degraded"


class TaskType(StrEnum):
    CHAT = "chat"
    CODING = "coding"
    DOCUMENT = "document"
    INFERENCE = "inference"


class NodeRole(StrEnum):
    CHAT = "chat"
    CODING = "coding"
    VISION = "vision"
    EMBEDDING = "embedding"
    TOOL_USE = "tool_use"
    CHUNK_ANALYSIS = "chunk_analysis"
    REDUCER = "reducer"
    SYNTHESIZER = "synthesizer"
    BATCH = "batch"


@dataclass(frozen=True)
class PlanRequest:
    request_id: str
    prompt: str
    model: str | None = None
    classification: dict[str, Any] = field(default_factory=dict)
    available_capability_summary: dict[str, Any] = field(default_factory=dict)
    policy: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PlanStep:
    id: str
    name: str
    responsibility: str
    required_output: str
    reason: str
    depends_on: list[str] = field(default_factory=list)
    preferred_roles: list[NodeRole] = field(default_factory=list)
    required_role: NodeRole | None = None
    fallback_roles: list[NodeRole] = field(default_factory=list)
    recommended_max_tokens: int | None = None
    minimum_max_tokens: int | None = None
    unavailable_timeout_seconds: int = 60
    on_unavailable: str = "fail_with_degradation"

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "id": self.id,
            "name": self.name,
            "responsibility": self.responsibility,
            "required_output": self.required_output,
            "reason": self.reason,
            "depends_on": list(self.depends_on),
            "preferred_roles": [role.value for role in self.preferred_roles],
            "fallback_roles": [role.value for role in self.fallback_roles],
            "unavailable_timeout_seconds": self.unavailable_timeout_seconds,
            "on_unavailable": self.on_unavailable,
        }
        if self.recommended_max_tokens is not None:
            payload["recommended_max_tokens"] = self.recommended_max_tokens
        if self.minimum_max_tokens is not None:
            payload["minimum_max_tokens"] = self.minimum_max_tokens
        if self.required_role:
            payload["required_role"] = self.required_role.value
        return payload


@dataclass(frozen=True)
class PlanResponse:
    request_id: str
    planner_provider: PlannerProvider
    planner_status: PlannerStatus
    plan: dict[str, Any]
    graph: dict[str, Any]
    scheduling_requirements: dict[str, Any]
    degraded_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "request_id": self.request_id,
            "planner_provider": self.planner_provider.value,
            "planner_status": self.planner_status.value,
            "plan": self.plan,
            "graph": self.graph,
            "scheduling_requirements": self.scheduling_requirements,
        }
        if self.degraded_reason:
            payload["degraded_reason"] = self.degraded_reason
        return payload
