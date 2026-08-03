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


class CapacityClass(StrEnum):
    MICRO = "micro"
    STANDARD = "standard"
    PERFORMANCE = "performance"
    HEAVY = "heavy"
    SYNTHESIS = "synthesis"
    SERVER = "server"


class ValidationLevel(StrEnum):
    STRUCTURAL = "structural"
    SYNTAX = "syntax"
    COMPILE = "compile"
    TEST = "test"


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
    expected_artifact_types: list[str] = field(default_factory=list)
    artifact_targets: list[str] = field(default_factory=list)
    max_input_artifacts: int | None = None
    recommended_max_tokens: int | None = None
    minimum_max_tokens: int | None = None
    unavailable_timeout_seconds: int = 60
    on_unavailable: str = "fail_with_degradation"
    minimum_capacity_class: CapacityClass = CapacityClass.MICRO
    recommended_capacity_class: CapacityClass = CapacityClass.STANDARD
    context_budget_tokens: int = 2048
    expected_artifact_count: int = 1
    expected_artifact_bytes: int = 65536
    model_quality_floor: str = "baseline"
    required_tools: list[str] = field(default_factory=list)
    requires_repository: bool = False
    requires_compile: bool = False
    requires_tests: bool = False
    validation_level: ValidationLevel = ValidationLevel.STRUCTURAL
    allowed_parallelism: int = 1
    reducer_credibility: str = "standard"
    synthesizer_credibility: str = "standard"

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
            "expected_artifact_types": list(self.expected_artifact_types),
            "artifact_targets": list(self.artifact_targets),
            "minimum_capacity_class": self.minimum_capacity_class.value,
            "recommended_capacity_class": self.recommended_capacity_class.value,
            "context_budget_tokens": self.context_budget_tokens,
            "expected_artifact_count": self.expected_artifact_count,
            "expected_artifact_bytes": self.expected_artifact_bytes,
            "model_quality_floor": self.model_quality_floor,
            "required_tools": list(self.required_tools),
            "requires_repository": self.requires_repository,
            "requires_compile": self.requires_compile,
            "requires_tests": self.requires_tests,
            "validation_level": self.validation_level.value,
            "allowed_parallelism": self.allowed_parallelism,
            "reducer_credibility": self.reducer_credibility,
            "synthesizer_credibility": self.synthesizer_credibility,
        }
        if self.max_input_artifacts is not None:
            payload["max_input_artifacts"] = self.max_input_artifacts
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
